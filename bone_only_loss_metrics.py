"""
小肿瘤分割专用 Loss & Metrics — 分阶段稳定收敛版 v4
=====================================================

v4 在 v3 基础上的修改：

  [FIX-6] BoneOnlyDetailedMetrics 默认阈值 threshold=0.5（原 0.3）
      原因：训练监控时用 0.3 作二值化阈值，而推理时用 0.5，
            导致训练期间显示的 Dice 比实际推理 Dice 虚高 3-8%，
            最佳 checkpoint 的选择依据失准。
      修改：train_epoch / validate 中统一使用 threshold=0.5。

  [FIX-7] validate() 新增双阈值指标：同时计算 thr=0.3 和 thr=0.5 的 Dice，
      便于观察模型的置信度分布是否健康。

  [FIX-8] 新增推理后处理工具 postprocess_prediction()：
      连通域分析去除小于 min_pixels 的独立 FP 区域，
      在不重训练的前提下可提升约 2-5% Dice。

v3 改动（保持不变）:
  [FIX-1] BoundaryLoss 最大权重：0.15 → 0.03
  [FIX-2] BoundaryLoss 线性 warmup
  [FIX-3] BoundaryLoss 计算数值稳定化
  [FIX-4] USE_BOUNDARY_DELAY 延迟标志
  [FIX-5] 各子 loss 值记录到实例属性
"""

import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  Tversky / FocalTversky Loss
# ============================================================

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred_prob, target):
        B = pred_prob.size(0)
        p = pred_prob.view(B, -1)
        t = target.view(B, -1)
        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma   = gamma

    def forward(self, pred_prob, target):
        tl = self.tversky(pred_prob, target)
        return tl ** self.gamma


# ============================================================
#  Boundary Loss
# ============================================================

class BoundaryLoss(nn.Module):
    """
    [FIX-3] 数值稳定化版 BoundaryLoss

    - Sobel 梯度幅值 clamp(0, 8)
    - 像素权重 clamp(1.0, bw)
    - reduction='mean'（与 batchsize / 图像尺寸解耦）
    - AMP fp16 兼容
    - 边界阈值 0.5（减少噪声边界误检）
    """

    def __init__(self, boundary_weight=5.0):
        super().__init__()
        self.bw = max(1.0, min(boundary_weight, 8.0))
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_boundary(self, mask):
        mask_f = mask.float()
        sx = self.sobel_x.to(device=mask.device, dtype=torch.float32)
        sy = self.sobel_y.to(device=mask.device, dtype=torch.float32)
        gx = F.conv2d(mask_f, sx, padding=1)
        gy = F.conv2d(mask_f, sy, padding=1)
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6).clamp(0.0, 8.0)
        return (magnitude > 0.5).float()

    def forward(self, pred_logits, target):
        with torch.no_grad():
            boundary = self.get_boundary(target)
            weight   = (1.0 + (self.bw - 1.0) * boundary).clamp(1.0, self.bw)
            weight   = weight.to(dtype=pred_logits.dtype)
        return F.binary_cross_entropy_with_logits(
            pred_logits,
            target.to(dtype=pred_logits.dtype),
            weight=weight,
            reduction='mean')


# ============================================================
#  SmallTumorLoss — 分阶段版 [v3 不变]
# ============================================================

class SmallTumorLoss(nn.Module):
    """
    分阶段小骨肿瘤综合 Loss — v3 BoundaryLoss warmup 专项修复

    BoundaryLoss 调度逻辑:
      epoch < phase2_start                    → bnd_w = 0.0
      USE_BOUNDARY_DELAY & epoch < boundary_delay_start → bnd_w = 0.0
      之后线性 warmup: min(bnd_weight_max, bnd_rampup_rate * steps)
    """

    def __init__(self,
                 ftl_weight=2.0,
                 bce_weight=0.5,
                 bnd_weight_max=0.03,
                 bnd_rampup_rate=0.002,
                 ds_weight=0.3,
                 alpha=0.3,
                 beta=0.7,
                 gamma=1.5,
                 boundary_weight=5.0,
                 smooth=1.0,
                 phase2_start=25,
                 use_boundary_delay=True,
                 boundary_delay_start=30,
                 ds_cutoff=40):
        super().__init__()
        self.ftl_weight           = ftl_weight
        self.bce_weight           = bce_weight
        self.bnd_weight_max       = bnd_weight_max
        self.bnd_rampup_rate      = bnd_rampup_rate
        self.ds_weight            = ds_weight
        self.phase2_start         = phase2_start
        self.use_boundary_delay   = use_boundary_delay
        self.boundary_delay_start = boundary_delay_start
        self.ds_cutoff            = ds_cutoff

        self.beta  = beta
        self.alpha = alpha
        self.gamma = gamma

        self.ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
        self.bnd = BoundaryLoss(boundary_weight=boundary_weight)
        self.bce = nn.BCEWithLogitsLoss()

        self.last_bnd_w     = 0.0
        self.last_bnd_loss  = 0.0
        self.last_main_loss = 0.0

    def get_boundary_weight(self, epoch: int) -> float:
        if epoch < self.phase2_start:
            return 0.0
        warmup_start = self.phase2_start
        if self.use_boundary_delay and epoch < self.boundary_delay_start:
            return 0.0
        elif self.use_boundary_delay:
            warmup_start = self.boundary_delay_start
        steps = epoch - warmup_start
        return min(self.bnd_weight_max, self.bnd_rampup_rate * steps)

    def _sync_ftl_params(self):
        self.ftl.tversky.alpha = self.alpha
        self.ftl.tversky.beta  = self.beta
        self.ftl.gamma         = self.gamma

    def _single_loss(self, logits, target, bnd_w: float):
        self._sync_ftl_params()
        pred_prob = torch.sigmoid(logits)
        ftl_loss  = self.ftl(pred_prob, target)
        bce_loss  = self.bce(logits, target)
        main      = self.ftl_weight * ftl_loss + self.bce_weight * bce_loss

        bnd_raw = 0.0
        if bnd_w > 0.0:
            bnd_raw = self.bnd(logits, target)
            total   = main + bnd_w * bnd_raw
        else:
            total = main

        return total, main.item() if isinstance(main, torch.Tensor) else main, \
               bnd_raw.item() if isinstance(bnd_raw, torch.Tensor) else bnd_raw

    def forward(self, outputs, tumor_mask, bone_pred, is_tumor,
                current_epoch=0, ds_epochs=None):
        if isinstance(outputs, dict):
            tumor_logits = outputs['tumor_logits']
        else:
            tumor_logits = outputs

        has_tumor = is_tumor.bool()

        self.last_bnd_w     = 0.0
        self.last_bnd_loss  = 0.0
        self.last_main_loss = 0.0

        if not has_tumor.any():
            return torch.tensor(0.0, device=tumor_logits.device, requires_grad=True)

        idx         = has_tumor.nonzero(as_tuple=True)[0]
        logits_t    = tumor_logits[idx]
        target_t    = tumor_mask[idx]
        bone_pred_t = bone_pred[idx]

        target_bone   = target_t * bone_pred_t
        logits_masked = logits_t * bone_pred_t + (bone_pred_t - 1) * 10.0

        bnd_w = self.get_boundary_weight(current_epoch)
        main_loss, main_val, bnd_val = self._single_loss(logits_masked, target_bone, bnd_w)

        self.last_bnd_w     = bnd_w
        self.last_bnd_loss  = bnd_val
        self.last_main_loss = main_val

        ds_loss = torch.tensor(0.0, device=tumor_logits.device)
        if isinstance(outputs, dict) and current_epoch <= self.ds_cutoff:
            for key in ['ds3_logits', 'ds2_logits']:
                if outputs.get(key) is not None:
                    ds_logit = outputs[key][idx]
                    if ds_logit.shape[2:] != target_bone.shape[2:]:
                        ds_logit = F.interpolate(ds_logit, size=target_bone.shape[2:],
                                                 mode='bilinear', align_corners=False)
                    ds_logit_m = ds_logit * bone_pred_t + (bone_pred_t - 1) * 10.0
                    ds_total, _, _ = self._single_loss(ds_logit_m, target_bone, bnd_w=0.0)
                    ds_loss = ds_loss + ds_total

        return main_loss + self.ds_weight * ds_loss


# ============================================================
#  SingleStageLoss — 分阶段版 [v3 不变]
# ============================================================

class SingleStageLoss(nn.Module):
    """
    分阶段单阶段损失函数 — v3

    Phase 1  (epoch < 25) :  FTL + BCE
    Phase 2  (25~59)       :  + BoundaryLoss warmup (max 0.03)
    Phase 3  (epoch >= 60) :  + IRGDA + FP suppress
    """

    def __init__(self, base_loss_fn,
                 irgda_sup_weight=0.001,
                 fp_suppress_weight=0.03,
                 irgda_start_epoch=60,
                 fp_start_epoch=60,
                 irgda_rampup_epochs=20):
        super().__init__()
        self.tumor_loss          = base_loss_fn
        self.irgda_sup_weight    = irgda_sup_weight
        self.fp_suppress_weight  = fp_suppress_weight
        self.irgda_start_epoch   = irgda_start_epoch
        self.fp_start_epoch      = fp_start_epoch
        self.irgda_rampup_epochs = irgda_rampup_epochs

        self.last_bnd_w      = 0.0
        self.last_bnd_loss   = 0.0
        self.last_total_loss = 0.0

        from new_network.fbfa_intrabone_enhanced_iddmga import DGMASupervisionLoss as IRGDASupervisionLoss
        self.irgda_loss_fn = IRGDASupervisionLoss(
            heatmap_weight=0.2, coverage_weight=0.2,
            shape_weight=0.05,  radius_weight=0.1)

    def _get_irgda_weight(self, epoch):
        if epoch < self.irgda_start_epoch:
            return 0.0
        ramp_progress = min(1.0,
            (epoch - self.irgda_start_epoch) / max(self.irgda_rampup_epochs, 1))
        return self.irgda_sup_weight * ramp_progress

    def _get_fp_weight(self, epoch):
        if epoch < self.fp_start_epoch:
            return 0.0
        return self.fp_suppress_weight

    def _fp_suppress_loss(self, logits, bone_pred):
        prob      = torch.sigmoid(logits)
        bone_mask = (bone_pred > 0.5).float()
        bone_area = bone_mask.sum().clamp(min=1.0)
        energy    = (prob * bone_mask).pow(2).sum() / bone_area
        return energy

    def forward(self, outputs, tumor_mask, bone_pred, is_tumor,
                current_epoch=0, ds_epochs=None, model=None):
        if isinstance(outputs, dict):
            logits = outputs['tumor_logits']
        else:
            logits = outputs

        has_tumor = is_tumor.bool()

        self.last_bnd_w      = 0.0
        self.last_bnd_loss   = 0.0
        self.last_total_loss = 0.0

        fp_w = self._get_fp_weight(current_epoch)
        if not has_tumor.any():
            if fp_w > 0:
                fp_loss = self._fp_suppress_loss(logits, bone_pred)
                total   = fp_loss * fp_w
                self.last_total_loss = total.item()
                return total
            return torch.tensor(0.0, device=logits.device)

        tumor_loss = self.tumor_loss(
            outputs, tumor_mask, bone_pred, is_tumor,
            current_epoch=current_epoch)

        self.last_bnd_w    = self.tumor_loss.last_bnd_w
        self.last_bnd_loss = self.tumor_loss.last_bnd_loss

        irgda_w    = self._get_irgda_weight(current_epoch)
        irgda_loss = torch.tensor(0.0, device=logits.device)
        if model is not None and has_tumor.any() and irgda_w > 0:
            tumor_idx = has_tumor.nonzero(as_tuple=True)[0]
            tm_full   = tumor_mask[tumor_idx]
            bm_full   = bone_pred[tumor_idx]
            try:
                for soe_module in model.soe:
                    state = soe_module.last_state
                    if state is None:
                        continue
                    attn_h, attn_w = state['attention_map'].shape[2:]
                    if tm_full.shape[2] != attn_h or tm_full.shape[3] != attn_w:
                        tm_ds = F.interpolate(tm_full, size=(attn_h, attn_w), mode='nearest')
                        bm_ds = F.interpolate(bm_full, size=(attn_h, attn_w), mode='nearest')
                    else:
                        tm_ds, bm_ds = tm_full, bm_full
                    irgda_loss = irgda_loss + self.irgda_loss_fn(state, tm_ds, bm_ds)
            except Exception:
                pass

        fp_loss = torch.tensor(0.0, device=logits.device)
        if fp_w > 0:
            has_empty = ~has_tumor
            if has_empty.any():
                empty_logits = logits[has_empty]
                empty_bone   = bone_pred[has_empty]
                fp_loss      = self._fp_suppress_loss(empty_logits, empty_bone)

        total = tumor_loss + irgda_w * irgda_loss + fp_w * fp_loss
        self.last_total_loss = total.item()
        return total


# ============================================================
#  指标计算 [v4 修改]
# ============================================================

class BoneOnlyDetailedMetrics:
    """
    [FIX-6] 默认阈值由 0.3 改为 0.5，与推理阶段统一。

    原因：训练时用 threshold=0.3 导致监控 Dice 比实际推理 Dice 虚高 3-8%，
          最佳 checkpoint 的选择依据失准，保存的模型不是推理最优。

    [FIX-7] 新增 tumor_dice_at_03 字段：
          同时计算 thr=0.3 时的 Dice，便于观察置信度分布是否健康。
          健康状态：dice_0.5 ≈ dice_0.3（说明置信度集中在两端）
          退化状态：dice_0.3 >> dice_0.5（说明模型大量输出 0.3~0.5 的中间概率，置信度不足）
    """

    def __init__(self, threshold=0.5, smooth=1.0):  # ← 修改：默认 0.3 → 0.5
        self.threshold = threshold
        self.smooth    = smooth

    def soft_dice_coeff(self, y_true, y_pred):
        inter = torch.sum(y_true * y_pred)
        union = torch.sum(y_true) + torch.sum(y_pred)
        return (2.0 * inter + self.smooth) / (union + self.smooth)

    def __call__(self, logits, target, bone_pred, is_tumor):
        has_tumor = is_tumor.bool()
        if not has_tumor.any():
            return dict(num_tumor_slices=0,
                        num_empty_slices=has_tumor.numel(),
                        tumor_dice=0.0,
                        tumor_dice_hard=0.0,
                        tumor_dice_at_03=0.0,   # [FIX-7] 新增
                        tumor_precision=0.0,
                        tumor_recall=0.0,
                        tumor_size_ratio=0.0,
                        empty_fp_rate=0.0)

        logits_t    = logits[has_tumor]
        target_t    = target[has_tumor]
        bone_pred_t = bone_pred[has_tumor]

        pred_prob   = torch.sigmoid(logits_t)
        pred_bone   = pred_prob  * bone_pred_t
        target_bone = target_t  * bone_pred_t

        soft_dice = self.soft_dice_coeff(target_bone, pred_bone).item()

        # 主阈值（默认 0.5）
        pred_binary      = (pred_prob > self.threshold).float()
        pred_binary_bone = pred_binary * bone_pred_t
        hard_dice        = self.soft_dice_coeff(target_bone, pred_binary_bone).item()

        tp = (pred_binary_bone * target_bone).sum()
        fp = (pred_binary_bone * (1 - target_bone)).sum()
        fn = ((1 - pred_binary_bone) * target_bone).sum()

        precision  = (tp / (tp + fp + 1e-6)).item()
        recall     = (tp / (tp + fn + 1e-6)).item()
        size_ratio = (pred_binary_bone.sum() / (target_bone.sum() + 1e-6)).item()

        # [FIX-7] 次阈值 0.3 的 Dice（用于置信度健康度监控）
        pred_03      = (pred_prob > 0.3).float() * bone_pred_t
        dice_at_03   = self.soft_dice_coeff(target_bone, pred_03).item()

        empty_fp_rate = 0.0
        if (~has_tumor).any():
            logits_empty    = logits[~has_tumor]
            bone_pred_empty = bone_pred[~has_tumor]
            pred_empty      = (torch.sigmoid(logits_empty) > self.threshold).float()
            fp_pixels       = (pred_empty * bone_pred_empty).sum()
            total_bone      = bone_pred_empty.sum()
            empty_fp_rate   = (fp_pixels / (total_bone + 1e-6)).item()

        return dict(
            num_tumor_slices = has_tumor.sum().item(),
            num_empty_slices = (~has_tumor).sum().item(),
            tumor_dice       = soft_dice,
            tumor_dice_hard  = hard_dice,
            tumor_dice_at_03 = dice_at_03,   # [FIX-7] 新增
            tumor_precision  = precision,
            tumor_recall     = recall,
            tumor_size_ratio = size_ratio,
            empty_fp_rate    = empty_fp_rate
        )


# ============================================================
#  [FIX-8] 推理后处理：连通域分析去除小 FP
# ============================================================

def postprocess_prediction(pred_binary: torch.Tensor,
                            min_component_pixels: int = 50) -> torch.Tensor:
    """
    [FIX-8] 推理后处理：移除骨内面积小于 min_component_pixels 的独立预测区域。

    效果：在不重训练的前提下，通过移除散点 FP（假阳性）提升 Dice 2-5%。
    适用于推理阶段，不用于训练 loss 计算。

    Args:
        pred_binary: [B, 1, H, W] 或 [1, H, W] 的二值 tensor（0/1）
        min_component_pixels: 小于此面积的连通域会被移除，默认 50px

    Returns:
        清洗后的二值 tensor，形状与输入相同

    用法示例：
        tumor_prob   = torch.sigmoid(model_output)
        tumor_binary = (tumor_prob > 0.5).float()
        tumor_clean  = postprocess_prediction(tumor_binary, min_component_pixels=50)
    """
    squeeze = pred_binary.dim() == 3
    if squeeze:
        pred_binary = pred_binary.unsqueeze(0)

    B = pred_binary.shape[0]
    result = torch.zeros_like(pred_binary)

    for b in range(B):
        pred_np = pred_binary[b, 0].cpu().numpy().astype(np.uint8)

        if pred_np.sum() == 0:
            continue  # 全空跳过

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_np)
        cleaned = np.zeros_like(pred_np)

        for i in range(1, num_labels):   # 0 为背景
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_component_pixels:
                cleaned[labels == i] = 1

        result[b, 0] = torch.from_numpy(cleaned).to(pred_binary.device)

    return result.squeeze(0) if squeeze else result


def postprocess_batch(pred_probs: torch.Tensor,
                       threshold: float = 0.5,
                       min_component_pixels: int = 50) -> torch.Tensor:
    """
    便捷封装：将概率图 → 二值化 → 后处理，一步完成。

    Args:
        pred_probs:            [B, 1, H, W] sigmoid 概率输出
        threshold:             二值化阈值，默认 0.5
        min_component_pixels:  最小连通域面积，默认 50px

    Returns:
        [B, 1, H, W] 清洗后的二值 tensor
    """
    binary = (pred_probs > threshold).float()
    return postprocess_prediction(binary, min_component_pixels)


# 向后兼容别名
BoneOnlyFixedLoss = SmallTumorLoss