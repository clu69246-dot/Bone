"""
小肿瘤分割专用 Loss & Metrics

针对小骨肿瘤的核心问题:
  [问题1] Dice Loss 对小目标不敏感
          - 小肿瘤像素极少(几十个)，被大量背景淹没
          - Dice 改善 0.01 在大目标≈ 1000像素，小目标≈ 5像素

  [问题2] 漏检(低Recall)比误检(低Precision)代价更大
          - 骨肿瘤漏检是临床大忌

  [改进1] Tversky Loss: 不对称惩罚 FN(漏检) > FP(误检)
          TL = TP / (TP + α·FP + β·FN)   β > α → 重惩漏检
          推荐: α=0.3, β=0.7

  [改进2] Focal Tversky Loss: 对难样本额外加权
          FTL = (1 - TL)^γ   γ∈[1,3]

  [改进3] Boundary Loss: 监督肿瘤边界，对小目标轮廓贡献大

  [改进4] 深度监督分别计算 Loss，浅层监督用较小权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
#  Tversky Loss (小肿瘤高召回率)
# ============================================================

class TverskyLoss(nn.Module):
    """
    Tversky Loss — 专为小目标高召回设计

    α < β → FN 惩罚 > FP 惩罚 → 模型倾向于多预测 → 减少漏检

    推荐配置:
      α=0.3, β=0.7  — 标准小目标配置
      α=0.2, β=0.8  — 极端追求召回时
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha  = alpha   # FP 权重
        self.beta   = beta    # FN 权重 (β > α → 重惩漏检)
        self.smooth = smooth

    def forward(self, pred_prob, target):
        """
        pred_prob: (B,1,H,W) sigmoid 后的概率
        target:    (B,1,H,W) 0/1 标签
        """
        B = pred_prob.size(0)
        p = pred_prob.view(B, -1)
        t = target.view(B, -1)

        tp = (p * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return (1 - tversky).mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss — 对难样本(小肿瘤)额外加权

    FTL = (1 - TI)^γ
    γ > 1 → 越难的样本权重越大
    推荐 γ=1.5~2.0
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.5, smooth=1.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma   = gamma

    def forward(self, pred_prob, target):
        tl = self.tversky(pred_prob, target)
        return tl ** self.gamma


# ============================================================
#  Boundary Loss (边界监督，小肿瘤边缘精度提升)
# ============================================================

class BoundaryLoss(nn.Module):
    """
    Boundary Loss — 专门监督肿瘤边界

    原理: 用 Sobel 算子提取 GT 边界，加权 BCE
    对小肿瘤: 边界像素占全部像素比例更高，所以边界 loss 对小目标更有效
    """
    def __init__(self, boundary_weight=5.0):
        super().__init__()
        self.bw = boundary_weight
        # Sobel 核提取边界
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def get_boundary(self, mask):
        """提取边界，强制 float32，同时对齐 device（兼容 CUDA + AMP）"""
        # mask 可能是 cuda half，sobel buffer 可能在 cpu
        # 统一：mask→float32，sobel→同 device 同 dtype
        mask_f = mask.float()
        sx = self.sobel_x.to(device=mask.device, dtype=torch.float32)
        sy = self.sobel_y.to(device=mask.device, dtype=torch.float32)
        gx = F.conv2d(mask_f, sx, padding=1)
        gy = F.conv2d(mask_f, sy, padding=1)
        boundary = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        return (boundary > 0.1).float()

    def forward(self, pred_logits, target):
        """
        pred_logits: (B,1,H,W) 未经 sigmoid
        target:      (B,1,H,W) 0/1
        """
        with torch.no_grad():
            boundary = self.get_boundary(target)       # (B,1,H,W) float32
            weight   = 1.0 + (self.bw - 1.0) * boundary
            # weight 转回与 pred_logits 相同 dtype，兼容 AMP
            weight   = weight.to(dtype=pred_logits.dtype)

        bce = F.binary_cross_entropy_with_logits(
            pred_logits, target.to(dtype=pred_logits.dtype),
            weight=weight, reduction='mean')
        return bce


# ============================================================
#  小肿瘤综合 Loss
# ============================================================

class SmallTumorLoss(nn.Module):
    """
    小骨肿瘤分割综合 Loss

    = w_ftl × FocalTversky
    + w_bce × WeightedBCE
    + w_bnd × BoundaryLoss
    + w_ds  × DeepSupervision(辅助头)

    设计原则:
    - FocalTversky 主导 (高召回 + 难样本聚焦)
    - BoundaryLoss 补充轮廓精度
    - BCE 保持数值稳定性
    - DS 监督中间层帮助梯度传播

    对比原 BoneOnlyFixedLoss:
    - 删除 Area Penalty (对小肿瘤无意义，体积太小)
    - Tversky 替代 Dice (β=0.7 减少漏检)
    - 新增 BoundaryLoss (小目标轮廓关键)
    - 新增 Focal 机制 (小样本难样本聚焦)
    """

    def __init__(self,
                 ftl_weight=2.0,       # Focal Tversky 权重 (主导)
                 bce_weight=0.5,       # BCE 权重
                 bnd_weight=0.3,       # Boundary Loss 权重
                 ds_weight=0.3,        # Deep Supervision 权重
                 alpha=0.3,            # Tversky FP 权重
                 beta=0.7,             # Tversky FN 权重 (大 → 减少漏检)
                 gamma=1.5,            # Focal 指数
                 boundary_weight=5.0,  # 边界像素额外权重
                 smooth=1.0):
        super().__init__()
        self.ftl_weight = ftl_weight
        self.bce_weight = bce_weight
        self.bnd_weight = bnd_weight
        self.ds_weight  = ds_weight

        self.ftl = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
        self.bnd = BoundaryLoss(boundary_weight=boundary_weight)
        self.bce = nn.BCEWithLogitsLoss()

    def _single_loss(self, logits, target):
        """单个输出的 loss"""
        pred_prob = torch.sigmoid(logits)
        ftl_loss  = self.ftl(pred_prob, target)
        bce_loss  = self.bce(logits, target)
        bnd_loss  = self.bnd(logits, target)
        return (self.ftl_weight * ftl_loss
                + self.bce_weight * bce_loss
                + self.bnd_weight * bnd_loss)

    def forward(self, outputs, tumor_mask, bone_pred, is_tumor,
                current_epoch=0, ds_epochs=0):
        """
        outputs:    dict {'tumor_logits', 'ds3_logits', 'ds2_logits', ...}
                    或直接 tensor (eval 时)
        tumor_mask: (B,1,H,W)
        bone_pred:  (B,1,H,W)
        is_tumor:   (B,) bool
        """
        # 取主输出 logits
        if isinstance(outputs, dict):
            tumor_logits = outputs['tumor_logits']
        else:
            tumor_logits = outputs

        has_tumor = is_tumor.bool()

        if not has_tumor.any():
            return torch.tensor(0.0, device=tumor_logits.device, requires_grad=True)

        # 只处理有肿瘤的样本
        idx = has_tumor.nonzero(as_tuple=True)[0]
        logits_t    = tumor_logits[idx]
        target_t    = tumor_mask[idx]
        bone_pred_t = bone_pred[idx]

        # 骨区域内约束
        target_bone = target_t * bone_pred_t

        # 对 logits 做骨区域掩码 (骨外强制为大负值，不参与梯度)
        logits_masked = logits_t * bone_pred_t + (bone_pred_t - 1) * 10.0

        # 主 loss
        main_loss = self._single_loss(logits_masked, target_bone)

        # 深度监督 (浅层辅助)
        ds_loss = torch.tensor(0.0, device=tumor_logits.device)
        if isinstance(outputs, dict) and current_epoch <= ds_epochs:
            for key in ['ds3_logits', 'ds2_logits']:
                if outputs.get(key) is not None:
                    ds_logit = outputs[key][idx]
                    # 插值到目标尺寸
                    if ds_logit.shape[2:] != target_bone.shape[2:]:
                        ds_logit = F.interpolate(ds_logit, size=target_bone.shape[2:],
                                                 mode='bilinear', align_corners=False)
                    ds_logit_m = ds_logit * bone_pred_t + (bone_pred_t - 1) * 10.0
                    ds_loss = ds_loss + self._single_loss(ds_logit_m, target_bone)

        total = main_loss + self.ds_weight * ds_loss
        return total


# ============================================================
#  指标计算 (保持兼容)
# ============================================================

class BoneOnlyDetailedMetrics:
    """详细指标计算"""

    def __init__(self, threshold=0.5, smooth=1.0):
        self.threshold = threshold
        self.smooth    = smooth

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = self.smooth
        inter  = torch.sum(y_true * y_pred)
        union  = torch.sum(y_true) + torch.sum(y_pred)
        return (2.0 * inter + smooth) / (union + smooth)

    def __call__(self, logits, target, bone_pred, is_tumor):
        has_tumor = is_tumor.bool()

        if not has_tumor.any():
            return dict(num_tumor_slices=0,
                        num_empty_slices=has_tumor.numel(),
                        tumor_dice=0.0, tumor_dice_hard=0.0,
                        tumor_precision=0.0, tumor_recall=0.0,
                        tumor_size_ratio=0.0, empty_fp_rate=0.0)

        logits_t    = logits[has_tumor]
        target_t    = target[has_tumor]
        bone_pred_t = bone_pred[has_tumor]

        pred_prob  = torch.sigmoid(logits_t)
        pred_bone  = pred_prob  * bone_pred_t
        target_bone = target_t * bone_pred_t

        soft_dice = self.soft_dice_coeff(target_bone, pred_bone).item()

        pred_binary      = (pred_prob > self.threshold).float()
        pred_binary_bone = pred_binary * bone_pred_t

        hard_dice  = self.soft_dice_coeff(target_bone, pred_binary_bone).item()

        tp = (pred_binary_bone * target_bone).sum()
        fp = (pred_binary_bone * (1 - target_bone)).sum()
        fn = ((1 - pred_binary_bone) * target_bone).sum()

        precision   = (tp / (tp + fp + 1e-6)).item()
        recall      = (tp / (tp + fn + 1e-6)).item()
        size_ratio  = (pred_binary_bone.sum() / (target_bone.sum() + 1e-6)).item()

        empty_fp_rate = 0.0
        if (~has_tumor).any():
            logits_empty    = logits[~has_tumor]
            bone_pred_empty = bone_pred[~has_tumor]
            pred_empty      = (torch.sigmoid(logits_empty) > self.threshold).float()
            fp_pixels       = (pred_empty * bone_pred_empty).sum()
            total_bone      = bone_pred_empty.sum()
            empty_fp_rate   = (fp_pixels / (total_bone + 1e-6)).item()

        return dict(num_tumor_slices=has_tumor.sum().item(),
                    num_empty_slices=(~has_tumor).sum().item(),
                    tumor_dice=soft_dice,
                    tumor_dice_hard=hard_dice,
                    tumor_precision=precision,
                    tumor_recall=recall,
                    tumor_size_ratio=size_ratio,
                    empty_fp_rate=empty_fp_rate)


# ============================================================
#  向后兼容别名
# ============================================================

BoneOnlyFixedLoss = SmallTumorLoss


def apply_bone_pred_to_logits(logits, bone_pred):
    return logits * bone_pred


def debug_bone_pred_application(logits, bone_pred, tumor_mask=None):
    return {
        'logits_shape':    logits.shape,
        'bone_pred_shape': bone_pred.shape,
        'bone_pred_sum':   bone_pred.sum().item(),
    }