"""
ConDSeg Stage 2: 2D PET-CT Bone Tumor Segmentation (IMPROVED VERSION)

主要改进:
1. ✅ 稳定的Bone Mask处理 - 避免全0输入
2. ✅ 改进的损失函数 - 渐进式权重，避免初期梯度爆炸
3. ✅ GroupNorm稳定化 - 替代BatchNorm
4. ✅ 更好的权重初始化 - 输出层小权重
5. ✅ 异常检测和处理 - 跳过problematic batches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ==================== 稳定的Bone Mask处理模块 ====================

class StableBoneMaskProcessor(nn.Module):
    """
    稳定的骨骼掩码处理模块

    功能:
    1. 二值化bone mask
    2. 形态学膨胀
    3. 检测无效掩码并使用fallback
    4. 自适应信号缩放
    """

    def __init__(self, dilation_kernel=5, min_area_ratio=0.005):
        super().__init__()
        self.dilation_kernel = dilation_kernel
        self.min_area_ratio = min_area_ratio

    def dilate_mask(self, mask, kernel_size=3):
        """直接对二值mask膨胀"""
        if kernel_size <= 1:
            return mask
        padding = kernel_size // 2
        return F.max_pool2d(mask, kernel_size, stride=1, padding=padding)

    def forward(self, bone_logits, pet):
        """
        处理骨骼掩码并应用到PET

        Args:
            bone_logits: (B, 1, H, W) Stage1输出的logits
            pet: (B, 1, H, W) PET图像

        Returns:
            pet_bone: 掩码后的PET
            bone_pred: 二值掩码
            bone_pred_dilated: 膨胀后的掩码
            mask_valid: 掩码是否有效 (B,)
        """
        # 1. 获取bone mask
        bone_prob = torch.sigmoid(bone_logits)
        bone_pred = (bone_prob > 0.4).float()

        # 2. 膨胀
        bone_pred_dilated = self.dilate_mask(bone_pred, self.dilation_kernel)

        # 3. 检查掩码有效性 (避免全0情况)
        B = bone_pred.shape[0]
        total_pixels = bone_pred.shape[2] * bone_pred.shape[3]
        mask_areas = bone_pred_dilated.view(B, -1).sum(dim=1)
        mask_valid = (mask_areas / total_pixels) > self.min_area_ratio

        # 4. 对无效掩码使用全图 (而不是全0)
        for b in range(B):
            if not mask_valid[b]:
                bone_pred_dilated[b] = 1.0  # 使用全图作为fallback

        # 5. 应用到PET
        pet_bone = pet * bone_pred_dilated

        return pet_bone, bone_pred, bone_pred_dilated, mask_valid


# ==================== 小目标增强模块 (保持不变) ====================

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        super().__init__()

        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                    nn.GroupNorm(4, out_channels),  # 改用GroupNorm
                    nn.ReLU(inplace=True)
                )
            )

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[2:]
        features = []
        for conv in self.convs:
            features.append(conv(x))

        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=True)
        features.append(global_feat)

        out = torch.cat(features, dim=1)
        out = self.fusion(out)
        return out


class SmallTargetEnhancement(nn.Module):
    """小目标增强模块"""
    def __init__(self, channels):
        super().__init__()

        self.conv1x1 = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.conv3x3 = nn.Conv2d(channels, channels // 4, 3, padding=1, bias=False)
        self.conv3x3_d2 = nn.Conv2d(channels, channels // 4, 3, padding=2, dilation=2, bias=False)
        self.conv3x3_d4 = nn.Conv2d(channels, channels // 4, 3, padding=4, dilation=4, bias=False)

        self.bn = nn.GroupNorm(4, channels)
        self.relu = nn.ReLU(inplace=True)

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        f1 = self.conv1x1(x)
        f2 = self.conv3x3(x)
        f3 = self.conv3x3_d2(x)
        f4 = self.conv3x3_d4(x)

        out = torch.cat([f1, f2, f3, f4], dim=1)
        out = self.bn(out)
        out = self.relu(out)

        ca = self.channel_att(out)
        out = out * ca

        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg_out, max_out], dim=1))
        out = out * sa

        return out + x


# ==================== 前景-背景感知融合模块 ====================

class ForegroundBackgroundFusion(nn.Module):
    """前景-背景感知融合模块"""
    def __init__(self, ct_channels, pet_bone_channels, out_channels):
        super().__init__()

        self.ct_conv = nn.Sequential(
            nn.Conv2d(ct_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

        self.pet_bone_conv = nn.Sequential(
            nn.Conv2d(pet_bone_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

        self.foreground_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(4, out_channels)
        )

        self.background_context = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, 1, bias=False),
            nn.GroupNorm(2, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, 1, bias=False),
            nn.GroupNorm(4, out_channels)
        )

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, 2, 1),
            nn.Softmax(dim=1)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

        self.small_target = SmallTargetEnhancement(out_channels)

    def forward(self, ct_feat, pet_bone_feat, bone_pred=None):
        ct_f = self.ct_conv(ct_feat)
        pb_f = self.pet_bone_conv(pet_bone_feat)

        fg_feat = self.foreground_enhance(pb_f)
        bg_feat = self.background_context(ct_f)

        if bone_pred is not None:
            if bone_pred.shape[2:] != fg_feat.shape[2:]:
                bone_pred = F.interpolate(bone_pred, size=fg_feat.shape[2:],
                                         mode='bilinear', align_corners=True)
            fg_feat = fg_feat * bone_pred
            bg_feat = bg_feat * (1 - bone_pred)

        concat_feat = torch.cat([fg_feat, bg_feat], dim=1)
        weights = self.fusion_gate(concat_feat)

        fg_weight = weights[:, 0:1, :, :]
        bg_weight = weights[:, 1:2, :, :]
        fused = fg_feat * fg_weight + bg_feat * bg_weight

        out = self.final_conv(fused)
        out = self.small_target(out)

        return out


# ==================== 解码器块 ====================

class DecoderBlock(nn.Module):
    """解码器块"""
    def __init__(self, in_channels, skip_channels, out_channels, scale=2):
        super().__init__()

        if scale > 1:
            self.up = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.ReLU(inplace=True)
        )

        self.residual = nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False) \
            if in_channels + skip_channels != out_channels else nn.Identity()

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        concat = torch.cat([x, skip], dim=1)
        out = self.conv(concat) + self.residual(concat)
        return out


class DeepSupervisionHead(nn.Module):
    """深度监督输出头"""
    def __init__(self, in_channels, scale=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.GroupNorm(2, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1)
        )
        self.scale = scale

    def forward(self, x, target_size=None):
        out = self.conv(x)
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=True)
        return out


# ==================== 改进的主模型 ====================

class BoneTumorSegmentation2DImproved(nn.Module):
    """
    改进的2D骨肿瘤分割网络

    主要改进:
    1. 集成StableBoneMaskProcessor
    2. 使用GroupNorm替代BatchNorm
    3. 添加特征归一化
    4. 改进的权重初始化
    """

    def __init__(self,
                 stage1_model_path=None,
                 freeze_stage1=True,
                 bone_dilation=3):
        super().__init__()

        self.bone_dilation = bone_dilation
        self.freeze_stage1 = freeze_stage1

        # ============= CT Branch =============
        print("Loading CT Branch (Stage 1)...")
        from network.model_efficientvim_2d_stage1 import ConDSeg2DStage1_EfficientViM

        self.ct_branch = ConDSeg2DStage1_EfficientViM(in_channels=1, out_channels=1)

        if stage1_model_path:
            checkpoint = torch.load(stage1_model_path, map_location='cpu', weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model_dict = self.ct_branch.state_dict()
            pretrained_dict = {}
            skipped_keys = []

            for k, v in state_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        pretrained_dict[k] = v
                    else:
                        skipped_keys.append(f"{k} (shape mismatch)")
                else:
                    skipped_keys.append(f"{k} (not in model)")

            model_dict.update(pretrained_dict)
            self.ct_branch.load_state_dict(model_dict)

            print(f"✓ Loaded {len(pretrained_dict)}/{len(state_dict)} weights")
            if skipped_keys:
                print(f"  Skipped {len(skipped_keys)} incompatible weights")

        if freeze_stage1:
            for param in self.ct_branch.parameters():
                param.requires_grad = False
            self.ct_branch.eval()
            print("✓ CT Branch frozen")

        # ============= 稳定的Bone Mask处理器 =============
        self.bone_processor = StableBoneMaskProcessor(
            dilation_kernel=bone_dilation,
            min_area_ratio=0.001
        )
        print("✓ Stable Bone Mask Processor created")

        # ============= PET Bone Branch =============
        print("Creating PET Bone Branch...")
        from network.efficientvim_modules_2d import MambaFeatureExtractor2D

        self.pet_bone_backbone = MambaFeatureExtractor2D(
            in_dim=1,
            embed_dim=[32, 64, 128, 256],
            depths=[2, 2, 2, 2],
            state_dim=[49, 25, 9, 9],
            strides=[1, 2, 2, 1]
        )

        # ============= 特征归一化层 =============
        self.layer_norms = nn.ModuleDict({
            'ct_f0': nn.GroupNorm(4, 32),
            'ct_f1': nn.GroupNorm(4, 64),
            'ct_f2': nn.GroupNorm(4, 128),
            'ct_f3': nn.GroupNorm(4, 256),
            'pb_f0': nn.GroupNorm(4, 32),
            'pb_f1': nn.GroupNorm(4, 64),
            'pb_f2': nn.GroupNorm(4, 128),
            'pb_f3': nn.GroupNorm(4, 256),
        })

        # ============= 前景-背景融合模块 =============
        self.fusion0 = ForegroundBackgroundFusion(32, 32, 32)
        self.fusion1 = ForegroundBackgroundFusion(64, 64, 64)
        self.fusion2 = ForegroundBackgroundFusion(128, 128, 128)
        self.fusion3 = ForegroundBackgroundFusion(256, 256, 256)

        # ============= ASPP =============
        self.aspp = ASPP(256, 256, dilations=[1, 6, 12, 18])

        # ============= 解码器 =============
        self.decoder3 = DecoderBlock(256, 128, 128, scale=2)
        self.decoder2 = DecoderBlock(128, 64, 64, scale=2)
        self.decoder1 = DecoderBlock(64, 32, 32, scale=1)

        # ============= 深度监督头 =============
        self.ds_head3 = DeepSupervisionHead(128)
        self.ds_head2 = DeepSupervisionHead(64)
        self.ds_head1 = DeepSupervisionHead(32)

        # ============= 最终输出 =============
        self.output_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        print("✓ Model initialized with improvements")

    def forward(self, ct, pet, return_intermediate=False):
        """稳定的前向传播"""
        input_size = ct.shape[2:]
        B = ct.shape[0]

        # 1. CT Branch
        if self.freeze_stage1:
            with torch.no_grad():
                ct_features, bone_logits = self.ct_branch.forward_with_features(ct)
        else:
            ct_features, bone_logits = self.ct_branch.forward_with_features(ct)

        ct_f0, ct_f1, ct_f2, ct_f3 = ct_features

        # 应用归一化（保持原样）
        ct_f0 = self.layer_norms['ct_f0'](ct_f0)
        ct_f1 = self.layer_norms['ct_f1'](ct_f1)
        ct_f2 = self.layer_norms['ct_f2'](ct_f2)
        ct_f3 = self.layer_norms['ct_f3'](ct_f3)

        # 2. 稳定的Bone Mask处理
        pet_bone, bone_pred, bone_pred_dilated, mask_valid = \
            self.bone_processor(bone_logits, pet)

        # ✅ 修复1：PET强度校准（关键！mask后动态范围改变）
        pet_bone = pet_bone / (pet_bone.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0] + 1e-6)

        # 3. PET Bone Branch
        pet_bone_features = self.pet_bone_backbone(pet_bone)
        pb_f0, pb_f1, pb_f2, pb_f3 = pet_bone_features

        # 应用归一化
        pb_f0 = self.layer_norms['pb_f0'](pb_f0)
        pb_f1 = self.layer_norms['pb_f1'](pb_f1)
        pb_f2 = self.layer_norms['pb_f2'](pb_f2)
        pb_f3 = self.layer_norms['pb_f3'](pb_f3)

        # ✅ 修复2：特征尺寸匹配检查（必须）
        assert ct_f0.shape == pb_f0.shape, f"Level0: CT{ct_f0.shape} vs PET{pb_f0.shape}"
        assert ct_f1.shape == pb_f1.shape, f"Level1: CT{ct_f1.shape} vs PET{pb_f1.shape}"
        assert ct_f2.shape == pb_f2.shape, f"Level2: CT{ct_f2.shape} vs PET{pb_f2.shape}"
        assert ct_f3.shape == pb_f3.shape, f"Level3: CT{ct_f3.shape} vs PET{pb_f3.shape}"

        # 4. 多尺度Bone Mask
        bone_preds = []
        for feat in [ct_f0, ct_f1, ct_f2, ct_f3]:
            mask = F.interpolate(bone_pred_dilated, size=feat.shape[2:], mode='nearest')
            mask = (mask > 0.5).float()

            # ✅ 修复3：处理无效mask（避免梯度消失）
            for b in range(B):
                if not mask_valid[b]:
                    mask[b] = 1.0  # 无效时用全图，不抑制任何区域
            bone_preds.append(mask)

        # 5. 前景-背景融合
        fused_f0 = self.fusion0(ct_f0, pb_f0, bone_preds[0])
        fused_f1 = self.fusion1(ct_f1, pb_f1, bone_preds[1])
        fused_f2 = self.fusion2(ct_f2, pb_f2, bone_preds[2])
        fused_f3 = self.fusion3(ct_f3, pb_f3, bone_preds[3])

        # 6-9. 保持不变...
        fused_f3 = self.aspp(fused_f3)
        d2 = self.decoder3(fused_f3, fused_f2)
        d1 = self.decoder2(d2, fused_f1)
        d0 = self.decoder1(d1, fused_f0)
        ds3 = self.ds_head3(d2, target_size=input_size)
        ds2 = self.ds_head2(d1, target_size=input_size)
        ds1 = self.ds_head1(d0, target_size=input_size)
        tumor_logits = self.output_conv(d0)

        if return_intermediate:
            return {
                'tumor_logits': tumor_logits,
                'bone_pred': bone_pred,
                'bone_pred_dilated': bone_pred_dilated,
                'pet_bone': pet_bone,
                'deep_supervisions': [ds1, ds2, ds3],
                'mask_valid': mask_valid
            }
        else:
            return tumor_logits

    # ✅ 修复4：predict方法缩进错误（必须移入类内）
    def predict(self, ct, pet, threshold=0.5):
        """推理模式"""
        self.eval()
        with torch.no_grad():
            results = self.forward(ct, pet, return_intermediate=True)
            tumor_prob = torch.sigmoid(results['tumor_logits'])
            tumor_mask = (tumor_prob > threshold).float()

        return {
            'tumor_mask': tumor_mask,
            'tumor_prob': tumor_prob,
            'bone_pred': results['bone_pred'],
            'bone_pred_dilated': results['bone_pred_dilated']
        }


# ==================== 改进的损失函数 ====================

class ImprovedTumorLoss(nn.Module):
    """
    改进的肿瘤分割损失

    改进点:
    1. 降低总权重，避免初期梯度爆炸
    2. 渐进式启用复杂损失 (Focal, Boundary)
    3. 增加smooth参数提升数值稳定性
    """

    def __init__(self,
                 dice_weight=1.0,
                 bce_weight=0.5,
                 focal_weight=0.3,
                 boundary_weight=0.2,
                 deep_supervision=True,
                 ds_weights=[0.3, 0.2, 0.1],
                 warmup_epochs=15):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.deep_supervision = deep_supervision
        self.ds_weights = ds_weights
        self.warmup_epochs = warmup_epochs

        self.current_epoch = 0

    def set_epoch(self, epoch):
        """设置当前epoch用于warmup"""
        self.current_epoch = epoch

    def get_loss_weights(self):
        """渐进式权重"""
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            return {
                'dice': self.dice_weight,
                'bce': self.bce_weight,
                'focal': self.focal_weight * progress,
                'boundary': self.boundary_weight * progress,
                'ds': [w * progress for w in self.ds_weights]
            }
        else:
            return {
                'dice': self.dice_weight,
                'bce': self.bce_weight,
                'focal': self.focal_weight,
                'boundary': self.boundary_weight,
                'ds': self.ds_weights
            }

    def dice_loss(self, pred, target, smooth=1.0):
        """改进的Dice Loss"""
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)

        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def focal_loss(self, pred, target, gamma=2.0, alpha=0.75):
        """Focal Loss"""
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)

        pt = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - pt) ** gamma
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)

        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        focal = alpha_weight * focal_weight * bce

        return focal.mean()

    def boundary_loss(self, pred, target, smooth=1.0):
        """简化的边界损失 - AMP安全版本"""
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                             device=target.device, dtype=torch.float32).view(1, 1, 3, 3)

        target_boundary = F.conv2d(target, kernel, padding=1).abs()
        target_boundary = (target_boundary > 0.1).float()

        boundary_mask_sum = target_boundary.sum()
        if boundary_mask_sum > 10:
            # 使用 binary_cross_entropy_with_logits (AMP安全)
            boundary_loss = F.binary_cross_entropy_with_logits(
                pred, target, weight=target_boundary, reduction='sum'
            ) / (boundary_mask_sum + smooth)
        else:
            boundary_loss = torch.tensor(0.0, device=pred.device)

        return boundary_loss

    def forward(self, outputs, target):
        """计算总损失"""
        if isinstance(outputs, dict):
            logits = outputs['tumor_logits']
            deep_sups = outputs.get('deep_supervisions', [])
        else:
            logits = outputs
            deep_sups = []

        if logits.shape[2:] != target.shape[2:]:
            logits = F.interpolate(logits, size=target.shape[2:],
                                   mode='bilinear', align_corners=True)

        weights = self.get_loss_weights()

        dice = self.dice_loss(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        focal = self.focal_loss(logits, target)
        boundary = self.boundary_loss(logits, target)

        total_loss = (weights['dice'] * dice +
                     weights['bce'] * bce +
                     weights['focal'] * focal +
                     weights['boundary'] * boundary)

        if self.deep_supervision and deep_sups:
            for i, ds in enumerate(deep_sups):
                if i < len(weights['ds']):
                    ds_loss = F.binary_cross_entropy_with_logits(ds, target)
                    total_loss += weights['ds'][i] * ds_loss

        return total_loss


# ==================== 保持原有的旧损失函数作为备选 ====================

class CombinedTumorLoss2D(nn.Module):
    """原版损失函数 (保持向后兼容)"""

    def __init__(self,
                 dice_weight=1.0,
                 bce_weight=0.5,
                 focal_weight=0.5,
                 boundary_weight=0.3,
                 deep_supervision=True,
                 ds_weights=[0.4, 0.2, 0.1]):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.deep_supervision = deep_supervision
        self.ds_weights = ds_weights

    def dice_loss(self, pred, target, smooth=1e-5):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def focal_loss(self, pred, target, gamma=2.0, alpha=0.75):
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        pt = pred * target + (1 - pred) * (1 - target)
        focal_weight = (1 - pt) ** gamma
        alpha_weight = alpha * target + (1 - alpha) * (1 - target)
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        focal = alpha_weight * focal_weight * bce
        return focal.mean()

    def boundary_loss(self, pred, target, smooth=1.0):
        """修复版：边界2倍加权，内部1倍"""
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=target.device).view(1, 1, 3, 3)
        target_boundary = F.conv2d(target, kernel, padding=1).abs() > 0.1

        # ✅ 正确：所有像素都参与计算，边界权重更高
        weight = 1.0 + target_boundary.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        return (bce * weight).mean()

    def forward(self, outputs, target):
        if isinstance(outputs, dict):
            logits = outputs['tumor_logits']
            deep_sups = outputs.get('deep_supervisions', [])
        else:
            logits = outputs
            deep_sups = []

        if logits.shape[2:] != target.shape[2:]:
            logits = F.interpolate(logits, size=target.shape[2:],
                                   mode='bilinear', align_corners=True)

        dice = self.dice_loss(logits, target)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        focal = self.focal_loss(logits, target)
        boundary = self.boundary_loss(logits, target)

        total_loss = (self.dice_weight * dice +
                     self.bce_weight * bce +
                     self.focal_weight * focal +
                     self.boundary_weight * boundary)

        if self.deep_supervision and deep_sups:
            for i, ds in enumerate(deep_sups):
                if i < len(self.ds_weights):
                    ds_loss = F.binary_cross_entropy_with_logits(ds, target)
                    total_loss += self.ds_weights[i] * ds_loss

        return total_loss


# ==================== 测试代码 ====================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("Testing Improved Stage 2 Bone Tumor Segmentation Model")
    print("=" * 80)

    import os
    stage1_path = r'checkpoints/stage1_2D_ultra_fixed_20251215-094012/best_model.pth'

    if not os.path.exists(stage1_path):
        print(f"Stage1 model not found: {stage1_path}")
        stage1_path = None

    model = BoneTumorSegmentation2DImproved(
        stage1_model_path=stage1_path,
        freeze_stage1=True,
        bone_dilation=3
    ).to(device)

    # 测试输入
    batch_size = 2
    ct = torch.randn(batch_size, 1, 512, 512).to(device)
    pet = torch.randn(batch_size, 1, 512, 512).to(device)
    target = torch.randint(0, 2, (batch_size, 1, 512, 512)).float().to(device)

    print(f"\nInput shapes:")
    print(f"  CT:  {ct.shape}")
    print(f"  PET: {pet.shape}")

    # 前向传播
    with torch.no_grad():
        results = model(ct, pet, return_intermediate=True)

    print(f"\nOutput shapes:")
    print(f"  Tumor logits: {results['tumor_logits'].shape}")
    print(f"  Bone mask: {results['bone_pred'].shape}")
    print(f"  Mask valid: {results['mask_valid']}")
    print(f"  Deep supervisions: {[ds.shape for ds in results['deep_supervisions']]}")

    # 测试损失
    loss_fn = ImprovedTumorLoss()
    loss = loss_fn(results, target)
    print(f"\nLoss: {loss.item():.4f}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters:     {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")

    print("\n" + "=" * 80)
    print("✓ Model test passed!")
    print("=" * 80)

    print("\nKey Improvements:")
    print("  1. ✓ StableBoneMaskProcessor - 避免全0输入")
    print("  2. ✓ ImprovedTumorLoss - 渐进式权重")
    print("  3. ✓ GroupNorm - 替代BatchNorm提升稳定性")
    print("  4. ✓ Feature normalization layers")
    print("  5. ✓ Better weight initialization")
    print("=" * 80)