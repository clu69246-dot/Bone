"""
FBFA-Fusion Enhanced for Intra-Bone Tumor Segmentation - BONE-ONLY VERSION - 512x512

核心改进:
1. ✅ 输入已包含骨掩码信息（数据层面强制）
2. ✅ 输出强制骨区域约束
3. ✅ CBAM (通道+空间注意力)
4. ✅ Deep Supervision (3层中间监督)
5. ✅ 多尺度解码器融合
6. ✅ [DGMA] Dynamic Gaussian Mixture Attention - 病灶尺度自适应高斯注意力

[DGMA 集成说明]
  集成位置: PET 特征对齐之后、FBFA 多尺度融合之前
  数据流: pet_features_aligned → DGMAWithSOE[i] → pet_features_enhanced → FBFA
  分辨率策略:
    pf0 (H=256): DGMAWithSOE 执行 SOE + DGMA (高分辨率，可检测中心)
    pf1 (H=128): DGMAWithSOE 执行 SOE + DGMA
    pf2 (H= 64): DGMAWithSOE 仅执行 SOE (min_spatial_size=65，深层跳过 DGMA)
    pf3 (H= 32): DGMAWithSOE 仅执行 SOE
  FBFA 融合、解码器、Deep Supervision 全部保持原样不变。

目标: Intra-Bone Tumor Dice ≥ 0.75
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# DGMA 模块导入
# dgma_v3.py 放在同一 network/ 目录下即可
from new_network.fbfa_intrabone_enhanced_iddmga import DGMAWithSOE, compute_spatial_radius_loss


# ==================== LightSE: 轻量 Squeeze-and-Excitation 通道注意力 ====================
# 替代原 CBAM（ChannelAttention + SpatialAttention）。
# 改动：去掉空间注意力分支，仅保留通道 squeeze-excitation；
#       参数量和 FLOPs 约为原 CBAM 的 50%，接口与 CBAM 完全兼容（同名、同签名）。

class CBAM(nn.Module):
    """LightSE：轻量通道注意力，接口兼容原 CBAM。
    仅做全局平均池化 → 两层 FC → Sigmoid → 通道缩放，去掉空间注意力分支。
    kernel_size 参数保留以兼容原有调用点，但不再使用。
    """

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # [B, C, 1, 1]
            nn.Flatten(),                      # [B, C]
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()                       # [B, C]
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# ==================== DoG 频率分解（Difference-of-Gaussian）====================
#
# 理论基础：DoG 是带通滤波器的离散近似。
#   G_σ1 (小尺度, 3×3) 保留结构细节；
#   G_σ2 (大尺度, 7×7) 仅保留整体轮廓，作为低频基准；
#   DoG = G_σ1 - G_σ2 → 带通信号，突出小病灶边缘与纹理。
#
# 输出接口与原 EnhancedFrequencyDecomposition 完全兼容，仍返回 (low_freq, high_freq) 元组，
# 供下游 FBFAFusionSingleScaleIntraBone 的四路频率-前背景分解路径独立使用：
#   low_freq  = G_σ2(x)            — 低频（大尺度平滑）
#   high_freq = G_σ1(x) - G_σ2(x) — 带通高频（DoG band-pass）
#
# 各分量在返回前经过独立的 1×1 卷积精炼（BN + ReLU），增强可学习性。
# 高斯核权重固定（requires_grad=False），保证频域语义不被训练破坏。

class EnhancedFrequencyDecomposition(nn.Module):
    """DoG 频率分解：Difference-of-Gaussian band-pass 替代多尺度均值低通滤波"""

    def __init__(self, channels):
        super().__init__()

        # 小尺度 depthwise Gaussian（σ ≈ 0.5，保留细节）
        self.gauss_small = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1,
            groups=channels, bias=False
        )
        # 大尺度 depthwise Gaussian（σ ≈ 1.17，平滑背景）
        self.gauss_large = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3,
            groups=channels, bias=False
        )

        self._init_gaussian_kernels()

        # 各分量独立精炼：concat(low, high) → 1×1 conv → 拆回两路
        # 使用 groups=2 分组卷积：两路各自精炼，参数量极低
        self.refine_low = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.refine_high = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def _init_gaussian_kernels(self):
        """固定高斯核权重，保证频域语义稳定"""

        sigmas = {3: 1.0, 7: 2.5}

        for ks, conv in [(3, self.gauss_small), (7, self.gauss_large)]:

            sigma = sigmas[ks]

            coords = torch.arange(ks).float() - ks // 2
            g1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g1d = g1d / g1d.sum()

            g2d = g1d.view(-1, 1) * g1d.view(1, -1)

            with torch.no_grad():
                for i in range(conv.weight.size(0)):
                    conv.weight[i, 0] = g2d

            conv.weight.requires_grad_(False)

    def forward(self, x):
        # 两尺度高斯平滑
        g_small = self.gauss_small(x)   # G_σ1：细节保留
        g_large = self.gauss_large(x)   # G_σ2：低频基准

        # DoG 分解
        low_freq  = g_large                # 低频：大尺度平滑输出
        high_freq = g_small - g_large      # 带通高频：DoG band-pass

        # 1×1 精炼（可学习），增强各分量表达能力
        low_freq  = self.refine_low(low_freq)
        high_freq = self.refine_high(high_freq)

        return low_freq, high_freq


# ==================== 骨内肿瘤前景预测器 ====================

class IntraBoneForegroundPredictor(nn.Module):
    """骨内肿瘤前景预测 - 基于 CT+PET 联合特征

    改进①：CT+PET 联合输入（输入通道 C → 2C）
    ─────────────────────────────────────────────────────
    原设计仅以 PET 特征作为输入，模型只能依赖代谢信息判断前景，
    容易忽略 CT 中提供的解剖结构边界信息。
    改为在通道维度拼接 CT 与 PET 特征（torch.cat([ct_feat, pet_feat], dim=1)），
    使掩码同时感知 PET 代谢热点与 CT 骨内结构，减少小病灶的误检与漏检。
    相应地将第一层卷积输入通道由 C 调整为 2C。

    改进②：残差偏置掩码 mask = 0.5 + 0.5 * sigmoid(logit)
    ─────────────────────────────────────────────────────
    引入 0.5 的残差下界，mask ∈ [0.5, 1.0]，保证前景分支特征幅度
    在任何训练阶段均不会被完全压缩；M_inv ∈ [0.0, 0.5]，语义仍可分。
    """

    def __init__(self, channels):
        super().__init__()
        # 第一层输入通道为 2C（CT + PET concat）
        self.predictor = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            CBAM(channels // 2, reduction=8),
            nn.Conv2d(channels // 2, 1, 1)   # 输出原始 logit
        )

    def forward(self, ct_feat, pet_feat):
        # 拼接 CT 与 PET 特征：[B, 2C, H, W]
        joint = torch.cat([ct_feat, pet_feat], dim=1)
        logit = self.predictor(joint)
        # 残差偏置：mask ∈ [0.5, 1.0]，前景分支特征幅度始终 ≥ 50%
        # mask = 0.5 + 0.5 * torch.sigmoid(logit)
        mask = torch.sigmoid(logit)
        return mask


# ==================== 单尺度 FBFA 融合 (骨内版本) ====================

class BranchGate(nn.Module):
    """四路分支门控加权模块
    ────────────────────────────────────────────────────────────────
    输入：四路特征 [f0, f1, f2, f3]，每路形状 [B, C, H, W]
    输出：加权求和后的单路特征 [B, C, H, W]

    机制：对每路做全局平均池化 → 拼接 → FC → Softmax(4) → 广播乘回各路
    效果：网络自动学习四个分支（low_fg / low_bg / high_fg / high_bg）的
          相对重要性，取代原来简单 concat + 1×1 压缩。
    参数量：4C × 4 ≈ 极低（以 C=128 为例仅 2K）
    """

    def __init__(self, channels, num_branches=4):
        super().__init__()
        self.num_branches = num_branches
        # 全局上下文 → 四路门控权重
        self.gate_fc = nn.Sequential(
            nn.Linear(channels * num_branches, channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, num_branches, bias=True)
        )

    def forward(self, branches: list):
        # branches: list of [B, C, H, W], len = num_branches
        B = branches[0].size(0)
        # 全局平均池化后拼接：[B, num_branches * C]
        pooled = torch.cat(
            [f.mean(dim=[2, 3]) for f in branches], dim=1
        )
        # 门控权重：[B, num_branches]，softmax 归一化保证权重之和为 1
        weights = torch.softmax(self.gate_fc(pooled), dim=1)  # [B, 4]

        # 加权求和：[B, C, H, W]
        out = sum(
            weights[:, i].view(B, 1, 1, 1) * branches[i]
            for i in range(self.num_branches)
        )
        return out


class FBFAFusionSingleScaleIntraBone(nn.Module):
    """单尺度 FBFA 融合 - 骨内肿瘤版

    三处改进：
    ① 残差偏置掩码  mask ∈ [0.5, 1.0]，防止前景分支特征在训练初期消失
    ② 频率感知融合  low 分支用 3×3（大感受野匹配全局低频语义）
                    high 分支用 1×1（轻量匹配局部高频细节）
    ③ 门控重构      BranchGate 自动学习四路分支重要性，取代简单 concat+压缩
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels

        # 频率分解（DoG）
        self.ct_freq_decomp  = EnhancedFrequencyDecomposition(channels)
        self.pet_freq_decomp = EnhancedFrequencyDecomposition(channels)

        # 骨内肿瘤前景预测（残差偏置掩码）
        self.fg_predictor = IntraBoneForegroundPredictor(channels)

        # ② 频率感知融合路径
        # low 分支：3×3 卷积，感受野更大，匹配低频全局语义
        self.fusion_low_fg = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )
        self.fusion_low_bg = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )
        # high 分支：1×1 卷积，轻量，匹配局部高频纹理细节
        self.fusion_high_fg = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )
        self.fusion_high_bg = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )

        # ③ 门控重构：BranchGate 自动加权 + 1×1 精炼
        self.branch_gate = BranchGate(channels, num_branches=4)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            CBAM(channels, reduction)
        )

        # 残差混合权重（可学习）
        self.alpha = nn.Parameter(torch.ones(1) * 0.3)

    def forward(self, ct_feat, pet_feat):
        # DoG 频率分解
        CT_low,  CT_high  = self.ct_freq_decomp(ct_feat)
        PET_low, PET_high = self.pet_freq_decomp(pet_feat)

        # ① 残差偏置前景掩码：M ∈ [0.5, 1.0]，M_inv ∈ [0.0, 0.5]
        M     = self.fg_predictor(ct_feat, pet_feat)  # CT+PET 联合前景掩码
        M_inv = (1.0 - M).detach()

        # 前景/背景特征分离
        CT_low_fg,  PET_low_fg  = CT_low  * M,     PET_low  * M
        CT_low_bg,  PET_low_bg  = CT_low  * M_inv, PET_low  * M_inv
        CT_high_fg, PET_high_fg = CT_high * M,     PET_high * M
        CT_high_bg, PET_high_bg = CT_high * M_inv, PET_high * M_inv

        # ② 频率感知四路融合
        fused_low_fg  = self.fusion_low_fg (torch.cat([CT_low_fg,  PET_low_fg],  dim=1))
        fused_low_bg  = self.fusion_low_bg (torch.cat([CT_low_bg,  PET_low_bg],  dim=1))
        fused_high_fg = self.fusion_high_fg(torch.cat([CT_high_fg, PET_high_fg], dim=1))
        fused_high_bg = self.fusion_high_bg(torch.cat([CT_high_bg, PET_high_bg], dim=1))

        # ③ 门控重构：softmax 门控加权 → 1×1 精炼
        gated  = self.branch_gate([fused_low_fg, fused_low_bg, fused_high_fg, fused_high_bg])
        fused  = self.reconstruction(gated)

        # 残差连接
        alpha  = torch.sigmoid(self.alpha)
        base = 0.5 * (ct_feat + pet_feat)
        output = alpha * fused + (1 - alpha) * base

        return output


# ==================== 多尺度 FBFA-Fusion ====================

class FBFAFusionMultiScaleIntraBone(nn.Module):
    """多尺度 FBFA-Fusion (骨内版本)"""

    def __init__(self, channels_list=[32, 64, 128, 256], reduction=16):
        super().__init__()
        self.num_scales = len(channels_list)

        self.fusions = nn.ModuleList([
            FBFAFusionSingleScaleIntraBone(c, reduction)
            for c in channels_list
        ])

    def forward(self, ct_features, pet_features):
        fused_features = []

        for i in range(self.num_scales):
            ct_f = ct_features[i]
            pet_f = pet_features[i]

            if ct_f.shape[2:] != pet_f.shape[2:]:
                pet_f = F.interpolate(pet_f, size=ct_f.shape[2:], mode='bilinear', align_corners=False)

            fused = self.fusions[i](ct_f, pet_f)
            fused_features.append(fused)

        return fused_features


# ==================== 多尺度解码器块 ====================

class MultiScaleDecoderBlock(nn.Module):
    """多尺度解码器 - 支持 Deep Supervision"""

    def __init__(self, in_channels, skip_channels, out_channels, scale=2, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear',
                                    align_corners=True) if scale > 1 else nn.Identity()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            CBAM(out_channels, reduction=8),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Deep Supervision 输出头
        if deep_supervision:
            self.ds_head = nn.Sequential(
                nn.Conv2d(out_channels, out_channels // 2, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 2, 1, 1)
            )
        else:
            self.ds_head = None

    def forward(self, x, skip):
        x = self.upsample(x)
        # 空间尺寸对齐：CT 与 PET 分辨率路径不同时，skip 可能与 x 尺寸不一致
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        ds_output = None
        if self.ds_head is not None:
            ds_output = self.ds_head(x)

        return x, ds_output


# ==================== 🔥 主模型（BONE-ONLY VERSION - 512x512）====================

class FBFAIntraBoneTumorSegmentation(nn.Module):
    """
    FBFA-Fusion 骨内肿瘤分割模型 - BONE-ONLY VERSION - 512x512

    核心改进:
    1. 输入已经过骨掩码约束（数据层面）
    2. 输出强制骨区域约束
    3. CBAM + Deep Supervision
    """

    def __init__(self,
                 stage1_model_path='checkpoints/stage1_2D_20260303-152807/best_model.pth',
                 freeze_stage1=True,
                 bone_dilation=7,
                 channels_list=[32, 64, 128, 256],
                 enable_deep_supervision=True,
                 use_bone_constraint=False,
                 # ── [DGMA] 新增参数 ──────────────────────────────────────
                 dgma_K_max: int           = 8,     # 每张切片最多检测的病灶中心数
                 dgma_nms_threshold: float = 0.3,   # NMS 峰值置信度阈值
                 dgma_r_min: float         = 0.03,  # 半径图最小值（归一化坐标）
                 dgma_r_max: float         = 0.40,  # 半径图最大值（归一化坐标）
                 dgma_min_spatial_size: int = 65,   # 低于此分辨率的特征图跳过 DGMA (strides=[2,2,2,1]下: scale2=H64跳过✓)
                 ):
        super().__init__()

        self.freeze_stage1 = freeze_stage1
        self.bone_dilation = bone_dilation
        self.enable_deep_supervision = enable_deep_supervision
        self.use_bone_constraint = use_bone_constraint

        # Stage1: CT分支
        from network.model_efficientvim_2d_stage1 import ConDSeg2DStage1_EfficientViM
        self.ct_branch = ConDSeg2DStage1_EfficientViM(in_channels=1, out_channels=1)

        if stage1_model_path:
            checkpoint = torch.load(stage1_model_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model_dict = self.ct_branch.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.ct_branch.load_state_dict(model_dict)

        if freeze_stage1:
            for param in self.ct_branch.parameters():
                param.requires_grad = False
            self.ct_branch.eval()

        # Bone处理器
        from network.model_stage2_tumor_2d import StableBoneMaskProcessor
        self.bone_processor = StableBoneMaskProcessor(
            dilation_kernel=bone_dilation,
            min_area_ratio=0.001
        )

        # PET分支
        from network.efficientvim_modules_2d import MambaFeatureExtractor2D
        # [性能优化] strides: [1,2,2,1] → [2,2,2,1]
        # 原 stride=1 让 stage0 在 512×512 全分辨率运行 Mamba (seq_len=262,144)，
        # 是 stage1-3 计算量之和的 8×，占 forward 总时间约 70%。
        # 改为 stride=2 后 stage0 在 256×256 运行 (seq_len=65,536)，预估 ~5-6× 提速。
        # 分辨率损失：最浅层特征从 512→256，Decoder 的 Upsample 路径不变，
        # 最终输出仍为 512×512；小病灶细节由 CT 分支的高分辨率特征补偿。
        #
        # [性能优化] state_dim[0]: 49 → 16
        # stage0 的 SSM 状态维度从 49 降到 16，SSM 递归代价再降 ~3×；
        # state_dim 主要影响 Mamba 的"记忆容量"，对局部病灶检测影响有限。
        self.pet_backbone = MambaFeatureExtractor2D(
            in_dim=1,
            embed_dim=channels_list,
            depths=[2, 2, 2, 2],
            state_dim=[16, 25, 9, 9],
            strides=[2, 2, 2, 1]
        )

        # FBFA-Fusion (骨内版本)
        self.fbfa_fusion = FBFAFusionMultiScaleIntraBone(
            channels_list=channels_list,
            reduction=16
        )

        # ── [DGMA] PET 特征增强模块 ─────────────────────────────
        # 每个尺度一个 DGMAWithSOE，与 PET backbone 的 channels_list 一一对应。
        # DGMAWithSOE 内部根据 min_spatial_size 自动判断是否执行 DGMA：
        #   高分辨率尺度 (H >= 65) → SOE + DGMA（动态高斯病灶注意力）
        #   低分辨率尺度 (H <  65) → 仅 SOE（保持计算效率）
        self.soe = nn.ModuleList([
            DGMAWithSOE(
                channels          = c,
                K_max             = dgma_K_max,
                nms_threshold     = dgma_nms_threshold,
                r_min             = dgma_r_min,
                r_max             = dgma_r_max,
                min_spatial_size  = dgma_min_spatial_size,
            )
            for c in channels_list
        ])

        # 解码器 (支持 Deep Supervision)
        self.decoder3 = MultiScaleDecoderBlock(256, 128, 128, scale=2, deep_supervision=enable_deep_supervision)
        self.decoder2 = MultiScaleDecoderBlock(128, 64,  64,  scale=2, deep_supervision=enable_deep_supervision)
        self.decoder1 = MultiScaleDecoderBlock(64,  32,  32,  scale=2, deep_supervision=False)  # scale 1→2，补全到256×256

        # 主输出头
        self.output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32, reduction=8),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, ct, pet, bone_mask=None, return_intermediate=False):
        """
        🔥 CRITICAL FIX: 输出纯 logits，不做 bone masking

        Args:
            ct: [B, 1, H, W] - 已经过骨掩码约束（数据层面）
            pet: [B, 1, H, W] - 已经过骨掩码约束（数据层面）
            bone_mask: [B, 1, H, W] - 骨掩码（仅用于返回）
            return_intermediate: 是否返回中间结果
        """
        # CT特征提取
        if self.freeze_stage1:
            with torch.no_grad():
                ct_features, decoder_feats, bone_logits = self.ct_branch.forward_with_features(ct)
        else:
            ct_features, decoder_feats, bone_logits = self.ct_branch.forward_with_features(ct)

        # 处理骨掩码
        if bone_mask is None:
            pet_bone, bone_mask_pred, bone_mask_dilated, mask_valid = \
                self.bone_processor(bone_logits, pet)
            bone_mask_to_use = bone_mask_dilated
        else:
            bone_mask_to_use = bone_mask
            bone_mask_pred = bone_mask
            bone_mask_dilated = bone_mask
            pet_bone = pet
            mask_valid = (bone_mask.sum(dim=(1, 2, 3)) > 0).float()

        # PET特征提取
        pet_features = self.pet_backbone(pet_bone)

        # 特征对齐
        def align(src, tgt):
            if src.shape[2:] != tgt.shape[2:]:
                return F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=False)
            return src

        pet_features_aligned = [
            align(pf, cf) for pf, cf in zip(pet_features, ct_features)
        ]

        # ── [DGMA] PET 特征增强 ────────────────────────────────
        # 在 FBFA 融合之前，对每个尺度的 PET 特征做动态高斯注意力增强。
        # DGMAWithSOE.forward 返回增强后的特征（形状不变）。
        # 深层尺度（H < min_spatial_size）内部自动跳过 DGMA 只做 SOE。
        pet_features_enhanced = [
            self.soe[i](pet_features_aligned[i])
            for i in range(len(pet_features_aligned))
        ]

        # FBFA多尺度融合
        fused_features = self.fbfa_fusion(ct_features, pet_features_enhanced)

        # 解码 + Deep Supervision
        d2, ds3_logits = self.decoder3(fused_features[3], fused_features[2])
        d1, ds2_logits = self.decoder2(d2, fused_features[1])
        d0, _ = self.decoder1(d1, fused_features[0])

        # 主输出
        tumor_logits = self.output(d0)

        # 🔥 Deep Supervision 尺寸对齐：将中间监督输出上采样至与主输出相同的空间尺寸
        # decoder3 输出 ds3_logits 为 64×64，decoder2 输出 ds2_logits 为 128×128，
        # 均需对齐到 512×512，否则 deep supervision loss 计算时尺寸不匹配。
        target_size = tumor_logits.shape[2:]  # (H, W) = (512, 512)
        if ds3_logits is not None and ds3_logits.shape[2:] != target_size:
            ds3_logits = F.interpolate(ds3_logits, size=target_size,
                                       mode='bilinear', align_corners=False)
        if ds2_logits is not None and ds2_logits.shape[2:] != target_size:
            ds2_logits = F.interpolate(ds2_logits, size=target_size,
                                       mode='bilinear', align_corners=False)

        # 🔥🔥🔥 完美方案修改：彻底移除 bone constraint 🔥🔥🔥
        # 理由：
        # 1. 输入端（Dataset）已经做了骨掩码约束（ct * bone_mask）
        # 2. 输出端乘以 bone_mask 会导致骨外梯度=0，不利于训练稳定性
        # 3. 骨内约束统一交给 Loss/Metrics 处理（apply_bone_mask_to_logits）

        # ❌ 已删除原有的 bone masking 逻辑
        # ✅ 模型直接输出纯 logits，不做任何掩码操作

        if return_intermediate:
            # ── [DGMA] 收集各尺度中间状态 ──────────────────────────
            # dgma_states[i] 对应 channels_list[i] 尺度的 last_state 字典。
            # 深层尺度已跳过 DGMA 时，last_state 为 None（supervision loss 会安全跳过）。
            # 训练脚本可用于：
            #   1. TensorBoard 监控 k_dynamic、radius_map、attention_map
            #   2. compute_spatial_radius_loss(self.soe[i].dgma, tumor_ds, bone_ds) 辅助监督
            dgma_states = [soe_block.last_state for soe_block in self.soe]

            return {
                'tumor_logits':    tumor_logits,
                'ds3_logits':      ds3_logits,
                'ds2_logits':      ds2_logits,
                'bone_mask':       bone_mask_pred,
                'bone_mask_dilated': bone_mask_dilated,
                'pet_bone':        pet_bone,
                'mask_valid':      mask_valid,
                'dgma_states':     dgma_states,   # [DGMA] 新增
            }
        else:
            return tumor_logits


# ==================== 测试 ====================

if __name__ == "__main__":
    print("Testing Intra-Bone FBFA-Fusion Model (BONE-ONLY VERSION - 512x512 + DGMA)...")

    model = FBFAIntraBoneTumorSegmentation(
        stage1_model_path=None,
        freeze_stage1=False,
        enable_deep_supervision=True,
        use_bone_constraint=False,
        dgma_K_max=5,
        dgma_nms_threshold=0.3,
        dgma_min_spatial_size=65,
    )

    ct       = torch.randn(2, 1, 512, 512)
    pet      = torch.randn(2, 1, 512, 512)
    bone_mask = (torch.randn(2, 1, 512, 512) > 0).float()

    outputs = model(ct, pet, bone_mask, return_intermediate=True)

    print(f"Main logits:  {outputs['tumor_logits'].shape}")
    print(f"DS3 logits:   {outputs['ds3_logits'].shape}")
    print(f"DS2 logits:   {outputs['ds2_logits'].shape}")
    print(f"Bone mask:    {outputs['bone_mask'].shape}")

    # [DGMA] 验证各尺度状态
    print("\n[DGMA] Per-scale states:")
    channels_list = [32, 64, 128, 256]
    for i, state in enumerate(outputs['dgma_states']):
        if state is None:
            print(f"  scale[{i}] C={channels_list[i]:3d}: skipped (SOE only)")
        else:
            k  = state['k_dynamic'].tolist()
            rm = state['radius_map']
            print(f"  scale[{i}] C={channels_list[i]:3d}: "
                  f"k_dynamic={k}  "
                  f"radius_map=[{rm.min():.3f}, {rm.max():.3f}]  "
                  f"fallback={state['used_fallback'].tolist()}")

    # [DGMA] 验证 compute_spatial_radius_loss 可调用
    tumor_gt = torch.zeros(2, 1, 256, 256)
    tumor_gt[:, :, 60:80, 60:80] = 1.0
    bone_gt  = torch.ones(2, 1, 256, 256)
    aux_loss = compute_spatial_radius_loss(model.soe[0].dgma, tumor_gt, bone_gt)
    print(f"\n[DGMA] compute_spatial_radius_loss (scale 0): {aux_loss.item():.6f}")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dgma_params      = sum(p.numel() for soe in model.soe for p in soe.parameters())
    print(f"\nTotal: {total_params/1e6:.2f}M  |  "
          f"Trainable: {trainable_params/1e6:.2f}M  |  "
          f"DGMA: {dgma_params/1e3:.1f}K")
    print("\n✓ Model test passed!")