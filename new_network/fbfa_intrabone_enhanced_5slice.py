"""
FBFA-Fusion Enhanced — 5-Slice 2.5D + Cross-Slice Attention 版本
=================================================================
在 fbfa_intrabone_enhanced.py 基础上升级：
  [v7-A] 5-Slice 输入支持
         ct=(5,H,W), pet=(5,H,W) — 对应 t-2/t-1/t/t+1/t+2
  [v7-B] CSASliceFusion 替换 ct_adapter + pet_adapter
         原版: ct_center_weight(1,3,1,1) × ct_3ch + 1×1 Conv(3→1)
         新版: CSASliceFusion(n_slices=5) → 跨切片注意力 → (B,1,H,W)
               · 共享轻量编码器（逐切片，权重共享）
               · 中心切片为 Q，4 个邻居为 K/V，跨切片交叉注意力
               · 可选 PET→CT 跨模态辅助注意力
  [v7-C] EncoderCSABlock（可选）插入 PET backbone Stage1/2 之间
         在编码器中间层补充跨切片语义关联（CSA-Net 推荐位置）
  [v7-D] 其余模块（FBFA、DGMA、解码器、Deep Supervision）完全不变

架构数据流：
  ct(5,H,W) + pet(5,H,W)
         │
  CSASliceFusion
         │
  ct_1ch(1,H,W) + pet_bone_1ch(1,H,W)
         │                  │
  ct_branch             pet_backbone
  (Stage1, frozen)      (MambaFeatureExtractor)
         │                  │
  ct_features[4]    pet_features[4] ──→ [opt: EncoderCSABlock]
         │                  │
         └──── DGMAWithSOE ──┘
                    │
              FBFA Fusion
                    │
                Decoder
                    │
             tumor_logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from new_network.fbfa_intrabone_enhanced_iddmga import (
    DGMAWithSOE, compute_spatial_radius_loss)
from new_network.cross_slice_attention import (
    CSASliceFusion, EncoderCSABlock)


# ─────────────────────────────────────────────────────────────────────────────
# 以下模块直接复用 fbfa_intrabone_enhanced.py 中的代码（不重复粘贴，通过 import）
# ─────────────────────────────────────────────────────────────────────────────

from new_network.fbfa_intrabone_enhanced import (
    CBAM,
    DepthwiseSeparableConv,
    EnhancedFrequencyDecomposition,
    IntraBoneForegroundPredictor,
    BranchGate,
    FBFAFusionSingleScaleIntraBone,
    FBFAFusionMultiScaleIntraBone,
    MultiScaleDecoderBlock,
)


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class FBFAIntraBoneTumorSegmentation5Slice(nn.Module):
    """
    FBFA-Fusion 骨内肿瘤分割 — 5-Slice 2.5D + Cross-Slice Attention 版

    与原版 FBFAIntraBoneTumorSegmentation 的差异：
    ─────────────────────────────────────────────
    1. [v7-A] 输入 ct/pet 从 (B,3,H,W) 升级为 (B,5,H,W)
    2. [v7-B] ct_adapter/pet_adapter → CSASliceFusion
              · 5切片 → per-slice shared encoder → CrossSliceAttention → 1ch
              · 可选 PET→CT 跨模态辅助注意力（use_cross_modal=True）
    3. [v7-C] 可选 EncoderCSABlock 插入 PET backbone 中间层
              · 分别对 5 个切片跑 PET backbone stage0~2
              · 在 stage1/2 输出处插入 CSA，只保留中心切片特征向后传
              · enable_encoder_csa=False 时退化为原始单切片处理（等价于原版）
    4. 其余完全不变：DGMA、FBFA、解码器、Deep Supervision

    注意：enable_encoder_csa=True 时内存开销会增大约 5× PET backbone 运算量，
    因为需要对 5 个切片各跑一次 backbone。
    建议只对浅层（stage0/1）做，或直接关闭（依赖 CSASliceFusion 即可）。
    """

    def __init__(
        self,
        stage1_model_path: str           = None,
        freeze_stage1:     bool          = True,
        bone_dilation:     int           = 7,
        channels_list:     List[int]     = [32, 64, 128, 256],
        enable_deep_supervision: bool    = True,
        use_bone_constraint: bool        = False,
        # ── CSA 参数 ──────────────────────────────────────────────
        n_slices:          int           = 5,    # [v7] 5-slice
        csa_feat_ch:       int           = 32,   # CSASliceFusion 中间特征维度
        csa_n_heads:       int           = 4,    # 注意力头数
        csa_pool_size:     int           = 16,   # 空间池化目标尺寸
        csa_use_cross_modal: bool        = True, # PET→CT 跨模态辅助注意力
        # ── 可选 Encoder-Level CSA ────────────────────────────────
        enable_encoder_csa: bool         = False,  # 慎开，5× backbone 开销
        encoder_csa_stages: List[int]    = [1, 2], # 在哪些 stage 后插入 CSA
        # ── DGMA 参数（与原版相同）──────────────────────────────────
        dgma_K_max:           int        = 8,
        dgma_nms_threshold:   float      = 0.3,
        dgma_r_min:           float      = 0.03,
        dgma_r_max:           float      = 0.25,
        dgma_min_spatial_size: int       = 65,
    ):
        super().__init__()

        assert n_slices % 2 == 1, "n_slices 必须为奇数（3 或 5）"
        self.n_slices              = n_slices
        self.center_idx            = n_slices // 2
        self.freeze_stage1         = freeze_stage1
        self.bone_dilation         = bone_dilation
        self.enable_deep_supervision = enable_deep_supervision
        self.use_bone_constraint   = use_bone_constraint
        self.enable_encoder_csa    = enable_encoder_csa
        self.encoder_csa_stages    = encoder_csa_stages

        # ── [v7-B] CSASliceFusion —— 替换 ct/pet_adapter ─────────
        self.csa_fusion = CSASliceFusion(
            n_slices         = n_slices,
            feat_ch          = csa_feat_ch,
            n_heads          = csa_n_heads,
            pool_size        = csa_pool_size,
            use_self_attn    = True,
            use_cross_modal  = csa_use_cross_modal,
        )

        # ── Stage1: CT 骨骼分割分支（冻结）────────────────────────
        from network.model_efficientvim_2d_stage1 import ConDSeg2DStage1_EfficientViM
        self.ct_branch = ConDSeg2DStage1_EfficientViM(in_channels=1, out_channels=1)

        if stage1_model_path:
            checkpoint = torch.load(stage1_model_path, map_location='cpu',
                                    weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model_dict = self.ct_branch.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items()
                               if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            self.ct_branch.load_state_dict(model_dict)
            print(f"[CSA5Slice] Loaded {len(pretrained_dict)} CT branch weights")

        if freeze_stage1:
            for param in self.ct_branch.parameters():
                param.requires_grad = False
            self.ct_branch.eval()

        # ── Bone 处理器 ───────────────────────────────────────────
        from network.model_stage2_tumor_2d import StableBoneMaskProcessor
        self.bone_processor = StableBoneMaskProcessor(
            dilation_kernel=bone_dilation, min_area_ratio=0.001)

        # ── PET backbone (单切片，接收 CSASliceFusion 输出的 1ch) ──
        from network.efficientvim_modules_2d import MambaFeatureExtractor2D
        self.pet_backbone = MambaFeatureExtractor2D(
            in_dim      = 1,
            embed_dim   = channels_list,
            depths      = [2, 2, 2, 2],
            state_dim   = [16, 25, 9, 9],
            strides     = [2, 2, 2, 1],
        )

        # ── [v7-C] Encoder-level CSA（可选）────────────────────────
        # 为在 backbone 中间层捕获跨切片语义关系，需要对 n_slices 个
        # 切片分别跑 backbone stage0 → stage_k，然后在 stage_k 输出处插入 CSA。
        # 开启后需要在 forward 中对每个切片独立跑 backbone，计算量 × n_slices。
        if enable_encoder_csa:
            # encoder_csa_blocks[k] 对应 channels_list[k] 通道的 CSA
            self.encoder_csa_blocks = nn.ModuleDict({
                str(s): EncoderCSABlock(
                    channels   = channels_list[s],
                    n_slices   = n_slices,
                    n_heads    = csa_n_heads,
                    pool_size  = csa_pool_size,
                )
                for s in encoder_csa_stages
                if s < len(channels_list)
            })

        # ── DGMA 病灶注意力 ───────────────────────────────────────
        self.soe = nn.ModuleList([
            DGMAWithSOE(
                channels           = c,
                K_max              = dgma_K_max,
                nms_threshold      = dgma_nms_threshold,
                r_min              = dgma_r_min,
                r_max              = dgma_r_max,
                min_spatial_size   = dgma_min_spatial_size,
            )
            for c in channels_list
        ])

        # ── FBFA 多尺度融合 ───────────────────────────────────────
        self.fbfa_fusion = FBFAFusionMultiScaleIntraBone(
            channels_list=channels_list, reduction=16)

        # ── 解码器 ────────────────────────────────────────────────
        self.decoder3 = MultiScaleDecoderBlock(
            256, 128, 128, scale=2, deep_supervision=enable_deep_supervision)
        self.decoder2 = MultiScaleDecoderBlock(
            128, 64,  64,  scale=2, deep_supervision=enable_deep_supervision)
        self.decoder1 = MultiScaleDecoderBlock(
            64,  32,  32,  scale=2, deep_supervision=False)

        # ── 主输出头 ──────────────────────────────────────────────
        self.output = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAM(32, reduction=8),
            nn.Conv2d(32, 1, 1),
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _pet_backbone_per_slice(
        self,
        pet5ch: torch.Tensor,      # (B, 5, H, W)  原始 5 切片 PET（bone-masked）
        bone_mask_to_use: torch.Tensor,  # (B, 1, H, W)
    ):
        """
        [v7-C] Encoder-level CSA 模式：
        对每个 slice 单独跑 PET backbone，在 encoder_csa_stages 指定的
        stage 输出处插入 EncoderCSABlock，只保留中心切片特征。

        返回: center_pet_features  List[(B, C, H', W')] × 4 stages
        """
        B, N, H, W = pet5ch.shape

        # 对每个切片做 bone mask，送入 PET backbone
        all_slice_feats = []   # shape: n_slices × [stage0, stage1, stage2, stage3]
        for i in range(N):
            pet_i = pet5ch[:, i:i+1, :, :] * bone_mask_to_use   # (B, 1, H, W)
            feats_i = self.pet_backbone(pet_i)                    # list of 4 tensors
            all_slice_feats.append(feats_i)
        # all_slice_feats[slice_idx][stage_idx]: (B, C, H', W')

        # 在指定 stage 插入 EncoderCSABlock，只处理中心切片
        center_feats = []
        for stage_idx in range(len(all_slice_feats[0])):
            stage_feats = [all_slice_feats[si][stage_idx] for si in range(N)]
            # stage_feats: list of N × (B, C, H', W')
            key = str(stage_idx)
            if self.enable_encoder_csa and key in self.encoder_csa_blocks:
                # CSA: center slice attends to all neighbors at this stage
                enriched = self.encoder_csa_blocks[key](stage_feats)
                center_feats.append(enriched)
            else:
                # 无 CSA：直接取中心切片特征
                center_feats.append(stage_feats[self.center_idx])

        return center_feats   # List[(B, C, H', W')] × 4

    # ──────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        ct:          torch.Tensor,           # (B, 5, H, W)  [v7] 5切片
        pet:         torch.Tensor,           # (B, 5, H, W)
        bone_mask:   Optional[torch.Tensor] = None,  # (B, 1, H, W)
        return_intermediate: bool           = False,
    ):
        B, N, H, W = ct.shape
        assert N == self.n_slices, \
            f"Expected {self.n_slices} slices, got {N}"

        # ── [v7-B] CSASliceFusion: 5ch → 1ch ──────────────────────
        # ct/pet 各自经过跨切片注意力，输出中心切片的增强单通道表示
        ct_1ch, pet_1ch_csa = self.csa_fusion(ct, pet)
        # ct_1ch:      (B, 1, H, W)
        # pet_1ch_csa: (B, 1, H, W)  — CSA 增强后的 PET 中心切片

        # ── CT 特征提取（Stage1，冻结）──────────────────────────────
        if self.freeze_stage1:
            with torch.no_grad():
                ct_features, decoder_feats, bone_logits = \
                    self.ct_branch.forward_with_features(ct_1ch)
        else:
            ct_features, decoder_feats, bone_logits = \
                self.ct_branch.forward_with_features(ct_1ch)

        # ── Bone mask 处理 ────────────────────────────────────────
        if bone_mask is None:
            pet_bone, bone_mask_pred, bone_mask_dilated, mask_valid = \
                self.bone_processor(bone_logits, pet_1ch_csa)
            bone_mask_to_use = bone_mask_dilated
        else:
            bone_mask_to_use = bone_mask
            bone_mask_pred   = bone_mask
            bone_mask_dilated = bone_mask
            pet_bone          = pet_1ch_csa
            mask_valid = (bone_mask.sum(dim=(1, 2, 3)) > 0).float()

        # ── PET 特征提取 ───────────────────────────────────────────
        if self.enable_encoder_csa:
            # [v7-C] 需要对 5 切片分别跑 backbone + 中间层 CSA
            # pet5_bone: 对每个切片独立做 bone mask
            # 注：这里的 pet 仍是 (B,5,H,W) 原始输入
            pet5_bone = pet * bone_mask_to_use.unsqueeze(1)  # broadcast over 5 slices
            pet_features = self._pet_backbone_per_slice(pet5_bone, bone_mask_to_use)
        else:
            # [默认] 使用 CSASliceFusion 的单通道输出 → 直接跑 backbone
            pet_bone_masked = pet_bone   # (B, 1, H, W)，已 bone-masked
            pet_features    = self.pet_backbone(pet_bone_masked)

        # ── 特征对齐 ──────────────────────────────────────────────
        def align(src, tgt):
            if src.shape[2:] != tgt.shape[2:]:
                return F.interpolate(src, size=tgt.shape[2:],
                                     mode='bilinear', align_corners=False)
            return src

        pet_features_aligned = [
            align(pf, cf) for pf, cf in zip(pet_features, ct_features)
        ]

        # ── DGMA 病灶注意力增强 ───────────────────────────────────
        pet_features_enhanced = [
            self.soe[i](pet_features_aligned[i])
            for i in range(len(pet_features_aligned))
        ]

        # ── FBFA 多尺度融合 ───────────────────────────────────────
        fused_features = self.fbfa_fusion(ct_features, pet_features_enhanced)

        # ── 解码 + Deep Supervision ───────────────────────────────
        d2, ds3_logits = self.decoder3(fused_features[3], fused_features[2])
        d1, ds2_logits = self.decoder2(d2, fused_features[1])
        d0, _          = self.decoder1(d1, fused_features[0])

        tumor_logits = self.output(d0)

        # Deep Supervision 尺寸对齐
        target_size = tumor_logits.shape[2:]
        if ds3_logits is not None and ds3_logits.shape[2:] != target_size:
            ds3_logits = F.interpolate(ds3_logits, size=target_size,
                                       mode='bilinear', align_corners=False)
        if ds2_logits is not None and ds2_logits.shape[2:] != target_size:
            ds2_logits = F.interpolate(ds2_logits, size=target_size,
                                       mode='bilinear', align_corners=False)

        if return_intermediate:
            dgma_states = [soe_block.last_state for soe_block in self.soe]
            return {
                'tumor_logits':      tumor_logits,
                'ds3_logits':        ds3_logits,
                'ds2_logits':        ds2_logits,
                'bone_mask':         bone_mask_pred,
                'bone_mask_dilated': bone_mask_dilated,
                'pet_bone':          pet_bone,
                'mask_valid':        mask_valid,
                'dgma_states':       dgma_states,
            }
        else:
            return tumor_logits

    @property
    def last_state(self):
        # 向后兼容 SingleStageLoss 中的 model.soe 访问
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 参数量统计工具
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> dict:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    csa_params = sum(p.numel() for p in model.csa_fusion.parameters())
    soe_params = sum(p.numel() for soe in model.soe for p in soe.parameters())
    return dict(
        total_M     = total     / 1e6,
        trainable_M = trainable / 1e6,
        csa_K       = csa_params / 1e3,
        dgma_K      = soe_params / 1e3,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    print("Testing FBFAIntraBoneTumorSegmentation5Slice...")
    print("(stage1_model_path=None → freeze_stage1=False for test)")

    model = FBFAIntraBoneTumorSegmentation5Slice(
        stage1_model_path       = None,
        freeze_stage1           = False,
        n_slices                = 5,
        csa_feat_ch             = 32,
        csa_n_heads             = 4,
        csa_pool_size           = 16,
        csa_use_cross_modal     = True,
        enable_encoder_csa      = False,  # 关闭以节省测试内存
        enable_deep_supervision = True,
        dgma_K_max              = 5,
        dgma_nms_threshold      = 0.3,
        dgma_min_spatial_size   = 65,
    )

    B, N, H, W = 2, 5, 512, 512
    ct        = torch.randn(B, N, H, W)
    pet       = torch.randn(B, N, H, W)
    bone_mask = (torch.randn(B, 1, H, W) > 0).float()

    t0 = time.time()
    outputs = model(ct, pet, bone_mask, return_intermediate=True)
    elapsed = time.time() - t0

    print(f"\nForward pass: {elapsed*1000:.0f}ms")
    print(f"  tumor_logits: {outputs['tumor_logits'].shape}")
    print(f"  ds3_logits:   {outputs['ds3_logits'].shape}")
    print(f"  ds2_logits:   {outputs['ds2_logits'].shape}")
    print(f"  bone_mask:    {outputs['bone_mask'].shape}")

    print("\n[DGMA] Per-scale states:")
    channels_list = [32, 64, 128, 256]
    for i, state in enumerate(outputs['dgma_states']):
        if state is None:
            print(f"  scale[{i}] C={channels_list[i]:3d}: skipped (SOE only)")
        else:
            k  = state['k_dynamic'].tolist()
            rm = state['radius_map']
            print(f"  scale[{i}] C={channels_list[i]:3d}: k_dynamic={k}  "
                  f"radius=[{rm.min():.3f}, {rm.max():.3f}]")

    stats = count_params(model)
    print(f"\nParams: total={stats['total_M']:.2f}M  "
          f"trainable={stats['trainable_M']:.2f}M  "
          f"CSA={stats['csa_K']:.1f}K  "
          f"DGMA={stats['dgma_K']:.1f}K")

    # Gradient test
    print("\n[Gradient] CSASliceFusion...")
    loss = outputs['tumor_logits'].mean()
    loss.backward()
    csa_grad = sum(p.grad.norm().item()
                   for p in model.csa_fusion.parameters()
                   if p.grad is not None)
    print(f"  CSA grad norm: {csa_grad:.6f}  "
          f"{'✓' if csa_grad > 1e-8 else '✗'}")

    print("\n✓ 5-Slice model test passed!")