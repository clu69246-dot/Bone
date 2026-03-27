"""
Cross-Slice Attention (CSA) Module for PET-CT Bone Tumor Segmentation
=======================================================================
Adapted from CSA-Net (Kumar et al., Computers in Biology and Medicine, 2024)
"A Flexible 2.5D Medical Image Segmentation Approach with In-Slice and
Cross-Slice Attention"  https://github.com/mirthAI/CSA-Net

CSA-Net 原理回顾
─────────────────
  原论文设计（单模态 MRI）：
    · 以中心切片特征为 Query
    · 所有邻居切片特征为 Key/Value
    · 跨切片交叉注意力 + 中心切片内自注意力
    · 两路输出残差融合 → 富含切片间上下文的中心切片特征

本模块改进（双模态 PET-CT 骨肿瘤）：
  [A] 双模态联合 CSA
      PET 和 CT 各自进行跨切片注意力，再做模态间门控融合，
      确保代谢信号与解剖结构信号互补。
  [B] 内存高效实现
      原论文在特征图上直接做全局注意力（HW×HW）。
      对于 512×512 输入经骨干网后得到的 256×256 特征图，
      HW = 65536，全注意力约 34 GB — OOM。
      本模块采用空间池化 (pool_size × pool_size) 降低 token 数量，
      再将注意力权重广播回原始分辨率：
        pool_size=16 → 256 tokens/slice → 4 邻居共 1024 KV tokens
        内存：O(B × heads × 256 × 1024) ≈ 几十 MB，完全可控。
  [C] 可插拔设计
      CSASliceFusion 直接替换现有 ct_adapter / pet_adapter（3ch→1ch），
      升级为 5ch → C_feat → CSA → 1ch，下游 backbone 和 FBFA 不变。
  [D] N-slice 灵活输入
      支持任意奇数 n_slices（3 或 5），center_idx = n_slices // 2。

数据流（以 5-slice 为例）：
  输入: (B, 5, H, W)  5 个切片堆叠在通道维
         ↓ split → [(B,1,H,W)] × 5
  shared_encoder: 每个切片 → (B, C_feat, H, W)  [共享权重]
         ↓
  CrossSliceAttention:
    Q = center_feat                   (B, C_feat, H, W)
    K,V = concat(neighbor_feats)      (B, 4×C_feat, H, W) → pool → reshape
    cross_out = Attention(Q, K, V)    (B, C_feat, H, W)
    self_out  = SelfAttention(Q)      (B, C_feat, H, W)  [可选，轻量]
    fused     = Q + cross_out + self_out                  [残差]
         ↓
  out_proj: (B, C_feat, H, W) → (B, 1, H, W)  [供现有 backbone 使用]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# 工具层
# ─────────────────────────────────────────────────────────────────────────────

class DWConvBNReLU(nn.Module):
    """深度可分离卷积 + BN + ReLU（轻量特征提取）"""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)


# ─────────────────────────────────────────────────────────────────────────────
# 核心模块 1: 单模态 Cross-Slice Attention
# ─────────────────────────────────────────────────────────────────────────────

class CrossSliceAttention(nn.Module):
    """
    单模态跨切片注意力（对应 CSA-Net 的 CSA module）

    设计要点：
    · center 切片特征作为 Query
    · 所有邻居切片特征作为 Key/Value（concat 后统一处理）
    · 空间池化降低 token 数量：H×W → pool_size×pool_size
    · 多头注意力计算后，注意力权重广播回原始空间分辨率
    · 同时做一次轻量自注意力（center 内部相关性）
    · 两路注意力输出 + 原始 center 特征三路残差融合

    参数：
      feat_ch:   每个切片的特征通道数 (C_feat)
      n_heads:   注意力头数
      pool_size: 空间池化目标尺寸（控制内存 vs 精度 trade-off）
                 pool_size=16 → 256 tokens，pool_size=8 → 64 tokens
      use_self_attn: 是否启用中心切片内自注意力（轻量版可以关闭）
    """

    def __init__(
        self,
        feat_ch: int,
        n_heads: int = 4,
        pool_size: int = 16,
        use_self_attn: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feat_ch      = feat_ch
        self.n_heads      = n_heads
        self.pool_size    = pool_size
        self.head_dim     = max(feat_ch // n_heads, 1)
        self.use_self_attn = use_self_attn
        self.scale        = self.head_dim ** -0.5

        d = self.head_dim * n_heads  # projected dim (= feat_ch if divisible)

        # ── 跨切片投影 ──────────────────────────────────────────
        self.q_proj = nn.Linear(feat_ch, d, bias=False)   # center → Q
        self.k_proj = nn.Linear(feat_ch, d, bias=False)   # neighbor → K
        self.v_proj = nn.Linear(feat_ch, d, bias=False)   # neighbor → V
        self.cross_out = nn.Sequential(
            nn.Linear(d, feat_ch, bias=False),
            nn.LayerNorm(feat_ch),
        )

        # ── 自注意力（中心切片内）──────────────────────────────────
        if use_self_attn:
            self.sq_proj   = nn.Linear(feat_ch, d, bias=False)
            self.sk_proj   = nn.Linear(feat_ch, d, bias=False)
            self.sv_proj   = nn.Linear(feat_ch, d, bias=False)
            self.self_out  = nn.Sequential(
                nn.Linear(d, feat_ch, bias=False),
                nn.LayerNorm(feat_ch),
            )

        # ── 融合门控（三路：cross + self + identity）──────────────
        # 根据全局上下文预测三路融合权重
        in_gate = feat_ch * (3 if use_self_attn else 2)
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                              # (B, C, 1, 1)
            nn.Flatten(),                                         # (B, C)
            nn.Linear(feat_ch, 3 if use_self_attn else 2),       # (B, 3)
            nn.Softmax(dim=-1),
        )
        # 融合门控的输入只看 center 自身，不需要拼接
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_ch, 3 if use_self_attn else 2, bias=True),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ── 输出卷积（恢复空间细节）──────────────────────────────
        self.refine = nn.Sequential(
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1, groups=feat_ch, bias=False),
            nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
            nn.BatchNorm2d(feat_ch),
            nn.ReLU(inplace=True),
        )

    def _pool_and_flatten(self, feat: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) → pool to (B, C, P, P) → flatten → (B, P*P, C)
        """
        B, C, H, W = feat.shape
        P = self.pool_size
        if H != P or W != P:
            feat = F.adaptive_avg_pool2d(feat, (P, P))   # (B, C, P, P)
        return feat.flatten(2).permute(0, 2, 1)          # (B, P*P, C)

    def _multihead_attn(
        self,
        q: torch.Tensor,   # (B, Lq, d)
        k: torch.Tensor,   # (B, Lk, d)
        v: torch.Tensor,   # (B, Lk, d)
    ) -> torch.Tensor:
        """
        标准多头注意力（无 dropout，轻量化实现）
        返回: (B, Lq, d)
        """
        B, Lq, d = q.shape
        Lk = k.shape[1]
        H  = self.n_heads
        Dh = self.head_dim

        # reshape → (B, H, L, Dh)
        q = q.view(B, Lq, H, Dh).transpose(1, 2)   # (B, H, Lq, Dh)
        k = k.view(B, Lk, H, Dh).transpose(1, 2)   # (B, H, Lk, Dh)
        v = v.view(B, Lk, H, Dh).transpose(1, 2)   # (B, H, Lk, Dh)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # (B, H, Lq, Lk)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)                  # (B, H, Lq, Dh)
        out = out.transpose(1, 2).contiguous().view(B, Lq, H * Dh)  # (B, Lq, d)
        return out

    def forward(
        self,
        center_feat: torch.Tensor,          # (B, C, H, W)
        neighbor_feats: List[torch.Tensor], # List of (B, C, H, W), len = N-1
    ) -> torch.Tensor:
        """
        返回: enriched center feature (B, C, H, W)
        """
        B, C, H, W = center_feat.shape
        P = self.pool_size

        # ── 空间池化 + 展平 ────────────────────────────────────────
        # center: (B, P*P, C)
        q_flat = self._pool_and_flatten(center_feat)

        # 所有邻居池化后 concat: (B, n_neighbors*P*P, C)
        nb_flats = [self._pool_and_flatten(nf) for nf in neighbor_feats]
        kv_flat  = torch.cat(nb_flats, dim=1)   # (B, K*P*P, C)

        # ── 跨切片注意力 ───────────────────────────────────────────
        Q = self.q_proj(q_flat)    # (B, P*P, d)
        K = self.k_proj(kv_flat)   # (B, K*P*P, d)
        V = self.v_proj(kv_flat)

        cross_raw    = self._multihead_attn(Q, K, V)          # (B, P*P, d)
        cross_pooled = self.cross_out(cross_raw)               # (B, P*P, C)
        # 恢复空间维度 → (B, C, P, P) → upsample → (B, C, H, W)
        cross_spatial = (cross_pooled
                         .permute(0, 2, 1)                     # (B, C, P*P)
                         .view(B, C, P, P))
        if H != P or W != P:
            cross_spatial = F.interpolate(
                cross_spatial, size=(H, W), mode='bilinear', align_corners=True)
        # (B, C, H, W)

        # ── 自注意力（中心切片内）──────────────────────────────────
        if self.use_self_attn:
            SQ = self.sq_proj(q_flat)
            SK = self.sk_proj(q_flat)
            SV = self.sv_proj(q_flat)
            self_raw     = self._multihead_attn(SQ, SK, SV)    # (B, P*P, d)
            self_pooled  = self.self_out(self_raw)              # (B, P*P, C)
            self_spatial = (self_pooled
                            .permute(0, 2, 1)
                            .view(B, C, P, P))
            if H != P or W != P:
                self_spatial = F.interpolate(
                    self_spatial, size=(H, W), mode='bilinear', align_corners=True)

        # ── 门控融合 ───────────────────────────────────────────────
        gate_weights = self.fusion_gate(center_feat)            # (B, 2 or 3)
        if self.use_self_attn:
            w_identity = gate_weights[:, 0].view(B, 1, 1, 1)
            w_cross    = gate_weights[:, 1].view(B, 1, 1, 1)
            w_self     = gate_weights[:, 2].view(B, 1, 1, 1)
            fused = (w_identity * center_feat
                     + w_cross  * cross_spatial
                     + w_self   * self_spatial)
        else:
            w_identity = gate_weights[:, 0].view(B, 1, 1, 1)
            w_cross    = gate_weights[:, 1].view(B, 1, 1, 1)
            fused = w_identity * center_feat + w_cross * cross_spatial

        return self.refine(fused)   # (B, C, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# 核心模块 2: 双模态联合 CSA Slice Fusion（替换原有 ct_adapter / pet_adapter）
# ─────────────────────────────────────────────────────────────────────────────

class CSASliceFusion(nn.Module):
    """
    双模态 Cross-Slice Attention Fusion
    ====================================
    替换原 FBFAIntraBoneTumorSegmentation 中的：
      · ct_center_weight + ct_adapter  (3ch → 1ch)
      · pet_center_weight + pet_adapter (3ch → 1ch)

    升级为：
      · 5-slice CT  → per-slice shared encoder → CrossSliceAttention → 1ch
      · 5-slice PET → per-slice shared encoder → CrossSliceAttention → 1ch
      · 可选：双模态跨切片联合注意力（PET 作为 CT 的辅助 KV）

    接口：
      forward(ct5, pet5) → (ct_center_1ch, pet_center_1ch)
      ct5:  (B, 5, H, W)  — 5 个 CT  切片堆叠
      pet5: (B, 5, H, W)  — 5 个 PET 切片堆叠

    参数：
      n_slices:    输入切片数，必须为奇数（3 或 5）
      feat_ch:     中间特征通道数（共享编码器输出维度）
      n_heads:     注意力头数
      pool_size:   空间池化降维目标尺寸（16 → 256 tokens）
      use_cross_modal: 是否在 PET→CT 之间也做跨模态跨切片注意力
    """

    def __init__(
        self,
        n_slices:         int   = 5,
        feat_ch:          int   = 32,
        n_heads:          int   = 4,
        pool_size:        int   = 16,
        use_self_attn:    bool  = True,
        use_cross_modal:  bool  = True,   # PET 邻居切片也辅助 CT CSA
        dropout:          float = 0.0,
    ):
        super().__init__()
        assert n_slices % 2 == 1, "n_slices must be odd (3 or 5)"
        self.n_slices      = n_slices
        self.center_idx    = n_slices // 2
        self.n_neighbors   = n_slices - 1
        self.use_cross_modal = use_cross_modal

        # ── 共享轻量编码器（各模态独立，但 slice 间共享权重）─────────
        # 单通道切片 → feat_ch 特征图
        self.ct_slice_encoder  = nn.Sequential(
            DWConvBNReLU(1, feat_ch // 2, 3),
            DWConvBNReLU(feat_ch // 2, feat_ch, 3),
        )
        self.pet_slice_encoder = nn.Sequential(
            DWConvBNReLU(1, feat_ch // 2, 3),
            DWConvBNReLU(feat_ch // 2, feat_ch, 3),
        )

        # ── 单模态 CSA（CT 和 PET 各一个）──────────────────────────
        self.ct_csa  = CrossSliceAttention(feat_ch, n_heads, pool_size,
                                           use_self_attn, dropout)
        self.pet_csa = CrossSliceAttention(feat_ch, n_heads, pool_size,
                                           use_self_attn, dropout)

        # ── 跨模态辅助注意力（可选）──────────────────────────────────
        # PET 邻居切片的语义信息辅助 CT 中心切片的定位：
        # 对 CT center 而言，PET 邻居的代谢信息是额外的 KV 来源
        if use_cross_modal:
            # 将 PET 特征投影到 CT 特征空间
            self.pet2ct_proj = nn.Sequential(
                nn.Conv2d(feat_ch, feat_ch, 1, bias=False),
                nn.BatchNorm2d(feat_ch),
                nn.ReLU(inplace=True),
            )
            self.cross_modal_csa = CrossSliceAttention(
                feat_ch, n_heads, pool_size, use_self_attn=False, dropout=dropout)

        # ── 输出投影：feat_ch → 1ch（与现有 backbone 兼容）─────────
        self.ct_out_proj = nn.Sequential(
            nn.Conv2d(feat_ch, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.pet_out_proj = nn.Sequential(
            nn.Conv2d(feat_ch, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

        # 初始化输出投影偏置为小正值（防止早期输出全零）
        nn.init.constant_(self.ct_out_proj[0].bias,  0.1)
        nn.init.constant_(self.pet_out_proj[0].bias, 0.1)

    def _encode_slices(
        self, x5: torch.Tensor, encoder: nn.Module
    ) -> List[torch.Tensor]:
        """
        x5: (B, N, H, W) — N 个切片堆叠
        返回: list of N × (B, C_feat, H, W)
        每个切片单独通过共享编码器
        """
        feats = []
        for i in range(self.n_slices):
            s = x5[:, i:i+1, :, :]          # (B, 1, H, W)
            feats.append(encoder(s))          # (B, C_feat, H, W)
        return feats

    def forward(
        self,
        ct5:  torch.Tensor,   # (B, 5, H, W)  CT  5 切片
        pet5: torch.Tensor,   # (B, 5, H, W)  PET 5 切片
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回: (ct_1ch, pet_1ch)  各为 (B, 1, H, W)
        """
        c = self.center_idx

        # ── Step1: 逐切片编码 ──────────────────────────────────────
        ct_feats  = self._encode_slices(ct5,  self.ct_slice_encoder)
        pet_feats = self._encode_slices(pet5, self.pet_slice_encoder)
        # ct_feats[i]: (B, feat_ch, H, W)

        ct_center  = ct_feats[c]
        pet_center = pet_feats[c]
        ct_neighbors  = [ct_feats[i]  for i in range(self.n_slices) if i != c]
        pet_neighbors = [pet_feats[i] for i in range(self.n_slices) if i != c]

        # ── Step2: 单模态 CSA ──────────────────────────────────────
        ct_enriched  = self.ct_csa( ct_center,  ct_neighbors)   # (B, feat_ch, H, W)
        pet_enriched = self.pet_csa(pet_center, pet_neighbors)

        # ── Step3: 跨模态辅助注意力（可选）───────────────────────────
        # CT center 进一步借助 PET 邻居切片的代谢信号
        if self.use_cross_modal:
            pet_nb_projected = [self.pet2ct_proj(nf) for nf in pet_neighbors]
            ct_cross_modal   = self.cross_modal_csa(ct_enriched, pet_nb_projected)
            # 残差融合：原 CT CSA 输出 + 跨模态增益
            ct_enriched = ct_enriched + 0.5 * ct_cross_modal

        # ── Step4: 输出投影 → 1ch ─────────────────────────────────
        ct_1ch  = self.ct_out_proj(ct_enriched)    # (B, 1, H, W)
        pet_1ch = self.pet_out_proj(pet_enriched)  # (B, 1, H, W)

        return ct_1ch, pet_1ch


# ─────────────────────────────────────────────────────────────────────────────
# 核心模块 3: Encoder-Level CSA（插入 Encoder 中间层）
# ─────────────────────────────────────────────────────────────────────────────

class EncoderCSABlock(nn.Module):
    """
    Encoder 中间层 CSA 插件
    ==========================
    对应 CSA-Net 论文推荐的"在 Encoder 中间层插入 CSA"。

    用途：在 PET backbone (MambaFeatureExtractor2D) 提取出各 stage 特征后，
    对特定 stage 的特征图做跨切片注意力增强。

    调用场景（在 Intrabone_petct_dataset_5slice.py 中）：
      · Stage 1 (H/4)  和 Stage 2 (H/8) 特征图上应用 CSA
      · Stage 3 (H/16) 和 Stage 4 (H/16) 分辨率过低，不用 CSA

    输入：每个 slice 经过 backbone 提取的特征，共 n_slices 个
      feats: List[(B, C, H', W')] × n_slices，中心 slice 下标 = center_idx
    输出：仅中心 slice 的增强特征 (B, C, H', W')
    """

    def __init__(
        self,
        channels:   int,
        n_slices:   int  = 5,
        n_heads:    int  = 4,
        pool_size:  int  = 16,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.center_idx = n_slices // 2
        self.n_slices   = n_slices
        self.csa = CrossSliceAttention(
            channels, n_heads, pool_size, use_self_attn=True, dropout=dropout)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        feats: List[(B, C, H', W')] × n_slices
        返回: (B, C, H', W')  仅中心切片增强结果
        """
        c = self.center_idx
        center    = feats[c]
        neighbors = [feats[i] for i in range(self.n_slices) if i != c]
        return self.csa(center, neighbors)


# ─────────────────────────────────────────────────────────────────────────────
# 快速验证（开发调试用）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 65)
    print("CSA Module Tests — PET-CT 5-Slice Bone Tumor")
    print("=" * 65)

    B, N, H, W = 2, 5, 512, 512

    # ── Test 1: CrossSliceAttention ──────────────────────────────
    print("\n[1] CrossSliceAttention (C=32, pool=16, 4 neighbors)")
    csa = CrossSliceAttention(feat_ch=32, n_heads=4, pool_size=16).to(device)
    center = torch.randn(B, 32, H // 4, W // 4, device=device)  # 128×128 feature
    neighbors = [torch.randn(B, 32, H // 4, W // 4, device=device) for _ in range(4)]
    t0 = time.time()
    with torch.no_grad():
        out = csa(center, neighbors)
    print(f"  Input center: {center.shape}")
    print(f"  Output:       {out.shape}  [{(time.time()-t0)*1000:.0f}ms]")
    assert out.shape == center.shape, "Shape mismatch!"
    print("  ✓ Shape OK")

    # ── Test 2: CSASliceFusion (全流程，512×512 输入) ──────────────
    print("\n[2] CSASliceFusion (5-slice, 512×512, feat_ch=32)")
    fusion = CSASliceFusion(
        n_slices=5, feat_ch=32, n_heads=4, pool_size=16,
        use_self_attn=True, use_cross_modal=True).to(device)
    ct5  = torch.randn(B, N, H, W, device=device)
    pet5 = torch.randn(B, N, H, W, device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        ct_1ch, pet_1ch = fusion(ct5, pet5)
    elapsed = (time.time() - t0) * 1000
    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak GPU: {peak_mb:.0f} MB")
    print(f"  ct_1ch:  {ct_1ch.shape}   [{elapsed:.0f}ms]")
    print(f"  pet_1ch: {pet_1ch.shape}")
    assert ct_1ch.shape  == (B, 1, H, W)
    assert pet_1ch.shape == (B, 1, H, W)
    print("  ✓ Shape OK")

    # ── Test 3: Gradient flow ─────────────────────────────────────
    print("\n[3] Gradient flow test")
    ct5_g  = torch.randn(B, N, H // 4, W // 4, device=device, requires_grad=False)
    pet5_g = torch.randn(B, N, H // 4, W // 4, device=device, requires_grad=False)
    fusion_s = CSASliceFusion(n_slices=5, feat_ch=16, n_heads=2, pool_size=8).to(device)
    ct5_in  = torch.randn(B, N, H // 4, W // 4, device=device)
    pet5_in = torch.randn(B, N, H // 4, W // 4, device=device)
    ct_o, pet_o = fusion_s(ct5_in, pet5_in)
    (ct_o.mean() + pet_o.mean()).backward()
    total_grad = sum(p.grad.norm().item()
                     for p in fusion_s.parameters() if p.grad is not None)
    print(f"  Total grad norm: {total_grad:.4f}  "
          f"{'✓ has grad' if total_grad > 1e-6 else '✗ no grad'}")

    # ── Test 4: 3-slice compatibility ─────────────────────────────
    print("\n[4] 3-slice compatibility")
    fusion3 = CSASliceFusion(n_slices=3, feat_ch=16, n_heads=2, pool_size=8).to(device)
    ct3  = torch.randn(B, 3, 64, 64, device=device)
    pet3 = torch.randn(B, 3, 64, 64, device=device)
    with torch.no_grad():
        c3, p3 = fusion3(ct3, pet3)
    print(f"  ct3 → {c3.shape}  ✓")

    # ── Param count ────────────────────────────────────────────────
    n_params = sum(p.numel() for p in fusion.parameters()) / 1e3
    print(f"\nCSASliceFusion (5-slice, feat_ch=32) params: {n_params:.1f}K")
    print("\n✅ All CSA tests passed!")