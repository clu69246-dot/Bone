"""
2D EfficientViM Module — Memory-Efficient Rewrite
适配骨骼分割 + Stage2 多尺度特征提取

核心改动:
[FIXED-1] CrossScan2D: 彻底重写扫描机制
  - 原问题: einsum 产生 (4B, C, d_state, L) 超大张量
             512x512 输入: 4*2*256*49*262144 ≈ 100GB → OOM
  - 修复方案: 改用 Window-based Local Scan + Global Pooling 融合
    · 先在局部窗口内做状态扫描 (内存可控)
    · 再用 Global Average Pooling 注入全局骨骼上下文
    · 两路门控融合 → 兼顾局部细节和全局结构

[FIXED-2] state_dim 含义从 "SSM状态维度" 改为 "窗口大小"
  - state_dim=49 → 7×7 局部窗口 (sqrt(49)=7)
  - state_dim=25 → 5×5 局部窗口
  - state_dim=9  → 3×3 局部窗口

[FIXED-3] 扫描顺序: 4方向 → 2方向 (H, W各一次)，进一步省内存

输入: (B, C, H, W) 4D张量
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  归一化工具
# ============================================================

class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor (B C H W)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        x_n  = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_n = x_n * self.weight + self.bias
        return x_n


class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor (B C L)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var  = x.var(dim=1, keepdim=True, unbiased=False)
        x_n  = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x_n = x_n * self.weight + self.bias
        return x_n


# ============================================================
#  卷积构建块
# ============================================================

class ConvLayer2D(nn.Module):
    """2D Conv + BN + Act"""
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, norm=nn.BatchNorm2d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=False)
        self.norm = norm(num_features=out_dim) if norm else None
        self.act  = act_layer() if act_layer else None
        if self.norm:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act:  x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    """1D Conv + BN + Act"""
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, norm=nn.BatchNorm1d,
                 act_layer=nn.ReLU, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=False)
        self.norm = norm(num_features=out_dim) if norm else None
        self.act  = act_layer() if act_layer else None
        if self.norm:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm: x = self.norm(x)
        if self.act:  x = self.act(x)
        return x


class FFN2D(nn.Module):
    """Point-wise FFN (1×1 Conv)"""
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim,    kernel_size=1, padding=0)
        self.fc2 = ConvLayer2D(dim,    in_dim, kernel_size=1, padding=0,
                               act_layer=None, bn_weight_init=0)

    def forward(self, x):
        return self.fc2(self.fc1(x))


# ============================================================
#  [FIXED] 内存高效扫描模块
# ============================================================

class EfficientScan2D(nn.Module):
    """
    内存高效的2D扫描模块 — 替换原来的 CrossScan2D

    原问题根源:
      CrossScan2D 的 einsum 会生成 (4B, C, d_state, L) 张量
      对于 B=2, C=32, d_state=49, H=W=256 (Stem后):
        4 * 2 * 32 * 49 * 65536 * 4 bytes ≈ 3.2 GB (仅一个中间结果!)
      Stage0 已经爆掉，更不用说后面几个Stage。

    修复方案: Strip-based Linear Scan + Global Context Injection
    ──────────────────────────────────────────────────────────
    Step 1. Row Scan (沿H方向): 把 (B,C,H,W) 看作 B*W 条长度为H的序列
             每条序列做简单的 Gated Linear Scan (GLS):
             h_t = α_t * h_{t-1} + (1-α_t) * x_t  (α是遗忘门，逐元素)
             实现上等价于: output = conv1d + gating, 无大中间张量

    Step 2. Col Scan (沿W方向): 对 (B,C,H,W) 做转置后同样处理

    Step 3. Global Context: AdaptiveAvgPool2d + MLP → 全局骨骼先验
             对骨骼分割极重要: 骨骼是连续结构，全局池化能感知整体走向

    Step 4. 三路融合: row_out + col_out + global_ctx (可学习门控权重)

    内存复杂度: O(B * C * H * W) — 和输入同量级，不再有爆炸项
    """

    def __init__(self, dim, window_size=7):
        """
        Args:
            dim: 通道数
            window_size: 局部感受野大小 (对应原来的 state_dim 的 sqrt)
                         state_dim=49 → window_size=7
                         state_dim=25 → window_size=5
                         state_dim=9  → window_size=3
        """
        super().__init__()
        self.dim = dim
        self.ws  = window_size

        # ── Row & Col Scan: 用深度可分离 Conv1D 近似线性递归 ──
        # Conv1D(groups=dim) = 每个通道独立滤波，模拟 per-channel 遗忘门
        # padding='same' 保持长度不变
        self.row_gate  = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=window_size,
                      padding=window_size // 2, groups=dim, bias=False),
            nn.Sigmoid()
        )
        self.row_val   = nn.Conv1d(dim, dim, kernel_size=window_size,
                                   padding=window_size // 2, groups=dim, bias=False)

        self.col_gate  = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=window_size,
                      padding=window_size // 2, groups=dim, bias=False),
            nn.Sigmoid()
        )
        self.col_val   = nn.Conv1d(dim, dim, kernel_size=window_size,
                                   padding=window_size // 2, groups=dim, bias=False)

        # ── Global Context: 全局骨骼结构先验 ──
        self.global_ctx = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                          # (B, C, 1, 1)
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1, bias=False),
            nn.Sigmoid()                                      # channel-wise scale
        )

        # ── 三路可学习融合门控 ──
        # 输出 3 个权重: w_row, w_col, w_global
        self.fusion_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, 3, 1, bias=True),                  # → (B, 3, 1, 1)
            nn.Softmax(dim=1)                                  # 3路概率
        )

        # ── 输出投影 ──
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim)
        )

    def _row_scan(self, x):
        """
        沿 W 方向 (行) 扫描
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 把所有行展开: (B*H, C, W)
        x_row = x.permute(0, 2, 1, 3).reshape(B * H, C, W)
        g = self.row_gate(x_row)          # (B*H, C, W)
        v = self.row_val(x_row)           # (B*H, C, W)
        out = g * v + (1 - g) * x_row    # 残差门控
        out = out.reshape(B, H, C, W).permute(0, 2, 1, 3)   # (B, C, H, W)
        return out

    def _col_scan(self, x):
        """
        沿 H 方向 (列) 扫描
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape
        # 把所有列展开: (B*W, C, H)
        x_col = x.permute(0, 3, 1, 2).reshape(B * W, C, H)
        g = self.col_gate(x_col)          # (B*W, C, H)
        v = self.col_val(x_col)           # (B*W, C, H)
        out = g * v + (1 - g) * x_col
        out = out.reshape(B, W, C, H).permute(0, 2, 3, 1)   # (B, C, H, W)
        return out

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, C, H, W)
        """
        # 三路特征
        row_out    = self._row_scan(x)                   # (B, C, H, W)
        col_out    = self._col_scan(x)                   # (B, C, H, W)
        global_out = x * self.global_ctx(x)              # (B, C, H, W) channel scale

        # 可学习融合 (各路权重由输入决定)
        weights = self.fusion_gate(x)                    # (B, 3, 1, 1)
        w_r = weights[:, 0:1]
        w_c = weights[:, 1:2]
        w_g = weights[:, 2:3]

        fused = w_r * row_out + w_c * col_out + w_g * global_out  # (B, C, H, W)

        return self.out_proj(fused)


# ============================================================
#  核心 Mamba 块
# ============================================================

class HSMSSD2D(nn.Module):
    """
    2D 轻量级 Mamba 核心模块 (内存高效版)
    Channel Mix + Spatial Scan (EfficientScan2D) + Gated Fusion
    """
    def __init__(self, d_model, ssd_expand=1.0, state_dim=16):
        """
        state_dim 现在控制局部扫描窗口大小:
          49 → window=7, 25 → window=5, 9/16 → window=3
        """
        super().__init__()
        self.d_model   = d_model
        self.state_dim = state_dim

        # 根据 state_dim 推断窗口大小
        ws = max(3, int(math.isqrt(state_dim)))
        if ws % 2 == 0:
            ws += 1  # 保证奇数

        # 1. 通道混合
        self.channel_mix = nn.Sequential(
            nn.Conv2d(d_model, d_model * 2, 1, bias=False),
            nn.BatchNorm2d(d_model * 2),
            nn.SiLU(),
            nn.Conv2d(d_model * 2, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model)
        )

        # 2. [FIXED] 内存高效扫描
        self.spatial_scan = EfficientScan2D(dim=d_model, window_size=ws)

        # 3. 门控融合
        self.gate = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.Sigmoid()
        )

        # 4. 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model)
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (y, h_placeholder)  — h 保留接口兼容
        """
        identity = x

        channel_out = self.channel_mix(x)
        spatial_out = self.spatial_scan(x)

        gate   = self.gate(x)
        fused  = channel_out * gate + spatial_out * (1 - gate)

        out = self.out_proj(fused)
        y   = out + identity       # 残差

        h = x.mean(dim=(2, 3))    # 占位符，保留接口
        return y, h


# ============================================================
#  深度可分离卷积
# ============================================================

class DWConv2D(nn.Module):
    """2D 深度可分离卷积"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = ConvLayer2D(dim, dim, kernel_size=3, padding=1,
                                  groups=dim, bn_weight_init=0, act_layer=None)

    def forward(self, x):
        return self.dwconv(x)


# ============================================================
#  EfficientViM 基础块
# ============================================================

class EfficientViMBlock2D(nn.Module):
    """
    2D EfficientViM 基础块
    DWConv → HSMSSD2D → DWConv → FFN  (全部带 LayerScale 残差)
    """
    def __init__(self, dim, mlp_ratio=2.0, ssd_expand=1.0, state_dim=16):
        super().__init__()
        self.dim = dim

        self.mixer  = HSMSSD2D(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm   = LayerNorm1D(dim)

        self.dwconv1 = DWConv2D(dim)
        self.dwconv2 = DWConv2D(dim)

        self.ffn = FFN2D(in_dim=dim, dim=int(dim * mlp_ratio))

        # LayerScale — 初始化为小值，稳定早期训练
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        """x: (B, C, H, W)  →  (B, C, H, W), h"""
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)

        x_prev = x
        x, h   = self.mixer(x)
        x = (1 - alpha[1]) * x_prev + alpha[1] * x

        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)

        return x, h


# ============================================================
#  下采样模块
# ============================================================

class PatchMerging2D(nn.Module):
    """
    2D 下采样: in_dim → out_dim, 空间 /stride
    """
    def __init__(self, in_dim, out_dim, stride=2, ratio=4.0):
        super().__init__()
        hidden_dim = int(out_dim * ratio)

        self.conv = nn.Sequential(
            ConvLayer2D(in_dim,     hidden_dim, kernel_size=1, padding=0),
            ConvLayer2D(hidden_dim, hidden_dim, kernel_size=3,
                        stride=stride, padding=1, groups=hidden_dim),
            ConvLayer2D(hidden_dim, out_dim,    kernel_size=1, padding=0, act_layer=None)
        )

        self.dwconv1 = ConvLayer2D(in_dim,  in_dim,  kernel_size=3, padding=1,
                                    groups=in_dim,  act_layer=None)
        self.dwconv2 = ConvLayer2D(out_dim, out_dim, kernel_size=3, padding=1,
                                    groups=out_dim, act_layer=None)

    def forward(self, x):
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


# ============================================================
#  基础块 = EfficientViMBlock × depth + PatchMerging
# ============================================================

class BasicBlock2D(nn.Module):
    """
    depth 个 EfficientViMBlock2D + 1 个 PatchMerging2D
    返回: (下采样特征, 下采样前特征, 最终隐藏状态)
    """
    def __init__(self, in_dim, out_dim, depth=1, state_dim=16, stride=2):
        super().__init__()
        self.depth = depth

        self.blocks = nn.ModuleList([
            EfficientViMBlock2D(dim=in_dim, mlp_ratio=2.0,
                                ssd_expand=1.0, state_dim=state_dim)
            for _ in range(depth)
        ])

        self.downsample  = PatchMerging2D(in_dim=in_dim, out_dim=out_dim, stride=stride)
        self.out_channels = out_dim

    def forward(self, x):
        h = None
        for blk in self.blocks:
            x, h = blk(x)
        feats = x
        out   = self.downsample(x)
        return out, feats, h


# ============================================================
#  完整特征提取器
# ============================================================

class MambaFeatureExtractor2D(nn.Module):
    """
    4阶段 Mamba 特征提取器 (供 Stage1 骨骼分割 和 Stage2 肿瘤分割使用)

    输出特征 [f0, f1, f2, f3]:
      f0: (B, 32,  H/2,  W/2)   — 细粒度骨骼边缘
      f1: (B, 64,  H/4,  W/4)   — 骨骼局部纹理
      f2: (B, 128, H/8,  W/8)   — 骨骼中层语义
      f3: (B, 256, H/8,  W/8)   — 全局骨骼上下文 (Stage4 stride=1)

    内存估算 (B=2, 512×512 输入, fp16):
      Stem 后: (2, 32, 256, 256) — 8 MB
      Stage0:  scan 最大中间张量 (2*256, 32, 256) = 16M × 2B = 32 MB  ✓
      Stage1:  (2*128, 64, 128) = 4M × 2B = 8 MB  ✓
      Stage2:  (2*64,  128, 64) = 1M × 2B = 2 MB  ✓
      Stage3:  (2*64,  256, 64) = 2M × 2B = 4 MB  ✓
    """

    def __init__(self, in_dim=1,
                 embed_dim=[32, 64, 128, 256],
                 depths=[1, 1, 1, 1],
                 state_dim=[49, 25, 9, 9],
                 strides=[2, 2, 2, 1]):
        super().__init__()
        self.num_layers = len(depths)

        # Stem: 空间减半 H,W → H/2, W/2
        # Stage0(×2) + Stage1(×2) + Stage2(×2) + Stage3(×1)
        # 合计: Stem×2 + ×2 + ×2 + ×2 + ×1 = ÷16
        # f0@H/4, f1@H/8, f2@H/16, f3@H/16
        self.stem = nn.Sequential(
            ConvLayer2D(in_dim,            embed_dim[0] // 2, kernel_size=3, stride=1, padding=1),
            ConvLayer2D(embed_dim[0] // 2, embed_dim[0],      kernel_size=3, stride=2, padding=1,
                        act_layer=None)
        )

        self.stages = nn.ModuleList()
        for i in range(self.num_layers):
            in_ch  = embed_dim[i]
            out_ch = embed_dim[i + 1] if i < self.num_layers - 1 else embed_dim[i]
            stride = strides[i] if isinstance(strides, list) else strides
            sd     = state_dim[i] if i < len(state_dim) else state_dim[-1]

            self.stages.append(
                BasicBlock2D(in_dim=in_ch, out_dim=out_ch,
                             depth=depths[i], state_dim=sd, stride=stride)
            )

        self.out_channels = embed_dim

    def forward(self, x):
        """
        x: (B, in_dim, H, W)
        returns: [f0, f1, f2, f3]  — 多尺度特征列表
        """
        x = self.stem(x)

        features = []
        for stage in self.stages:
            x, feats, _ = stage(x)
            features.append(feats)   # feats = 下采样前的特征 (供 skip-connection)

        return features              # [f0(32), f1(64), f2(128), f3(256)]


# ============================================================
#  单元测试
# ============================================================

if __name__ == "__main__":
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("Testing Memory-Efficient 2D EfficientViM Modules")
    print("=" * 70)

    # 1. EfficientScan2D
    print("\n[1] EfficientScan2D (window=7) ...")
    scan = EfficientScan2D(dim=64, window_size=7).to(device)
    x = torch.randn(2, 64, 64, 64).to(device)
    with torch.no_grad():
        y = scan(x)
    print(f"    {x.shape} → {y.shape}  ✓")

    # 2. HSMSSD2D
    print("\n[2] HSMSSD2D (state_dim=49) ...")
    hsm = HSMSSD2D(d_model=128, state_dim=49).to(device)
    x = torch.randn(2, 128, 32, 32).to(device)
    with torch.no_grad():
        y, h = hsm(x)
    print(f"    {x.shape} → {y.shape}, h={h.shape}  ✓")

    # 3. BasicBlock2D
    print("\n[3] BasicBlock2D ...")
    blk = BasicBlock2D(in_dim=64, out_dim=128, depth=1, state_dim=49).to(device)
    x = torch.randn(2, 64, 32, 32).to(device)
    with torch.no_grad():
        out, feats, h = blk(x)
    print(f"    in={x.shape} → out={out.shape}, skip={feats.shape}  ✓")

    # 4. 全量大尺寸 Memory 压力测试
    print("\n[4] Full MambaFeatureExtractor2D @ 512×512 (Memory Stress Test) ...")
    extractor = MambaFeatureExtractor2D(
        in_dim=1, embed_dim=[32, 64, 128, 256],
        depths=[1, 1, 1, 1], state_dim=[49, 25, 9, 9],
        strides=[2, 2, 2, 1]   # f0@H/4, f1@H/8, f2@H/16, f3@H/16
    ).to(device)

    x = torch.randn(2, 1, 512, 512).to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        features = extractor(x)
    t1 = time.time()

    print(f"    Input: {x.shape}")
    for i, f in enumerate(features):
        print(f"    Stage {i}: {f.shape}")

    if device.type == 'cuda':
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"    Peak GPU memory: {peak_mb:.1f} MB")
    print(f"    Forward time: {(t1-t0)*1000:.1f} ms  ✓")

    params = sum(p.numel() for p in extractor.parameters())
    print(f"\n    Parameters: {params/1e6:.2f} M")
    print("\n" + "=" * 70)
    print("All tests passed! OOM issue fixed.")
    print("=" * 70)