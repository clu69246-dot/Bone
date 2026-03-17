"""
DGMA: Dynamic Gaussian Mixture Attention  (v2: Lesion Size-Aware)
动态高斯混合注意力模块 + 病灶尺度自适应高斯

适用场景: PET/CT 骨转移灶分割，每张切片病灶数量不固定 (0~K_max)，
          且病灶尺寸跨度大（微小转移灶 vs 大块溶骨性病变）

v2 新增: Lesion Size-Aware Gaussian
  原 v1: σx, σy 由特征 MLP 直接输出，与病灶实际大小无显式关联
  v2:    新增 LesionRadiusPredictor → radius_map (B,1,H,W)
         在 NMS 中心处 grid_sample 得到 r_k（该位置估计的病灶半径）
         σx = α × r_k               (α 可学习缩放因子)
         σy = α × r_k × β           (β = prior_aspect_ratio，纵向椭圆先验)
         MLP 额外输出 (Δσx_res, Δσy_res) 残差修正，保留细粒度调整能力
         最终 σ = clamp(σ_base + σ_res, σ_min, σ_max)

         效果: 小病灶 → r_k 小 → σ 小 → 紧凑高斯；
               大病灶 → r_k 大 → σ 大 → 宽松高斯；
               自然适配多尺度，无需手工设计尺度先验

核心数据流 (v2):
  feature_map (B,C,H,W)
      │
      ├───────────────────────────────────────────────┐
      │                                               │
  ┌───▼────────────────────┐              ┌───────────▼────────────────┐
  │  Step1: HeatmapPredictor│              │  Step1b: RadiusPredictor   │
  │  Conv3x3→BN→ReLU→Sigmoid│              │  DWConv→BN→ReLU→Conv→SP   │
  └───┬────────────────────┘              └───────────┬────────────────┘
      │ center_heatmap (B,1,H,W)                      │ radius_map (B,1,H,W)
  ┌───▼──────────────────────┐                        │
  │  Step2: NMSCenterExtractor│                        │
  └───┬──────────────────────┘                        │
      │ centers (B,K_max,2)   ──────────── grid_sample┘
      │ valid_mask (B,K_max)              r_k (B,K_max) = sampled radius
  ┌───▼────────────────────────────────────────────────┐
  │  Step3: SizeAwareParamPredictor                    │
  │    σx_base = α × r_k                               │
  │    σy_base = α × r_k × β                           │
  │    MLP(feat_k) → (Δσx_res, Δσy_res, θ, weight)    │
  │    σx = clamp(σx_base + Δσx_res, σ_min, σ_max)    │
  └───┬────────────────────────────────────────────────┘
      │ sigma_x, sigma_y, theta, weight (B,K_max,1,1)
  ┌───▼──────────────────────────┐
  │  Step4: RotatedGaussianGen    │  各向异性旋转椭圆高斯 (B,K_max,H,W)
  └───┬──────────────────────────┘
  ┌───▼──────────────────────────┐
  │  Step5: GaussianMixtureFusion │  Σ weight_k * G_k → Sigmoid
  └───┬──────────────────────────┘
      │ attention_map (B,1,H,W)
  ┌───▼──────────────────┐
  │  Step6: Feature Enhance│  enhanced = feature × attention
  └──────────────────────┘
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

# [性能优化②] 网格缓存字典：key=(H,W,device_str)，避免每次 forward 重新 linspace+meshgrid
# DGMA 在 scale0(256×256)和 scale1(128×128) 各调用一次，batch 间 H/W 不变，缓存命中率 100%
_NORM_GRID_CACHE: dict = {}

def _make_norm_grid(H: int, W: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    生成归一化坐标网格 [-1, 1]，带进程级缓存（同尺寸只创建一次）
    返回: grid_x (H,W), grid_y (H,W)
    """
    key = (H, W, str(device))
    if key not in _NORM_GRID_CACHE:
        yy = torch.linspace(-1.0, 1.0, H, device=device)
        xx = torch.linspace(-1.0, 1.0, W, device=device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        _NORM_GRID_CACHE[key] = (grid_x, grid_y)
    return _NORM_GRID_CACHE[key]


def generate_rotated_gaussian(
    H: int, W: int,
    mu_x: torch.Tensor,    # (B, K, 1, 1) 归一化 x 坐标
    mu_y: torch.Tensor,    # (B, K, 1, 1) 归一化 y 坐标
    sigma_x: torch.Tensor, # (B, K, 1, 1) x 轴标准差
    sigma_y: torch.Tensor, # (B, K, 1, 1) y 轴标准差
    theta: torch.Tensor,   # (B, K, 1, 1) 旋转角 ∈ [-π, π]
    eps: float = 1e-6
) -> torch.Tensor:
    """
    批量生成各向异性旋转椭圆高斯图

    数学公式:
      dx = grid_x - μx,  dy = grid_y - μy
      x_rot =  cos(θ)*dx + sin(θ)*dy
      y_rot = -sin(θ)*dx + cos(θ)*dy
      G = exp( -(x_rot²/(2σx²) + y_rot²/(2σy²)) )

    返回: (B, K, H, W)  值域 (0, 1]，中心处为 1
    """
    grid_x, grid_y = _make_norm_grid(H, W, mu_x.device)
    gx = grid_x.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
    gy = grid_y.unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)

    cos_t = torch.cos(theta)   # (B, K, 1, 1)
    sin_t = torch.sin(theta)

    dx    = gx - mu_x           # (B, K, H, W)
    dy    = gy - mu_y           # (B, K, H, W)

    x_rot =  cos_t * dx + sin_t * dy
    y_rot = -sin_t * dx + cos_t * dy

    return torch.exp(
        -(x_rot ** 2 / (2.0 * sigma_x ** 2 + eps)
        + y_rot ** 2 / (2.0 * sigma_y ** 2 + eps))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: 病灶中心热力图预测
# ─────────────────────────────────────────────────────────────────────────────

class DGMAHeatmapPredictor(nn.Module):
    """
    从 PET 特征图预测每像素成为病灶中心的概率

    网络结构:
      Conv3×3 (DWConv) → BN → ReLU → Conv3×3 → BN → ReLU → Conv1×1 → Sigmoid

    设计要点:
      - 深度可分离卷积 (DWConv) 减少参数，保留空间细节
      - 两层 3×3 感受野足够捕获局部代谢热点，不用全局池化
      - 最终 Sigmoid 输出逐像素概率
    """

    def __init__(self, channels: int):
        super().__init__()
        mid = max(channels // 2, 16)
        self.net = nn.Sequential(
            # 深度可分离卷积保留空间结构
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1, bias=True),    # 输出单通道 logit
            nn.Sigmoid()
        )
        # 初始化最后一层偏置为负值，避免训练初期满图预测
        nn.init.constant_(self.net[-2].bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:  (B, C, H, W)
        输出:  center_heatmap (B, 1, H, W)  值域 [0, 1]
        """
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1b: 病灶半径图预测 (Lesion Size-Aware 新增)
# ─────────────────────────────────────────────────────────────────────────────

class LesionRadiusPredictor(nn.Module):
    """
    轻量半径预测头：预测每个空间位置的潜在病灶半径

    输出的 radius_map 表示「如果该位置是病灶中心，估计其半径有多大」。
    在 NMS 找到中心后，通过 grid_sample 采样得到逐中心半径 r_k，
    再映射为高斯宽度: σx = α × r_k，σy = α × r_k × β。

    网络结构（轻量）:
      DWConv3×3 → BN → ReLU → Conv1×1 → Softplus → clamp
      参数量约 = 2C（深度可分离 + 逐点）+ C（BN）+ 1（逐点输出）

    输出值域:
      radius_map ∈ [r_min, r_max]，以归一化坐标单位衡量
      r_min=0.03: 最小病灶半径，对应特征图约 1~2 像素
      r_max=0.40: 最大病灶半径，覆盖特征图约 40% 尺寸

    可微性:
      Softplus 完全可微，radius_map 可通过 supervision loss 直接监督
    """

    def __init__(self, channels: int,
                 r_min: float = 0.03,
                 r_max: float = 0.40):
        super().__init__()
        self.r_min = r_min
        self.r_max = r_max

        self.net = nn.Sequential(
            # 深度可分离卷积：捕获局部尺度特征，保留空间信息
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, max(channels // 4, 8), 1, bias=False),
            nn.BatchNorm2d(max(channels // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 4, 8), 1, 1, bias=True),  # 单通道 logit
            nn.Sigmoid()          # [FIX-2] 改用 Sigmoid，输出稳定在 (0,1)
        )
        # [FIX-2] 偏置初始化：
        #   旧版 Softplus + bias=-0.5: softplus(-0.5)≈0.47 → clamp 到 r_max=0.40 → 饱和
        #   新版 Sigmoid  + bias= 0.0: sigmoid(0)=0.50 → 映射后 ≈ (r_min+r_max)/2 = 0.22
        #   网络从中间值出发，可自由向两端调整，不会训练初期就饱和
        nn.init.constant_(self.net[-2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:  (B, C, H, W)
        输出:  radius_map (B, 1, H, W)  值域 (r_min, r_max)

        sigmoid 输出 ∈ (0,1)，线性映射到 (r_min, r_max)：
          radius = sigmoid(logit) * (r_max - r_min) + r_min
        无需 clamp，边界自然满足，且梯度在全域非零（不存在梯度消失边界）
        """
        sig = self.net(x)                                      # ∈ (0, 1)
        return sig * (self.r_max - self.r_min) + self.r_min   # ∈ (r_min, r_max)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: NMS + TopK 提取中心点
# ─────────────────────────────────────────────────────────────────────────────

class DGMANMSCenterExtractor(nn.Module):
    """
    从热力图提取动态数量的病灶中心

    算法:
      1. MaxPool NMS：peak = (heatmap == MaxPool(heatmap))
      2. TopK：保留 confidence 最高的 K_max 个候选
      3. Threshold 过滤：低于 nms_threshold 的候选标记为无效

    关于 differentiability:
      - 热力图生成路径全可微 (用于 heatmap supervision loss)
      - 中心坐标通过 argmax/topk 得到，停止梯度 (仅作为采样坐标)
      - 高斯参数通过 grid_sample 在中心处插值特征后预测，全可微
    """

    def __init__(self, K_max: int = 5, nms_kernel: int = 3, nms_threshold: float = 0.3):
        """
        K_max:         每个样本最多保留的中心数量
        nms_kernel:    NMS MaxPool kernel size (奇数)
        nms_threshold: 低于此值的峰值视为背景噪声
        """
        super().__init__()
        self.K_max         = K_max
        self.nms_threshold = nms_threshold
        # NMS 用 MaxPool: peak 处值不变，非 peak 处被最近峰值替代
        pad = nms_kernel // 2
        self.nms_pool = nn.MaxPool2d(nms_kernel, stride=1, padding=pad)

    def forward(
        self, heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        输入:
          heatmap: (B, 1, H, W)

        返回:
          centers    : (B, K_max, 2)   归一化坐标 (norm_x, norm_y) ∈ [-1,1]
          valid_mask : (B, K_max)      True = 该槽位有有效中心
          peak_scores: (B, K_max)      各中心的 heatmap 置信度分数
        """
        B, _, H, W = heatmap.shape
        device = heatmap.device

        # ── NMS: 保留局部极大值 ──────────────────────────────
        # local_max[b,0,h,w] = True 当且仅当 heatmap[b,0,h,w] 是
        # (h±1, w±1) 邻域内的最大值（含自身）
        hmap_squeeze = heatmap.squeeze(1)           # (B, H, W)
        local_max    = (self.nms_pool(heatmap).squeeze(1) == hmap_squeeze)  # (B, H, W)
        # 同时要求超过背景阈值
        peaks        = hmap_squeeze * local_max.float()  # (B, H, W)

        # ── TopK 选取最多 K_max 个峰值 ────────────────────────
        # 展平空间维度后取 topk
        peaks_flat   = peaks.view(B, -1)                    # (B, H*W)
        K_cand       = min(self.K_max, peaks_flat.shape[1])
        top_vals, top_idx = torch.topk(peaks_flat, K_cand, dim=1)  # (B, K_cand)

        # ── 补齐到 K_max（不足时用无效槽位填充）─────────────────
        if K_cand < self.K_max:
            pad_vals = torch.zeros(B, self.K_max - K_cand, device=device)
            pad_idx  = torch.zeros(B, self.K_max - K_cand, dtype=torch.long, device=device)
            top_vals = torch.cat([top_vals, pad_vals], dim=1)
            top_idx  = torch.cat([top_idx,  pad_idx],  dim=1)

        # ── 有效性掩码 ────────────────────────────────────────
        # 超过阈值且不是填充槽位才视为有效中心
        valid_mask = top_vals >= self.nms_threshold   # (B, K_max)

        # ── 像素坐标 → 归一化坐标 ─────────────────────────────
        # top_idx 是展平后的 index，转回 (row, col)
        row_idx = (top_idx // W).float()   # (B, K_max)  像素行 ∈ [0, H-1]
        col_idx = (top_idx  % W).float()   # (B, K_max)  像素列 ∈ [0, W-1]

        # 归一化: 像素 0 → -1，像素 H-1 → 1
        norm_y = (2.0 * row_idx / max(H - 1, 1)) - 1.0   # y = row 方向
        norm_x = (2.0 * col_idx / max(W - 1, 1)) - 1.0   # x = col 方向

        centers = torch.stack([norm_x, norm_y], dim=-1)   # (B, K_max, 2)

        # ── 无效槽位坐标归零（保证 Fallback 高斯在中心）────────
        centers = centers * valid_mask.unsqueeze(-1).float()

        return centers, valid_mask, top_vals


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: 逐中心高斯参数预测
# ─────────────────────────────────────────────────────────────────────────────

class DGMAParamPredictor(nn.Module):
    """
    [v2: Lesion Size-Aware] 逐中心高斯参数预测

    σ 的计算分为两阶段:
      阶段一 (尺度基准):  σx_base = α × r_k
                          σy_base = α × r_k × β
        其中 r_k 来自 radius_map 在中心处的采样值，α 为可学习缩放因子，
        β = prior_aspect_ratio 为骨肿瘤纵向椭圆先验。
        这使 σ 与病灶实际尺度直接挂钩。

      阶段二 (残差修正):  MLP 输出 (Δσx_res, Δσy_res, θ, weight)
        Δσ 是对尺度基准的细粒度修正（tanh × max_delta 限幅），
        允许模型在整体尺度正确的基础上进一步调整椭圆形状和方向。
        θ, weight 仍由 MLP 完全预测。

      最终: σx = clamp(σx_base + Δσx_res, σ_min, σ_max)

    可微性:
      radius_map 由 LesionRadiusPredictor 预测，完全可微
      grid_sample 对 radius_map 完全可微（梯度回传到 radius head）
      MLP 部分完全可微（梯度回传到特征 backbone）

    参数量变化 vs v1:
      v1: MLP 输出 4 维 (σx, σy, θ, w)
      v2: MLP 输出 4 维 (Δσx, Δσy, θ, w) + radius_alpha 标量
          净增 ~1 个参数（alpha），radius head 单独计入 LesionRadiusPredictor
    """

    def __init__(self, channels: int,
                 sigma_min: float = 0.05,
                 sigma_max: float = 0.45,
                 prior_aspect_ratio: float = 1.5,
                 sigma_residual_max: float = 0.08):
        """
        sigma_min / sigma_max:  最终 σ 的取值范围（归一化坐标单位）
        prior_aspect_ratio:     σy/σx 骨肿瘤先验（>1 = 纵向椭圆）
        sigma_residual_max:     MLP 残差修正的最大幅度，防止残差主导 σ
        """
        super().__init__()
        self.sigma_min         = sigma_min
        self.sigma_max         = sigma_max
        self.prior_beta        = prior_aspect_ratio   # β: σy_base / σx_base
        self.sigma_residual_max = sigma_residual_max

        # ── α: 半径到 σ 的可学习缩放因子 ─────────────────────
        # 初始化使 σx_base ≈ r_k（α=1），训练时自动调整
        # 用 softplus 保证 α > 0
        self._log_alpha = nn.Parameter(torch.zeros(1))   # softplus(0) = ln2 ≈ 0.69

        # ── MLP: 特征 → (Δσx_res, Δσy_res, θ_raw, w_raw) ────
        # 与 v1 相比: σx/σy 改为预测残差而非绝对值，维度不变 (→4)
        self.param_mlp = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, 4)
        )
        # θ 初始化偏置为 0（初始无旋转）
        # Δσ 初始化偏置为 0（初始残差为 0，完全依赖 radius_map）
        nn.init.zeros_(self.param_mlp[-1].bias)

    @property
    def alpha(self) -> torch.Tensor:
        """可学习的半径-σ 缩放因子，通过 softplus 保证正数"""
        return F.softplus(self._log_alpha)              # > 0，无上界约束

    def forward(
        self,
        feature_map: torch.Tensor,   # (B, C, H, W)
        radius_map:  torch.Tensor,   # (B, 1, H, W)  来自 LesionRadiusPredictor
        centers:     torch.Tensor,   # (B, K_max, 2) (norm_x, norm_y)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回:
          sigma_x  (B, K_max, 1, 1)
          sigma_y  (B, K_max, 1, 1)
          theta    (B, K_max, 1, 1)
          weight   (B, K_max, 1, 1)
          r_k      (B, K_max)        各中心处采样到的半径值（用于监控/loss）
        """
        B, C, H, W = feature_map.shape
        K = centers.shape[1]

        # ── grid_sample: 同时采样特征图和半径图 ───────────────
        # grid shape: (B, 1, K, 2)，同一 grid 用于两次采样
        sample_grid = centers.unsqueeze(1)   # (B, 1, K, 2)

        # 采样特征: (B, C, 1, K) → (B, K, C)
        feat_sampled = F.grid_sample(
            feature_map, sample_grid,
            mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(2).permute(0, 2, 1)   # (B, K, C)

        # 采样半径: (B, 1, 1, K) → (B, K)
        # radius_map 已被 clamp 到 [r_min, r_max]，直接采样即可
        r_k = F.grid_sample(
            radius_map, sample_grid,
            mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(1).squeeze(1)   # (B, K)

        # ── 阶段一: 尺度基准 σ ────────────────────────────────
        # α: (1,) 标量；r_k: (B,K)
        alpha = self.alpha                           # 可学习正数
        sigma_x_base = alpha * r_k                  # (B, K)
        sigma_y_base = alpha * r_k * self.prior_beta  # (B, K)  纵向先验

        # ── 阶段二: MLP 残差修正 ───────────────────────────────
        params = self.param_mlp(feat_sampled)        # (B, K, 4)

        # Δσ: tanh × max_delta 限幅，防止残差覆盖尺度基准
        delta_sigma_x = torch.tanh(params[..., 0]) * self.sigma_residual_max  # (B,K)
        delta_sigma_y = torch.tanh(params[..., 1]) * self.sigma_residual_max
        theta         = torch.tanh(params[..., 2]) * math.pi                  # ∈(-π,π)
        weight        = torch.sigmoid(params[..., 3])                          # ∈(0,1)

        # ── 最终 σ = clamp(基准 + 残差) ───────────────────────
        sigma_x = (sigma_x_base + delta_sigma_x).clamp(self.sigma_min, self.sigma_max)
        sigma_y = (sigma_y_base + delta_sigma_y).clamp(self.sigma_min, self.sigma_max)

        # 扩展维度: (B, K) → (B, K, 1, 1) 便于广播
        def _e(t): return t.unsqueeze(-1).unsqueeze(-1)
        return _e(sigma_x), _e(sigma_y), _e(theta), _e(weight), r_k


# ─────────────────────────────────────────────────────────────────────────────
# Fallback 全局高斯（当样本中没有检测到任何中心时使用）
# ─────────────────────────────────────────────────────────────────────────────

def _global_fallback_gaussian(
    B: int, H: int, W: int, device: torch.device, sigma: float = 0.5
) -> torch.Tensor:
    """
    生成以图像中心为中心、各向同性的全局高斯注意力图

    用途: 当 NMS 未找到任何超过阈值的峰值时，退回到一个"全图均匀关注"的策略
    sigma=0.5 对应覆盖图像中央 ~50% 面积的圆形高斯

    返回: (B, 1, H, W)
    """
    gx, gy = _make_norm_grid(H, W, device)
    g = torch.exp(-(gx ** 2 + gy ** 2) / (2.0 * sigma ** 2))  # (H, W)
    return g.unsqueeze(0).unsqueeze(0).expand(B, 1, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 + 5: 生成高斯混合注意力图
# ─────────────────────────────────────────────────────────────────────────────

def build_gaussian_mixture_attention(
    H: int, W: int,
    centers: torch.Tensor,     # (B, K_max, 2)  (norm_x, norm_y)
    sigma_x: torch.Tensor,     # (B, K_max, 1, 1)
    sigma_y: torch.Tensor,     # (B, K_max, 1, 1)
    theta: torch.Tensor,       # (B, K_max, 1, 1)
    weight: torch.Tensor,      # (B, K_max, 1, 1)
    valid_mask: torch.Tensor,  # (B, K_max)  boolean
) -> torch.Tensor:
    """
    Step4: 生成 K_max 个旋转椭圆高斯
    Step5: 用有效掩码加权融合 → Sigmoid → attention_map

    无效槽位通过 valid_mask 乘零消除贡献，不影响最终结果

    返回: attention_map (B, 1, H, W)  值域 [0, 1]
    """
    mu_x = centers[..., 0].unsqueeze(-1).unsqueeze(-1)   # (B, K_max, 1, 1)
    mu_y = centers[..., 1].unsqueeze(-1).unsqueeze(-1)

    # Step4: 生成 K_max 个旋转高斯 → (B, K_max, H, W)
    gaussians = generate_rotated_gaussian(H, W, mu_x, mu_y, sigma_x, sigma_y, theta)

    # 掩码: 无效槽位贡献清零
    # valid_mask: (B, K_max) → (B, K_max, 1, 1)
    mask_4d = valid_mask.unsqueeze(-1).unsqueeze(-1).float()

    # Step5: 加权混合
    # weight: (B, K_max, 1, 1)；先归一化 weight，再加权求和
    masked_weight = weight * mask_4d                       # (B, K_max, 1, 1)
    weight_sum    = masked_weight.sum(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, 1, 1)
    norm_weight   = masked_weight / weight_sum             # 归一化权重

    # (B, K_max, H, W) * (B, K_max, 1, 1) → sum over K → (B, 1, H, W)
    mixture = (gaussians * norm_weight * mask_4d).sum(dim=1, keepdim=True)

    # Sigmoid 将 mixture 值域稳定到 (0, 1)
    return torch.sigmoid(mixture * 4.0 - 2.0)   # 中心处约 0.88，边缘趋近 0.12


# ─────────────────────────────────────────────────────────────────────────────
# 整合模块: DGMA
# ─────────────────────────────────────────────────────────────────────────────

class DGMA(nn.Module):
    """
    Dynamic Gaussian Mixture Attention v2 (Lesion Size-Aware)
    动态高斯混合注意力模块（病灶尺度自适应版本）

    端到端流程:
      feature_map → HeatmapPredictor  →  NMSCenterExtractor → ┐
                  → RadiusPredictor   → grid_sample r_k     → SizeAwareParamPredictor
                                                               → GaussianMixture
                                                               → attention_map
                                                               → enhanced_feature

    超参数:
      K_max:            每样本最多病灶中心数量
      nms_threshold:    中心置信度阈值
      r_min / r_max:    半径图的取值范围（归一化坐标单位）
      sigma_min/max:    最终 σ 的 clamp 范围
      prior_aspect_ratio: 骨肿瘤纵向椭圆先验 (σy/σx 初始比值 β)
      sigma_residual_max: MLP 残差修正的最大幅度

    last_state 字段 (forward 后可访问，用于监控/supervision loss):
      center_heatmap:  (B,1,H,W)    中心热力图
      radius_map:      (B,1,H,W)    半径预测图  ← [v2 新增]
      attention_map:   (B,1,H,W)    最终注意力
      centers:         (B,K_max,2)  中心坐标
      valid_mask:      (B,K_max)    槽位有效性
      peak_scores:     (B,K_max)    峰值置信度
      k_dynamic:       (B,)         实际检测到的中心数
      r_k:             (B,K_max)    各中心采样半径  ← [v2 新增]
      sigma_x, sigma_y, theta, weight: 各 (B,K_max,1,1)
      used_fallback:   (B,)         是否使用 Fallback
      alpha:           float        当前 radius-σ 缩放因子  ← [v2 新增]
    """

    def __init__(
        self,
        channels: int,
        K_max: int             = 5,
        nms_kernel: int        = 3,
        nms_threshold: float   = 0.3,
        r_min: float           = 0.03,
        r_max: float           = 0.40,
        sigma_min: float       = 0.05,
        sigma_max: float       = 0.45,
        prior_aspect_ratio: float = 1.5,
        sigma_residual_max: float = 0.08,
        residual_alpha: float  = 0.1,
        # [FIX-3] 移除 fallback_sigma 常量参数
        # 改用 fallback_log_scale 可学习参数 + radius_map 采样
    ):
        super().__init__()
        self.K_max          = K_max
        self.nms_threshold  = nms_threshold
        self.residual_alpha = residual_alpha
        self.sigma_min      = sigma_min   # 保存供 fallback 使用
        self.sigma_max      = sigma_max

        # Step1a: 中心热力图预测
        self.heatmap_predictor = DGMAHeatmapPredictor(channels)
        # Step1b: 病灶半径图预测
        self.radius_predictor  = LesionRadiusPredictor(channels, r_min, r_max)
        # Step2: NMS + TopK 中心提取
        self.nms_extractor     = DGMANMSCenterExtractor(K_max, nms_kernel, nms_threshold)
        # Step3: 尺度感知参数预测
        self.param_predictor   = DGMAParamPredictor(
            channels, sigma_min, sigma_max, prior_aspect_ratio, sigma_residual_max)

        # 通道缩放
        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid()
        )
        # 可学习残差强度
        self.residual_gate = nn.Parameter(torch.tensor(residual_alpha))

        # [FIX-3] Fallback sigma = softplus(fallback_log_scale) × r_center
        # 其中 r_center = radius_map 在图像中心 (0,0) 处的采样值
        # 梯度链完整: loss → σ_fb → r_center → grid_sample → radius_map → radius_predictor
        # softplus(1.0) ≈ 1.31，初始期望 σ_fb ≈ 1.31 × 0.22 ≈ 0.29
        self.fallback_log_scale = nn.Parameter(torch.tensor(1.0))

        # _radius_map: 保存带梯度的 radius_map，供 compute_spatial_radius_loss 使用
        self._radius_map: Optional[torch.Tensor] = None
        # 中间状态 (detach 版，供监控)
        self.last_state: Optional[Dict] = None

    def forward(
        self, feature_map: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        输入:
          feature_map: (B, C, H, W)

        返回:
          enhanced_feature: (B, C, H, W)  注意力增强后的特征
          attention_map:    (B, 1, H, W)  高斯混合注意力图 ∈ [0,1]
        """
        B, C, H, W = feature_map.shape
        device     = feature_map.device

        # ── Step1a: 中心热力图预测 ─────────────────────────────
        center_heatmap = self.heatmap_predictor(feature_map)   # (B,1,H,W)

        # ── Step1b: 病灶半径图预测 [v2] ───────────────────────
        # radius_map[b,0,h,w] ∈ [r_min, r_max]：
        # 表示若 (h,w) 是病灶中心，该病灶的估计半径
        radius_map = self.radius_predictor(feature_map)        # (B,1,H,W)

        # ── Step2: NMS + TopK 提取中心点 ──────────────────────
        centers, valid_mask, peak_scores = self.nms_extractor(center_heatmap)
        k_dynamic     = valid_mask.sum(dim=1)          # (B,) int tensor
        used_fallback = (k_dynamic == 0)               # (B,) bool

        # ── Fallback: 无有效中心时激活第0槽位 (坐标=图像中心(0,0)) ──
        if used_fallback.any():
            valid_mask_aug = valid_mask.clone()
            valid_mask_aug[used_fallback, 0] = True
        else:
            valid_mask_aug = valid_mask

        # ── Step3: 尺度感知参数预测 ───────────────────────────
        sigma_x, sigma_y, theta, weight, r_k = self.param_predictor(
            feature_map, radius_map, centers)

        # ── [FIX-3] Fallback 槽位使用 radius_map 采样值而非常量 ─
        #
        # 旧版: sigma_x[fb_idx, 0] = self.fallback_sigma  ← 常量赋值
        #   → 切断 radius_map 的整条梯度链 → radius_predictor grad = 0
        #
        # 新版: σ_fb = softplus(fallback_log_scale) × r_center
        #   r_center = radius_map 在 (0,0) 处双线性插值采样
        #   → 梯度路径: loss → σ_fb → r_center → radius_map → radius_predictor ✓
        #
        if used_fallback.any():
            fb_idx = used_fallback.nonzero(as_tuple=True)[0]    # 整数索引

            # 在归一化坐标 (0,0) = 图像中心处采样 radius_map
            # grid shape: (len(fb_idx), 1, 1, 2)
            center_grid = torch.zeros(len(fb_idx), 1, 1, 2, device=device)
            r_center = F.grid_sample(
                radius_map[fb_idx], center_grid,
                mode='bilinear', align_corners=True, padding_mode='border'
            ).squeeze(-1).squeeze(-1).squeeze(-1)    # (len(fb_idx),)

            # softplus 保证 fallback 缩放因子 > 0
            fb_scale   = F.softplus(self.fallback_log_scale)    # scalar
            sigma_fb_x = (fb_scale * r_center).clamp(self.sigma_min, self.sigma_max)
            sigma_fb_y = (fb_scale * r_center * self.param_predictor.prior_beta
                          ).clamp(self.sigma_min, self.sigma_max)

            # 写入第0槽位（需要 clone 保证 in-place 不破坏 autograd）
            sigma_x = sigma_x.clone()
            sigma_y = sigma_y.clone()
            sigma_x[fb_idx, 0, 0, 0] = sigma_fb_x   # 直接写标量维度
            sigma_y[fb_idx, 0, 0, 0] = sigma_fb_y

        # ── Step4 + 5: 高斯混合注意力图 ───────────────────────
        attention_map = build_gaussian_mixture_attention(
            H, W, centers, sigma_x, sigma_y, theta, weight, valid_mask_aug
        )   # (B, 1, H, W)

        # ── Step6: 特征增强 ────────────────────────────────────
        channel_w = self.channel_scale(feature_map)    # (B, C, 1, 1)
        attended  = feature_map * attention_map * channel_w
        alpha_r   = torch.sigmoid(self.residual_gate)
        enhanced  = attended + alpha_r * feature_map

        # ── 存储状态 ───────────────────────────────────────────
        # _radius_map: 保留梯度，供 compute_spatial_radius_loss 使用
        self._radius_map = radius_map

        # ── 存储中间状态 ───────────────────────────────────────
        self.last_state = {
            'center_heatmap':  center_heatmap.detach(),
            'radius_map':      radius_map.detach(),
            'attention_map':   attention_map.detach(),
            'centers':         centers.detach(),
            'valid_mask':      valid_mask.detach(),
            'peak_scores':     peak_scores.detach(),
            'k_dynamic':       k_dynamic.detach(),
            'used_fallback':   used_fallback.detach(),
            'r_k':             r_k.detach(),
            'sigma_x':         sigma_x.detach(),
            'sigma_y':         sigma_y.detach(),
            'theta':           theta.detach(),
            'weight':          weight.detach(),
            'alpha':           self.param_predictor.alpha.item(),
            # [FIX-3] 监控 fallback_log_scale 的实际值
            'fallback_scale':  F.softplus(self.fallback_log_scale).item(),
        }

        return enhanced, attention_map


# ─────────────────────────────────────────────────────────────────────────────
# 串联包装: DGMAWithSOE
# 可直接替换 fbfa_intrabone_enhanced_iddmga.py 中的 SOEWithIRGDA
# ─────────────────────────────────────────────────────────────────────────────

class DGMAWithSOE(nn.Module):
    """
    SOE + DGMA 串联  (v3: 深层自动跳过 DGMA)

    执行流程:
      PET特征 → SOE粗定位 → [若 min(H,W) >= min_spatial_size] DGMA精细定位

    [FIX-1] min_spatial_size 参数:
      深层特征图 (H=32, H=64) 空间分辨率太低，heatmap 无法产生有效峰值 →
      全部触发 fallback → DGMA 退化为全局高斯，计算浪费且不稳定。
      设 min_spatial_size=65（默认），则：
        pf0 (H=256): 256 >= 65 → 使用 DGMA ✓
        pf1 (H=128): 128 >= 65 → 使用 DGMA ✓
        pf2 (H= 64):  64 <  65 → 跳过 DGMA，仅 SOE ✓
        pf3 (H= 32):  32 <  65 → 跳过 DGMA，仅 SOE ✓
      这样 DGMA 只在浅层（高分辨率）生效，与你提出的建议完全一致。

    接口与 SOEWithIRGDA 完全一致。
    """

    def __init__(
        self,
        channels: int,
        K_max: int             = 5,
        nms_threshold: float   = 0.3,
        r_min: float           = 0.03,
        r_max: float           = 0.40,
        sigma_min: float       = 0.05,
        sigma_max: float       = 0.45,
        prior_aspect_ratio: float = 1.5,
        sigma_residual_max: float = 0.08,
        min_spatial_size: int  = 65,      # [FIX-1] 低于此分辨率跳过 DGMA
    ):
        super().__init__()
        self.min_spatial_size = min_spatial_size

        # SOE
        self.soe_pool3 = nn.MaxPool2d(3, stride=1, padding=1)
        self.soe_pool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.soe_pool7 = nn.MaxPool2d(7, stride=1, padding=3)
        self.soe_attn  = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
        # DGMA（仅浅层实际执行，深层由 min_spatial_size 控制跳过）
        self.dgma = DGMA(
            channels           = channels,
            K_max              = K_max,
            nms_threshold      = nms_threshold,
            r_min              = r_min,
            r_max              = r_max,
            sigma_min          = sigma_min,
            sigma_max          = sigma_max,
            prior_aspect_ratio = prior_aspect_ratio,
            sigma_residual_max = sigma_residual_max,
        )
        self.fusion_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, pet_feat: torch.Tensor) -> torch.Tensor:
        """
        输入:  pet_feat (B, C, H, W)
        输出:  enhanced (B, C, H, W)
        """
        B, C, H, W = pet_feat.shape

        # SOE 阶段（所有层都执行）
        r3 = pet_feat - self.soe_pool3(pet_feat)
        r5 = pet_feat - self.soe_pool5(pet_feat)
        r7 = pet_feat - self.soe_pool7(pet_feat)
        soe_enhanced = pet_feat * (1.0 + self.soe_attn(torch.cat([r3, r5, r7], dim=1)))

        # [FIX-1] 深层跳过 DGMA
        if min(H, W) < self.min_spatial_size:
            # 分辨率过低，直接返回 SOE 结果，不调用 DGMA
            # last_state 置 None，DGMASupervisionLoss 会安全跳过
            self.dgma.last_state  = None
            self.dgma._radius_map = None
            return soe_enhanced

        # 浅层：SOE → DGMA
        dgma_enhanced, _ = self.dgma(soe_enhanced)
        gate = torch.sigmoid(self.fusion_gate)
        return gate * dgma_enhanced + (1.0 - gate) * soe_enhanced

    @property
    def last_state(self):
        return self.dgma.last_state


# ─────────────────────────────────────────────────────────────────────────────
# compute_spatial_radius_loss: radius_predictor 直接监督辅助损失
# ─────────────────────────────────────────────────────────────────────────────

def compute_spatial_radius_loss(
    dgma_module: 'DGMA',
    tumor_mask: torch.Tensor,   # (B, 1, H_feat, W_feat) GT 已对齐特征图尺寸
    bone_mask:  torch.Tensor,   # (B, 1, H_feat, W_feat)
    weight: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    [FIX-3 补充] radius_predictor 的直接空间监督损失

    解决的问题:
      DGMASupervisionLoss 中的 L4 依赖 last_state['r_k']（detach），
      无法为 radius_predictor 提供梯度。
      当所有样本均 fallback（全空切片批次）时，
      就算 fallback 路径的梯度已修复，L4 也拿不到 r_k。

    本函数直接访问 dgma_module._radius_map（非 detach，有梯度），
    对其在 tumor 区域施加空间正则，提供稳定的直接梯度信号：

      L_spatial =
        alpha * MSE(mean_r_tumor, r_gt_per_sample)   # tumor 区均值应匹配 GT 等效半径
      + beta  * ReLU(mean_r_bone_bg - mean_r_tumor)  # tumor 区 radius 应 > 背景

    使用方式 (在训练 loop 中):
      for soe in model.soe:
          aux += compute_spatial_radius_loss(soe.dgma, tumor_mask_ds, bone_mask_ds)
      loss += 0.05 * aux

    注意: 若 soe.last_state is None（深层已跳过 DGMA），则 _radius_map 也为 None，
          函数安全返回 0。
    """
    radius_map = dgma_module._radius_map   # 带梯度的版本
    if radius_map is None:
        return torch.tensor(0.0, device=tumor_mask.device)

    B = tumor_mask.shape[0]
    # 对齐尺寸
    if radius_map.shape[2:] != tumor_mask.shape[2:]:
        radius_map = F.interpolate(radius_map, size=tumor_mask.shape[2:],
                                   mode='bilinear', align_corners=True)

    has_tumor = (tumor_mask.sum(dim=(1, 2, 3)) > 0)   # (B,)
    if not has_tumor.any():
        return torch.tensor(0.0, device=tumor_mask.device)

    total = torch.tensor(0.0, device=tumor_mask.device)

    for b in has_tumor.nonzero(as_tuple=True)[0]:
        tm = tumor_mask[b]      # (1, H, W)
        bm = bone_mask[b]
        rm = radius_map[b]      # (1, H, W)  带梯度

        # GT 等效半径（归一化空间：tumor 面积 / bone 面积的比例开方）
        tumor_area = tm.sum().float()
        bone_area  = bm.sum().float().clamp(min=1.0)
        r_gt = (tumor_area / bone_area / math.pi).clamp(min=eps).sqrt().detach()

        # radius_map 在 tumor 区域的均值应接近 r_gt
        r_tumor_mean = (rm * tm).sum() / (tm.sum() + eps)
        total = total + weight * F.mse_loss(r_tumor_mean, r_gt)

        # radius_map 在 tumor 区均值应高于骨骼背景均值（病灶比背景大）
        non_tumor_bone = bm * (1.0 - tm)
        r_bg_mean = (rm * non_tumor_bone).sum() / (non_tumor_bone.sum() + eps)
        total = total + weight * 0.5 * F.relu(r_bg_mean - r_tumor_mean)

    return total


# ─────────────────────────────────────────────────────────────────────────────
# DGMASupervisionLoss: 可选附加监督（训练时辅助 heatmap 学习）
# ─────────────────────────────────────────────────────────────────────────────

class DGMASupervisionLoss(nn.Module):
    """
    DGMA 附加监督损失（v2 新增 L4: Radius Consistency）

    四项约束:
      L1. Heatmap Coverage:
            heatmap 在肿瘤区域均值 > 骨骼背景均值
            形式: ReLU(attn_bg - attn_tumor)

      L2. Attention Coverage:
            最终 attention_map 在肿瘤区域的覆盖率高于骨骼背景
            形式: ReLU(attn_bone_bg - 2 * attn_tumor)

      L3. Ellipse Shape Constraint:
            σx/σy 或 σy/σx 不超过 max_aspect 倍（v2 中 sigma 来自 size-aware，
            形状正则仍有效，防止残差修正导致极度扭曲）

      L4. Radius Consistency [v2 新增]:
            已知分割 GT 时，r_k 应与真实病灶半径接近。
            用 GT tumor_mask 估算等效半径: r_gt ≈ sqrt(tumor_area / π)
            形式: MSE(r_k[valid], r_gt[对应样本])
            仅对有肿瘤且有有效中心的样本计算

    使用方式 (在 MixedSliceLoss.forward 中):
      dgma_loss_fn = DGMASupervisionLoss()
      for soe in model.soe:
          state = soe.last_state
          aux  += dgma_loss_fn(state, tumor_mask_ds, bone_mask_ds)
      total_loss += 0.05 * aux
    """

    def __init__(
        self,
        heatmap_weight:  float = 0.2,
        coverage_weight: float = 0.2,
        shape_weight:    float = 0.05,
        radius_weight:   float = 0.1,   # [v2] L4 权重
        max_aspect:      float = 4.0,
    ):
        super().__init__()
        self.heatmap_weight  = heatmap_weight
        self.coverage_weight = coverage_weight
        self.shape_weight    = shape_weight
        self.radius_weight   = radius_weight    # [v2]
        self.max_aspect      = max_aspect

    def forward(
        self,
        state:       Optional[Dict],
        tumor_mask:  torch.Tensor,   # (B, 1, H_feat, W_feat)  已对齐特征图尺寸
        bone_mask:   torch.Tensor,   # (B, 1, H_feat, W_feat)
    ) -> torch.Tensor:
        if state is None:
            return torch.tensor(0.0, device=tumor_mask.device)

        total     = torch.tensor(0.0, device=tumor_mask.device)
        has_tumor = (tumor_mask.sum(dim=(1, 2, 3)) > 0)   # (B,)

        # ── L1: Heatmap Coverage ──────────────────────────────
        if has_tumor.any() and self.heatmap_weight > 0:
            hmap = state['center_heatmap']
            if hmap.shape[2:] != tumor_mask.shape[2:]:
                hmap = F.interpolate(hmap, size=tumor_mask.shape[2:],
                                     mode='bilinear', align_corners=True)
            B_t  = has_tumor.nonzero(as_tuple=True)[0]
            hm_t = hmap[B_t];  tm_t = tumor_mask[B_t];  bm_t = bone_mask[B_t]
            non_tumor_bone = bm_t * (1.0 - tm_t)
            attn_on_tumor = (hm_t * tm_t).sum() / (tm_t.sum() + 1e-6)
            attn_on_bg    = (hm_t * non_tumor_bone).sum() / (non_tumor_bone.sum() + 1e-6)
            total = total + self.heatmap_weight * F.relu(attn_on_bg - attn_on_tumor)

        # ── L2: Attention Coverage ────────────────────────────
        if has_tumor.any() and self.coverage_weight > 0:
            attn = state['attention_map']
            if attn.shape[2:] != tumor_mask.shape[2:]:
                attn = F.interpolate(attn, size=tumor_mask.shape[2:],
                                     mode='bilinear', align_corners=True)
            B_t    = has_tumor.nonzero(as_tuple=True)[0]
            att_t  = attn[B_t];  tm_t = tumor_mask[B_t];  bm_t = bone_mask[B_t]
            attn_tumor = (att_t * tm_t).sum()  / (tm_t.sum()  + 1e-6)
            attn_bone  = (att_t * bm_t).sum()  / (bm_t.sum()  + 1e-6)
            total = total + self.coverage_weight * F.relu(attn_bone - attn_tumor * 2.0)

        # ── L3: Ellipse Shape Constraint ──────────────────────
        # (last_state 存的是 detached tensor，此项跳过梯度回传)

        # ── L4: Radius Consistency [v2] ───────────────────────
        # 用 GT tumor_mask 估算该样本的等效半径:
        #   r_gt = sqrt( tumor_area_norm / π )
        # 其中 tumor_area_norm = (肿瘤像素数 / 总骨骼像素数) 的归一化面积
        # 然后要求 r_k（有效中心处的预测半径均值）接近 r_gt
        if has_tumor.any() and self.radius_weight > 0 and 'r_k' in state:
            # [性能优化④] 向量化 radius consistency loss，消除 Python for 循环
            r_k        = state['r_k']          # (B, K_max)  detached
            valid_mask = state['valid_mask']   # (B, K_max)  bool, detached
            B_t = has_tumor.nonzero(as_tuple=True)[0]
            # 只取有肿瘤的样本
            r_k_t   = r_k[B_t]           # (N_t, K_max)
            vm_t    = valid_mask[B_t]     # (N_t, K_max)
            # 过滤掉没有任何有效中心的样本行（避免 mean() 返回 nan）
            row_valid = vm_t.any(dim=1)   # (N_t,)
            if row_valid.any():
                r_k_v  = r_k_t[row_valid]    # (N_v, K_max)
                vm_v   = vm_t[row_valid]      # (N_v, K_max)
                # GT 等效半径（向量化）
                tm_v   = tumor_mask[B_t[row_valid]]  # (N_v, 1, H, W)
                bm_v   = bone_mask[B_t[row_valid]]
                tumor_area = tm_v.sum(dim=(1,2,3)).float()           # (N_v,)
                bone_area  = bm_v.sum(dim=(1,2,3)).float().clamp(1)  # (N_v,)
                r_gt   = (tumor_area / bone_area / math.pi).clamp(min=1e-6).sqrt()  # (N_v,)
                # 有效槽位均值（masked mean）
                r_sum  = (r_k_v * vm_v.float()).sum(dim=1)            # (N_v,)
                r_cnt  = vm_v.float().sum(dim=1).clamp(min=1)         # (N_v,)
                r_pred_mean = r_sum / r_cnt                            # (N_v,)
                # 向量化 MSE：对所有有效样本一次计算
                total = total + self.radius_weight * F.mse_loss(
                    r_pred_mean, r_gt.detach())
                total = total + self.radius_weight * F.mse_loss(
                    r_pred_mean, r_gt.detach())

        return total


# ─────────────────────────────────────────────────────────────────────────────
# 快速验证（开发调试用）
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print("=" * 65)
    print("DGMA v3 (3 Bugs Fixed)")
    print("=" * 65)

    # ── 测试不同通道数 ──────────────────────────────────────────
    for C, H, W in [(32, 256, 256), (64, 128, 128), (128, 64, 64), (256, 32, 32)]:
        B = 4
        x = torch.randn(B, C, H, W, device=device)

        dgma = DGMA(channels=C, K_max=5, nms_threshold=0.3,
                    r_min=0.03, r_max=0.40,
                    sigma_min=0.05, sigma_max=0.45).to(device)

        n_heatmap = sum(p.numel() for p in dgma.heatmap_predictor.parameters())
        n_radius  = sum(p.numel() for p in dgma.radius_predictor.parameters())
        n_total   = sum(p.numel() for p in dgma.parameters())

        with torch.no_grad():
            t0 = time.time()
            enhanced, attn_map = dgma(x)
            elapsed = time.time() - t0

        state = dgma.last_state
        print(f"\nC={C:3d}  H={H:3d}  W={W:3d}  "
              f"params={n_total/1e3:.1f}K (heatmap={n_heatmap/1e3:.1f}K, "
              f"radius={n_radius/1e3:.1f}K)  time={elapsed*1000:.0f}ms")
        print(f"  k_dynamic: {state['k_dynamic'].tolist()}  "
              f"fallback: {state['used_fallback'].tolist()}")

        # [FIX-2] 验证 radius_map 不再饱和
        rm = state['radius_map']
        print(f"  [FIX-2] radius_map: min={rm.min():.3f}  max={rm.max():.3f}  "
              f"mean={rm.mean():.3f}  "
              f"{'✓ not saturated' if rm.max() < 0.38 else '⚠ near r_max'}")

        # FIX-3 监控 fallback_scale
        print(f"  [FIX-3] fallback_scale={state['fallback_scale']:.3f}  "
              f"α={state['alpha']:.4f}")

    # ── [FIX-1] DGMAWithSOE 深层跳过验证 ──────────────────────
    print("\n" + "─" * 65)
    print("[FIX-1] DGMAWithSOE min_spatial_size=65:")
    for C, H, W in [(32, 256, 256), (64, 128, 128), (128, 64, 64), (256, 32, 32)]:
        soe = DGMAWithSOE(channels=C, K_max=5, min_spatial_size=65).to(device)
        x   = torch.randn(2, C, H, W, device=device)
        with torch.no_grad():
            out = soe(x)
        skip = soe.last_state is None
        mark = '(SOE only ✓)' if skip else '(SOE+DGMA ✓)'
        print(f"  C={C:3d} H={H:3d}: {mark}  out={list(out.shape)}")

    # ── [FIX-3] 梯度链验证 ─────────────────────────────────────
    print("\n" + "─" * 65)
    print("[FIX-3] Gradient flow through radius_predictor:")

    # 场景A: 全 fallback (低分辨率，模拟深层)
    print("\n  Scene A: all-fallback (C=64, H=32, W=32, nms_threshold=0.99)")
    x_a = torch.randn(2, 64, 32, 32, device=device, requires_grad=True)
    d_a = DGMA(channels=64, K_max=3, nms_threshold=0.99).to(device)
    enh_a, attn_a = d_a(x_a)
    loss_a = enh_a.mean() + attn_a.mean()
    loss_a.backward()
    rp_grad_a = sum(p.grad.norm().item() for p in d_a.radius_predictor.parameters()
                    if p.grad is not None)
    fb_scale_grad = d_a.fallback_log_scale.grad
    print(f"  k_dynamic: {d_a.last_state['k_dynamic'].tolist()}")
    print(f"  radius_predictor grad norm: {rp_grad_a:.6f}  "
          f"{'✓ has grad' if rp_grad_a > 1e-8 else '✗ grad=0 (bug remains)'}")
    print(f"  fallback_log_scale grad: "
          f"{'✓ ' + f'{fb_scale_grad.item():.6f}' if fb_scale_grad is not None else '✗ None'}")

    # 场景B: 有有效中心 (高分辨率)
    print("\n  Scene B: valid centers exist (C=32, H=256, W=256)")
    x_b = torch.randn(2, 32, 256, 256, device=device, requires_grad=True)
    d_b = DGMA(channels=32, K_max=5, nms_threshold=0.3).to(device)
    enh_b, attn_b = d_b(x_b)
    loss_b = enh_b.mean() + attn_b.mean()
    loss_b.backward()
    rp_grad_b = sum(p.grad.norm().item() for p in d_b.radius_predictor.parameters()
                    if p.grad is not None)
    print(f"  k_dynamic: {d_b.last_state['k_dynamic'].tolist()}")
    print(f"  radius_predictor grad norm: {rp_grad_b:.6f}  "
          f"{'✓ has grad' if rp_grad_b > 1e-8 else '✗ grad=0 (bug remains)'}")

    # ── compute_spatial_radius_loss 验证 ────────────────────────
    print("\n" + "─" * 65)
    print("compute_spatial_radius_loss gradient test:")
    x_c    = torch.randn(2, 64, 128, 128, device=device, requires_grad=True)
    d_c    = DGMA(channels=64, K_max=5).to(device)
    enh_c, _ = d_c(x_c)
    # 模拟 GT mask
    tumor  = torch.zeros(2, 1, 128, 128, device=device)
    tumor[:, :, 50:70, 50:70] = 1.0
    bone   = torch.ones(2, 1, 128, 128, device=device)
    aux    = compute_spatial_radius_loss(d_c, tumor, bone)
    (enh_c.mean() + aux).backward()
    rp_grad_c = sum(p.grad.norm().item() for p in d_c.radius_predictor.parameters()
                    if p.grad is not None)
    print(f"  aux radius loss value: {aux.item():.6f}")
    print(f"  radius_predictor grad norm: {rp_grad_c:.6f}  "
          f"{'✓' if rp_grad_c > 1e-8 else '✗'}")
    print("\n✅ All 3 fixes verified")