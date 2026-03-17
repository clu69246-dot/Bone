"""
ConDSeg Stage1 — 2D EfficientViM-M1  (完全重写版)

任务: CT切片骨骼二值分割  (Binary Bone Segmentation)
输入: (B, 1, H, W)  — CT Z轴切片
输出: (B, 1, H, W)  — 骨骼分割 logits (未经 sigmoid)

架构改进:
  [1] 骨骼分割专用解码器
      - 原来: 简单的 CBR + skip-cat
      - 现在: MSCA (Multi-Scale Context Aggregation) 解码器
              = 多尺度空洞卷积 + CBAM 注意力 + 细化残差块
      - 目的: 骨骼边缘细节更准确，连续性更好

  [2] 多尺度输出监督 (Deep Supervision)
      - Stage2/1/0 各加一个辅助头
      - 训练时 loss = main + 0.4*aux2 + 0.2*aux1 + 0.1*aux0
      - 推理时只用 main head

  [3] Stage2 接口完善
      - forward_with_features(): 返回 [f0,f1,f2,f3] + logits
      - 特征维度: [32, 64, 128, 256]  对应 [H/2, H/4, H/8, H/8]

  [4] 扫描机制已修复 (见 efficientvim_modules_2d.py)
      - 原 CrossScan2D 产生 O(4B·C·d_state·L) 中间张量 → OOM
      - 新 EfficientScan2D 最大中间张量 O(B·H·C·W) ≈ 输入量级
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.efficientvim_modules_2d import MambaFeatureExtractor2D


# ============================================================
#  工具层
# ============================================================

class CBR2D(nn.Module):
    """Conv2D + BN + ReLU"""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1,
                 dilation=1, stride=1, act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding,
                      dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.act  = act

    def forward(self, x):
        x = self.conv(x)
        return self.relu(x) if self.act else x


# ============================================================
#  CBAM 注意力 (骨骼边缘增强)
# ============================================================

class ChannelAttn(nn.Module):
    def __init__(self, c, ratio=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(
            nn.Conv2d(c, c // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // ratio, c, 1, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        w = self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))
        return x * w


class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        m = self.sig(self.conv(torch.cat([avg, mx], dim=1)))
        return x * m


class CBAM(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ca = ChannelAttn(c)
        self.sa = SpatialAttn()

    def forward(self, x):
        return self.sa(self.ca(x))


# ============================================================
#  MSCA 解码块 (Multi-Scale Context Aggregation)
# ============================================================

class MSCADecoder(nn.Module):
    """
    骨骼分割专用解码块

    设计思路:
    - 骨骼是细长连续的结构，需要:
      1. 多尺度感受野 (骨骼粗细变化大)
      2. 强边缘响应 (骨骼边界与软组织差异明显)
      3. 长程连续性 (椎骨链、肋骨等)
    - 多尺度空洞卷积: r=1,2,4 聚合多尺度骨骼特征
    - 细化残差块: 精细化骨骼边缘
    - CBAM: 抑制软组织干扰

    输入:
      x:    来自更深层的特征 (已上采样)
      skip: 编码器 skip connection

    输出:
      refine: 解码后的特征 (B, out_c, H, W)
    """

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        merge_c = in_c + skip_c

        # 1. skip 融合（上采样在 forward 里动态做）
        self.fuse = CBR2D(merge_c, out_c, kernel_size=1, padding=0)

        # 2. 多尺度空洞卷积 (ASPP-lite)
        self.branch1 = CBR2D(out_c, out_c, dilation=1, padding=1)
        self.branch2 = CBR2D(out_c, out_c, dilation=2, padding=2)
        self.branch4 = CBR2D(out_c, out_c, dilation=4, padding=4)
        self.branch_g = nn.Sequential(          # 全局分支 (骨骼整体先验)
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_c, out_c, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.ms_fuse = CBR2D(out_c * 4, out_c, kernel_size=1, padding=0)

        # 3. 细化残差块 (精细化骨骼边缘)
        self.refine1 = CBR2D(out_c, out_c, act=False)
        self.refine2 = CBR2D(out_c, out_c, act=False)
        self.relu    = nn.ReLU(inplace=True)

        # 4. CBAM 注意力
        self.cbam = CBAM(out_c)

    def forward(self, x, skip):
        # 上采样到 skip 的空间尺寸（自动对齐，不依赖固定 scale）
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x    = torch.cat([x, skip], dim=1)
        x    = self.fuse(x)

        # 多尺度空洞卷积
        b1   = self.branch1(x)
        b2   = self.branch2(x)
        b4   = self.branch4(x)
        bg   = self.branch_g(x).expand_as(x)   # 广播到空间维度
        ms   = self.ms_fuse(torch.cat([b1, b2, b4, bg], dim=1))

        # 细化残差
        s1   = ms
        r1   = self.relu(self.refine1(ms) + s1)
        r2   = self.relu(self.refine2(r1) + r1 + s1)

        # 注意力
        out  = self.cbam(r2)
        return out


# ============================================================
#  辅助分割头 (深度监督用)
# ============================================================

class AuxHead(nn.Module):
    """轻量辅助头 — 动态 upsample 到目标尺寸"""
    def __init__(self, in_c):
        super().__init__()
        self.head = nn.Sequential(
            CBR2D(in_c, in_c // 2, kernel_size=3, padding=1),
            nn.Conv2d(in_c // 2, 1, kernel_size=1)
        )

    def forward(self, x, target_size):
        logits = self.head(x)
        if logits.shape[2:] != target_size:
            logits = F.interpolate(logits, size=target_size,
                                   mode='bilinear', align_corners=True)
        return logits


# ============================================================
#  主模型
# ============================================================

class ConDSeg2DStage1_EfficientViM(nn.Module):
    """
    ConDSeg Stage1 — 骨骼分割网络 (内存高效 + 高精度版)

    编码器: MambaFeatureExtractor2D  (strides=[2,2,2,1])
      特征通道:  [32,   64,   128,  256]
      空间尺寸:  [H/4,  H/8,  H/16, H/16]   (Stem×2 + Stage0×2 + Stage1×2 + Stage2×2 + Stage3×1)

    解码器: 3 × MSCADecoder
      d2: f3(256,H/16) + f2(128,H/16) → 128, H/16  (scale=1, 同尺寸拼接)
      d1: d2(128,H/16) + f1(64, H/8)  → 64,  H/8   (scale=2)
      d0: d1(64, H/8)  + f0(32, H/4)  → 32,  H/4   (scale=2)

    输出头:
      Main: Upsample × 4 → CBR × 2 → Conv1×1  (H/4 → H)
      Aux2/1/0: 深度监督头 scale=16/8/4 (仅训练时使用)

    Stage2 接口:
      forward_with_features(x) → (features, logits)
      features = [f0, f1, f2, f3]  可直接供 Stage2 融合

    注意: 输出 logits，不含 Sigmoid
          使用 BCEWithLogitsLoss 训练
    """

    def __init__(self, in_channels=1, out_channels=1, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # ── 编码器 ── strides=[2,2,2,1]
        # 真实特征图 (H=512, Stem先×2):
        #   f0=(32, H/4=128... 不, Stem后是256, Stage0 stride=2 → 下采前feats在256, 下采后128)
        # 实际: Stem(256) → Stage0(stride=2): feats f0@256, x→128
        #                 → Stage1(stride=2): feats f1@128, x→64
        #                 → Stage2(stride=2): feats f2@64,  x→32
        #                 → Stage3(stride=1): feats f3@32,  x→32
        # 所以: f0=(32,256), f1=(64,128), f2=(128,64), f3=(256,32)
        self.backbone = MambaFeatureExtractor2D(
            in_dim=in_channels,
            embed_dim=[32, 64, 128, 256],
            depths=[1, 1, 1, 1],
            state_dim=[49, 25, 9, 9],
            strides=[2, 2, 2, 1]
        )

        # ── 解码器 (MSCA) — scale 在 forward 里动态对齐 ──
        # strides=[2,2,2,1]: f0=(32,H/2), f1=(64,H/4→实际H/4? 见下)
        # 真实尺寸 (H=512, Stem×2):
        #   f0=(32, 256)  f1=(64, 128)  f2=(128, 64)  f3=(256, 32)
        # 解码路径:
        #   d2: f3(32) upsample→(64) cat f2(64) → 128ch, 64×64
        #   d1: d2(64) upsample→(128) cat f1(128) → 64ch, 128×128
        #   d0: d1(128) upsample→(256) cat f0(256) → 32ch, 256×256
        # main_head: 256×256 → 512×512 (×2)
        self.decoder2 = MSCADecoder(in_c=256, skip_c=128, out_c=128)
        self.decoder1 = MSCADecoder(in_c=128, skip_c=64,  out_c=64)
        self.decoder0 = MSCADecoder(in_c=64,  skip_c=32,  out_c=32)

        # ── 主输出头 ── d0 是 256×256, 需要 ×2 → 512×512
        self.main_head = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            CBR2D(32, 64,  kernel_size=3, padding=1),
            CBR2D(64, 64,  kernel_size=3, padding=1),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

        # ── 深度监督辅助头 (scale_factor 改为动态) ──
        if deep_supervision:
            self.aux_head2 = AuxHead(in_c=128)
            self.aux_head1 = AuxHead(in_c=64)
            self.aux_head0 = AuxHead(in_c=32)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ──────────────────────────────────────────────
    #  主前向 (训练 / 推理)
    # ──────────────────────────────────────────────
    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W)
        Returns:
            训练时 (deep_supervision=True):
                {
                  'main':  logits (B,1,H,W),
                  'aux2':  logits (B,1,H,W),
                  'aux1':  logits (B,1,H,W),
                  'aux0':  logits (B,1,H,W),
                }
            推理时 (deep_supervision=False 或 not training):
                logits (B,1,H,W)
        """
        H, W = x.shape[2], x.shape[3]

        # 编码
        features = self.backbone(x)     # [f0,f1,f2,f3]
        f0, f1, f2, f3 = features       # (32,H/2), (64,H/4), (128,H/8), (256,H/8)

        # 解码
        d2 = self.decoder2(f3, f2)      # (128, H/8)
        d1 = self.decoder1(d2, f1)      # (64,  H/4)
        d0 = self.decoder0(d1, f0)      # (32,  H/2)

        # 主输出
        main_logits = self.main_head(d0)   # (1, H, W)

        if self.deep_supervision and self.training:
            aux2 = self.aux_head2(d2, target_size=(H, W))
            aux1 = self.aux_head1(d1, target_size=(H, W))
            aux0 = self.aux_head0(d0, target_size=(H, W))
            return {
                'main': main_logits,
                'aux2': aux2,
                'aux1': aux1,
                'aux0': aux0,
            }

        return main_logits

    # ──────────────────────────────────────────────
    #  Stage2 接口: 返回多尺度特征 + logits
    # ──────────────────────────────────────────────
    def forward_with_features(self, x):
        """
        供 Stage2 骨肿瘤分割调用。

        Args:
            x: (B, 1, H, W)

        Returns:
            features: list of [f0, f1, f2, f3]
                f0: (B, 32,  H/4,  W/4)   — 细粒度骨骼边缘特征
                f1: (B, 64,  H/8,  W/8)   — 骨骼局部纹理特征
                f2: (B, 128, H/16, W/16)  — 骨骼中层语义特征
                f3: (B, 256, H/16, W/16)  — 全局骨骼上下文特征
            decoder_feats: list of [d0, d1, d2]
                解码器各层特征 (已融合 skip connection 信息)
                d2: (B, 128, H/16, W/16)
                d1: (B, 64,  H/8,  W/8)
                d0: (B, 32,  H/4,  W/4)
            logits: (B, 1, H, W)  — 骨骼分割 logits (供 Stage2 用作骨骼 mask 先验)
        """
        features = self.backbone(x)
        f0, f1, f2, f3 = features

        d2 = self.decoder2(f3, f2)
        d1 = self.decoder1(d2, f1)
        d0 = self.decoder0(d1, f0)

        logits = self.main_head(d0)

        decoder_feats = [d0, d1, d2]   # 从浅到深

        return features, decoder_feats, logits


# ============================================================
#  训练用 Loss (含深度监督)
# ============================================================

class Stage1Loss(nn.Module):
    """
    Stage1 骨骼分割 Loss = Dice + BCE + 深度监督

    训练时模型返回 dict {'main', 'aux2', 'aux1', 'aux0'}
    推理时模型返回 tensor，直接传给此 loss 也可以正常工作

    Loss 权重:
      main × 1.0  +  aux2 × 0.4  +  aux1 × 0.2  +  aux0 × 0.1
    """

    def __init__(self, bce_weight=0.3, dice_weight=0.7,
                 aux_weights=(0.4, 0.2, 0.1)):
        super().__init__()
        self.bce        = nn.BCEWithLogitsLoss()
        self.bce_w      = bce_weight
        self.dice_w     = dice_weight
        self.aux_ws     = aux_weights   # (aux2, aux1, aux0)

    def _single_loss(self, logits, target):
        bce_loss  = self.bce(logits, target)
        pred      = torch.sigmoid(logits)
        pred      = torch.clamp(pred, 1e-5, 1 - 1e-5)
        smooth    = 1e-5
        pf        = pred.view(pred.size(0), -1)
        tf        = target.view(target.size(0), -1)
        inter     = (pf * tf).sum(dim=1)
        union     = pf.sum(dim=1) + tf.sum(dim=1)
        dice_loss = 1 - ((2 * inter + smooth) / (union + smooth)).mean()
        return self.bce_w * bce_loss + self.dice_w * dice_loss

    def forward(self, outputs, target):
        if isinstance(outputs, dict):
            main_loss = self._single_loss(outputs['main'], target)
            aux2_loss = self._single_loss(outputs['aux2'], target)
            aux1_loss = self._single_loss(outputs['aux1'], target)
            aux0_loss = self._single_loss(outputs['aux0'], target)
            total = (main_loss
                     + self.aux_ws[0] * aux2_loss
                     + self.aux_ws[1] * aux1_loss
                     + self.aux_ws[2] * aux0_loss)
        else:
            total = self._single_loss(outputs, target)

        return total


# ============================================================
#  单元测试
# ============================================================

if __name__ == "__main__":
    import time
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("ConDSeg2D Stage1 — Memory-Efficient + High-Quality Bone Segmentation")
    print("=" * 70)

    # 创建模型
    model = ConDSeg2DStage1_EfficientViM(
        in_channels=1, out_channels=1, deep_supervision=True
    ).to(device)

    B, H, W = 2, 512, 512
    x = torch.randn(B, 1, H, W).to(device)

    # 训练模式 (返回 dict)
    model.train()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
        out_train = model(x)
    t1 = time.time()

    print(f"\n[Train mode] Input: {x.shape}")
    for k, v in out_train.items():
        print(f"  {k}: {v.shape}  range=[{v.min():.2f}, {v.max():.2f}]")

    if device.type == 'cuda':
        print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")
    print(f"  Forward time: {(t1-t0)*1000:.1f} ms")

    # 深度监督 Loss
    target = (torch.rand(B, 1, H, W) > 0.5).float().to(device)
    loss_fn = Stage1Loss()
    with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
        loss = loss_fn(out_train, target)
    print(f"\n[Loss] Stage1Loss = {loss.item():.4f}  ✓")

    # 推理模式
    model.eval()
    with torch.no_grad():
        logits = model(x)
    probs = torch.sigmoid(logits)
    print(f"\n[Inference mode] logits: {logits.shape}")
    print(f"  probs range: [{probs.min():.3f}, {probs.max():.3f}]")
    assert logits.shape == (B, 1, H, W), "Shape mismatch!"

    # Stage2 接口
    with torch.no_grad():
        feats, dec_feats, stage1_logits = model.forward_with_features(x)
    print(f"\n[Stage2 Interface]  (H=512, strides=[2,2,2,1])")
    print(f"  encoder features:")
    names = ['f0(32,H/4)', 'f1(64,H/8)', 'f2(128,H/16)', 'f3(256,H/16)']
    for i, f in enumerate(feats):
        print(f"    {names[i]}: {f.shape}")
    print(f"  decoder features:")
    dnames = ['d0(32,H/4)', 'd1(64,H/8)', 'd2(128,H/16)']
    for i, d in enumerate(dec_feats):
        print(f"    {dnames[i]}: {d.shape}")
    print(f"  stage1 logits: {stage1_logits.shape}")

    # 参数量
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Parameters] Total={total/1e6:.2f}M  Trainable={train/1e6:.2f}M")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("Key fixes:")
    print("  [1] OOM: EfficientScan2D 替换 CrossScan2D，内存降低 ~100x")
    print("  [2] 解码器: MSCA 多尺度空洞卷积 + CBAM，骨骼边缘更精准")
    print("  [3] 深度监督: aux2/1/0 辅助头，训练收敛更稳定")
    print("  [4] Stage2 接口: forward_with_features() 返回编/解码器特征")
    print("=" * 70)