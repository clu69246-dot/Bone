"""
Intra-Bone Tumor Training — Single Stage v1
============================================
核心设计原则：
  - 单阶段训练，去除 curriculum / HNM / FP suppression
  - 训练集 tumor:empty = 1:1（固定，不动态切换）
  - 验证集 tumor:empty = 1:5（接近真实分布）
  - Loss = Tversky(beta=0.85) + BCE + Boundary + IRGDA supervision
  - 保留 IDDMGA warmup（前5轮）和 Deep Supervision（全程）
  - ReduceLROnPlateau(patience=5, factor=0.5)
"""

import os
import sys
import argparse
import datetime
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset, Subset
import warnings
import gc
from tqdm import tqdm
import io
import cv2
import torch.nn.functional as F

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_train.Intrabone_petct_dataset_tumor_only import (
    get_intrabone_dataloader_512,
    PerfectIntraBoneDataset512Fixed,
    EnhancedCTNormalizer,
    EnhancedPETNormalizer,
    get_augmentation,
)

from new_network.fbfa_intrabone_enhanced import FBFAIntraBoneTumorSegmentation
from new_network.fbfa_intrabone_enhanced_iddmga import DGMASupervisionLoss as IRGDASupervisionLoss

from bone_only_loss_metrics import (
    SmallTumorLoss,
    BoneOnlyDetailedMetrics,
)


# ============================================================
#  工具函数
# ============================================================

def setup_logger(log_file):
    logger = logging.getLogger('train')
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def freeze_bn_stats(model):
    """CT 分支 BN 保持 eval 统计量（pretrained）"""
    for m in model.ct_branch.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.weight.requires_grad_(True)
            m.bias.requires_grad_(True)


class EarlyStopping:
    def __init__(self, patience=30, min_delta=0.001, mode='max'):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_score = None
        self.counter    = 0

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return False
        improved = (metric - self.best_score > self.min_delta) if self.mode == 'max' \
                   else (self.best_score - metric > self.min_delta)
        if improved:
            self.best_score = metric
            self.counter    = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ============================================================
#  EmptySliceDataset（仅读取空切片，供混合用）
# ============================================================

class EmptySliceDataset(torch.utils.data.Dataset):
    """读取空切片（is_tumor=False），与 PerfectIntraBoneDataset512Fixed 接口兼容"""

    def __init__(self, image_list, img_root, mode='train', is_16bit=True):
        self.img_root = img_root
        self.mode     = mode
        self.is_16bit = is_16bit
        self.ct_norm  = EnhancedCTNormalizer()
        self.pet_norm = EnhancedPETNormalizer()
        self.transform = get_augmentation(is_train=(mode == 'train'))
        self.image_list = self._filter_empty(image_list)
        print(f"  [EmptySlice] {len(self.image_list)} empty slices loaded")

    def _get_path(self, image_id, suffix):
        p = os.path.join(self.img_root, image_id + suffix)
        if os.path.exists(p):
            return p
        parts = image_id.split('_')
        for n in range(len(parts), 0, -1):
            alt = os.path.join(self.img_root, '_'.join(parts[:n]), image_id + suffix)
            if os.path.exists(alt):
                return alt
        return p

    def _get_bone_path(self, image_id):
        p = self._get_path(image_id, '_bone_pred.png')
        if os.path.exists(p):
            return p
        return self._get_path(image_id, '_bone_pred.png')

    def _filter_empty(self, raw_list):
        result = []
        for image_id in raw_list:
            tmask = self._get_path(image_id, '_mask.png')
            bone  = self._get_bone_path(image_id)
            if not (os.path.exists(tmask) and os.path.exists(bone)):
                continue
            tm = cv2.imread(tmask, cv2.IMREAD_GRAYSCALE)
            bm = cv2.imread(bone,  cv2.IMREAD_GRAYSCALE)
            if tm is None or bm is None:
                continue
            bone_area = (bm > 127).sum()
            if bone_area < 500:
                continue
            tumor_in_bone = ((tm > 127) & (bm > 127)).sum()
            if tumor_in_bone == 0:
                result.append(image_id)
        return result

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id = self.image_list[idx]
        ct_path    = self._get_path(image_id, '_CT.png')
        pet_path   = self._get_path(image_id, '_PET.png')
        tmask_path = self._get_path(image_id, '_mask.png')
        bone_path  = self._get_bone_path(image_id)

        def read_ct(p):
            if self.is_16bit:
                ct = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                return self.ct_norm.normalize(ct, method='bone')
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        ct         = read_ct(ct_path)
        pet_raw    = cv2.imread(pet_path, cv2.IMREAD_UNCHANGED)
        pet        = self.pet_norm.normalize(pet_raw)
        tmask_raw  = cv2.imread(tmask_path, cv2.IMREAD_GRAYSCALE)
        bone_raw   = cv2.imread(bone_path,  cv2.IMREAD_GRAYSCALE)
        tumor_mask = (tmask_raw > 127).astype(np.float32)
        bone_pred  = (bone_raw  > 127).astype(np.float32)

        aug = self.transform(image=ct, pet=pet,
                              bone_pred=bone_pred, tumor_mask=tumor_mask)
        ct         = aug['image'].float()
        pet        = aug['pet'].float()
        bone_pred  = aug['bone_pred'].float()
        tumor_mask = aug['tumor_mask'].float()

        def e3(t): return t.unsqueeze(0) if t.dim() == 2 else t
        ct, pet, bone_pred, tumor_mask = map(e3, [ct, pet, bone_pred, tumor_mask])

        ct         = torch.clamp(ct, 0, 1) * bone_pred
        pet        = torch.clamp(pet, 0, 1) * bone_pred
        bone_pred  = (bone_pred > 0.5).float()
        tumor_mask = (tumor_mask > 0.5).float() * bone_pred

        return {
            'ct':          ct,
            'pet':         pet,
            'bone_pred':   bone_pred,
            'tumor_mask':  tumor_mask,
            'name':        image_id,
            'tumor_ratio': torch.tensor(0.0, dtype=torch.float32),
            'is_tumor':    torch.tensor(False, dtype=torch.bool),
        }


# ============================================================
#  Loss：SmallTumorLoss + IRGDA supervision（无 FP suppression）
# ============================================================

class SingleStageLoss(nn.Module):
    """
    单阶段损失函数

    Total = SmallTumorLoss(tumor slices)
          + irgda_sup_weight * IRGDASupervisionLoss(tumor slices, phase>=1)

    删除了 FP suppression 项：
      该项会在 Recall 和 Precision 之间引入对抗，
      导致模型在不同 epoch 交替"偏向多预测/少预测"，Recall 不稳定。
    """

    def __init__(self, base_loss_fn, irgda_sup_weight=0.05):
        super().__init__()
        self.tumor_loss       = base_loss_fn
        self.irgda_sup_weight = irgda_sup_weight
        self.irgda_loss_fn    = IRGDASupervisionLoss(
            heatmap_weight=0.2, coverage_weight=0.2,
            shape_weight=0.05,  radius_weight=0.1)

    def forward(self, outputs, tumor_mask, bone_pred, is_tumor,
                current_epoch=0, ds_epochs=0, model=None):

        if isinstance(outputs, dict):
            logits = outputs['tumor_logits']
        else:
            logits = outputs

        has_tumor = is_tumor.bool()

        # 用 logits.sum()*0 构造与计算图相连的零张量，
        # 避免全空切片 batch 时 backward 报 "no grad_fn" 错误。
        zero = logits.sum() * 0.0

        # ── 主 loss（只算有肿瘤的 slice）──────────────────────────
        tumor_loss = zero
        if has_tumor.any():
            tumor_loss = self.tumor_loss(
                outputs, tumor_mask, bone_pred, is_tumor,
                current_epoch=current_epoch, ds_epochs=ds_epochs)

        # ── IRGDA 监督 loss ────────────────────────────────────────
        irgda_loss = zero
        if model is not None and has_tumor.any() and self.irgda_sup_weight > 0:
            tumor_idx = has_tumor.nonzero(as_tuple=True)[0]
            tm_full = tumor_mask[tumor_idx]
            bm_full = bone_pred[tumor_idx]
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

        return tumor_loss + self.irgda_sup_weight * irgda_loss


# ============================================================
#  IDDMGA LR Warmup（保留，仅前5轮线性升温）
# ============================================================

def apply_iddmga_lr_warmup(optimizer, epoch, warmup_end_epoch,
                            iddmga_lr_full, iddmga_group_idxs):
    ratio = epoch / warmup_end_epoch
    lr    = iddmga_lr_full * ratio
    for idx in iddmga_group_idxs:
        optimizer.param_groups[idx]['lr'] = lr


# ============================================================
#  Train / Validate
# ============================================================

def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler,
                epoch, logger, accumulation_steps=8, clip_grad_norm=0.5,
                ds_epochs=150):
    model.train()
    freeze_bn_stats(model)
    metrics_fn = BoneOnlyDetailedMetrics(threshold=0.3)

    total_loss  = 0.0
    all_tdice, all_tprec, all_trecall, all_fp = [], [], [], []

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(dataloader, desc=f'Epoch {epoch:03d} [Train]')

    for batch_idx, batch in enumerate(pbar):
        ct         = batch['ct'].to(device, non_blocking=True)
        pet        = batch['pet'].to(device, non_blocking=True)
        bone_pred  = batch['bone_pred'].to(device, non_blocking=True)
        tumor_mask = batch['tumor_mask'].to(device, non_blocking=True)
        is_tumor   = batch['is_tumor'].to(device, non_blocking=True)

        with autocast():
            outputs = model(ct, pet, bone_pred, return_intermediate=True)
            loss    = loss_fn(outputs, tumor_mask, bone_pred, is_tumor,
                              current_epoch=epoch, ds_epochs=ds_epochs,
                              model=model)
            loss    = loss / accumulation_steps

        scaler.scale(loss).backward()

        step = (batch_idx + 1) % accumulation_steps == 0
        last = (batch_idx + 1) == len(dataloader)
        if step or last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 每 10 步计算一次 metrics，减少 GPU→CPU 同步
        if batch_idx % 10 == 0:
            with torch.no_grad():
                m = metrics_fn(outputs['tumor_logits'], tumor_mask, bone_pred, is_tumor)
                if m['num_tumor_slices'] > 0:
                    all_tdice.append(m['tumor_dice'])
                    all_tprec.append(m['tumor_precision'])
                    all_trecall.append(m['tumor_recall'])
                if m['num_empty_slices'] > 0:
                    all_fp.append(m['empty_fp_rate'])

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({
            'loss':   f'{loss.item()*accumulation_steps:.4f}',
            'dice':   f'{np.mean(all_tdice):.3f}'   if all_tdice   else '—',
            'recall': f'{np.mean(all_trecall):.3f}' if all_trecall else '—',
            'fp':     f'{np.mean(all_fp):.3f}'      if all_fp      else '—',
        })
        del ct, pet, bone_pred, tumor_mask, outputs, loss

    n = len(dataloader)
    logger.info(f"\n{'='*70}")
    logger.info(f"Epoch {epoch:03d} [TRAIN]")
    logger.info(f"  Loss:      {total_loss/n:.4f}")
    logger.info(f"  Dice:      {np.mean(all_tdice):.4f}"   if all_tdice   else "  Dice:      N/A")
    logger.info(f"  Precision: {np.mean(all_tprec):.4f}"   if all_tprec   else "  Precision: N/A")
    logger.info(f"  Recall:    {np.mean(all_trecall):.4f}" if all_trecall else "  Recall:    N/A")
    logger.info(f"  FP Rate:   {np.mean(all_fp):.4f}"      if all_fp      else "  FP Rate:   N/A")
    logger.info(f"{'='*70}\n")

    return {
        'loss':            total_loss / n,
        'tumor_dice':      np.mean(all_tdice)   if all_tdice   else 0.0,
        'tumor_precision': np.mean(all_tprec)   if all_tprec   else 0.0,
        'tumor_recall':    np.mean(all_trecall) if all_trecall else 0.0,
        'empty_fp_rate':   np.mean(all_fp)      if all_fp      else 0.0,
    }


def validate(model, dataloader, loss_fn, device, epoch, logger, ds_epochs=150):
    model.eval()
    metrics_fn = BoneOnlyDetailedMetrics(threshold=0.3)

    total_loss  = 0.0
    all_tdice, all_tprec, all_trecall, all_fp = [], [], [], []

    pbar = tqdm(dataloader, desc=f'Epoch {epoch:03d} [Val]')
    with torch.no_grad():
        for batch in pbar:
            ct         = batch['ct'].to(device, non_blocking=True)
            pet        = batch['pet'].to(device, non_blocking=True)
            bone_pred  = batch['bone_pred'].to(device, non_blocking=True)
            tumor_mask = batch['tumor_mask'].to(device, non_blocking=True)
            is_tumor   = batch['is_tumor'].to(device, non_blocking=True)

            with autocast():
                outputs = model(ct, pet, bone_pred, return_intermediate=True)
                # 验证时不加 IRGDA loss（model=None）
                loss    = loss_fn(outputs, tumor_mask, bone_pred, is_tumor,
                                  current_epoch=epoch, ds_epochs=ds_epochs,
                                  model=None)

            m = metrics_fn(outputs['tumor_logits'], tumor_mask, bone_pred, is_tumor)
            if m['num_tumor_slices'] > 0:
                all_tdice.append(m['tumor_dice'])
                all_tprec.append(m['tumor_precision'])
                all_trecall.append(m['tumor_recall'])
            if m['num_empty_slices'] > 0:
                all_fp.append(m['empty_fp_rate'])

            total_loss += loss.item()
            pbar.set_postfix({
                'dice':   f'{np.mean(all_tdice):.3f}'   if all_tdice   else '—',
                'recall': f'{np.mean(all_trecall):.3f}' if all_trecall else '—',
                'fp':     f'{np.mean(all_fp):.3f}'      if all_fp      else '—',
            })
            del ct, pet, bone_pred, tumor_mask, outputs, loss

    n = len(dataloader)
    logger.info(f"\n{'='*70}")
    logger.info(f"Epoch {epoch:03d} [VAL]")
    logger.info(f"  Loss:      {total_loss/n:.4f}")
    logger.info(f"  Dice:      {np.mean(all_tdice):.4f}"   if all_tdice   else "  Dice:      N/A")
    logger.info(f"  Precision: {np.mean(all_tprec):.4f}"   if all_tprec   else "  Precision: N/A")
    logger.info(f"  Recall:    {np.mean(all_trecall):.4f}" if all_trecall else "  Recall:    N/A")
    logger.info(f"  FP Rate:   {np.mean(all_fp):.4f}"      if all_fp      else "  FP Rate:   N/A")
    logger.info(f"{'='*70}\n")

    return {
        'loss':            total_loss / n,
        'tumor_dice':      np.mean(all_tdice)   if all_tdice   else 0.0,
        'tumor_precision': np.mean(all_tprec)   if all_tprec   else 0.0,
        'tumor_recall':    np.mean(all_trecall) if all_trecall else 0.0,
        'empty_fp_rate':   np.mean(all_fp)      if all_fp      else 0.0,
    }


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='IntraBone Single-Stage Training')

    # 路径
    parser.add_argument('--data_root',         type=str,      default='data/img1')
    parser.add_argument('--stage1_model_path', type=str,      default='checkpoints/stage1_2D_20260303-152807/best_model.pth')
    parser.add_argument('--save_dir',          type=str,      default='checkpoints')
    parser.add_argument('--exp_name',          type=str,      default='intrabone_single_small')
    parser.add_argument('--resume_from',       type=str,      default=None)

    # 数据
    parser.add_argument('--batch_size',        type=int,      default=4)
    parser.add_argument('--num_workers',       type=int,      default=4)
    parser.add_argument('--min_tumor_pixels',  type=int,      default=20)
    parser.add_argument('--is_16bit',          type=str2bool, default=True)
    parser.add_argument('--train_empty_ratio', type=float,    default=1.0,
                        help='训练集 empty:tumor 比例，1.0 = 1:1')
    parser.add_argument('--val_empty_ratio',   type=float,    default=5.0,
                        help='验证集 empty:tumor 比例，5.0 = 1:5（接近真实分布）')

    # 模型
    parser.add_argument('--freeze_stage1',     type=str2bool, default=True)
    parser.add_argument('--bone_dilation',     type=int,      default=5)
    parser.add_argument('--iddmga_K',          type=int,      default=3)

    # 学习率
    parser.add_argument('--ct_lr',             type=float,    default=3e-6)
    parser.add_argument('--main_lr',           type=float,    default=3e-5)
    parser.add_argument('--iddmga_lr',         type=float,    default=2e-4)
    parser.add_argument('--lr_min',            type=float,    default=1e-6)
    parser.add_argument('--weight_decay',      type=float,    default=1e-4)
    parser.add_argument('--main_weight_decay', type=float,    default=5e-4)
    parser.add_argument('--iddmga_warmup_epochs', type=int,   default=5)

    # 训练
    parser.add_argument('--epochs',              type=int,    default=100)
    parser.add_argument('--accumulation_steps',  type=int,    default=8)
    parser.add_argument('--clip_grad_norm',      type=float,  default=0.5)
    parser.add_argument('--seed',                type=int,    default=42)
    parser.add_argument('--early_stop_patience', type=int,    default=30)

    # Loss
    parser.add_argument('--irgda_sup_weight',    type=float,  default=0.03)

    # Scheduler
    parser.add_argument('--scheduler_patience',  type=int,    default=4)
    parser.add_argument('--scheduler_factor',    type=float,  default=0.5)

    args = parser.parse_args()
    set_seed(args.seed)

    # ── 保存路径 ──
    ts       = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(args.save_dir, f"{args.exp_name}_{ts}")
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logger(os.path.join(save_dir, 'training.log'))
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'tensorboard'))

    logger.info(f"""
{'='*70}
Intra-Bone Tumor Training — Single Stage v1
{'='*70}
[策略]
  单阶段训练，150 epochs
  删除：三阶段 curriculum / HNM / FP suppression loss
  训练集 tumor:empty = 1:{args.train_empty_ratio:.1f}（固定）
  验证集 tumor:empty = 1:{args.val_empty_ratio:.1f}（接近真实分布）

[Loss]
  2.0 × FocalTversky(alpha=0.3, beta=0.85, gamma=1.5)
  + 0.5 × BCE
  + 0.3 × BoundaryLoss
  + {args.irgda_sup_weight} × IRGDASupervisionLoss（全程）

[Deep Supervision]  全程开启（ds_epochs={args.epochs}）
[IDDMGA Warmup]     前 {args.iddmga_warmup_epochs} epoch 线性升温
[Early Stop]        patience={args.early_stop_patience}
{'='*70}
""")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")

    # ── 数据 ──────────────────────────────────────────────────────────
    train_file = os.path.join(args.data_root, 'train_tumor.txt')
    val_file   = os.path.join(args.data_root, 'val_tumor.txt')

    # 肿瘤数据集
    _, train_tumor_ds = get_intrabone_dataloader_512(
        data_root=args.data_root, split_file=train_file, mode='train',
        batch_size=args.batch_size, num_workers=args.num_workers,
        min_tumor_pixels=args.min_tumor_pixels, is_16bit=args.is_16bit)

    _, val_tumor_ds = get_intrabone_dataloader_512(
        data_root=args.data_root, split_file=val_file, mode='val',
        batch_size=args.batch_size, num_workers=args.num_workers,
        min_tumor_pixels=args.min_tumor_pixels, is_16bit=args.is_16bit)

    # 空切片数据集
    with open(train_file) as f:
        all_train_ids = [l.strip() for l in f if l.strip()]
    with open(val_file) as f:
        all_val_ids = [l.strip() for l in f if l.strip()]

    train_empty_ds = EmptySliceDataset(all_train_ids, args.data_root,
                                       mode='train', is_16bit=args.is_16bit)
    val_empty_ds   = EmptySliceDataset(all_val_ids,   args.data_root,
                                       mode='val',   is_16bit=args.is_16bit)

    # ── 训练集：tumor:empty = 1:train_empty_ratio（固定，不变）──────
    n_train_tumor = len(train_tumor_ds)
    n_train_empty = min(len(train_empty_ds), int(n_train_tumor * args.train_empty_ratio))
    _rng = random.Random(args.seed)
    train_empty_sub = Subset(train_empty_ds,
                              _rng.sample(range(len(train_empty_ds)), n_train_empty))
    train_combined  = ConcatDataset([train_tumor_ds, train_empty_sub])
    train_loader    = DataLoader(
        train_combined, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None)
    logger.info(f"Train: {n_train_tumor} tumor + {n_train_empty} empty "
                f"(ratio 1:{n_train_empty/max(n_train_tumor,1):.2f})\n")

    # ── 验证集：tumor:empty = 1:val_empty_ratio（固定 seed）──────────
    n_val_tumor = len(val_tumor_ds)
    n_val_empty = min(len(val_empty_ds), int(n_val_tumor * args.val_empty_ratio))
    _val_rng = random.Random(args.seed + 1)
    val_empty_sub = Subset(val_empty_ds,
                            _val_rng.sample(range(len(val_empty_ds)), n_val_empty))
    val_combined  = ConcatDataset([val_tumor_ds, val_empty_sub])
    val_loader    = DataLoader(
        val_combined, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None)
    logger.info(f"Val:   {n_val_tumor} tumor + {n_val_empty} empty "
                f"(ratio 1:{n_val_empty/max(n_val_tumor,1):.2f})\n")

    # ── 模型 ──────────────────────────────────────────────────────────
    model = FBFAIntraBoneTumorSegmentation(
        stage1_model_path       = args.stage1_model_path,
        freeze_stage1           = args.freeze_stage1,
        bone_dilation           = args.bone_dilation,
        enable_deep_supervision = True,
        dgma_K_max              = args.iddmga_K,
    ).to(device)

    # Resume
    start_epoch = 1
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"[Resume] Loading from {args.resume_from}")
        ckpt     = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        sd_saved = ckpt.get('model_state_dict', ckpt)
        sd_model = model.state_dict()
        matched  = {k: v for k, v in sd_saved.items()
                    if k in sd_model and sd_model[k].shape == v.shape}
        sd_model.update(matched)
        model.load_state_dict(sd_model)
        if 'optimizer_state_dict' in ckpt:
            try:
                # optimizer 下面才初始化，先存 ckpt 备用
                _resume_opt_state = ckpt['optimizer_state_dict']
            except Exception:
                _resume_opt_state = None
        else:
            _resume_opt_state = None
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        logger.info(f"  Loaded {len(matched)} params | Resuming from epoch {start_epoch}\n")
    else:
        _resume_opt_state = None

    freeze_bn_stats(model)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    iddmga_p    = sum(p.numel() for n, p in model.named_parameters()
                      if 'soe' in n and p.requires_grad)
    logger.info(f"Model: {total_p/1e6:.2f}M total | "
                f"{trainable_p/1e6:.2f}M trainable | "
                f"ID-DMGA {iddmga_p/1e3:.1f}K ({100*iddmga_p/max(trainable_p,1):.1f}%)\n")

    # ── Optimizer（6组 weight-decay 分离）────────────────────────────
    def _split_wd(named_params_list):
        no_wd, wd = [], []
        for name, param in named_params_list:
            if not param.requires_grad:
                continue
            if (param.ndim <= 1
                    or name.endswith('.bias')
                    or 'bn' in name.lower()
                    or '.norm' in name.lower()
                    or 'batch_norm' in name.lower()):
                no_wd.append(param)
            else:
                wd.append(param)
        return no_wd, wd

    ct_named     = [(n, p) for n, p in model.named_parameters() if 'ct_branch' in n]
    iddmga_named = [(n, p) for n, p in model.named_parameters()
                    if 'soe' in n and 'ct_branch' not in n]
    main_named   = [(n, p) for n, p in model.named_parameters()
                    if 'ct_branch' not in n and 'soe' not in n]

    ct_no_wd,     ct_wd     = _split_wd(ct_named)
    main_no_wd,   main_wd   = _split_wd(main_named)
    iddmga_no_wd, iddmga_wd = _split_wd(iddmga_named)

    all_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    covered = len(ct_no_wd)+len(ct_wd)+len(main_no_wd)+len(main_wd)+len(iddmga_no_wd)+len(iddmga_wd)
    assert all_trainable == covered, f"参数未完全覆盖! total={all_trainable}, covered={covered}"

    IDDMGA_GROUP_IDXS = [4, 5]

    optimizer = torch.optim.AdamW([
        {'params': ct_no_wd,     'lr': args.ct_lr,     'weight_decay': 0.0,                    'name': 'ct_no_wd'},
        {'params': ct_wd,        'lr': args.ct_lr,     'weight_decay': args.weight_decay,       'name': 'ct_wd'},
        {'params': main_no_wd,   'lr': args.main_lr,   'weight_decay': 0.0,                    'name': 'main_no_wd'},
        {'params': main_wd,      'lr': args.main_lr,   'weight_decay': args.main_weight_decay,  'name': 'main_wd'},
        {'params': iddmga_no_wd, 'lr': args.iddmga_lr, 'weight_decay': 0.0,                    'name': 'iddmga_no_wd'},
        {'params': iddmga_wd,    'lr': args.iddmga_lr, 'weight_decay': args.weight_decay,       'name': 'iddmga_wd'},
    ], betas=(0.9, 0.999), eps=1e-8)

    # Resume optimizer state（如果有）
    if _resume_opt_state is not None:
        try:
            optimizer.load_state_dict(_resume_opt_state)
            lr_map = {
                'ct_no_wd':     args.ct_lr,
                'ct_wd':        args.ct_lr,
                'main_no_wd':   args.main_lr,
                'main_wd':      args.main_lr,
                'iddmga_no_wd': args.iddmga_lr,
                'iddmga_wd':    args.iddmga_lr,
            }
            for g in optimizer.param_groups:
                name = g.get('name', '')
                if name in lr_map:
                    g['lr'] = lr_map[name]
            logger.info(f"  Optimizer restored; LR overridden → "
                        f"ct={args.ct_lr:.1e}  main={args.main_lr:.1e}  iddmga={args.iddmga_lr:.1e}")
        except Exception as e:
            logger.warning(f"  Optimizer load failed ({e}), using fresh optimizer")

    logger.info("Optimizer groups (6组 weight-decay 分离):")
    logger.info(f"  [0] ct_no_wd:     {len(ct_no_wd):4d} params, lr={args.ct_lr:.1e}, wd=0")
    logger.info(f"  [1] ct_wd:        {len(ct_wd):4d} params, lr={args.ct_lr:.1e}, wd={args.weight_decay:.0e}")
    logger.info(f"  [2] main_no_wd:   {len(main_no_wd):4d} params, lr={args.main_lr:.1e}, wd=0")
    logger.info(f"  [3] main_wd:      {len(main_wd):4d} params, lr={args.main_lr:.1e}, wd={args.main_weight_decay:.0e}")
    logger.info(f"  [4] iddmga_no_wd: {len(iddmga_no_wd):4d} params, lr={args.iddmga_lr:.1e}, wd=0")
    logger.info(f"  [5] iddmga_wd:    {len(iddmga_wd):4d} params, lr={args.iddmga_lr:.1e}, wd={args.weight_decay:.0e}\n")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=args.lr_min, verbose=True)

    # ── Loss ──────────────────────────────────────────────────────────
    base_loss_fn = SmallTumorLoss(
        ftl_weight=2.0, bce_weight=0.5, bnd_weight=0.3,
        alpha=0.3, beta=0.85, gamma=1.5, boundary_weight=5.0)
    loss_fn = SingleStageLoss(
        base_loss_fn,
        irgda_sup_weight=args.irgda_sup_weight)

    scaler        = GradScaler()
    early_stopper = EarlyStopping(patience=args.early_stop_patience, mode='max')

    best_dice  = 0.0
    best_epoch = 0

    logger.info("=" * 70)
    logger.info("Starting Training (Single Stage)...")
    logger.info("=" * 70 + "\n")

    # ── 训练循环（单阶段）────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # IDDMGA LR warmup（前5轮）
        if epoch <= args.iddmga_warmup_epochs:
            apply_iddmga_lr_warmup(
                optimizer, epoch,
                warmup_end_epoch  = args.iddmga_warmup_epochs,
                iddmga_lr_full    = args.iddmga_lr,
                iddmga_group_idxs = IDDMGA_GROUP_IDXS)
            logger.info(f"[IDDMGA Warmup] epoch={epoch}/{args.iddmga_warmup_epochs} "
                        f"lr={optimizer.param_groups[IDDMGA_GROUP_IDXS[1]]['lr']:.2e}")

        # ── 训练 + 验证 ──
        train_m = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler,
            epoch, logger,
            accumulation_steps = args.accumulation_steps,
            clip_grad_norm     = args.clip_grad_norm,
            ds_epochs          = args.epochs)   # Deep Supervision 全程开启

        val_m = validate(
            model, val_loader, loss_fn, device, epoch, logger,
            ds_epochs=args.epochs)

        # ── Scheduler / TensorBoard ──
        cur_dice = val_m['tumor_dice']
        scheduler.step(cur_dice)

        writer.add_scalar('Train/Loss',    train_m['loss'],            epoch)
        writer.add_scalar('Train/Dice',    train_m['tumor_dice'],      epoch)
        writer.add_scalar('Train/Recall',  train_m['tumor_recall'],    epoch)
        writer.add_scalar('Train/FP',      train_m['empty_fp_rate'],   epoch)
        writer.add_scalar('Val/Loss',      val_m['loss'],              epoch)
        writer.add_scalar('Val/Dice',      val_m['tumor_dice'],        epoch)
        writer.add_scalar('Val/Recall',    val_m['tumor_recall'],      epoch)
        writer.add_scalar('Val/FP',        val_m['empty_fp_rate'],     epoch)
        writer.add_scalar('LR/CT',    optimizer.param_groups[1]['lr'], epoch)
        writer.add_scalar('LR/Main',  optimizer.param_groups[3]['lr'], epoch)
        writer.add_scalar('LR/IDDMGA',optimizer.param_groups[5]['lr'], epoch)

        # ── 保存最佳 ──
        if cur_dice > best_dice:
            best_dice  = cur_dice
            best_epoch = epoch
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice':                 best_dice,
                'recall':               val_m['tumor_recall'],
                'fp_rate':              val_m['empty_fp_rate'],
                'args':                 vars(args),
            }, os.path.join(save_dir, 'best_model.pth'))
            logger.info(f"  [Best] Dice={best_dice:.4f} "
                        f"Recall={val_m['tumor_recall']:.4f} "
                        f"FP={val_m['empty_fp_rate']:.6f}\n")

        logger.info(f"  Epoch {epoch} | Time={int(time.time()-t0)}s | "
                    f"LR_ct={optimizer.param_groups[1]['lr']:.2e} | "
                    f"LR_main={optimizer.param_groups[3]['lr']:.2e} | "
                    f"LR_iddmga={optimizer.param_groups[5]['lr']:.2e}\n"
                    f"  Best: Dice={best_dice:.4f} (epoch {best_epoch})\n")

        # ── Early Stopping ──
        if early_stopper(cur_dice):
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(patience={args.early_stop_patience})")
            break

        # 每 5 epoch 清理一次显存
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Training done. Best Dice={best_dice:.4f} at epoch {best_epoch}")
    writer.close()


if __name__ == '__main__':
    main()