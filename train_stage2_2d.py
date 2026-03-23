"""
Intra-Bone Tumor Training — 分阶段稳定收敛版 v6
=================================================

v5 在 v4 基础上针对 Precision 过低（≈0.51）的专项修复，目标 Dice ≥ 0.70：

  [v5-A] Tversky alpha/beta 初始值调整：alpha 0.3→0.4, beta 0.7→0.6
         增加对 FP 的惩罚，减少过度预测
  [v5-B] gamma 从 1.5 提高至 2.0，让难分 FP 区域获得更大惩罚梯度
  [v5-C] Phase2 动态切换（epoch25）：alpha→0.55, beta→0.45（精度优先模式）
  [v5-D] BoundaryLoss 提前至 epoch15 启动（原 epoch30），bnd_weight_max 0.03→0.05
  [v5-E] phase2_start 提前至 epoch20（原 25），phase3_start 提前至 epoch35（原 60）
  [v5-F] validate() 接入 postprocess_batch 后处理（CC 过滤，min_pixels=30）
  [v5-G] main_lr 提高至 2e-4（原 1e-4）
  [v5-H] cosine_T0 从 30 拉长至 50（防学习率过快衰减，改用 CosineAnnealingWarmRestarts）
  [v5-I] train_empty_ratio 1.0→2.0（每 batch tumor:empty = 1:2）
  [v5-J] 验证指标新增 Dice@0.5 / Dice@0.3 双阈值 + confidence_gap 监控
  [v5-K] 保存条件：prec≥0.35, recall≥0.55（放宽自 0.45/0.65）

v4 沿用项:
  [v4-A] freeze_stage1=True
  [v4-B] beta 切换 epoch 60→（v5 已提前至 25）
  [v4-C] GroupNorm 替换 BatchNorm2d
  [v4-D] state_dim [32, 25, 9, 9]

训练阶段:
  Phase 1  epoch  1~19 : FTL + BCE（建立分割基础）
  Phase 2  epoch 20~34 : + BoundaryLoss warmup（0 → 0.05）+ Precision 优先 Tversky
  Phase 3  epoch 35+   : + IRGDA supervision + FP suppress

Deep Supervision:  epoch 1~40 开启，epoch > 40 关闭
Sampler:           BalancedBatchSampler（N/3 tumor + 2N/3 empty per batch，1:2）
Scheduler:         CosineAnnealingWarmRestarts T0=50 Tmult=2
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
from torch.utils.data import DataLoader, ConcatDataset, Subset, Sampler
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
    HardNegativeDataset,          # [v6] hard negative
    EnhancedCTNormalizer,
    EnhancedPETNormalizer,
    get_augmentation,
)

from new_network.fbfa_intrabone_enhanced import FBFAIntraBoneTumorSegmentation

# [FIX] 使用 v3 分阶段 loss（BoundaryLoss warmup + 数值稳定）
from bone_only_loss_metrics import (
    SmallTumorLoss,
    SingleStageLoss,
    BoneOnlyDetailedMetrics,
    postprocess_batch,          # [v5-F] 验证后处理 CC 过滤
)


# ============================================================
#  工具函数（不变）
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
#  EmptySliceDataset（不变）
# ============================================================

class EmptySliceDataset(torch.utils.data.Dataset):
    """读取空切片（is_tumor=False）— [v6] 已升级为 2.5D 三通道接口"""

    def __init__(self, image_list, img_root, mode='train', is_16bit=True):
        self.img_root  = img_root
        self.mode      = mode
        self.is_16bit  = is_16bit
        self.ct_norm   = EnhancedCTNormalizer()
        self.pet_norm  = EnhancedPETNormalizer()
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
        image_id   = self.image_list[idx]
        ct_path    = self._get_path(image_id, '_CT.png')
        pet_path   = self._get_path(image_id, '_PET.png')
        tmask_path = self._get_path(image_id, '_mask.png')
        bone_path  = self._get_bone_path(image_id)

        def read_ct(p):
            if self.is_16bit:
                ct = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                return self.ct_norm.normalize(ct, method='bone')
            return cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        ct_2d     = read_ct(ct_path)                             # (H, W)
        pet_raw   = cv2.imread(pet_path, cv2.IMREAD_UNCHANGED)
        pet_2d    = self.pet_norm.normalize(pet_raw)             # (H, W)
        tmask_raw = cv2.imread(tmask_path, cv2.IMREAD_GRAYSCALE)
        bone_raw  = cv2.imread(bone_path,  cv2.IMREAD_GRAYSCALE)

        tumor_mask = (tmask_raw > 127).astype(np.float32)
        bone_pred  = (bone_raw  > 127).astype(np.float32)

        # [v6] 单切片复制为三通道 (H,W,3) 与 2.5D 接口一致
        ct3ch  = np.stack([ct_2d,  ct_2d,  ct_2d],  axis=-1)   # (H, W, 3)
        pet3ch = np.stack([pet_2d, pet_2d, pet_2d],  axis=-1)  # (H, W, 3)

        aug        = self.transform(image=ct3ch, pet3ch=pet3ch,
                                    bone_pred=bone_pred, tumor_mask=tumor_mask)
        ct         = aug['image'].float()       # (3, H, W) after ToTensorV2
        pet        = aug['pet3ch'].float()      # (3, H, W)
        bone_pred  = aug['bone_pred'].float()
        tumor_mask = aug['tumor_mask'].float()

        def e3(t): return t.unsqueeze(0) if t.dim() == 2 else t
        bone_pred  = e3(bone_pred)
        tumor_mask = e3(tumor_mask)

        ct         = torch.clamp(ct,  0, 1)   # [v6-fix] 不遮盖，保留上下文
        pet        = torch.clamp(pet, 0, 1)
        bone_pred  = (bone_pred  > 0.5).float()
        tumor_mask = (tumor_mask > 0.5).float() * bone_pred  # label 仍限在骨内

        return {
            'ct':          ct,           # (3, H, W)
            'pet':         pet,          # (3, H, W)
            'bone_pred':   bone_pred,    # (1, H, W)
            'tumor_mask':  tumor_mask,   # (1, H, W)
            'name':        image_id,
            'tumor_ratio': torch.tensor(0.0,   dtype=torch.float32),
            'is_tumor':    torch.tensor(False,  dtype=torch.bool),
        }


# ============================================================
#  [MODIFIED] BalancedBatchSampler — 替换 WeightedRandomSampler
# ============================================================

class BalancedBatchSampler(Sampler):
    """
    [MODIFIED] 每个 batch 保证 tumor_per_batch 个 tumor + empty_per_batch 个 empty，
    彻底消除全空 batch / 全肿瘤 batch 的可能。

    原理：
      1. 将 ConcatDataset 的索引按 tumor / empty 分成两个池
      2. 每次 yield 一个 batch：从 tumor 池随机取 n_t 个，从 empty 池随机取 n_e 个
      3. 两个池各自循环洗牌（无放回），池耗尽后重新 shuffle

    参数:
      dataset         : ConcatDataset([tumor_ds, empty_ds])
      n_tumor         : tumor 数据集的样本数（前 n_tumor 个索引属于 tumor）
      batch_size      : batch 大小
      tumor_fraction  : 每个 batch 中 tumor 占比（默认 0.5 → N/2 tumor + N/2 empty）
    """

    def __init__(self, dataset, n_tumor, batch_size, tumor_fraction=0.5, seed=42):
        super().__init__(dataset)
        self.n_total   = len(dataset)
        self.n_tumor   = n_tumor
        self.n_empty   = self.n_total - n_tumor
        self.bs        = batch_size
        self.n_t       = max(1, int(batch_size * tumor_fraction))  # tumor per batch
        self.n_e       = batch_size - self.n_t                     # empty per batch
        self.seed      = seed
        self.rng       = random.Random(seed)

        assert self.n_tumor > 0,  "No tumor samples found"
        assert self.n_empty > 0,  "No empty samples found"

        self.tumor_idxs = list(range(n_tumor))
        self.empty_idxs = list(range(n_tumor, self.n_total))

        # 每轮 epoch 总 iter 数：以 tumor 池能产出的轮次为准
        self.n_batches  = max(1, self.n_tumor // self.n_t)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        # 打乱两个池
        t_pool = self.tumor_idxs.copy()
        e_pool = self.empty_idxs.copy()
        self.rng.shuffle(t_pool)
        self.rng.shuffle(e_pool)

        t_ptr = 0
        e_ptr = 0

        for _ in range(self.n_batches):
            # tumor: 池不够时重新 shuffle 循环
            if t_ptr + self.n_t > len(t_pool):
                self.rng.shuffle(t_pool)
                t_ptr = 0
            # empty: 同上
            if e_ptr + self.n_e > len(e_pool):
                self.rng.shuffle(e_pool)
                e_ptr = 0

            batch = (t_pool[t_ptr:t_ptr + self.n_t]
                     + e_pool[e_ptr:e_ptr + self.n_e])
            self.rng.shuffle(batch)   # 打乱 tumor/empty 顺序
            yield batch

            t_ptr += self.n_t
            e_ptr += self.n_e


# ============================================================
#  IDDMGA LR Warmup（不变）
# ============================================================

def apply_iddmga_lr_warmup(optimizer, epoch, warmup_end_epoch,
                            iddmga_lr_full, iddmga_group_idxs):
    ratio = epoch / warmup_end_epoch
    lr    = iddmga_lr_full * ratio
    for idx in iddmga_group_idxs:
        optimizer.param_groups[idx]['lr'] = lr


# ============================================================
#  [MODIFIED] Train Epoch — 分阶段控制
# ============================================================

def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler,
                epoch, logger,
                accumulation_steps=4,
                clip_grad_norm=1.0):
    model.train()
    freeze_bn_stats(model)
    # [FIX] 统一用 threshold=0.5，与推理一致
    metrics_fn = BoneOnlyDetailedMetrics(threshold=0.5)

    total_loss = 0.0
    all_tdice, all_tprec, all_trecall, all_fp = [], [], [], []
    all_bnd_loss  = []
    all_bnd_total = []

    bnd_w_this_epoch = loss_fn.tumor_loss.get_boundary_weight(epoch)

    phase = 1 if epoch < 25 else (2 if epoch < 60 else 3)
    phase_desc = {
        1: "Phase1[FTL+BCE+FP_suppress]",
        2: f"Phase2[+BndLoss bnd_w={bnd_w_this_epoch:.4f}]",
        3: "Phase3[+IRGDA+FP]",
    }[phase]
    logger.info(f"  Epoch {epoch} → {phase_desc}  "
                f"DS={'ON' if epoch <= loss_fn.tumor_loss.ds_cutoff else 'OFF'}")

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
                              current_epoch=epoch, model=model)

        # [FIX] 不再跳过空batch，因为空切片现在有FP loss梯度
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            del ct, pet, bone_pred, tumor_mask, outputs, loss
            continue

        if loss_fn.last_bnd_loss > 0.0:
            all_bnd_loss.append(loss_fn.last_bnd_loss)
        if loss_fn.last_total_loss > 0.0:
            all_bnd_total.append(loss_fn.last_total_loss)

        (loss / accumulation_steps).backward()

        step = (batch_idx + 1) % accumulation_steps == 0
        last = (batch_idx + 1) == len(dataloader)
        if step or last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if batch_idx % 10 == 0:
            with torch.no_grad():
                # [FIX] 使用硬Dice
                m = metrics_fn(outputs['tumor_logits'], tumor_mask, bone_pred, is_tumor)
                if m['num_tumor_slices'] > 0:
                    all_tdice.append(m['tumor_dice'])      # 现在是硬Dice
                    all_tprec.append(m['tumor_precision'])
                    all_trecall.append(m['tumor_recall'])
                if m['num_empty_slices'] > 0:
                    all_fp.append(m['empty_fp_rate'])

        total_loss += loss.item()
        pbar.set_postfix({
            'loss':   f'{loss.item():.4f}',
            'dice':   f'{np.mean(all_tdice):.3f}'   if all_tdice   else '—',
            'prec':   f'{np.mean(all_tprec):.3f}'   if all_tprec   else '—',
            'recall': f'{np.mean(all_trecall):.3f}' if all_trecall else '—',
            'fp':     f'{np.mean(all_fp):.4f}'      if all_fp      else '—',
        })
        del ct, pet, bone_pred, tumor_mask, outputs, loss

    n = len(dataloader)
    mean_bnd_loss = np.mean(all_bnd_loss)  if all_bnd_loss  else 0.0
    mean_total    = np.mean(all_bnd_total) if all_bnd_total else 1.0
    bnd_ratio     = (bnd_w_this_epoch * mean_bnd_loss) / (mean_total + 1e-8)

    dice_val   = np.mean(all_tdice)   if all_tdice   else 0.0
    prec_val   = np.mean(all_tprec)   if all_tprec   else 0.0
    recall_val = np.mean(all_trecall) if all_trecall else 0.0
    fp_val     = np.mean(all_fp)      if all_fp      else 0.0

    logger.info(f"\n{'='*70}")
    logger.info(f"Epoch {epoch:03d} [TRAIN]  {phase_desc}")
    logger.info(f"  Loss:      {total_loss/n:.4f}")
    logger.info(f"  Dice@0.5:  {dice_val:.4f}   ← 硬Dice（已修复）")
    logger.info(f"  Precision: {prec_val:.4f}")
    logger.info(f"  Recall:    {recall_val:.4f}")
    logger.info(f"  FP Rate:   {fp_val:.4f}")
    logger.info(f"  [Bnd] weight={bnd_w_this_epoch:.4f}  "
                f"raw={mean_bnd_loss:.4f}  ratio={bnd_ratio:.4f}")
    if prec_val < 0.40 and dice_val > 0:
        logger.warning(f"  ⚠ Precision={prec_val:.4f} < 0.40：过度预测")
    logger.info(f"{'='*70}\n")

    return {
        'loss':            total_loss / n,
        'tumor_dice':      dice_val,
        'tumor_precision': prec_val,
        'tumor_recall':    recall_val,
        'empty_fp_rate':   fp_val,
        'bnd_weight':      bnd_w_this_epoch,
        'bnd_loss':        mean_bnd_loss,
        'bnd_ratio':       bnd_ratio,
    }


def validate(model, dataloader, loss_fn, device, epoch, logger):
    model.eval()
    # [FIX] 只用一个metrics_fn，threshold=0.5，用硬Dice
    metrics_fn = BoneOnlyDetailedMetrics(threshold=0.5)

    total_loss = 0.0
    all_dice, all_prec, all_recall, all_fp = [], [], [], []

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
                loss    = loss_fn(outputs, tumor_mask, bone_pred, is_tumor,
                                  current_epoch=epoch, model=None)

            logits = outputs['tumor_logits']
            # [FIX] tumor_dice 现在是硬Dice
            m = metrics_fn(logits, tumor_mask, bone_pred, is_tumor)

            if m['num_tumor_slices'] > 0:
                all_dice.append(m['tumor_dice'])
                all_prec.append(m['tumor_precision'])
                all_recall.append(m['tumor_recall'])
            if m['num_empty_slices'] > 0:
                all_fp.append(m['empty_fp_rate'])

            total_loss += loss.item()
            pbar.set_postfix({
                'dice':   f'{np.mean(all_dice):.3f}'   if all_dice   else '—',
                'prec':   f'{np.mean(all_prec):.3f}'   if all_prec   else '—',
                'recall': f'{np.mean(all_recall):.3f}' if all_recall else '—',
                'fp':     f'{np.mean(all_fp):.6f}'     if all_fp     else '—',
            })
            del ct, pet, bone_pred, tumor_mask, outputs, loss, logits

    n      = len(dataloader)
    dice   = np.mean(all_dice)   if all_dice   else 0.0
    prec   = np.mean(all_prec)   if all_prec   else 0.0
    recall = np.mean(all_recall) if all_recall else 0.0
    fp     = np.mean(all_fp)     if all_fp     else 0.0

    logger.info(f"\n{'='*70}")
    logger.info(f"Epoch {epoch:03d} [VAL]")
    logger.info(f"  Loss:      {total_loss/n:.4f}")
    logger.info(f"  Dice@0.5:  {dice:.4f}   ← 硬Dice主指标（已修复）")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  FP Rate:   {fp:.6f}")
    logger.info(f"{'='*70}\n")

    return {
        'loss':            total_loss / n,
        'tumor_dice':      dice,
        'tumor_precision': prec,
        'tumor_recall':    recall,
        'empty_fp_rate':   fp,
    }


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='IntraBone Phased Training v2')

    # 路径
    parser.add_argument('--data_root',         type=str,      default='data/img1')
    parser.add_argument('--stage1_model_path', type=str,      default='checkpoints/stage1_2D_20260303-152807/best_model.pth')
    parser.add_argument('--save_dir',          type=str,      default='checkpoints')
    parser.add_argument('--exp_name',          type=str,      default='intrabone_phased_v2')
    parser.add_argument('--resume_from',       type=str,      default=None)

    # 数据
    parser.add_argument('--batch_size',        type=int,      default=4)
    parser.add_argument('--num_workers',       type=int,      default=4)
    parser.add_argument('--min_tumor_pixels',  type=int,      default=100)
    parser.add_argument('--is_16bit',          type=str2bool, default=True)
    parser.add_argument('--train_empty_ratio', type=float,    default=2.0,
                        help='训练集 empty:tumor 比例，2.0 = 1:2 (v5: 提高背景负样本)')  # [v5-I]
    parser.add_argument('--val_empty_ratio',   type=float,    default=5.0)

    # 模型
    parser.add_argument('--freeze_stage1',     type=str2bool, default=True)   # [v4-A]
    parser.add_argument('--bone_dilation',     type=int,      default=5)
    parser.add_argument('--iddmga_K',          type=int,      default=3)

    # 学习率
    parser.add_argument('--ct_lr',             type=float,    default=3e-6)
    parser.add_argument('--main_lr',           type=float,    default=2e-4)   # [v5-G] 1e-4 → 2e-4
    parser.add_argument('--iddmga_lr',         type=float,    default=2e-4)
    parser.add_argument('--lr_min',            type=float,    default=1e-7)
    parser.add_argument('--weight_decay',      type=float,    default=1e-4)
    parser.add_argument('--main_weight_decay', type=float,    default=3e-4)

    # IDDMGA warmup（保留，保证 IRGDA 模块参数仍然走 LR warmup）
    parser.add_argument('--iddmga_warmup_epochs', type=int,   default=10)
    parser.add_argument('--iddmga_warmup_start',  type=int,   default=5)

    # [v5-E] 阶段边界提前
    parser.add_argument('--phase2_start',      type=int,      default=20,
                        help='BoundaryLoss + Precision Tversky 启用 epoch')
    parser.add_argument('--phase3_start',      type=int,      default=35,
                        help='IRGDA + FP suppress 启用 epoch')
    parser.add_argument('--ds_cutoff',         type=int,      default=40,
                        help='Deep Supervision 关闭的 epoch（> ds_cutoff 后关闭）')
    parser.add_argument('--irgda_rampup',      type=int,      default=15,
                        help='IRGDA 权重线性升温轮数')

    # [v5-D] BoundaryLoss 提前
    parser.add_argument('--boundary_delay_start',type=int,    default=15,
                        help='BoundaryLoss warmup 实际开始的 epoch（原 30 → 15）')
    parser.add_argument('--bnd_weight_max',      type=float,  default=0.05,
                        help='BoundaryLoss 最大权重上限（原 0.03 → 0.05）')
    parser.add_argument('--bnd_rampup_rate',     type=float,  default=0.003,
                        help='BoundaryLoss 每 epoch 权重增量（原 0.002 → 0.003）')
    parser.add_argument('--use_boundary_delay',  type=str2bool, default=True)

    # [v5-H] Cosine Scheduler 参数
    parser.add_argument('--use_cosine',        type=str2bool, default=True,
                        help='使用 CosineAnnealingWarmRestarts 替代 ReduceLROnPlateau')
    parser.add_argument('--cosine_T0',         type=int,      default=80,
                        help='Cosine 第一周期长度（v6: 80，避免 epoch50 重启破坏精调）')
    parser.add_argument('--cosine_Tmult',      type=int,      default=2,
                        help='Cosine 周期倍增系数')

    # 训练
    parser.add_argument('--epochs',              type=int,    default=150)
    parser.add_argument('--accumulation_steps',  type=int,    default=4)
    parser.add_argument('--clip_grad_norm',      type=float,  default=1.0)
    parser.add_argument('--seed',                type=int,    default=42)
    parser.add_argument('--early_stop_patience', type=int,    default=50,
                        help='v6: 50（保证 Phase3 IRGDA 充分训练后再早停）')

    # [v6] Hard Negative 参数
    parser.add_argument('--hard_neg_range',       type=int,   default=5,
                        help='HardNeg 邻域半径（距肿瘤切片 ±N slice）')
    parser.add_argument('--hard_neg_max_per_tumor', type=int, default=3,
                        help='每个肿瘤切片最多采几个 hard negative')

    # Loss 权重
    parser.add_argument('--irgda_sup_weight',    type=float,  default=0.001)
    parser.add_argument('--fp_suppress_weight',  type=float,  default=0.03)

    # Scheduler (ReduceLROnPlateau fallback)
    parser.add_argument('--scheduler_patience',  type=int,    default=15)
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
Intra-Bone Tumor Training — 分阶段稳定收敛版 v3
{'='*70}
[阶段划分]
  Phase 1  epoch  1~{args.phase2_start-1}: FTL + BCE（仅核心 loss）
  Phase 2  epoch {args.phase2_start}~{args.phase3_start-1}: + BoundaryLoss warmup
  Phase 3  epoch {args.phase3_start}+   : + IRGDA + FP suppress

[BoundaryLoss v3 调度]
  最大权重:     {args.bnd_weight_max}   (降幅: 0.15 → 0.03)
  Warmup 速率:  +{args.bnd_rampup_rate}/epoch (15 epoch 到达上限)
  延迟开关:     USE_BOUNDARY_DELAY={args.use_boundary_delay}
  延迟起始:     epoch {args.boundary_delay_start}
  实际激活时间: epoch {args.boundary_delay_start if args.use_boundary_delay else args.phase2_start}

[Deep Supervision]  epoch 1~{args.ds_cutoff}（之后关闭）
[IRGDA]     epoch >= {args.phase3_start} 启用，线性升温 {args.irgda_rampup} epoch
[FP suppress] epoch >= {args.phase3_start} 启用
[Sampler]   BalancedBatchSampler (N/2 tumor + N/2 empty per batch)

[LR]  main={args.main_lr:.1e}  ct={args.ct_lr:.1e}  iddmga={args.iddmga_lr:.1e}
[Grad] accum={args.accumulation_steps}  clip={args.clip_grad_norm}
{'='*70}
""")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")

    # ── 数据 ──
    train_file = os.path.join(args.data_root, 'train_tumor.txt')
    val_file   = os.path.join(args.data_root, 'val_tumor.txt')

    _, train_tumor_ds = get_intrabone_dataloader_512(
        data_root=args.data_root, split_file=train_file, mode='train',
        batch_size=args.batch_size, num_workers=args.num_workers,
        min_tumor_pixels=args.min_tumor_pixels, is_16bit=args.is_16bit)

    _, val_tumor_ds = get_intrabone_dataloader_512(
        data_root=args.data_root, split_file=val_file, mode='val',
        batch_size=args.batch_size, num_workers=args.num_workers,
        min_tumor_pixels=args.min_tumor_pixels, is_16bit=args.is_16bit)

    with open(train_file) as f:
        all_train_ids = [l.strip() for l in f if l.strip()]
    with open(val_file) as f:
        all_val_ids = [l.strip() for l in f if l.strip()]

    train_empty_ds = EmptySliceDataset(all_train_ids, args.data_root,
                                       mode='train', is_16bit=args.is_16bit)
    val_empty_ds   = EmptySliceDataset(all_val_ids,   args.data_root,
                                       mode='val',   is_16bit=args.is_16bit)

    # ── [v6] Hard Negative Dataset 替换随机空切片 ──
    # 使用距肿瘤切片 ±neg_range 的邻近无肿瘤切片作为负样本
    # 这些"难负样本"骨结构与肿瘤切片相似，是 FP 的真实来源
    train_hard_neg = HardNegativeDataset(
        tumor_image_ids = list(train_tumor_ds.image_list),
        all_image_ids   = all_train_ids,
        img_root        = args.data_root,
        neg_range       = args.hard_neg_range,
        mode            = 'train',
        is_16bit        = args.is_16bit,
        max_per_tumor   = args.hard_neg_max_per_tumor,
    )
    # 若 hard negative 数量不足，用随机 empty 补充到目标比例
    n_train_tumor  = len(train_tumor_ds)
    n_hard_neg     = len(train_hard_neg)
    n_target_neg   = int(n_train_tumor * args.train_empty_ratio)
    if n_hard_neg < n_target_neg:
        n_supplement = n_target_neg - n_hard_neg
        n_supplement = min(n_supplement, len(train_empty_ds))
        _rng = random.Random(args.seed)
        supplement_ds = Subset(train_empty_ds,
                               _rng.sample(range(len(train_empty_ds)), n_supplement))
        neg_combined = ConcatDataset([train_hard_neg, supplement_ds])
        logger.info(f"  HardNeg: {n_hard_neg} hard + {n_supplement} random supplement\n")
    else:
        # 截取到目标数量
        _rng = random.Random(args.seed)
        hard_sub = Subset(train_hard_neg,
                          _rng.sample(range(n_hard_neg), min(n_target_neg, n_hard_neg)))
        neg_combined = hard_sub
        logger.info(f"  HardNeg: {min(n_target_neg, n_hard_neg)} hard negative slices\n")

    n_train_neg   = len(neg_combined)
    train_combined = ConcatDataset([train_tumor_ds, neg_combined])

    # [v6] tumor_fraction: tumor/(tumor+neg)
    tumor_frac = 1.0 / (1.0 + args.train_empty_ratio)
    train_sampler = BalancedBatchSampler(
        train_combined,
        n_tumor       = n_train_tumor,
        batch_size    = args.batch_size,
        tumor_fraction= tumor_frac,
        seed          = args.seed)

    train_loader = DataLoader(
        train_combined,
        batch_sampler      = train_sampler,
        num_workers        = args.num_workers,
        pin_memory         = True,
        persistent_workers = (args.num_workers > 0),
        prefetch_factor    = 2 if args.num_workers > 0 else None)

    logger.info(f"Train: {n_train_tumor} tumor + {n_train_neg} neg "
                f"(hard_neg={n_hard_neg}, ratio 1:{args.train_empty_ratio:.1f})  "
                f"batches/epoch={len(train_sampler)}\n")

    # ── 验证集：同样使用 HardNeg（val split 的邻近切片）──
    val_hard_neg = HardNegativeDataset(
        tumor_image_ids = list(val_tumor_ds.image_list),
        all_image_ids   = all_val_ids,
        img_root        = args.data_root,
        neg_range       = args.hard_neg_range,
        mode            = 'val',
        is_16bit        = args.is_16bit,
        max_per_tumor   = args.hard_neg_max_per_tumor,
    )
    n_val_tumor = len(val_tumor_ds)
    n_val_hard  = len(val_hard_neg)
    # 验证集补充随机 empty 到 1:3.87（原始比例）
    n_val_target = int(n_val_tumor * args.val_empty_ratio)
    if n_val_hard < n_val_target:
        n_val_sup = min(n_val_target - n_val_hard, len(val_empty_ds))
        _val_rng  = random.Random(args.seed + 1)
        val_sup   = Subset(val_empty_ds,
                           _val_rng.sample(range(len(val_empty_ds)), n_val_sup))
        val_neg   = ConcatDataset([val_hard_neg, val_sup])
    else:
        _val_rng = random.Random(args.seed + 1)
        val_neg  = Subset(val_hard_neg,
                          _val_rng.sample(range(n_val_hard),
                                          min(n_val_target, n_val_hard)))
    val_combined = ConcatDataset([val_tumor_ds, val_neg])
    val_loader   = DataLoader(
        val_combined, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None)
    logger.info(f"Val:   {n_val_tumor} tumor + {len(val_neg)} neg "
                f"(hard={n_val_hard}, ratio 1:{len(val_neg)/max(n_val_tumor,1):.2f})\n")

    # ── 模型 ──
    model = FBFAIntraBoneTumorSegmentation(
        stage1_model_path       = args.stage1_model_path,
        freeze_stage1           = args.freeze_stage1,
        bone_dilation           = args.bone_dilation,
        enable_deep_supervision = True,
        dgma_K_max              = args.iddmga_K,
    ).to(device)

    # Resume
    start_epoch    = 1
    _resume_opt_state = None
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
            _resume_opt_state = ckpt.get('optimizer_state_dict')
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        logger.info(f"  Loaded {len(matched)} params | Resuming from epoch {start_epoch}\n")

    freeze_bn_stats(model)

    # ── Optimizer（6组 weight-decay 分离，不变）──
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

    IDDMGA_GROUP_IDXS = [4, 5]

    optimizer = torch.optim.AdamW([
        {'params': ct_no_wd,     'lr': args.ct_lr,     'weight_decay': 0.0,                   'name': 'ct_no_wd'},
        {'params': ct_wd,        'lr': args.ct_lr,     'weight_decay': args.weight_decay,      'name': 'ct_wd'},
        {'params': main_no_wd,   'lr': args.main_lr,   'weight_decay': 0.0,                   'name': 'main_no_wd'},
        {'params': main_wd,      'lr': args.main_lr,   'weight_decay': args.main_weight_decay, 'name': 'main_wd'},
        {'params': iddmga_no_wd, 'lr': args.iddmga_lr, 'weight_decay': 0.0,                   'name': 'iddmga_no_wd'},
        {'params': iddmga_wd,    'lr': args.iddmga_lr, 'weight_decay': args.weight_decay,      'name': 'iddmga_wd'},
    ], betas=(0.9, 0.999), eps=1e-8)

    if _resume_opt_state is not None:
        try:
            optimizer.load_state_dict(_resume_opt_state)
            lr_map = {
                'ct_no_wd':     args.ct_lr,   'ct_wd':        args.ct_lr,
                'main_no_wd':   args.main_lr,  'main_wd':      args.main_lr,
                'iddmga_no_wd': args.iddmga_lr,'iddmga_wd':    args.iddmga_lr,
            }
            for g in optimizer.param_groups:
                name = g.get('name', '')
                if name in lr_map:
                    g['lr'] = lr_map[name]
            logger.info(f"  Optimizer restored; LR overridden")
        except Exception as e:
            logger.warning(f"  Optimizer load failed ({e}), using fresh optimizer")

    # ── Loss ──
    base_loss_fn = SmallTumorLoss(
        ftl_weight=2.0,
        bce_weight=0.5,
        bnd_weight_max=args.bnd_weight_max,       # [v5-D] 0.03 → 0.05
        bnd_rampup_rate=args.bnd_rampup_rate,     # [v5-D] 0.002 → 0.003
        alpha=0.4,            # [v5-A] 0.3 → 0.4（增加 FP 惩罚）
        beta=0.6,             # [v5-A] 0.70 → 0.6
        gamma=2.0,            # [v5-B] 1.5 → 2.0（难分 FP 更大梯度）
        boundary_weight=5.0,
        phase2_start=args.phase2_start,           # [v5-E] 25 → 20
        use_boundary_delay=args.use_boundary_delay,
        boundary_delay_start=args.boundary_delay_start,  # [v5-D] 30 → 15
        ds_cutoff=args.ds_cutoff)

    loss_fn = SingleStageLoss(
        base_loss_fn,
        irgda_sup_weight   = args.irgda_sup_weight,   # [MODIFIED]
        fp_suppress_weight = args.fp_suppress_weight,  # [MODIFIED]
        irgda_start_epoch  = args.phase3_start,        # [MODIFIED]
        fp_start_epoch     = args.phase3_start,        # [MODIFIED]
        irgda_rampup_epochs= args.irgda_rampup)        # [MODIFIED]

    # [v5-H] Cosine scheduler（T0=50 防早期 LR 衰减过快）
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmult,
            eta_min=args.lr_min)
        logger.info(f"Scheduler: CosineAnnealingWarmRestarts T0={args.cosine_T0} Tmult={args.cosine_Tmult}\n")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.scheduler_factor,
            patience=args.scheduler_patience, min_lr=args.lr_min, verbose=True)

    scaler        = GradScaler()
    early_stopper = EarlyStopping(patience=args.early_stop_patience, mode='max')

    best_dice  = 0.0
    best_epoch = 0

    logger.info("=" * 70)
    logger.info("Starting Training (Phased v2)...")
    logger.info("=" * 70 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ── IDDMGA LR Warmup（逻辑不变）──
        warmup_start = args.iddmga_warmup_start
        warmup_end   = warmup_start + args.iddmga_warmup_epochs
        if epoch < warmup_start:
            for idx in IDDMGA_GROUP_IDXS:
                optimizer.param_groups[idx]['lr'] = 0.0
        elif epoch <= warmup_end:
            apply_iddmga_lr_warmup(
                optimizer, epoch - warmup_start + 1,
                warmup_end_epoch  = args.iddmga_warmup_epochs,
                iddmga_lr_full    = args.iddmga_lr,
                iddmga_group_idxs = IDDMGA_GROUP_IDXS)

        # [v6] Phase2 精度优先模式：切换幅度收窄（v5 beta=0.45 导致 Recall 崩至 0.63）
        if epoch == args.phase2_start:
            loss_fn.tumor_loss.alpha = 0.48   # 原 0.55，收窄避免 Recall 下崩
            loss_fn.tumor_loss.beta  = 0.52   # 原 0.45，保持对 FN 的适当惩罚
            loss_fn.tumor_loss.gamma = 2.0    # 原 2.5
            loss_fn.tumor_loss.ftl.tversky.alpha = 0.48
            loss_fn.tumor_loss.ftl.tversky.beta  = 0.52
            loss_fn.tumor_loss.ftl.gamma         = 2.0
            logger.info(f"[v6] Phase2 Switch @ epoch {epoch}: "
                        f"alpha=0.48, beta=0.52, gamma=2.0 → 温和精度优化（保 Recall≥0.68）")

        # ── [MODIFIED] 每轮重新 shuffle BalancedBatchSampler（更新 rng）──
        # BalancedBatchSampler 的 __iter__ 会自动重新 shuffle，无需额外操作

        train_m = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler,
            epoch, logger,
            accumulation_steps = args.accumulation_steps,  # [MODIFIED] 4
            clip_grad_norm     = args.clip_grad_norm)       # [MODIFIED] 1.0

        val_m = validate(model, val_loader, loss_fn, device, epoch, logger)

        cur_dice = val_m['tumor_dice']
        # [v5-H] Cosine 按 epoch 步进；Plateau 按 dice 步进
        if args.use_cosine:
            scheduler.step(epoch)
        else:
            scheduler.step(cur_dice)

        writer.add_scalar('Train/Loss',    train_m['loss'],          epoch)
        writer.add_scalar('Train/Dice',    train_m['tumor_dice'],    epoch)
        writer.add_scalar('Train/Recall',  train_m['tumor_recall'],  epoch)
        writer.add_scalar('Train/FP',      train_m['empty_fp_rate'], epoch)
        # BoundaryLoss 专项曲线
        writer.add_scalar('BndLoss/Weight', train_m['bnd_weight'],   epoch)
        writer.add_scalar('BndLoss/RawVal', train_m['bnd_loss'],     epoch)
        writer.add_scalar('BndLoss/Ratio',  train_m['bnd_ratio'],    epoch)
        # [v5-J] 验证双阈值 Dice + 置信度差值
        writer.add_scalar('Val/Loss',           val_m['loss'],            epoch)
        writer.add_scalar('Val/Dice_05_raw',   val_m['tumor_dice'],      epoch)
        writer.add_scalar('Val/Dice_05_post',  val_m['tumor_dice_pp'],   epoch)
        writer.add_scalar('Val/Dice_03',       val_m['tumor_dice_03'],   epoch)
        writer.add_scalar('Val/ConfidenceGap', val_m['confidence_gap'],  epoch)
        writer.add_scalar('Val/Precision',     val_m['tumor_precision'], epoch)
        writer.add_scalar('Val/Recall',        val_m['tumor_recall'],    epoch)
        writer.add_scalar('Val/FP',            val_m['empty_fp_rate'],   epoch)
        writer.add_scalar('LR/CT',    optimizer.param_groups[1]['lr'], epoch)
        writer.add_scalar('LR/Main',  optimizer.param_groups[3]['lr'], epoch)
        writer.add_scalar('LR/IDDMGA',optimizer.param_groups[5]['lr'], epoch)

        # [v5-K] 保存条件：prec≥0.35, recall≥0.55（放宽自 0.45/0.65）
        cur_prec   = val_m['tumor_precision']
        cur_recall = val_m['tumor_recall']
        prec_ok   = cur_prec   >= 0.35
        recall_ok = cur_recall >= 0.55
        is_best   = (cur_dice > best_dice) and prec_ok and recall_ok
        if not is_best and cur_dice > best_dice and epoch <= 20:
            is_best = True

        if is_best:
            best_dice  = cur_dice
            best_epoch = epoch
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice_raw':             best_dice,           # 主指标 raw @0.5
                'dice_post':            val_m['tumor_dice_pp'],
                'dice_03':              val_m['tumor_dice_03'],
                'confidence_gap':       val_m['confidence_gap'],
                'precision':            val_m['tumor_precision'],
                'recall':               val_m['tumor_recall'],
                'fp_rate':              val_m['empty_fp_rate'],
                'args':                 vars(args),
            }, os.path.join(save_dir, 'best_model.pth'))
            logger.info(f"  [Best] Dice@0.5(raw)={best_dice:.4f} "
                        f"Dice@0.5(post)={val_m['tumor_dice_pp']:.4f} "
                        f"Dice@0.3={val_m['tumor_dice_03']:.4f} "
                        f"Prec={val_m['tumor_precision']:.4f} "
                        f"Recall={val_m['tumor_recall']:.4f} "
                        f"FP={val_m['empty_fp_rate']:.6f}\n")

        logger.info(f"  Epoch {epoch} | Time={int(time.time()-t0)}s | "
                    f"LR_ct={optimizer.param_groups[1]['lr']:.2e} | "
                    f"LR_main={optimizer.param_groups[3]['lr']:.2e} | "
                    f"LR_iddmga={optimizer.param_groups[5]['lr']:.2e}\n"
                    f"  Best: Dice@0.5={best_dice:.4f} (epoch {best_epoch})\n")

        if early_stopper(cur_dice):
            logger.info(f"Early stopping at epoch {epoch} "
                        f"(patience={args.early_stop_patience})")
            break

        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    logger.info(f"Training done. Best Dice={best_dice:.4f} at epoch {best_epoch}")
    writer.close()


if __name__ == '__main__':
    main()