"""
Intra-Bone Tumor Training — 分阶段稳定收敛版 v7 (5-Slice CSA)
==============================================================

v7 在 v6 基础上完成 5-Slice CSA 迁移：

  [v7-A] Dataset 升级为 5-Slice 版本（PerfectIntraBoneDataset512_5Slice）
         输出 ct=(5,H,W), pet=(5,H,W)，兼容 CSASliceFusion
  [v7-B] 模型引入 CSASliceFusion 替换原 ct_adapter/pet_adapter
         参数：n_slices=5, csa_feat_ch=32, csa_n_heads=4,
               csa_pool_size=16, csa_use_cross_modal=True
  [v7-C] Optimizer 新增 CSA 参数组（与 main 同 LR，5 epoch warmup）
  [v7-D] argparse 新增 CSA 超参（--n_slices, --csa_feat_ch, 等）
  [v7-E] 消除 min_tumor_pixels 过滤，包含所有肿瘤切片（含极小肿瘤）

v6 沿用项（均保留）:
  [v6-A] HardNegativeDataset（难负样本）替换随机空切片
  [v6-B] CosineAnnealingWarmRestarts T0=80 Tmult=2
  [v6-C] Phase2 温和精度优化（alpha=0.48, beta=0.52，保 Recall≥0.68）
  [v6-D] EmptySliceDataset 随机补充不足时使用
  [v6-E] early_stop_patience=50（保证 Phase3 IRGDA 充分训练后再早停）

训练阶段:
  Phase 1  epoch  1~19 : FTL + BCE（建立分割基础）
  Phase 2  epoch 20~34 : + BoundaryLoss warmup（0 → 0.05）+ Precision 优先 Tversky
  Phase 3  epoch 35+   : + IRGDA supervision + FP suppress

Deep Supervision:  epoch 1~40 开启，epoch > 40 关闭
Sampler:           BalancedBatchSampler（N/3 tumor + 2N/3 empty per batch，1:2）
Scheduler:         CosineAnnealingWarmRestarts T0=80 Tmult=2
CSA Warmup:        前 5 epoch CSA LR 从 0 线性升至 main_lr
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

# ─────────────────────────────────────────────────────────────────────────────
# [CHANGE-1] Import — 5-Slice CSA 版（已替换原 tumor_only 版本）
# ─────────────────────────────────────────────────────────────────────────────
from new_train.Intrabone_petct_dataset_5slice import (
    get_intrabone_dataloader_5slice          as get_intrabone_dataloader_512,  # 向后兼容别名
    PerfectIntraBoneDataset512_5Slice        as PerfectIntraBoneDataset512Fixed,
    HardNegativeDataset5Slice                as HardNegativeDataset,
    EnhancedCTNormalizer,
    EnhancedPETNormalizer,
    get_augmentation_5slice                  as get_augmentation,
    parse_patient_slice,
    build_patient_slice_map,
)
from new_network.fbfa_intrabone_enhanced_5slice import (
    FBFAIntraBoneTumorSegmentation5Slice     as FBFAIntraBoneTumorSegmentation,
)

# Loss / Metrics
from bone_only_loss_metrics import (
    SmallTumorLoss,
    SingleStageLoss,
    BoneOnlyDetailedMetrics,
    postprocess_batch,          # [v5-F] 验证后处理 CC 过滤
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
    """冻结 ct_branch BN 统计量；CSA 模块有独立 BN，不受影响。"""
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
# [CHANGE-2] EmptySliceDataset — 5-Slice 版
# ============================================================

class EmptySliceDataset(torch.utils.data.Dataset):
    """
    读取空切片（骨内无肿瘤）— [v7] 5-Slice 版

    输出 ct=(5,H,W), pet=(5,H,W)，与 PerfectIntraBoneDataset512_5Slice 接口一致。
    注：_filter_empty 仅过滤「骨区域面积 < 500」的无效切片，
        保留所有骨内无肿瘤的切片作为负样本，不做肿瘤像素数过滤。
    """

    def __init__(self, image_list, img_root, mode='train', is_16bit=True):
        self.img_root  = img_root
        self.mode      = mode
        self.is_16bit  = is_16bit
        self.ct_norm   = EnhancedCTNormalizer()
        self.pet_norm  = EnhancedPETNormalizer()
        self.transform = get_augmentation(is_train=(mode == 'train'))

        # 构建邻切片索引
        pmap = build_patient_slice_map(image_list)
        self._zmap = {}
        for patient, entries in pmap.items():
            for z, iid in entries:
                self._zmap[(patient, z)] = iid

        self.image_list = self._filter_empty(image_list)
        print(f"  [EmptySlice 5-slice] {len(self.image_list)} empty slices loaded")

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
        return self._get_path(image_id, '_bone_pred.png')

    def _neighbor_id(self, image_id, delta):
        patient, z = parse_patient_slice(image_id)
        if z is None:
            return image_id
        nb = self._zmap.get((patient, z + delta))
        return nb if nb is not None else image_id

    def _filter_empty(self, raw_list):
        """保留骨区域 >= 500 且骨内无肿瘤的切片作为负样本。"""
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

    def _load_ct5(self, image_id):
        slices = []
        for d in (-2, -1, 0, 1, 2):
            nid = self._neighbor_id(image_id, d)
            try:
                p = self._get_path(nid, '_CT.png')
                if self.is_16bit:
                    raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                    s = self.ct_norm.normalize(raw, method='bone')
                else:
                    raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    s = raw.astype(np.float32) / 255.0
            except Exception:
                p0  = self._get_path(image_id, '_CT.png')
                raw = cv2.imread(p0, cv2.IMREAD_UNCHANGED if self.is_16bit
                                 else cv2.IMREAD_GRAYSCALE)
                s = self.ct_norm.normalize(raw, 'bone') if self.is_16bit \
                    else raw.astype(np.float32) / 255.0
            slices.append(s)
        return np.stack(slices, axis=-1)   # (H, W, 5)

    def _load_pet5(self, image_id):
        slices = []
        for d in (-2, -1, 0, 1, 2):
            nid = self._neighbor_id(image_id, d)
            try:
                p   = self._get_path(nid, '_PET.png')
                raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                s   = self.pet_norm.normalize(raw)
            except Exception:
                p0  = self._get_path(image_id, '_PET.png')
                raw = cv2.imread(p0, cv2.IMREAD_UNCHANGED)
                s   = self.pet_norm.normalize(raw)
            slices.append(s)
        return np.stack(slices, axis=-1)   # (H, W, 5)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id   = self.image_list[idx]
        ct5ch      = self._load_ct5(image_id)     # (H, W, 5)
        pet5ch     = self._load_pet5(image_id)    # (H, W, 5)
        bone_path  = self._get_bone_path(image_id)
        bone_raw   = cv2.imread(bone_path, cv2.IMREAD_GRAYSCALE)
        bone_pred  = (bone_raw  > 127).astype(np.float32) \
                     if bone_raw  is not None else np.zeros((512, 512), np.float32)
        tumor_mask = np.zeros_like(bone_pred)

        aug        = self.transform(image=ct5ch, pet5ch=pet5ch,
                                    bone_pred=bone_pred, tumor_mask=tumor_mask)
        ct         = aug['image'].float()       # (5, H, W)
        pet        = aug['pet5ch'].float()      # (5, H, W)
        bone_pred  = aug['bone_pred'].float()
        tumor_mask = aug['tumor_mask'].float()

        def e3(t): return t.unsqueeze(0) if t.dim() == 2 else t
        bone_pred  = e3(bone_pred)
        tumor_mask = e3(tumor_mask)

        ct         = torch.clamp(ct,  0, 1)
        pet        = torch.clamp(pet, 0, 1)
        bone_pred  = (bone_pred  > 0.5).float()
        tumor_mask = (tumor_mask > 0.5).float() * bone_pred

        return {
            'ct':          ct,
            'pet':         pet,
            'bone_pred':   bone_pred,
            'tumor_mask':  tumor_mask,
            'name':        image_id,
            'tumor_ratio': torch.tensor(0.0,  dtype=torch.float32),
            'is_tumor':    torch.tensor(False, dtype=torch.bool),
        }


# ============================================================
#  BalancedBatchSampler
# ============================================================

class BalancedBatchSampler(Sampler):
    """
    每个 batch 保证 tumor_per_batch 个 tumor + empty_per_batch 个 empty，
    彻底消除全空 batch / 全肿瘤 batch 的可能。

    参数:
      dataset         : ConcatDataset([tumor_ds, neg_ds])
      n_tumor         : tumor 数据集的样本数（前 n_tumor 个索引属于 tumor）
      batch_size      : batch 大小
      tumor_fraction  : 每个 batch 中 tumor 占比（默认 1/3 → 1:2 比例）
    """

    def __init__(self, dataset, n_tumor, batch_size, tumor_fraction=0.5, seed=42):
        super().__init__(dataset)
        self.n_total   = len(dataset)
        self.n_tumor   = n_tumor
        self.n_empty   = self.n_total - n_tumor
        self.bs        = batch_size
        self.n_t       = max(1, int(batch_size * tumor_fraction))
        self.n_e       = batch_size - self.n_t
        self.seed      = seed
        self.rng       = random.Random(seed)

        assert self.n_tumor > 0, "No tumor samples found"
        assert self.n_empty > 0, "No empty samples found"

        self.tumor_idxs = list(range(n_tumor))
        self.empty_idxs = list(range(n_tumor, self.n_total))
        self.n_batches  = max(1, self.n_tumor // self.n_t)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        t_pool = self.tumor_idxs.copy()
        e_pool = self.empty_idxs.copy()
        self.rng.shuffle(t_pool)
        self.rng.shuffle(e_pool)

        t_ptr = 0
        e_ptr = 0

        for _ in range(self.n_batches):
            if t_ptr + self.n_t > len(t_pool):
                self.rng.shuffle(t_pool)
                t_ptr = 0
            if e_ptr + self.n_e > len(e_pool):
                self.rng.shuffle(e_pool)
                e_ptr = 0

            batch = (t_pool[t_ptr:t_ptr + self.n_t]
                     + e_pool[e_ptr:e_ptr + self.n_e])
            self.rng.shuffle(batch)
            yield batch

            t_ptr += self.n_t
            e_ptr += self.n_e


# ============================================================
#  IDDMGA / CSA LR Warmup
# ============================================================

def apply_iddmga_lr_warmup(optimizer, epoch, warmup_end_epoch,
                            iddmga_lr_full, iddmga_group_idxs):
    ratio = epoch / warmup_end_epoch
    lr    = iddmga_lr_full * ratio
    for idx in iddmga_group_idxs:
        optimizer.param_groups[idx]['lr'] = lr


def apply_csa_lr_warmup(optimizer, epoch, warmup_epochs,
                         csa_lr_full, csa_group_idxs):
    """[v7-C] CSA 参数组前 warmup_epochs 个 epoch 线性升至 csa_lr_full。"""
    ratio = min(1.0, epoch / max(warmup_epochs, 1))
    lr    = csa_lr_full * ratio
    for idx in csa_group_idxs:
        optimizer.param_groups[idx]['lr'] = lr


# ============================================================
#  Train Epoch
# ============================================================

def train_epoch(model, dataloader, optimizer, loss_fn, device, scaler,
                epoch, logger,
                accumulation_steps=4,
                clip_grad_norm=1.0):
    model.train()
    freeze_bn_stats(model)
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

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            del ct, pet, bone_pred, tumor_mask, outputs, loss
            continue

        if loss_fn.last_bnd_loss > 0.0:
            all_bnd_loss.append(loss_fn.last_bnd_loss)
        if loss_fn.last_total_loss > 0.0:
            all_bnd_total.append(loss_fn.last_total_loss)

        scaler.scale(loss / accumulation_steps).backward()

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
                m = metrics_fn(outputs['tumor_logits'], tumor_mask, bone_pred, is_tumor)
                if m['num_tumor_slices'] > 0:
                    all_tdice.append(m['tumor_dice'])
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
    logger.info(f"  Dice@0.5:  {dice_val:.4f}")
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


# ============================================================
#  Validate
# ============================================================

def validate(model, dataloader, loss_fn, device, epoch, logger):
    model.eval()
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
            del ct, pet, tumor_mask, outputs, loss, logits

    n      = len(dataloader)
    dice   = np.mean(all_dice)   if all_dice   else 0.0
    prec   = np.mean(all_prec)   if all_prec   else 0.0
    recall = np.mean(all_recall) if all_recall else 0.0
    fp     = np.mean(all_fp)     if all_fp     else 0.0

    logger.info(f"\n{'='*70}")
    logger.info(f"Epoch {epoch:03d} [VAL]")
    logger.info(f"  Loss:      {total_loss/n:.4f}")
    logger.info(f"  Dice@0.5:  {dice:.4f}")
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
    parser = argparse.ArgumentParser(description='IntraBone Phased Training v7 (5-Slice CSA)')

    # 路径
    parser.add_argument('--data_root',         type=str,      default='data/img1')
    parser.add_argument('--stage1_model_path', type=str,
                        default='checkpoints/stage1_2D_20260303-152807/best_model.pth')
    parser.add_argument('--save_dir',          type=str,      default='checkpoints')
    parser.add_argument('--exp_name',          type=str,      default='intrabone_phased_v7')
    parser.add_argument('--resume_from',       type=str,      default=None)

    # 数据
    parser.add_argument('--batch_size',        type=int,      default=4)
    parser.add_argument('--num_workers',       type=int,      default=4)
    parser.add_argument('--is_16bit',          type=str2bool, default=True)
    parser.add_argument('--train_empty_ratio', type=float,    default=2.0,
                        help='训练集 neg:tumor 比例，2.0 = 1:2')
    parser.add_argument('--val_empty_ratio',   type=float,    default=5.0)
    # [v7-E] 消除 min_tumor_pixels 过滤：默认 0 = 不过滤，包含所有肿瘤切片
    parser.add_argument('--min_tumor_pixels',  type=int,      default=0,
                        help='肿瘤切片最小像素阈值，0 表示不过滤（v7: 默认关闭过滤）')

    # 模型
    parser.add_argument('--freeze_stage1',     type=str2bool, default=True)
    parser.add_argument('--bone_dilation',     type=int,      default=5)
    parser.add_argument('--iddmga_K',          type=int,      default=3)

    # ─────────────────────────────────────────────────────────
    # [CHANGE-5] CSA 超参（v7 新增）
    # ─────────────────────────────────────────────────────────
    parser.add_argument('--n_slices',            type=int,      default=5,
                        help='输入切片数（目前固定 5）')
    parser.add_argument('--csa_feat_ch',         type=int,      default=32,
                        help='CSASliceFusion 中间特征维度')
    parser.add_argument('--csa_n_heads',         type=int,      default=4,
                        help='CSA 多头注意力头数')
    parser.add_argument('--csa_pool_size',       type=int,      default=16,
                        help='CSA 空间池化目标尺寸（控制内存/精度 trade-off）')
    parser.add_argument('--csa_use_cross_modal', type=str2bool, default=True,
                        help='PET→CT 跨模态辅助注意力')
    parser.add_argument('--enable_encoder_csa',  type=str2bool, default=False,
                        help='Encoder 中间层 CSA（开启后 5× backbone 计算量）')
    parser.add_argument('--csa_warmup_epochs',   type=int,      default=5,
                        help='CSA 参数组 LR 从 0 线性升至 main_lr 所需 epoch 数')

    # 学习率
    parser.add_argument('--ct_lr',             type=float,    default=3e-6)
    parser.add_argument('--main_lr',           type=float,    default=2e-4)
    parser.add_argument('--iddmga_lr',         type=float,    default=2e-4)
    parser.add_argument('--lr_min',            type=float,    default=1e-7)
    parser.add_argument('--weight_decay',      type=float,    default=1e-4)
    parser.add_argument('--main_weight_decay', type=float,    default=3e-4)

    # IDDMGA warmup
    parser.add_argument('--iddmga_warmup_epochs', type=int,   default=10)
    parser.add_argument('--iddmga_warmup_start',  type=int,   default=5)

    # 阶段边界
    parser.add_argument('--phase2_start',      type=int,      default=20)
    parser.add_argument('--phase3_start',      type=int,      default=35)
    parser.add_argument('--ds_cutoff',         type=int,      default=40,
                        help='Deep Supervision 关闭的 epoch')
    parser.add_argument('--irgda_rampup',      type=int,      default=15)

    # BoundaryLoss
    parser.add_argument('--boundary_delay_start', type=int,   default=15)
    parser.add_argument('--bnd_weight_max',        type=float, default=0.05)
    parser.add_argument('--bnd_rampup_rate',       type=float, default=0.003)
    parser.add_argument('--use_boundary_delay',    type=str2bool, default=True)

    # Cosine Scheduler
    parser.add_argument('--use_cosine',        type=str2bool, default=True)
    parser.add_argument('--cosine_T0',         type=int,      default=80)
    parser.add_argument('--cosine_Tmult',      type=int,      default=2)

    # 训练
    parser.add_argument('--epochs',              type=int,    default=150)
    parser.add_argument('--accumulation_steps',  type=int,    default=4)
    parser.add_argument('--clip_grad_norm',      type=float,  default=1.0)
    parser.add_argument('--seed',                type=int,    default=42)
    parser.add_argument('--early_stop_patience', type=int,    default=50)

    # Hard Negative
    parser.add_argument('--hard_neg_range',          type=int,   default=5)
    parser.add_argument('--hard_neg_max_per_tumor',  type=int,   default=3)

    # Loss 权重
    parser.add_argument('--irgda_sup_weight',   type=float,  default=0.001)
    parser.add_argument('--fp_suppress_weight', type=float,  default=0.03)

    # Scheduler fallback
    parser.add_argument('--scheduler_patience', type=int,    default=15)
    parser.add_argument('--scheduler_factor',   type=float,  default=0.5)

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
Intra-Bone Tumor Training — v7 (5-Slice CSA)
{'='*70}
[阶段划分]
  Phase 1  epoch  1~{args.phase2_start-1}: FTL + BCE
  Phase 2  epoch {args.phase2_start}~{args.phase3_start-1}: + BoundaryLoss warmup
  Phase 3  epoch {args.phase3_start}+   : + IRGDA + FP suppress

[BoundaryLoss]
  最大权重:     {args.bnd_weight_max}
  Warmup 速率:  +{args.bnd_rampup_rate}/epoch
  延迟开关:     {args.use_boundary_delay}
  延迟起始:     epoch {args.boundary_delay_start}

[Deep Supervision]  epoch 1~{args.ds_cutoff}（之后关闭）
[IRGDA]     epoch >= {args.phase3_start} 启用，线性升温 {args.irgda_rampup} epoch
[Sampler]   BalancedBatchSampler (1:{args.train_empty_ratio:.1f} tumor:neg per batch)

[CSA v7]
  n_slices={args.n_slices}  feat_ch={args.csa_feat_ch}  n_heads={args.csa_n_heads}
  pool_size={args.csa_pool_size}  cross_modal={args.csa_use_cross_modal}
  encoder_csa={args.enable_encoder_csa}  csa_warmup={args.csa_warmup_epochs} epochs

[v7-E] 肿瘤过滤已关闭 (min_tumor_pixels={args.min_tumor_pixels})

[LR]  main={args.main_lr:.1e}  ct={args.ct_lr:.1e}  iddmga={args.iddmga_lr:.1e}
[Grad] accum={args.accumulation_steps}  clip={args.clip_grad_norm}
{'='*70}
""")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}\n")

    # ── 数据 ──
    train_file = os.path.join(args.data_root, 'train_tumor.txt')
    val_file   = os.path.join(args.data_root, 'val_tumor.txt')

    # [v7-E] min_tumor_pixels=0：不过滤小肿瘤切片，保留所有包含肿瘤的切片
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

    # ── Hard Negative Dataset ──
    train_hard_neg = HardNegativeDataset(
        tumor_image_ids = list(train_tumor_ds.image_list),
        all_image_ids   = all_train_ids,
        img_root        = args.data_root,
        neg_range       = args.hard_neg_range,
        mode            = 'train',
        is_16bit        = args.is_16bit,
        max_per_tumor   = args.hard_neg_max_per_tumor,
    )
    n_train_tumor  = len(train_tumor_ds)
    n_hard_neg     = len(train_hard_neg)
    n_target_neg   = int(n_train_tumor * args.train_empty_ratio)

    if n_hard_neg < n_target_neg:
        n_supplement = min(n_target_neg - n_hard_neg, len(train_empty_ds))
        _rng = random.Random(args.seed)
        supplement_ds = Subset(train_empty_ds,
                               _rng.sample(range(len(train_empty_ds)), n_supplement))
        neg_combined = ConcatDataset([train_hard_neg, supplement_ds])
        logger.info(f"  HardNeg: {n_hard_neg} hard + {n_supplement} random supplement\n")
    else:
        _rng = random.Random(args.seed)
        hard_sub = Subset(train_hard_neg,
                          _rng.sample(range(n_hard_neg), min(n_target_neg, n_hard_neg)))
        neg_combined = hard_sub
        logger.info(f"  HardNeg: {min(n_target_neg, n_hard_neg)} hard negative slices\n")

    n_train_neg    = len(neg_combined)
    train_combined = ConcatDataset([train_tumor_ds, neg_combined])

    tumor_frac    = 1.0 / (1.0 + args.train_empty_ratio)
    train_sampler = BalancedBatchSampler(
        train_combined,
        n_tumor        = n_train_tumor,
        batch_size     = args.batch_size,
        tumor_fraction = tumor_frac,
        seed           = args.seed)

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

    # ── 验证集 ──
    val_hard_neg = HardNegativeDataset(
        tumor_image_ids = list(val_tumor_ds.image_list),
        all_image_ids   = all_val_ids,
        img_root        = args.data_root,
        neg_range       = args.hard_neg_range,
        mode            = 'val',
        is_16bit        = args.is_16bit,
        max_per_tumor   = args.hard_neg_max_per_tumor,
    )
    n_val_tumor  = len(val_tumor_ds)
    n_val_hard   = len(val_hard_neg)
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

    # ─────────────────────────────────────────────────────────
    # [CHANGE-3] 模型 — 5-Slice CSA 版
    # ─────────────────────────────────────────────────────────
    model = FBFAIntraBoneTumorSegmentation(
        stage1_model_path       = args.stage1_model_path,
        freeze_stage1           = args.freeze_stage1,
        bone_dilation           = args.bone_dilation,
        enable_deep_supervision = True,
        # CSA 参数
        n_slices                = args.n_slices,
        csa_feat_ch             = args.csa_feat_ch,
        csa_n_heads             = args.csa_n_heads,
        csa_pool_size           = args.csa_pool_size,
        csa_use_cross_modal     = args.csa_use_cross_modal,
        enable_encoder_csa      = args.enable_encoder_csa,
        # DGMA
        dgma_K_max              = args.iddmga_K,
        dgma_nms_threshold      = 0.3,
        dgma_min_spatial_size   = 65,
    ).to(device)

    # ── Resume ──
    start_epoch       = 1
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

    # ─────────────────────────────────────────────────────────
    # [CHANGE-4] Optimizer — 8组（新增 csa_no_wd / csa_wd）
    # ─────────────────────────────────────────────────────────
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
    # [v7-C] CSA 参数组（csa_fusion 模块，不属于 ct_branch / soe）
    csa_named    = [(n, p) for n, p in model.named_parameters() if 'csa_fusion' in n]
    main_named   = [(n, p) for n, p in model.named_parameters()
                    if 'ct_branch' not in n
                    and 'soe'       not in n
                    and 'csa_fusion' not in n]

    ct_no_wd,     ct_wd     = _split_wd(ct_named)
    main_no_wd,   main_wd   = _split_wd(main_named)
    iddmga_no_wd, iddmga_wd = _split_wd(iddmga_named)
    csa_no_wd,    csa_wd    = _split_wd(csa_named)

    IDDMGA_GROUP_IDXS = [4, 5]
    CSA_GROUP_IDXS    = [6, 7]   # [v7-C]

    optimizer = torch.optim.AdamW([
        # idx 0
        {'params': ct_no_wd,     'lr': args.ct_lr,     'weight_decay': 0.0,                   'name': 'ct_no_wd'},
        # idx 1
        {'params': ct_wd,        'lr': args.ct_lr,     'weight_decay': args.weight_decay,      'name': 'ct_wd'},
        # idx 2
        {'params': main_no_wd,   'lr': args.main_lr,   'weight_decay': 0.0,                   'name': 'main_no_wd'},
        # idx 3
        {'params': main_wd,      'lr': args.main_lr,   'weight_decay': args.main_weight_decay, 'name': 'main_wd'},
        # idx 4
        {'params': iddmga_no_wd, 'lr': args.iddmga_lr, 'weight_decay': 0.0,                   'name': 'iddmga_no_wd'},
        # idx 5
        {'params': iddmga_wd,    'lr': args.iddmga_lr, 'weight_decay': args.weight_decay,      'name': 'iddmga_wd'},
        # idx 6  [v7-C] CSA warmup：初始 lr=0，由 apply_csa_lr_warmup 线性升至 main_lr
        {'params': csa_no_wd,    'lr': 0.0,            'weight_decay': 0.0,                   'name': 'csa_no_wd'},
        # idx 7
        {'params': csa_wd,       'lr': 0.0,            'weight_decay': args.main_weight_decay, 'name': 'csa_wd'},
    ], betas=(0.9, 0.999), eps=1e-8)

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
                'csa_no_wd':    args.main_lr,
                'csa_wd':       args.main_lr,
            }
            for g in optimizer.param_groups:
                name = g.get('name', '')
                if name in lr_map:
                    g['lr'] = lr_map[name]
            logger.info("  Optimizer restored; LR overridden")
        except Exception as e:
            logger.warning(f"  Optimizer load failed ({e}), using fresh optimizer")

    # ── Loss ──
    base_loss_fn = SmallTumorLoss(
        ftl_weight           = 2.0,
        bce_weight           = 0.5,
        bnd_weight_max       = args.bnd_weight_max,
        bnd_rampup_rate      = args.bnd_rampup_rate,
        alpha                = 0.4,
        beta                 = 0.6,
        gamma                = 2.0,
        boundary_weight      = 5.0,
        phase2_start         = args.phase2_start,
        use_boundary_delay   = args.use_boundary_delay,
        boundary_delay_start = args.boundary_delay_start,
        ds_cutoff            = args.ds_cutoff,
        fp_weight_base       = 0.02)

    loss_fn = SingleStageLoss(
        base_loss_fn,
        irgda_sup_weight    = args.irgda_sup_weight,
        fp_suppress_weight  = args.fp_suppress_weight,
        irgda_start_epoch   = args.phase3_start,
        fp_start_epoch      = args.phase3_start,
        irgda_rampup_epochs = args.irgda_rampup)

    # ── Scheduler ──
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.cosine_T0, T_mult=args.cosine_Tmult,
            eta_min=args.lr_min)
        logger.info(f"Scheduler: CosineAnnealingWarmRestarts "
                    f"T0={args.cosine_T0} Tmult={args.cosine_Tmult}\n")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=args.scheduler_factor,
            patience=args.scheduler_patience, min_lr=args.lr_min, verbose=True)

    scaler        = GradScaler()
    early_stopper = EarlyStopping(patience=args.early_stop_patience, mode='max')

    best_dice  = 0.0
    best_epoch = 0

    logger.info("=" * 70)
    logger.info("Starting Training (v7 5-Slice CSA)...")
    logger.info("=" * 70 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # ── IDDMGA LR Warmup ──
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

        # ── [v7-C] CSA LR Warmup（前 csa_warmup_epochs 个 epoch）──
        if epoch <= args.csa_warmup_epochs:
            apply_csa_lr_warmup(
                optimizer, epoch,
                warmup_epochs  = args.csa_warmup_epochs,
                csa_lr_full    = args.main_lr,
                csa_group_idxs = CSA_GROUP_IDXS)

        # ── [v6] Phase2 温和精度优先模式 ──
        if epoch == args.phase2_start:
            loss_fn.tumor_loss.alpha = 0.48
            loss_fn.tumor_loss.beta  = 0.52
            loss_fn.tumor_loss.gamma = 2.0
            loss_fn.tumor_loss.ftl.tversky.alpha = 0.48
            loss_fn.tumor_loss.ftl.tversky.beta  = 0.52
            loss_fn.tumor_loss.ftl.gamma         = 2.0
            logger.info(f"[v6] Phase2 Switch @ epoch {epoch}: "
                        f"alpha=0.48, beta=0.52, gamma=2.0")

        train_m = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler,
            epoch, logger,
            accumulation_steps = args.accumulation_steps,
            clip_grad_norm     = args.clip_grad_norm)

        val_m = validate(model, val_loader, loss_fn, device, epoch, logger)

        cur_dice = val_m['tumor_dice']
        if args.use_cosine:
            scheduler.step(epoch)
        else:
            scheduler.step(cur_dice)

        # ── TensorBoard ──
        writer.add_scalar('Train/Loss',       train_m['loss'],          epoch)
        writer.add_scalar('Train/Dice',       train_m['tumor_dice'],    epoch)
        writer.add_scalar('Train/Recall',     train_m['tumor_recall'],  epoch)
        writer.add_scalar('Train/FP',         train_m['empty_fp_rate'], epoch)
        writer.add_scalar('BndLoss/Weight',   train_m['bnd_weight'],    epoch)
        writer.add_scalar('BndLoss/RawVal',   train_m['bnd_loss'],      epoch)
        writer.add_scalar('BndLoss/Ratio',    train_m['bnd_ratio'],     epoch)
        writer.add_scalar('Val/Loss',         val_m['loss'],            epoch)
        writer.add_scalar('Val/Dice',         val_m['tumor_dice'],      epoch)
        writer.add_scalar('Val/Precision',    val_m['tumor_precision'], epoch)
        writer.add_scalar('Val/Recall',       val_m['tumor_recall'],    epoch)
        writer.add_scalar('Val/FP',           val_m['empty_fp_rate'],   epoch)
        writer.add_scalar('LR/CT',     optimizer.param_groups[1]['lr'], epoch)
        writer.add_scalar('LR/Main',   optimizer.param_groups[3]['lr'], epoch)
        writer.add_scalar('LR/IDDMGA', optimizer.param_groups[5]['lr'], epoch)
        writer.add_scalar('LR/CSA',    optimizer.param_groups[7]['lr'], epoch)  # [v7-C]

        # ── 保存 ──
        cur_prec   = val_m['tumor_precision']
        cur_recall = val_m['tumor_recall']
        prec_ok    = cur_prec   >= 0.35
        recall_ok  = cur_recall >= 0.55
        is_best    = (cur_dice > best_dice) and prec_ok and recall_ok
        if not is_best and cur_dice > best_dice and epoch <= 20:
            is_best = True

        if is_best:
            best_dice  = cur_dice
            best_epoch = epoch
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice':                 best_dice,
                'precision':            val_m['tumor_precision'],
                'recall':               val_m['tumor_recall'],
                'fp_rate':              val_m['empty_fp_rate'],
                'args':                 vars(args),
            }, os.path.join(save_dir, 'best_model.pth'))
            logger.info(f"  [Best] Dice@0.5={best_dice:.4f} "
                        f"Prec={val_m['tumor_precision']:.4f} "
                        f"Recall={val_m['tumor_recall']:.4f} "
                        f"FP={val_m['empty_fp_rate']:.6f}")

        logger.info(
            f"  Epoch {epoch} | Time={int(time.time()-t0)}s | "
            f"LR_ct={optimizer.param_groups[1]['lr']:.2e} | "
            f"LR_main={optimizer.param_groups[3]['lr']:.2e} | "
            f"LR_iddmga={optimizer.param_groups[5]['lr']:.2e} | "
            f"LR_csa={optimizer.param_groups[7]['lr']:.2e}\n"   # [v7-C]
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