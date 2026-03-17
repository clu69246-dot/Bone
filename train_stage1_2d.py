"""
Training script — ConDSeg Stage1 (2D Bone Segmentation)
适配深度监督版模型 + 内存高效扫描模块

主要变化:
  [1] Loss: 使用模型内置 Stage1Loss (含深度监督 aux 头)
  [2] train(): 适配模型返回 dict (训练) / tensor (推理) 两种模式
  [3] evaluate(): 切换 model.eval() 自动关闭深度监督
  [4] 梯度裁剪 + NaN 防护保留
  [5] 余弦退火 + Warmup 保留
"""

import os
import sys
import random
import time
import datetime
import warnings
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from network.model_efficientvim_2d_stage1 import (
    ConDSeg2DStage1_EfficientViM,
    Stage1Loss,
)


# ============================================================
#  日志
# ============================================================

def setup_logger(log_path):
    logger = logging.getLogger('ConDSeg')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def my_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# ============================================================
#  数据集
# ============================================================

class BoneSegmentation2DDataset(Dataset):
    """
    CT 骨骼分割数据集 (16-bit PNG)
    CT文件:   {id}_CT.png     (16-bit, 存储 HU+1024)
    Mask文件: {id}_bone_pred.png (8-bit, 0/255 二值)
    """

    def __init__(self, image_list, img_root, image_size=512,
                 mode='train', use_augmentation=True,
                 window_min=-200, window_max=1000):
        super().__init__()
        self.image_list  = image_list
        self.img_root    = img_root
        self.image_size  = image_size
        self.mode        = mode
        self.ct_suffix   = "_CT.png"
        self.mask_suffix = "_bone_pred.png"
        self.window_min  = window_min
        self.window_max  = window_max

        if use_augmentation and mode == 'train':
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5,
                         border_mode=cv2.BORDER_CONSTANT, value=0),
                A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
                A.RandomResizedCrop(height=image_size, width=image_size,
                                    scale=(0.85, 1.0), p=0.4),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=image_size, width=image_size),
                ToTensorV2()
            ])

    def _read_data(self, image_id):
        ct   = cv2.imread(
            os.path.join(self.img_root, image_id + self.ct_suffix),
            cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(
            os.path.join(self.img_root, image_id + self.mask_suffix),
            cv2.IMREAD_GRAYSCALE)
        assert ct   is not None, f"Failed to read CT:   {image_id}"
        assert mask is not None, f"Failed to read mask: {image_id}"
        return ct, mask

    def __getitem__(self, index):
        image_id   = self.image_list[index]
        ct, mask   = self._read_data(image_id)

        # HU 恢复 + 窗宽窗位 + 归一化
        ct = ct.astype(np.float32) - 1024.0
        ct = np.clip(ct, self.window_min, self.window_max)
        ct = (ct - self.window_min) / (self.window_max - self.window_min)
        ct_uint8 = (ct * 255).astype(np.uint8)

        mask_binary = (mask > 127).astype(np.uint8)

        aug = self.transform(image=ct_uint8, mask=mask_binary)
        ct_t   = aug['image'].float() / 255.0
        mask_t = aug['mask'].float()

        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)

        return ct_t, mask_t

    def __len__(self):
        return len(self.image_list)


# ============================================================
#  指标计算
# ============================================================

def calculate_metrics(logits, target, threshold=0.5):
    """logits → 二值化 → IoU, F1, Recall, Precision, Dice"""
    # logits 可以是 tensor 也可以是 dict['main'] (训练时)
    if isinstance(logits, dict):
        logits = logits['main']

    pred   = torch.sigmoid(logits)
    pred_b = (pred > threshold).float()
    tgt_b  = (target > threshold).float()

    tp = (pred_b * tgt_b).sum().item()
    fp = (pred_b * (1 - tgt_b)).sum().item()
    fn = ((1 - pred_b) * tgt_b).sum().item()

    iou       = (tp + 1e-5) / (tp + fp + fn + 1e-5)
    precision = (tp + 1e-5) / (tp + fp + 1e-5)
    recall    = (tp + 1e-5) / (tp + fn + 1e-5)
    f1        = 2 * precision * recall / (precision + recall + 1e-5)
    dice      = (2 * tp + 1e-5) / (2 * tp + fp + fn + 1e-5)
    return iou, f1, recall, precision, dice


# ============================================================
#  早停
# ============================================================

class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.001):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_score = None

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ============================================================
#  训练循环 (适配深度监督)
# ============================================================

def train(model, train_loader, optimizer, loss_fn, device, scaler,
          epoch, accumulation_steps, logger, current_lr):
    model.train()
    total_loss = 0.0
    metrics    = [0.0] * 5
    nan_count  = 0
    valid_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks  = masks.to(device)

        with autocast():
            # 训练时模型返回 dict {'main', 'aux2', 'aux1', 'aux0'}
            outputs = model(images)

            # NaN 检查 (检查主输出)
            main_logits = outputs['main'] if isinstance(outputs, dict) else outputs
            if torch.isnan(main_logits).any() or torch.isinf(main_logits).any():
                logger.warning(f"Invalid prediction at batch {batch_idx}, skipping")
                optimizer.zero_grad()
                continue

            loss = loss_fn(outputs, masks) / accumulation_steps

        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()

        step = (batch_idx + 1) % accumulation_steps == 0
        last = (batch_idx + 1) == len(train_loader)
        if step or last:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)

            if torch.isnan(grad_norm) or grad_norm > 100:
                logger.warning(f"Abnormal gradient norm={grad_norm:.2f}, skip step")
                scaler.update()
                optimizer.zero_grad()
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        batch_m = calculate_metrics(outputs, masks)
        total_loss += loss.item() * accumulation_steps
        for i in range(5):
            metrics[i] += batch_m[i]
        valid_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item()*accumulation_steps:.4f}',
            'IoU':  f'{batch_m[0]:.4f}',
            'Dice': f'{batch_m[4]:.4f}',
        })

    if nan_count > 0:
        logger.warning(f"NaN batches: {nan_count}/{len(train_loader)}")

    n = max(valid_batches, 1)
    return total_loss / n, [m / n for m in metrics]


# ============================================================
#  验证循环 (eval 模式自动关闭深度监督)
# ============================================================

def evaluate(model, val_loader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0.0
    metrics    = [0.0] * 5

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)

            # eval 模式: 模型直接返回 tensor
            logits = model(images)
            loss   = loss_fn(logits, masks)
            bm     = calculate_metrics(logits, masks)

            total_loss += loss.item()
            for i in range(5):
                metrics[i] += bm[i]

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'IoU':  f'{bm[0]:.4f}',
                'Dice': f'{bm[4]:.4f}',
            })

    n = len(val_loader)
    return total_loss / n, [m / n for m in metrics]


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # ── 路径配置 ──
    dataset_name = 'img_stage1'
    data_root    = f'data/{dataset_name}'

    seed = 42
    my_seeding(seed)

    # ── 超参数 ──
    image_size            = 512
    batch_size            = 8
    accumulation_steps    = 2       # 等效 batch_size = 8
    num_epochs            = 150
    base_lr               = 2e-4
    weight_decay          = 1e-4
    warmup_epochs         = 5
    early_stopping_patience = 20
    bce_weight            = 0.3
    dice_weight           = 0.7

    # ── 保存路径 ──
    current_time       = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name        = f"stage1_2D_{current_time}"
    save_dir           = os.path.join("checkpoints", folder_name)
    os.makedirs(save_dir, exist_ok=True)
    train_log_path     = os.path.join(save_dir, "train_log.txt")
    checkpoint_path    = os.path.join(save_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(save_dir, "last_model.pth")

    logger = setup_logger(train_log_path)

    log_msg = f"""
Training Configuration:
{'=' * 70}
Dataset:    {dataset_name}
Task:       Bone segmentation (binary, deep supervision)
Model:      ConDSeg2DStage1_EfficientViM (EfficientScan2D, MSCA Decoder)

Input:      {image_size}x{image_size}
Batch Size: {batch_size}  (accumulation x{accumulation_steps} → eff. {batch_size*accumulation_steps})
Epochs:     {num_epochs}
LR:         {base_lr}  (Warmup {warmup_epochs} ep + CosineAnnealing)
WD:         {weight_decay}
Loss:       Dice({dice_weight}) + BCE({bce_weight}) + DeepSupervision(0.4/0.2/0.1)
Early Stop: {early_stopping_patience} epochs
{'=' * 70}
"""
    logger.info(log_msg)

    # ── 数据集 ──
    logger.info("\nLoading Dataset...")
    with open(os.path.join(data_root, 'train.txt')) as f:
        train_list = [l.strip() for l in f]
    with open(os.path.join(data_root, 'val.txt')) as f:
        val_list = [l.strip() for l in f]

    train_dataset = BoneSegmentation2DDataset(
        train_list, data_root, image_size, 'train', True)
    val_dataset   = BoneSegmentation2DDataset(
        val_list, data_root, image_size, 'val', False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4,
                              pin_memory=True, drop_last=False)

    logger.info(f"  Train: {len(train_dataset)} samples  ({len(train_loader)} batches)")
    logger.info(f"  Val:   {len(val_dataset)}   samples  ({len(val_loader)}  batches)")

    # ── 模型 ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"\n[OK] Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"[OK] VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    model = ConDSeg2DStage1_EfficientViM(
        in_channels=1, out_channels=1, deep_supervision=True
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[OK] Model params: {n_params/1e6:.2f} M")

    # ── 优化器 + 调度器 ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr,
        weight_decay=weight_decay, betas=(0.9, 0.999))

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs,
        eta_min=base_lr * 0.01)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_sched, cosine_sched],
        milestones=[warmup_epochs])

    # ── Loss (含深度监督) ──
    loss_fn = Stage1Loss(bce_weight=bce_weight, dice_weight=dice_weight,
                         aux_weights=(0.4, 0.2, 0.1))

    scaler        = GradScaler()
    early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=0.001)

    # ── 训练循环 ──
    logger.info("\n" + "=" * 70)
    logger.info("Start Training ...")
    logger.info("=" * 70)

    best_dice = 0.0
    best_iou  = 0.0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_m = train(
            model, train_loader, optimizer, loss_fn,
            device, scaler, epoch, accumulation_steps, logger, lr)

        val_loss,   val_m   = evaluate(
            model, val_loader, loss_fn, device, epoch)

        scheduler.step()

        cur_dice = val_m[4]
        cur_iou  = val_m[0]
        improved = cur_dice > best_dice

        if improved:
            logger.info(f"\n[Best] Dice: {best_dice:.4f} -> {cur_dice:.4f} "
                        f"(+{cur_dice-best_dice:.4f})")
            best_dice = cur_dice
            best_iou  = cur_iou
            torch.save({
                'epoch':              epoch,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou':           best_iou,
                'best_dice':          best_dice,
            }, checkpoint_path)

        if epoch % 10 == 0:
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()},
                       last_checkpoint_path)

        t = int(time.time() - t0)
        logger.info(
            f"\nEpoch {epoch:03d}/{num_epochs} | Time: {t}s | LR: {lr:.6f}\n"
            f"  Train: Loss={train_loss:.4f} | IoU={train_m[0]:.4f} | Dice={train_m[4]:.4f}\n"
            f"  Val:   Loss={val_loss:.4f} | IoU={val_m[0]:.4f} | Dice={val_m[4]:.4f}\n"
            f"  Best:  IoU={best_iou:.4f} | Dice={best_dice:.4f}\n"
            f"  EarlyStop: {early_stopper.counter}/{early_stopping_patience}"
        )

        if early_stopper(cur_dice):
            logger.info(f"\n[OK] Early stopping at epoch {epoch}")
            break

    logger.info(f"""
{'=' * 70}
Training Completed!
  Best Val IoU:  {best_iou:.4f}
  Best Val Dice: {best_dice:.4f}
  Best model:    {checkpoint_path}
{'=' * 70}
""")