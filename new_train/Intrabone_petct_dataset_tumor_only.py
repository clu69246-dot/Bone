"""
Intra-Bone Tumor Dataset — bone_pred 版本

核心改动:
  [1] bone_pred 来源: _bone_pred.png → _bone_pred.png
      原因: Stage1 已生成预测骨骼 mask，Stage2 应使用预测结果
      (与实际推理时完全一致，避免训练/推理不匹配)

  [2] 保留所有原有过滤逻辑 (min_tumor_pixels, tumor_ratio)

  [3] 数据增强保持 FIXED 版本 (无 ElasticTransform / GridDistortion)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


# ============================================================
#  归一化器
# ============================================================

class EnhancedCTNormalizer:
    def __init__(self, auto_detect_offset=True):
        self.auto_detect_offset = auto_detect_offset
        self.bone_window = {'center': 500.0, 'width': 2500.0}

    def detect_hu_offset(self, ct_raw):
        if ct_raw.dtype == np.uint16:
            return ct_raw.astype(np.float32) - 1024.0
        return ct_raw.astype(np.float32)

    def normalize_with_window(self, ct_hu, center, width):
        wmin = center - width / 2
        wmax = center + width / 2
        ct_c = np.clip(ct_hu, wmin, wmax)
        if wmax > wmin:
            return ((ct_c - wmin) / (wmax - wmin)).astype(np.float32)
        return np.zeros_like(ct_hu, dtype=np.float32)

    def normalize(self, ct_raw, method='bone'):
        ct_hu = self.detect_hu_offset(ct_raw) if self.auto_detect_offset \
                else ct_raw.astype(np.float32) - 1024.0
        return self.normalize_with_window(
            ct_hu, self.bone_window['center'], self.bone_window['width'])


class EnhancedPETNormalizer:
    def __init__(self, percentile_low=5.0, percentile_high=99.5, gamma=1.3):
        self.pl    = percentile_low
        self.ph    = percentile_high
        self.gamma = gamma

    def normalize(self, pet_raw):
        pet = pet_raw.astype(np.float32)
        p_lo = np.percentile(pet, self.pl)
        p_hi = np.percentile(pet, self.ph)
        pet  = np.clip(pet, p_lo, p_hi)
        mn, mx = pet.min(), pet.max()
        if mx > mn:
            pet = (pet - mn) / (mx - mn)
        else:
            pet = np.zeros_like(pet, dtype=np.float32)
        return np.clip(np.power(pet, 1.0 / self.gamma), 0, 1).astype(np.float32)


# ============================================================
#  数据增强
# ============================================================

def get_augmentation(is_train=True):
    """增强版增强策略 — 针对过拟合和小肿瘤泛化优化

    [FIX-OVERFIT] 原版增强太弱 (HFlip + VFlip + Rotate±15 + BrightContrast)，
    75 epoch 后 train-val gap 扩大到 0.17。新增三类增强：
      1. CoarseDropout: 随机遮挡骨区域块，防止模型记忆全局骨骼纹理
      2. ElasticTransform: 弹性形变，增加小肿瘤形状多样性
      3. ShiftScaleRotate: 扩大旋转和平移范围，增强空间泛化

    所有增强同时作用于 CT / PET / mask，保证配准一致性（additional_targets）。
    保守设置 p 值，避免过强增强破坏小肿瘤(<50px)结构。
    """
    if is_train:
        return A.Compose([
            # ── 几何变换（同时作用于图像+mask）──────────────────
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            # 扩大旋转/平移/缩放范围，增强空间泛化
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=25,
                border_mode=cv2.BORDER_CONSTANT, value=0,
                p=0.6),
            # 弹性形变：小幅度，避免破坏小肿瘤
            A.ElasticTransform(
                alpha=30, sigma=5,
                border_mode=cv2.BORDER_CONSTANT, value=0,
                p=0.3),

            # ── 强度变换（仅作用于图像，mask不受影响）──────────
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.4),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),

            # ── 遮挡增强（防止全局纹理记忆）────────────────────
            # [FIX-OVERFIT] 核心改动：随机遮挡骨区域内若干块
            # max_holes=8: 同时最多遮挡8块
            # max_height/width=48: 每块最大48px（约1/10图像宽）
            # min_holes=2: 至少遮挡2块，确保增强生效
            # fill_value=0: 用0填充（与骨外区域一致，不引入虚假信号）
            A.CoarseDropout(
                max_holes=8, max_height=48, max_width=48,
                min_holes=2, min_height=16, min_width=16,
                fill_value=0, p=0.4),

            ToTensorV2()
        ], additional_targets={'pet': 'image', 'bone_pred': 'mask',
                               'tumor_mask': 'mask'})
    else:
        return A.Compose([
            ToTensorV2()
        ], additional_targets={'pet': 'image', 'bone_pred': 'mask',
                               'tumor_mask': 'mask'})


# ============================================================
#  数据集
# ============================================================

class PerfectIntraBoneDataset512Fixed(Dataset):
    """
    骨内肿瘤数据集 — bone_pred 版本

    文件读取顺序:
      CT:          {image_id}_CT.png
      PET:         {image_id}_PET.png
      Tumor mask:  {image_id}_mask.png
      Bone mask:   {image_id}_bone_pred.png   ← [改动] 用 Stage1 预测结果
                   (若不存在则 fallback 到 _bone_pred.png)
    """

    def __init__(self,
                 image_list,
                 img_root,
                 mode='train',
                 is_16bit=True,
                 ct_normalizer=None,
                 pet_normalizer=None,
                 min_tumor_pixels=10,
                 min_tumor_ratio=0.0001,
                 empty_keep_ratio=0.0):

        self.img_root        = img_root
        self.mode            = mode
        self.is_16bit        = is_16bit
        self.min_tumor_ratio = min_tumor_ratio
        self.min_tumor_pixels = min_tumor_pixels
        self.empty_keep_ratio = 0.0   # 强制为0：只保留肿瘤切片

        self.ct_norm  = ct_normalizer  or EnhancedCTNormalizer()
        self.pet_norm = pet_normalizer or EnhancedPETNormalizer()

        self.transform = get_augmentation(is_train=(mode == 'train'))

        print(f"\n{'='*70}")
        print(f"IntraBone Dataset - bone_pred ({mode.upper()})")
        print(f"  bone mask source: _bone_pred.png (fallback: _bone_pred.png)")
        print(f"  min tumor pixels: {min_tumor_pixels}")

        self.image_list, self.tumor_ratios, self.is_tumor_slice = \
            self._filter(image_list)

        print(f"  kept: {len(self.image_list)} tumor slices")
        print(f"{'='*70}\n")

    # ── 路径查找 ──────────────────────────────────────────

    def _get_path(self, image_id, suffix):
        """先在 img_root 下找，再在子文件夹下找"""
        # 直接路径
        p = os.path.join(self.img_root, image_id + suffix)
        if os.path.exists(p):
            return p
        # 子文件夹 (patient_id/image_id)
        parts = image_id.split('_')
        for n in range(len(parts), 0, -1):
            patient_id = '_'.join(parts[:n])
            alt = os.path.join(self.img_root, patient_id, image_id + suffix)
            if os.path.exists(alt):
                return alt
        return p   # 找不到也返回，让后续报错

    def _get_bone_path(self, image_id):
        """
        [核心改动] 优先使用 bone_pred，不存在则 fallback 到 bone_pred
        """
        pred_path = self._get_path(image_id, '_bone_pred.png')
        if os.path.exists(pred_path):
            return pred_path, 'pred'
        mask_path = self._get_path(image_id, '_bone_pred.png')
        return mask_path, 'gt'

    # ── 过滤逻辑 ──────────────────────────────────────────

    def _filter(self, raw_list):
        print(f"  filtering {len(raw_list)} slices...")

        tumor_slices = []
        n_small = 0
        n_empty = 0

        for image_id in raw_list:
            tmask_path         = self._get_path(image_id, '_mask.png')
            bone_path, src     = self._get_bone_path(image_id)

            if not (os.path.exists(tmask_path) and os.path.exists(bone_path)):
                continue

            tumor_mask = cv2.imread(tmask_path,  cv2.IMREAD_GRAYSCALE)
            bone_pred  = cv2.imread(bone_path,   cv2.IMREAD_GRAYSCALE)

            if tumor_mask is None or bone_pred is None:
                continue

            tumor_bin = (tumor_mask > 127).astype(np.float32)
            bone_bin  = (bone_pred  > 127).astype(np.float32)

            bone_area = bone_bin.sum()
            if bone_area < 500:
                continue

            tumor_in_bone = (tumor_bin * bone_bin).sum()
            tumor_ratio   = tumor_in_bone / bone_area

            if tumor_ratio >= self.min_tumor_ratio and \
               tumor_in_bone >= self.min_tumor_pixels:
                tumor_slices.append((image_id, tumor_ratio))
            elif 0 < tumor_in_bone < self.min_tumor_pixels:
                n_small += 1
            else:
                n_empty += 1

        print(f"  tumor: {len(tumor_slices)}  small(filtered): {n_small}  empty: {n_empty}")

        image_list   = [x[0] for x in tumor_slices]
        tumor_ratios = [x[1] for x in tumor_slices]
        is_tumor     = [True]  * len(tumor_slices)
        return image_list, tumor_ratios, is_tumor

    # ── 读取 ──────────────────────────────────────────────

    def _read_ct(self, path):
        if self.is_16bit:
            ct = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if ct is None:
                raise FileNotFoundError(f"CT not found: {path}")
            return self.ct_norm.normalize(ct, method='bone')
        ct = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ct is None:
            raise FileNotFoundError(f"CT not found: {path}")
        return ct.astype(np.float32) / 255.0

    def _read_pet(self, path):
        pet = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if pet is None:
            raise ValueError(f"PET not found: {path}")
        return self.pet_norm.normalize(pet)

    def _read_mask(self, path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"Mask not found: {path}")
        return (m > 127).astype(np.float32)

    # ── __getitem__ ────────────────────────────────────────

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id = self.image_list[idx]

        ct_path    = self._get_path(image_id, '_CT.png')
        pet_path   = self._get_path(image_id, '_PET.png')
        tmask_path = self._get_path(image_id, '_mask.png')
        bone_path, _ = self._get_bone_path(image_id)   # ← 用 bone_pred

        ct         = self._read_ct(ct_path)
        pet        = self._read_pet(pet_path)
        tumor_mask = self._read_mask(tmask_path)
        bone_pred  = self._read_mask(bone_path)

        # 数据增强
        aug = self.transform(image=ct, pet=pet,
                              bone_pred=bone_pred, tumor_mask=tumor_mask)
        ct         = aug['image'].float()
        pet        = aug['pet'].float()
        bone_pred  = aug['bone_pred'].float()
        tumor_mask = aug['tumor_mask'].float()

        ct         = torch.clamp(ct, 0, 1)
        pet        = torch.clamp(pet, 0, 1)
        bone_pred  = (bone_pred  > 0.5).float()
        tumor_mask = (tumor_mask > 0.5).float()

        # 骨内约束
        ct         = ct  * bone_pred
        pet        = pet * bone_pred
        tumor_mask = tumor_mask * bone_pred

        # 确保 (1,H,W)
        for t in [ct, pet, bone_pred, tumor_mask]:
            if t.dim() == 2:
                t = t.unsqueeze(0)

        def ensure_3d(t):
            return t.unsqueeze(0) if t.dim() == 2 else t

        ct, pet, bone_pred, tumor_mask = map(ensure_3d, [ct, pet, bone_pred, tumor_mask])

        bone_area   = bone_pred.sum().item()
        tumor_ratio = (tumor_mask * bone_pred).sum().item() / max(bone_area, 1)

        return {
            'ct':          ct,
            'pet':         pet,
            'bone_pred':   bone_pred,
            'tumor_mask':  tumor_mask,
            'name':        image_id,
            'tumor_ratio': torch.tensor(tumor_ratio, dtype=torch.float32),
            'is_tumor':    torch.tensor(self.is_tumor_slice[idx], dtype=torch.bool)
        }


# ============================================================
#  DataLoader
# ============================================================

def get_intrabone_dataloader_512(data_root, split_file, mode='train',
                                  batch_size=4, num_workers=4,
                                  use_weighted_sampler=False,
                                  tumor_prob=0.7,
                                  empty_keep_ratio=0.0,
                                  min_tumor_pixels=10,
                                  is_16bit=True,
                                  **kwargs):
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file) as f:
        image_list = [l.strip() for l in f if l.strip()]

    dataset = PerfectIntraBoneDataset512Fixed(
        image_list=image_list,
        img_root=data_root,
        mode=mode,
        is_16bit=is_16bit,
        min_tumor_pixels=min_tumor_pixels,
        empty_keep_ratio=0.0
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
        persistent_workers=(num_workers > 0)
    )

    print(f"DataLoader [{mode}]: {len(dataset)} samples, {len(loader)} batches")
    return loader, dataset


# 向后兼容
get_perfect_dataloader_512_fixed = get_intrabone_dataloader_512
IntraBoneTumorDataset512          = PerfectIntraBoneDataset512Fixed