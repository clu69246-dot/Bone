"""
Perfect Intra-Bone Tumor Dataset - NO ROI CROP VERSION

🔥 核心原则：
1. ❌ 废除 BoneROICropper（不再 Crop 小图）
2. ✅ 直接 Resize: 512x512 → 256x256（保留真实纹理）
3. ✅ 输入端骨掩码约束（ct * bone_pred）
4. ✅ 避免"虚构像素"问题

目标: Dice ≥ 0.75
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random


# ==================== CT 归一化 ====================

class EnhancedCTNormalizer:
    """CT归一化器"""
    def __init__(self, auto_detect_offset=True):
        self.auto_detect_offset = auto_detect_offset
        self.bone_window = {'center': 500, 'width': 2500}

    def detect_hu_offset(self, ct_raw):
        ct_min = ct_raw.min()
        ct_max = ct_raw.max()

        if ct_min >= 0 and ct_max <= 4095:
            ct_hu = ct_raw.astype(np.float32) - 1024
        elif ct_min >= -1024 and ct_max <= 3071:
            ct_hu = ct_raw.astype(np.float32)
        else:
            ct_hu = ct_raw.astype(np.float32) - 1024

        return ct_hu

    def normalize_with_window(self, ct_hu, center, width):
        window_min = center - width / 2
        window_max = center + width / 2
        ct_clipped = np.clip(ct_hu, window_min, window_max)

        if window_max > window_min:
            normalized = (ct_clipped - window_min) / (window_max - window_min)
        else:
            normalized = np.zeros_like(ct_hu, dtype=np.float32)

        return normalized.astype(np.float32)

    def normalize(self, ct_raw, method='bone'):
        if self.auto_detect_offset:
            ct_hu = self.detect_hu_offset(ct_raw)
        else:
            ct_hu = ct_raw.astype(np.float32) - 1024

        return self.normalize_with_window(ct_hu, self.bone_window['center'], self.bone_window['width'])


# ==================== PET 归一化 ====================

class EnhancedPETNormalizer:
    """PET归一化器"""
    def __init__(self, percentile_low=5.0, percentile_high=99.5, gamma=1.3):
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.gamma = gamma

    def normalize(self, pet_raw):
        pet = pet_raw.astype(np.float32)

        p_low = np.percentile(pet, self.percentile_low)
        p_high = np.percentile(pet, self.percentile_high)
        pet_clipped = np.clip(pet, p_low, p_high)

        min_val = pet_clipped.min()
        max_val = pet_clipped.max()

        if max_val > min_val:
            pet_norm = (pet_clipped - min_val) / (max_val - min_val)
        else:
            pet_norm = np.zeros_like(pet, dtype=np.float32)

        pet_norm = np.power(pet_norm, 1.0 / self.gamma)

        return np.clip(pet_norm, 0, 1).astype(np.float32)


# ==================== 🔥 数据增强（关键修改）====================

def get_perfect_augmentation(image_size=256, is_train=True):
    """
    🔥 完美增强管道

    关键改动：
    - 第一步就是 Resize（512 -> 256），避免虚构像素
    - 保留真实纹理和代谢信息
    """
    if is_train:
        return A.Compose([
            # 🔥 关键修改：第一步就 Resize，从 512 缩放到 256
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),

            # 几何增强
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=15,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),

            # 弹性变换
            A.ElasticTransform(alpha=50, sigma=5, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3, border_mode=cv2.BORDER_CONSTANT, value=0),

            # 强度增强
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
            ], p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),

            ToTensorV2()
        ], additional_targets={'pet': 'image', 'bone_pred': 'mask', 'tumor_mask': 'mask'})
    else:
        return A.Compose([
            # 🔥 验证集也是直接 Resize
            A.Resize(height=image_size, width=image_size, interpolation=cv2.INTER_LINEAR),
            ToTensorV2()
        ], additional_targets={'pet': 'image', 'bone_pred': 'mask', 'tumor_mask': 'mask'})


# ==================== 🔥 完美数据集（关键修改）====================

class PerfectIntraBoneDataset(Dataset):
    """
    🔥 完美骨内肿瘤数据集

    核心改动：
    1. ❌ 废除 BoneROICropper
    2. ✅ 直接 512 -> 256 Resize
    3. ✅ 输入端骨掩码约束
    """

    def __init__(self,
                 image_list,
                 img_root,
                 image_size=256,
                 mode='train',
                 is_16bit=True,
                 ct_normalizer=None,
                 pet_normalizer=None,
                 min_tumor_ratio=0.0001,
                 empty_keep_ratio=0.15):

        self.img_root = img_root
        self.image_size = image_size
        self.mode = mode
        self.is_16bit = is_16bit
        self.min_tumor_ratio = min_tumor_ratio
        self.empty_keep_ratio = empty_keep_ratio if mode == 'train' else 1.0

        self.ct_normalizer = ct_normalizer if ct_normalizer else EnhancedCTNormalizer()
        self.pet_normalizer = pet_normalizer if pet_normalizer else EnhancedPETNormalizer()

        # 🔥 关键改动：不再使用 BoneROICropper
        # self.roi_cropper = None

        # 🔥 使用新的增强管道
        self.transform = get_perfect_augmentation(image_size=image_size, is_train=(mode == 'train'))

        print(f"\n{'='*80}")
        print(f"🔥 PERFECT INTRA-BONE Dataset ({mode.upper()})")
        print(f"{'='*80}")
        print(f"📌 Image Size: 512 → {image_size} (Direct Resize)")
        print(f"📌 NO ROI Crop (Preserving Real Texture)")

        self.image_list, self.tumor_ratios, self.is_tumor_slice = self._hard_filter_slices(image_list)

        print(f"{'='*80}\n")

    def _hard_filter_slices(self, raw_image_list):
        """硬过滤逻辑（保持不变）"""
        print(f"📊 Raw dataset: {len(raw_image_list)} slices")

        tumor_slices = []
        empty_slices = []

        for idx, image_id in enumerate(raw_image_list):
            tumor_mask_path = self._get_file_path(image_id, "_mask.png")
            bone_pred_path = self._get_file_path(image_id, "_bone_pred.png")

            if os.path.exists(tumor_mask_path) and os.path.exists(bone_pred_path):
                tumor_mask = cv2.imread(tumor_mask_path, cv2.IMREAD_GRAYSCALE)
                bone_pred = cv2.imread(bone_pred_path, cv2.IMREAD_GRAYSCALE)

                if tumor_mask is not None and bone_pred is not None:
                    tumor_binary = (tumor_mask > 127).astype(np.float32)
                    bone_binary = (bone_pred > 127).astype(np.float32)

                    bone_area = bone_binary.sum()
                    if bone_area < 100:
                        continue

                    tumor_in_bone = (tumor_binary * bone_binary).sum()
                    tumor_ratio = tumor_in_bone / bone_area

                    if tumor_ratio >= self.min_tumor_ratio:
                        tumor_slices.append((image_id, tumor_ratio))
                    else:
                        empty_slices.append((image_id, 0.0))

        print(f"  Tumor slices: {len(tumor_slices)}")
        print(f"  Empty slices: {len(empty_slices)}")

        # 过滤逻辑
        filtered_list = []
        filtered_ratios = []
        filtered_is_tumor = []

        for image_id, ratio in tumor_slices:
            filtered_list.append(image_id)
            filtered_ratios.append(ratio)
            filtered_is_tumor.append(True)

        # Empty slice 采样
        if self.mode == 'train':
            num_keep = int(len(empty_slices) * self.empty_keep_ratio)
            sampled_empty = random.sample(empty_slices, min(num_keep, len(empty_slices)))
            print(f"\n🎯 Training mode: Sampling {len(sampled_empty)}/{len(empty_slices)} empty slices")
        else:
            sampled_empty = empty_slices
            print(f"\n✅ Validation mode: Keeping ALL slices")

        for image_id, ratio in sampled_empty:
            filtered_list.append(image_id)
            filtered_ratios.append(ratio)
            filtered_is_tumor.append(False)

        num_tumor = sum(filtered_is_tumor)
        num_empty = len(filtered_is_tumor) - num_tumor
        total = len(filtered_list)

        print(f"\n✅ After Filtering:")
        print(f"  Total: {total} slices")
        print(f"  Tumor: {num_tumor} ({100*num_tumor/total:.1f}%) ⭐")
        print(f"  Empty: {num_empty} ({100*num_empty/total:.1f}%)")

        return filtered_list, filtered_ratios, filtered_is_tumor

    def _get_file_path(self, image_id, suffix):
        parts = image_id.split('_')
        patient_id = '_'.join(parts[:-1])
        path = os.path.join(self.img_root, patient_id, image_id + suffix)

        if os.path.exists(path):
            return path

        if len(parts) >= 3:
            patient_id = '_'.join(parts[:2])
            path = os.path.join(self.img_root, patient_id, image_id + suffix)
            if os.path.exists(path):
                return path

        return os.path.join(self.img_root, '_'.join(parts[:-1]), image_id + suffix)

    def _read_ct(self, ct_path):
        if self.is_16bit:
            ct = cv2.imread(ct_path, cv2.IMREAD_UNCHANGED)
            if ct is None:
                raise FileNotFoundError(f"Cannot read CT: {ct_path}")
            ct_normalized = self.ct_normalizer.normalize(ct, method='bone')
        else:
            ct = cv2.imread(ct_path, cv2.IMREAD_GRAYSCALE)
            if ct is None:
                raise FileNotFoundError(f"Cannot read CT: {ct_path}")
            ct_normalized = ct.astype(np.float32) / 255.0

        return ct_normalized

    def _read_pet(self, pet_path):
        pet = cv2.imread(pet_path, cv2.IMREAD_UNCHANGED)
        if pet is None:
            raise ValueError(f"Failed to read PET: {pet_path}")

        pet_normalized = self.pet_normalizer.normalize(pet)
        return pet_normalized

    def _read_mask(self, path):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {path}")
        return (mask > 127).astype(np.float32)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id = self.image_list[idx]

        ct_path = self._get_file_path(image_id, "_CT.png")
        pet_path = self._get_file_path(image_id, "_PET.png")
        tumor_mask_path = self._get_file_path(image_id, "_mask.png")
        bone_pred_path = self._get_file_path(image_id, "_bone_pred.png")

        # 🔥 Step 1: 读取原始 512x512 大图
        ct = self._read_ct(ct_path)
        pet = self._read_pet(pet_path)
        tumor_mask = self._read_mask(tumor_mask_path)
        bone_pred = self._read_mask(bone_pred_path)

        # 🔥 Step 2: 数据增强（包含 Resize 512->256）
        if self.transform:
            augmented = self.transform(
                image=ct,
                pet=pet,
                bone_pred=bone_pred,
                tumor_mask=tumor_mask
            )
            ct = augmented['image'].float()
            pet = augmented['pet'].float()
            bone_pred = augmented['bone_pred'].float()
            tumor_mask = augmented['tumor_mask'].float()
        else:
            ct = torch.from_numpy(ct).float()
            pet = torch.from_numpy(pet).float()
            bone_pred = torch.from_numpy(bone_pred).float()
            tumor_mask = torch.from_numpy(tumor_mask).float()

        # 🔥 Step 3: 归一化和二值化
        ct = torch.clamp(ct, 0, 1)
        pet = torch.clamp(pet, 0, 1)
        bone_pred = (bone_pred > 0.5).float()
        tumor_mask = (tumor_mask > 0.5).float()

        # 🔥 Step 4: 骨内约束（关键！）
        ct = ct * bone_pred
        pet = pet * bone_pred
        tumor_mask = tumor_mask * bone_pred

        # 🔥 Step 5: 确保维度 [1, H, W]
        if ct.dim() == 2:
            ct = ct.unsqueeze(0)
        if pet.dim() == 2:
            pet = pet.unsqueeze(0)
        if bone_pred.dim() == 2:
            bone_pred = bone_pred.unsqueeze(0)
        if tumor_mask.dim() == 2:
            tumor_mask = tumor_mask.unsqueeze(0)

        bone_area = bone_pred.sum().item()
        if bone_area > 0:
            tumor_ratio = (tumor_mask * bone_pred).sum().item() / bone_area
        else:
            tumor_ratio = 0.0

        return {
            'ct': ct,
            'pet': pet,
            'bone_pred': bone_pred,
            'tumor_mask': tumor_mask,
            'name': image_id,
            'tumor_ratio': torch.tensor(tumor_ratio, dtype=torch.float32),
            'is_tumor': torch.tensor(self.is_tumor_slice[idx], dtype=torch.bool)
        }


# ==================== Sampler ====================

def create_tumor_weighted_sampler(dataset, tumor_prob=0.7, verbose=True):
    """创建加权采样器"""
    num_tumor = sum(dataset.is_tumor_slice)
    num_empty = len(dataset.is_tumor_slice) - num_tumor

    if verbose:
        print(f"\n{'='*80}")
        print(f"🎲 Creating WeightedRandomSampler")
        print(f"  Tumor: {num_tumor}, Empty: {num_empty}")

    weights = []
    for is_tumor in dataset.is_tumor_slice:
        if is_tumor:
            weights.append(tumor_prob)
        else:
            weights.append(1.0 - tumor_prob)

    total_weight = sum(weights)
    weights = [w / total_weight * len(weights) for w in weights]

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )

    if verbose:
        print(f"  Target P(tumor) ≈ {tumor_prob:.1%}")
        print(f"{'='*80}\n")

    return sampler


# ==================== DataLoader ====================

def get_perfect_dataloader(data_root,
                           split_file,
                           mode='train',
                           batch_size=4,
                           num_workers=4,
                           image_size=256,
                           use_weighted_sampler=True,
                           tumor_prob=0.7,
                           empty_keep_ratio=0.15,
                           is_16bit=True,
                           **kwargs):
    """
    🔥 创建完美数据加载器

    关键参数：
    - image_size: 目标尺寸（推荐 256）
    - use_roi_crop: 已废除，不再使用
    """

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, 'r') as f:
        image_list = [line.strip() for line in f.readlines() if line.strip()]

    # 🔥 使用新的 PerfectIntraBoneDataset
    dataset = PerfectIntraBoneDataset(
        image_list=image_list,
        img_root=data_root,
        image_size=image_size,
        mode=mode,
        is_16bit=is_16bit,
        empty_keep_ratio=empty_keep_ratio
    )

    if mode == 'train' and use_weighted_sampler:
        sampler = create_tumor_weighted_sampler(dataset, tumor_prob=tumor_prob, verbose=True)
        shuffle = False
    else:
        sampler = None
        shuffle = (mode == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(mode == 'train'),
        persistent_workers=(num_workers > 0)
    )

    print(f"✅ Perfect DataLoader Ready ({mode.upper()}):")
    print(f"  Batches: {len(dataloader)}, Samples: {len(dataset)}\n")

    return dataloader, dataset


# ==================== 向后兼容 ====================

# 🔥 为了兼容旧代码，提供别名
get_intrabone_dataloader = get_perfect_dataloader
IntraBoneTumorDatasetHardFiltered = PerfectIntraBoneDataset