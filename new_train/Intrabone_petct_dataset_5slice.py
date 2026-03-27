"""
Intra-Bone Tumor Dataset — v7  5-Slice 2.5D + CSA-Net 版本

与 v6 的差异：
  [v7-1] n_slices 从 3 升级为 5（±2 邻居），对应 CSA-Net 推荐的 5-slice 策略
  [v7-2] _load_ct5 / _load_pet5：加载 t-2, t-1, t, t+1, t+2 共 5 个切片
  [v7-3] DataLoader 输出 ct=(5,H,W), pet=(5,H,W)，而非原来的 (3,H,W)
  [v7-4] CSASliceFusion 替换 ct_adapter/pet_adapter（见 Intrabone_petct_dataset_5slice.py）
  [v7-5] HardNegativeDataset 同步升级为 5-slice 输出
  [v7-6] 增强管道（albumentations）对 5 通道同步几何变换（与 v6 相同逻辑，只换通道数）
  [v7-7] additional_targets 键名从 'pet3ch' 改为 'pet5ch'，'image' 为 ct5ch

保留 v6 所有过滤逻辑（min_tumor_pixels, tumor_ratio, bone_area）。
保留 v6 的 patient-slice 邻切片索引机制。
"""

import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from collections import defaultdict


# ============================================================
#  切片 ID 解析（与 v6 完全相同）
# ============================================================

def parse_patient_slice(image_id: str):
    m = re.match(r'^(.+?)_(\d+)$', image_id)
    if m:
        return m.group(1), int(m.group(2))
    return image_id, None


def build_patient_slice_map(image_ids):
    pmap = defaultdict(list)
    for iid in image_ids:
        patient, z = parse_patient_slice(iid)
        pmap[patient].append((z, iid))
    for k in pmap:
        pmap[k].sort(key=lambda x: (x[0] is None, x[0]))
    return dict(pmap)


# ============================================================
#  归一化器（与 v6 相同）
# ============================================================

class EnhancedCTNormalizer:
    def __init__(self, auto_detect_offset=True):
        self.auto_detect_offset = auto_detect_offset
        self.bone_window = {'center': 400.0, 'width': 1200.0}

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
#  [v7] 数据增强 — 5 通道同步版
# ============================================================

def get_augmentation_5slice(is_train=True):
    """
    [v7] 数据增强 — 5 通道 (H,W,5) ndarray 同步增强

    albumentations 对 additional_targets 中的 'image' 类型 tensor
    自动对任意通道数 ndarray 同步几何变换。
    pet5ch 作为第二个 'image' target，与 ct5ch 完全同步。
    bone_pred / tumor_mask 作为 'mask' target 同步几何变换（但不做强度增强）。

    注意：CoarseDropout 只对主 image (ct5ch) 做 dropout；
          pet5ch 通过 additional_targets 同步，不会独立 dropout。
    """
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=25,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.6),
            A.ElasticTransform(
                alpha=30, sigma=5,
                border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.4),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.CoarseDropout(
                max_holes=3, max_height=24, max_width=24,
                min_holes=1, min_height=8, min_width=8,
                fill_value=0, p=0.2),
            ToTensorV2()
        ], additional_targets={
            'pet5ch':    'image',   # [v7] 5-channel PET
            'bone_pred': 'mask',
            'tumor_mask':'mask',
        })
    else:
        return A.Compose([
            ToTensorV2()
        ], additional_targets={
            'pet5ch':    'image',
            'bone_pred': 'mask',
            'tumor_mask':'mask',
        })


# ============================================================
#  [v7] 主数据集 — 5-Slice 版
# ============================================================

class PerfectIntraBoneDataset512_5Slice(Dataset):
    """
    [v7] 骨内肿瘤数据集 — 5-Slice 2.5D 版

    输出：
      ct:         (5, H, W)  — 切片 t-2/t-1/t/t+1/t+2
      pet:        (5, H, W)
      bone_pred:  (1, H, W)  — 仅当前切片 t 的骨骼 mask
      tumor_mask: (1, H, W)  — 仅当前切片 t 的肿瘤 mask

    邻切片查找规则：
      同 v6：先解析 image_id 末尾数字为 z；在同 patient 下找 z±1/z±2。
      找不到时用当前切片边界复制（boundary padding）。
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
                 all_ids_for_neighbors=None):
        self.img_root         = img_root
        self.mode             = mode
        self.is_16bit         = is_16bit
        self.min_tumor_ratio  = min_tumor_ratio
        self.min_tumor_pixels = min_tumor_pixels

        self.ct_norm  = ct_normalizer  or EnhancedCTNormalizer()
        self.pet_norm = pet_normalizer or EnhancedPETNormalizer()
        self.transform = get_augmentation_5slice(is_train=(mode == 'train'))

        print(f"\n{'='*70}")
        print(f"IntraBone Dataset - 5-Slice 2.5D + CSA ({mode.upper()})")
        print(f"  Input:  ct=(5,H,W), pet=(5,H,W)  [±2 neighbors]")
        print(f"  bone mask source: _bone_pred.png")
        print(f"  min tumor pixels: {min_tumor_pixels}")

        self.image_list, self.tumor_ratios, self.is_tumor_slice = \
            self._filter(image_list)

        # 构建邻切片索引
        ref_ids = all_ids_for_neighbors if all_ids_for_neighbors else image_list
        self._pmap  = build_patient_slice_map(ref_ids)
        self._zmap  = {}
        for patient, entries in self._pmap.items():
            for z, iid in entries:
                self._zmap[(patient, z)] = iid

        print(f"  kept: {len(self.image_list)} tumor slices")
        print(f"{'='*70}\n")

    # ── 路径查找 ──────────────────────────────────────────────────

    def _get_path(self, image_id, suffix):
        p = os.path.join(self.img_root, image_id + suffix)
        if os.path.exists(p):
            return p
        parts = image_id.split('_')
        for n in range(len(parts), 0, -1):
            alt = os.path.join(self.img_root, '_'.join(parts[:n]),
                               image_id + suffix)
            if os.path.exists(alt):
                return alt
        return p

    def _get_bone_path(self, image_id):
        pred_path = self._get_path(image_id, '_bone_pred.png')
        if os.path.exists(pred_path):
            return pred_path
        return self._get_path(image_id, '_bone_pred.png')

    # ── [v7] 邻切片查找（支持 ±2）────────────────────────────────

    def _neighbor_id(self, image_id: str, delta: int) -> str:
        """返回 z+delta 的 image_id；找不到时返回 image_id（边界复制）。"""
        patient, z = parse_patient_slice(image_id)
        if z is None:
            return image_id
        nb_id = self._zmap.get((patient, z + delta))
        return nb_id if nb_id is not None else image_id

    # ── 过滤逻辑（与 v6 相同）────────────────────────────────────

    def _filter(self, raw_list):
        print(f"  filtering {len(raw_list)} slices...")
        tumor_slices = []
        n_small = 0
        n_empty = 0
        for image_id in raw_list:
            tmask_path    = self._get_path(image_id, '_mask.png')
            bone_path, _  = self._get_bone_path(image_id), 'pred'
            bone_path     = self._get_bone_path(image_id)
            if not (os.path.exists(tmask_path) and os.path.exists(bone_path)):
                continue
            tumor_mask = cv2.imread(tmask_path, cv2.IMREAD_GRAYSCALE)
            bone_pred  = cv2.imread(bone_path,  cv2.IMREAD_GRAYSCALE)
            if tumor_mask is None or bone_pred is None:
                continue
            bone_area     = (bone_pred  > 127).sum()
            if bone_area < 500:
                continue
            tumor_in_bone = ((tumor_mask > 127) & (bone_pred > 127)).sum()
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
        is_tumor     = [True] * len(tumor_slices)
        return image_list, tumor_ratios, is_tumor

    # ── 读取单切片 ────────────────────────────────────────────────

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

    # ── [v7] 加载 5 切片并堆叠为 (H,W,5) ─────────────────────────

    def _load_ct5(self, image_id: str) -> np.ndarray:
        """返回 (H, W, 5) float32，通道顺序 t-2,t-1,t,t+1,t+2"""
        deltas = (-2, -1, 0, 1, 2)
        ids    = [self._neighbor_id(image_id, d) for d in deltas]
        slices = []
        for iid in ids:
            try:
                s = self._read_ct(self._get_path(iid, '_CT.png'))
            except Exception:
                s = self._read_ct(self._get_path(image_id, '_CT.png'))
            slices.append(s)
        return np.stack(slices, axis=-1)   # (H, W, 5)

    def _load_pet5(self, image_id: str) -> np.ndarray:
        """返回 (H, W, 5) float32，通道顺序 t-2,t-1,t,t+1,t+2"""
        deltas = (-2, -1, 0, 1, 2)
        ids    = [self._neighbor_id(image_id, d) for d in deltas]
        slices = []
        for iid in ids:
            try:
                s = self._read_pet(self._get_path(iid, '_PET.png'))
            except Exception:
                s = self._read_pet(self._get_path(image_id, '_PET.png'))
            slices.append(s)
        return np.stack(slices, axis=-1)   # (H, W, 5)

    # ── __getitem__ ────────────────────────────────────────────────

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id = self.image_list[idx]

        # [v7] 5-切片输入
        ct5ch        = self._load_ct5(image_id)    # (H, W, 5)
        pet5ch       = self._load_pet5(image_id)   # (H, W, 5)
        bone_path    = self._get_bone_path(image_id)
        tumor_mask   = self._read_mask(self._get_path(image_id, '_mask.png'))
        bone_pred    = self._read_mask(bone_path)

        # 数据增强（5 通道同步）
        aug = self.transform(
            image      = ct5ch,
            pet5ch     = pet5ch,
            bone_pred  = bone_pred,
            tumor_mask = tumor_mask,
        )
        ct5ch      = aug['image'].float()        # Tensor (5, H, W) after ToTensorV2
        pet5ch     = aug['pet5ch'].float()       # Tensor (5, H, W)
        bone_pred  = aug['bone_pred'].float()
        tumor_mask = aug['tumor_mask'].float()

        ct5ch      = torch.clamp(ct5ch,  0, 1)
        pet5ch     = torch.clamp(pet5ch, 0, 1)

        def ensure_3d(t):
            return t.unsqueeze(0) if t.dim() == 2 else t
        bone_pred  = ensure_3d(bone_pred)
        tumor_mask = ensure_3d(tumor_mask)

        # [v6-fix 保留] 不遮盖 5ch 输入，保留上下文；仅 label 限在骨内
        bone_pred  = (bone_pred  > 0.5).float()
        tumor_mask = (tumor_mask > 0.5).float() * bone_pred

        bone_area   = bone_pred.sum().item()
        tumor_ratio = (tumor_mask * bone_pred).sum().item() / max(bone_area, 1)

        return {
            'ct':          ct5ch,          # (5, H, W)  ← [v7] 改为 5 通道
            'pet':         pet5ch,         # (5, H, W)  ← [v7]
            'bone_pred':   bone_pred,      # (1, H, W)
            'tumor_mask':  tumor_mask,     # (1, H, W)
            'name':        image_id,
            'tumor_ratio': torch.tensor(tumor_ratio, dtype=torch.float32),
            'is_tumor':    torch.tensor(self.is_tumor_slice[idx], dtype=torch.bool),
        }


# ============================================================
#  [v7] Hard Negative Dataset — 5-slice 版
# ============================================================

class HardNegativeDataset5Slice(Dataset):
    """
    [v7] Hard Negative 数据集（5-slice 版）

    与 v6 的 HardNegativeDataset 逻辑相同，仅将 3-slice 输出升级为 5-slice。
    """

    def __init__(self,
                 tumor_image_ids,
                 all_image_ids,
                 img_root,
                 neg_range=5,
                 mode='train',
                 is_16bit=True,
                 max_per_tumor=3):
        self.img_root  = img_root
        self.mode      = mode
        self.is_16bit  = is_16bit
        self.ct_norm   = EnhancedCTNormalizer()
        self.pet_norm  = EnhancedPETNormalizer()
        self.transform = get_augmentation_5slice(is_train=(mode == 'train'))

        pmap = build_patient_slice_map(all_image_ids)
        zmap = {}
        for patient, entries in pmap.items():
            for z, iid in entries:
                zmap[(patient, z)] = iid
        self._zmap = zmap

        self.image_list = self._collect_hard_negatives(
            tumor_image_ids, pmap, zmap, neg_range, max_per_tumor)

        print(f"  [HardNeg 5-slice] {len(self.image_list)} hard negative slices "
              f"(neg_range=±{neg_range}, max_per_tumor={max_per_tumor})")

    def _get_path(self, image_id, suffix):
        p = os.path.join(self.img_root, image_id + suffix)
        if os.path.exists(p):
            return p
        parts = image_id.split('_')
        for n in range(len(parts), 0, -1):
            alt = os.path.join(self.img_root, '_'.join(parts[:n]),
                               image_id + suffix)
            if os.path.exists(alt):
                return alt
        return p

    def _get_bone_path(self, image_id):
        p = self._get_path(image_id, '_bone_pred.png')
        return p

    def _is_valid_hard_neg(self, image_id):
        tmask_p = self._get_path(image_id, '_mask.png')
        bone_p  = self._get_bone_path(image_id)
        if not (os.path.exists(tmask_p) and os.path.exists(bone_p)):
            return False
        tm = cv2.imread(tmask_p, cv2.IMREAD_GRAYSCALE)
        bm = cv2.imread(bone_p,  cv2.IMREAD_GRAYSCALE)
        if tm is None or bm is None:
            return False
        bone_area    = (bm > 127).sum()
        tumor_pixels = ((tm > 127) & (bm > 127)).sum()
        return bone_area >= 500 and tumor_pixels == 0

    def _collect_hard_negatives(self, tumor_ids, pmap, zmap,
                                 neg_range, max_per_tumor):
        collected = set()
        rng = random.Random(42)
        for iid in tumor_ids:
            patient, z = parse_patient_slice(iid)
            if z is None:
                continue
            candidates = []
            for dz in range(-neg_range, neg_range + 1):
                if dz == 0:
                    continue
                nb = zmap.get((patient, z + dz))
                if nb and nb not in collected and self._is_valid_hard_neg(nb):
                    candidates.append(nb)
            rng.shuffle(candidates)
            for nb in candidates[:max_per_tumor]:
                collected.add(nb)
        return list(collected)

    def _neighbor_id(self, image_id, delta):
        patient, z = parse_patient_slice(image_id)
        if z is None:
            return image_id
        nb = self._zmap.get((patient, z + delta))
        return nb if nb is not None else image_id

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
        image_id = self.image_list[idx]

        ct5ch  = self._load_ct5(image_id)   # (H, W, 5)
        pet5ch = self._load_pet5(image_id)  # (H, W, 5)
        bone_p = self._get_bone_path(image_id)
        bone_raw   = cv2.imread(bone_p, cv2.IMREAD_GRAYSCALE)
        bone_pred  = (bone_raw > 127).astype(np.float32) \
                     if bone_raw is not None else np.zeros((512, 512), np.float32)
        tumor_mask = np.zeros_like(bone_pred)

        aug = self.transform(
            image      = ct5ch,
            pet5ch     = pet5ch,
            bone_pred  = bone_pred,
            tumor_mask = tumor_mask,
        )
        ct5ch      = aug['image'].float()     # (5, H, W)
        pet5ch     = aug['pet5ch'].float()    # (5, H, W)
        bone_pred  = aug['bone_pred'].float()
        tumor_mask = aug['tumor_mask'].float()

        ct5ch      = torch.clamp(ct5ch,  0, 1)
        pet5ch     = torch.clamp(pet5ch, 0, 1)

        def ensure_3d(t):
            return t.unsqueeze(0) if t.dim() == 2 else t
        bone_pred  = ensure_3d((bone_pred  > 0.5).float())
        tumor_mask = ensure_3d((tumor_mask > 0.5).float())

        return {
            'ct':          ct5ch,
            'pet':         pet5ch,
            'bone_pred':   bone_pred,
            'tumor_mask':  tumor_mask,
            'name':        image_id,
            'tumor_ratio': torch.tensor(0.0,   dtype=torch.float32),
            'is_tumor':    torch.tensor(False,  dtype=torch.bool),
        }


# ============================================================
#  DataLoader
# ============================================================

def get_intrabone_dataloader_5slice(data_root, split_file, mode='train',
                                     batch_size=4, num_workers=4,
                                     min_tumor_pixels=10,
                                     is_16bit=True,
                                     **kwargs):
    """
    [v7] 5-Slice DataLoader
    drop-in replacement for get_intrabone_dataloader_512
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file) as f:
        image_list = [l.strip() for l in f if l.strip()]

    dataset = PerfectIntraBoneDataset512_5Slice(
        image_list=image_list,
        img_root=data_root,
        mode=mode,
        is_16bit=is_16bit,
        min_tumor_pixels=min_tumor_pixels,
        all_ids_for_neighbors=image_list,
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


# 向后兼容别名
HardNegativeDataset = HardNegativeDataset5Slice