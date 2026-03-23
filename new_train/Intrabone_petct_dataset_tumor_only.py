"""
Intra-Bone Tumor Dataset — v6  2.5D + Hard Negative 版本

[v6-1] 2.5D 多切片输入
  每次返回 ct=(3,H,W)  pet=(3,H,W)，通道 0/1/2 分别对应 t-1 / t / t+1。
  命名约定：image_id 末尾为整数切片编号（以最后一个 '_' 分隔）
  例: "patient001_023" → patient="patient001", z=23
  找不到邻切片时用当前切片重复（边界 padding）。

[v6-2] Hard Negative Dataset（取代随机 Empty）
  专门采样"邻近肿瘤切片但不含肿瘤"的切片（距肿瘤±1~±5 slice）。
  这些切片骨区域结构复杂，是模型的主要 FP 来源，作为负样本
  可在不牺牲 Recall 的前提下压低 FP，直接提升 Precision。

[v6-3] 保留所有原有过滤逻辑 (min_tumor_pixels, tumor_ratio)

[v6-4] 数据增强：2.5D 三通道同步增强（所有通道共享同一几何变换）
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
#  [v6-1] 切片 ID 解析工具
# ============================================================

def parse_patient_slice(image_id: str):
    """
    解析 image_id，返回 (patient_prefix, slice_index)。
    支持末尾为数字的任意命名：
      "P001_023"      → ("P001", 23)
      "case01_s_042"  → ("case01_s", 42)
      "abc"           → ("abc", None)  ← 无法解析时返回 None
    """
    m = re.match(r'^(.+?)_(\d+)$', image_id)
    if m:
        return m.group(1), int(m.group(2))
    return image_id, None


def build_patient_slice_map(image_ids):
    """
    给定 image_id 列表，构建 {patient: sorted [(z, image_id), ...]} 映射。
    用于查找邻近切片。
    """
    pmap = defaultdict(list)
    for iid in image_ids:
        patient, z = parse_patient_slice(iid)
        pmap[patient].append((z, iid))
    for k in pmap:
        pmap[k].sort(key=lambda x: (x[0] is None, x[0]))
    return dict(pmap)


# ============================================================
#  归一化器
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
#  数据增强
# ============================================================

def get_augmentation(is_train=True):
    """[v6] 数据增强 — 2.5D 三通道同步版

    CoarseDropout 参数收紧（v5 已修正）：max_holes=3, max_height=24。
    几何变换对所有通道（ct3ch / pet3ch / bone_pred / tumor_mask）同步执行。
    注意：additional_targets 中 ct3ch / pet3ch 作为 'image' 类型，
    Albumentations 会自动对多通道 ndarray 同步变换。
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
            # 强度变换仅影响 image（ct3ch），额外 targets 按类型决定
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.4),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            # [v5修正] CoarseDropout 参数收紧防止破坏小肿瘤
            A.CoarseDropout(
                max_holes=3, max_height=24, max_width=24,
                min_holes=1, min_height=8, min_width=8,
                fill_value=0, p=0.2),
            ToTensorV2()
        ], additional_targets={
            'pet3ch':    'image',
            'bone_pred': 'mask',
            'tumor_mask':'mask',
        })
    else:
        return A.Compose([
            ToTensorV2()
        ], additional_targets={
            'pet3ch':    'image',
            'bone_pred': 'mask',
            'tumor_mask':'mask',
        })


# ============================================================
#  数据集
# ============================================================

class PerfectIntraBoneDataset512Fixed(Dataset):
    """
    [v6] 骨内肿瘤数据集 — 2.5D 多切片输入版

    输出:
      ct:         (3, H, W)  — 通道 0/1/2 = 切片 t-1 / t / t+1
      pet:        (3, H, W)
      bone_pred:  (1, H, W)  — 当前切片 t 的骨骼 mask
      tumor_mask: (1, H, W)  — 当前切片 t 的肿瘤 mask

    邻切片查找规则：
      - 先解析 image_id 末尾数字为 z；在同 patient 下找 z-1 / z+1
      - 找不到（边界或命名不规则）时用当前切片重复（边界复制 padding）
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
                 empty_keep_ratio=0.0,
                 all_ids_for_neighbors=None):   # [v6] 全量 id 用于邻切片查找
        """
        all_ids_for_neighbors: 包含当前 split 所有 image_id 的列表，
                               用于构建邻切片索引。若 None 则用 image_list。
        """
        self.img_root         = img_root
        self.mode             = mode
        self.is_16bit         = is_16bit
        self.min_tumor_ratio  = min_tumor_ratio
        self.min_tumor_pixels = min_tumor_pixels
        self.empty_keep_ratio = 0.0

        self.ct_norm  = ct_normalizer  or EnhancedCTNormalizer()
        self.pet_norm = pet_normalizer or EnhancedPETNormalizer()
        self.transform = get_augmentation(is_train=(mode == 'train'))

        print(f"\n{'='*70}")
        print(f"IntraBone Dataset - bone_pred 2.5D ({mode.upper()})")
        print(f"  bone mask source: _bone_pred.png")
        print(f"  min tumor pixels: {min_tumor_pixels}")

        self.image_list, self.tumor_ratios, self.is_tumor_slice = \
            self._filter(image_list)

        # [v6] 构建 patient→[(z, image_id)] 邻切片索引
        ref_ids = all_ids_for_neighbors if all_ids_for_neighbors else image_list
        self._pmap  = build_patient_slice_map(ref_ids)
        # 快速查找: (patient, z) → image_id
        self._zmap  = {}
        for patient, entries in self._pmap.items():
            for z, iid in entries:
                self._zmap[(patient, z)] = iid

        print(f"  kept: {len(self.image_list)} tumor slices")
        print(f"{'='*70}\n")

    # ── 路径查找 ──────────────────────────────────────────

    def _get_path(self, image_id, suffix):
        p = os.path.join(self.img_root, image_id + suffix)
        if os.path.exists(p):
            return p
        parts = image_id.split('_')
        for n in range(len(parts), 0, -1):
            patient_id = '_'.join(parts[:n])
            alt = os.path.join(self.img_root, patient_id, image_id + suffix)
            if os.path.exists(alt):
                return alt
        return p

    def _get_bone_path(self, image_id):
        pred_path = self._get_path(image_id, '_bone_pred.png')
        if os.path.exists(pred_path):
            return pred_path, 'pred'
        mask_path = self._get_path(image_id, '_bone_pred.png')
        return mask_path, 'gt'

    # ── [v6] 邻切片 image_id 查找 ─────────────────────────

    def _neighbor_id(self, image_id: str, delta: int) -> str:
        """返回 z+delta 的 image_id；找不到时返回 image_id（边界复制）。"""
        patient, z = parse_patient_slice(image_id)
        if z is None:
            return image_id
        nb_id = self._zmap.get((patient, z + delta))
        return nb_id if nb_id is not None else image_id

    # ── 过滤逻辑 ──────────────────────────────────────────

    def _filter(self, raw_list):
        print(f"  filtering {len(raw_list)} slices...")
        tumor_slices = []
        n_small = 0
        n_empty = 0
        for image_id in raw_list:
            tmask_path       = self._get_path(image_id, '_mask.png')
            bone_path, _     = self._get_bone_path(image_id)
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

    # ── 读取单切片 ────────────────────────────────────────

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

    # ── [v6] 加载三切片并堆叠为 (H,W,3) ──────────────────

    def _load_ct3(self, image_id: str) -> np.ndarray:
        """返回 (H, W, 3) float32，通道=t-1,t,t+1"""
        ids = [self._neighbor_id(image_id, d) for d in (-1, 0, 1)]
        slices = []
        for iid in ids:
            try:
                s = self._read_ct(self._get_path(iid, '_CT.png'))
            except Exception:
                # 邻切片文件不存在时用当前切片
                s = self._read_ct(self._get_path(image_id, '_CT.png'))
            slices.append(s)
        return np.stack(slices, axis=-1)   # (H, W, 3)

    def _load_pet3(self, image_id: str) -> np.ndarray:
        """返回 (H, W, 3) float32，通道=t-1,t,t+1"""
        ids = [self._neighbor_id(image_id, d) for d in (-1, 0, 1)]
        slices = []
        for iid in ids:
            try:
                s = self._read_pet(self._get_path(iid, '_PET.png'))
            except Exception:
                s = self._read_pet(self._get_path(image_id, '_PET.png'))
            slices.append(s)
        return np.stack(slices, axis=-1)   # (H, W, 3)

    # ── __getitem__ ────────────────────────────────────────

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id = self.image_list[idx]

        # [v6] 三通道输入
        ct3ch      = self._load_ct3(image_id)     # (H, W, 3)
        pet3ch     = self._load_pet3(image_id)    # (H, W, 3)
        bone_path, _ = self._get_bone_path(image_id)
        tumor_mask = self._read_mask(self._get_path(image_id, '_mask.png'))
        bone_pred  = self._read_mask(bone_path)

        # 数据增强（几何变换对三通道同步）
        # albumentations 对 ndarray(..., C) image 会自动处理多通道
        aug = self.transform(
            image      = ct3ch,
            pet3ch     = pet3ch,
            bone_pred  = bone_pred,
            tumor_mask = tumor_mask,
        )
        ct3ch      = aug['image'].float()       # Tensor (3, H, W) after ToTensorV2
        pet3ch     = aug['pet3ch'].float()
        bone_pred  = aug['bone_pred'].float()
        tumor_mask = aug['tumor_mask'].float()

        ct3ch      = torch.clamp(ct3ch, 0, 1)
        pet3ch     = torch.clamp(pet3ch, 0, 1)
        bone_pred  = (bone_pred  > 0.5).float()
        tumor_mask = (tumor_mask > 0.5).float()

        # ensure (1,H,W) for masks
        def ensure_3d(t):
            return t.unsqueeze(0) if t.dim() == 2 else t
        bone_pred  = ensure_3d(bone_pred)
        tumor_mask = ensure_3d(tumor_mask)

        # [v6-fix] 不用 bone_pred 遮盖三通道输入：
        #   - ct/pet 保持原始值（让模型自己学习骨内外的边界上下文）
        #   - 仅 tumor_mask 用 t 时刻的 bone_pred 约束（label 范围正确）
        #   - bone_pred 作为辅助通道传给模型，不作为乘法 mask
        tumor_mask = tumor_mask * bone_pred   # label 仍限在骨内

        bone_area   = bone_pred.sum().item()
        tumor_ratio = (tumor_mask * bone_pred).sum().item() / max(bone_area, 1)

        return {
            'ct':          ct3ch,          # (3, H, W)
            'pet':         pet3ch,         # (3, H, W)
            'bone_pred':   bone_pred,      # (1, H, W)
            'tumor_mask':  tumor_mask,     # (1, H, W)
            'name':        image_id,
            'tumor_ratio': torch.tensor(tumor_ratio, dtype=torch.float32),
            'is_tumor':    torch.tensor(self.is_tumor_slice[idx], dtype=torch.bool),
        }


# ============================================================
#  [v6-2] Hard Negative Dataset（邻近肿瘤切片的真难负样本）
# ============================================================

class HardNegativeDataset(Dataset):
    """
    [v6-2] Hard Negative 数据集

    采样策略：从每个肿瘤切片的 ±neg_range 邻域中，
    找出骨区域存在但肿瘤像素为 0 的切片作为负样本。
    这些切片的骨骼结构与肿瘤切片高度相似，
    是 FP 的主要来源，也是最有价值的负样本。

    Parameters
    ----------
    tumor_image_ids : 肿瘤切片 id 列表
    all_image_ids   : 同一 split 的全量 id（用于邻切片查找）
    img_root        : 图像根目录
    neg_range       : 邻域半径（默认 ±5）
    mode            : 'train' | 'val'
    is_16bit        : CT 是否 16bit
    max_per_tumor   : 每个肿瘤切片最多采几个负样本（防数量爆炸）
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
        self.transform = get_augmentation(is_train=(mode == 'train'))

        # 构建邻切片索引
        pmap = build_patient_slice_map(all_image_ids)
        zmap = {}
        for patient, entries in pmap.items():
            for z, iid in entries:
                zmap[(patient, z)] = iid
        self._zmap = zmap

        self.image_list = self._collect_hard_negatives(
            tumor_image_ids, pmap, zmap, neg_range, max_per_tumor)

        print(f"  [HardNeg] {len(self.image_list)} hard negative slices "
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
        if os.path.exists(p):
            return p
        return self._get_path(image_id, '_bone_pred.png')

    def _is_valid_hard_neg(self, image_id):
        """骨区域存在 且 无肿瘤 → 有效负样本"""
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

    def _load_ct3(self, image_id):
        slices = []
        for d in (-1, 0, 1):
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
                p0 = self._get_path(image_id, '_CT.png')
                raw = cv2.imread(p0, cv2.IMREAD_UNCHANGED if self.is_16bit
                                 else cv2.IMREAD_GRAYSCALE)
                s = self.ct_norm.normalize(raw, 'bone') if self.is_16bit \
                    else raw.astype(np.float32) / 255.0
            slices.append(s)
        return np.stack(slices, axis=-1)

    def _load_pet3(self, image_id):
        slices = []
        for d in (-1, 0, 1):
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
        return np.stack(slices, axis=-1)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_id = self.image_list[idx]

        ct3ch  = self._load_ct3(image_id)
        pet3ch = self._load_pet3(image_id)
        bone_p = self._get_bone_path(image_id)
        bone_raw = cv2.imread(bone_p, cv2.IMREAD_GRAYSCALE)
        bone_pred = (bone_raw > 127).astype(np.float32) \
                    if bone_raw is not None else np.zeros((512, 512), np.float32)
        tumor_mask = np.zeros_like(bone_pred)   # 真负样本，肿瘤全 0

        aug = self.transform(
            image      = ct3ch,
            pet3ch     = pet3ch,
            bone_pred  = bone_pred,
            tumor_mask = tumor_mask,
        )
        ct3ch      = aug['image'].float()
        pet3ch     = aug['pet3ch'].float()
        bone_pred  = aug['bone_pred'].float()
        tumor_mask = aug['tumor_mask'].float()

        ct3ch      = torch.clamp(ct3ch, 0, 1)
        pet3ch     = torch.clamp(pet3ch, 0, 1)

        def ensure_3d(t):
            return t.unsqueeze(0) if t.dim() == 2 else t
        bone_pred  = ensure_3d((bone_pred  > 0.5).float())
        tumor_mask = ensure_3d((tumor_mask > 0.5).float())

        # [v6-fix] 不对输入做 bone_pred 遮盖，保留上下文信息
        # tumor_mask 全 0（本来就是负样本），无需额外约束

        return {
            'ct':          ct3ch,
            'pet':         pet3ch,
            'bone_pred':   bone_pred,
            'tumor_mask':  tumor_mask,
            'name':        image_id,
            'tumor_ratio': torch.tensor(0.0, dtype=torch.float32),
            'is_tumor':    torch.tensor(False, dtype=torch.bool),
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
        empty_keep_ratio=0.0,
        all_ids_for_neighbors=image_list,  # [v6] 邻切片索引
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