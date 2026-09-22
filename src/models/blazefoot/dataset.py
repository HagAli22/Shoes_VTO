"""
dataset.py
──────────
PyTorch Dataset and Anchor Matching for BlazeFoot training.
PyTorch Dataset and Anchor Matching for BlazeFoot 4-Keypoint Training.
High-Performance PyTorch Dataset and Anchor Matching for BlazeFoot 4-Keypoint Training.

Loads annotations from data/shuffled_v3 (YOLO-Pose format):
  Line format: cls cx cy w h kx0 ky0 kv0 ... kx15 ky15 kv15
Extracts only the 4 Coarse Keypoints from the 16 annotated landmarks:
  - Keypoint 0: toe_tip
  - Keypoint 2: heel_back
  - Keypoint 4: ball_medial
  - Keypoint 5: ball_lateral
Features:
  - In-Memory RAM Caching (--cache_ram): Preloads all images into RAM for 100% GPU saturation on Colab/Servers.
  - Multi-worker parallel processing with pin_memory.
  - 4 Coarse Keypoints: toe_tip (0), heel_back (2), ball_medial (4), ball_lateral (5).
"""

import os
import glob
import math
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# The 4 coarse keypoints indices in the 16-keypoint annotation order
COARSE_KP_INDICES = [0, 2, 4, 5]


def generate_blaze_anchors(img_size: int = 320, num_anchors_per_scale: int = 2) -> torch.Tensor:
    """
    Generates anchor grid centers [N, 4] (cx, cy, base_w, base_h) normalized to [0, 1].
    Scales:
      P3: 20x20, stride 16 (anchor base scale: 0.15, 0.25)
      P4: 10x10, stride 32 (anchor base scale: 0.35, 0.50)
      P5:  5x5,  stride 64 (anchor base scale: 0.65, 0.85)
    Total Anchors = (20*20 + 10*10 + 5*5) * 2 = (400 + 100 + 25) * 2 = 1050 anchors.
    """
    grid_configs = [
        (20, [0.12, 0.22]), # P3
        (10, [0.35, 0.50]), # P4
        (5,  [0.65, 0.85])  # P5
        (20, [0.12, 0.22]), # P3 (20x20) -> 800 anchors
        (10, [0.35, 0.50]), # P4 (10x10) -> 200 anchors
        (5,  [0.65, 0.85])  # P5 (5x5)   -> 50 anchors
    ]
    anchors = []

    for grid_size, scales in grid_configs:
        stride = 1.0 / grid_size
        for i in range(grid_size):
            cy = (i + 0.5) * stride
            for j in range(grid_size):
                cx = (j + 0.5) * stride
                for s in scales:
                    anchors.append([cx, cy, s, s])

    return torch.tensor(anchors, dtype=torch.float32)


class BlazeFootDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", img_size: int = 320, is_train: bool = True):
    def __init__(self, data_root: str, split: str = "train", img_size: int = 320, is_train: bool = True, cache_ram: bool = True):
        self.img_size = img_size
        self.is_train = is_train
        self.cache_ram = cache_ram
        self.img_dir = os.path.join(data_root, split, "images")
        self.lbl_dir = os.path.join(data_root, split, "labels")

        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.img_paths = [p for p in self.img_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

        self.anchors = generate_blaze_anchors(img_size)
        self.ram_cache = {}

        # Albumentations pipeline
        # Preload images into RAM if requested
        if self.cache_ram:
            print(f"  [RAM Cache] Preloading {len(self.img_paths)} '{split}' images into memory...")
            for idx, p in enumerate(self.img_paths):
                im = cv2.imread(p)
                if im is not None:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                else:
                    im = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                self.ram_cache[idx] = im

        if is_train:
            self.transform = A.Compose([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.015, p=0.7),
                A.HorizontalFlip(p=0.5),
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
               bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
                ToTensorV2()
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
               bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_ram and idx in self.ram_cache:
            img = self.ram_cache[idx].copy()
        else:
            img_path = self.img_paths[idx]
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        img_path = self.img_paths[idx]
        lbl_path = os.path.join(self.lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]

        # Parse labels
        boxes_yolo = []
        category_ids = []
        kpts_list = []
        kpts_4_list = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx, cy, w, h = [float(v) for v in parts[1:5]]
                    cx = min(max(cx, 0.0), 1.0)
                    cy = min(max(cy, 0.0), 1.0)
                    w  = min(max(w, 0.001), 1.0)
                    h  = min(max(h, 0.001), 1.0)

                    boxes_yolo.append([cx, cy, w, h])
                    category_ids.append(cls_id)

                    kps = []
                    # Extract strictly the 4 Coarse Keypoints: [0, 2, 4, 5]
                    kps_4 = []
                    if len(parts) >= 5 + 48:
                        for ki in range(16):
                            kx = float(parts[5 + ki*3]) * w_orig
                            ky = float(parts[6 + ki*3]) * h_orig
                            kv = int(float(parts[7 + ki*3]))
                            kps.append((kx, ky, kv))
                        for target_idx in COARSE_KP_INDICES:
                            kx = float(parts[5 + target_idx*3]) * w_orig
                            ky = float(parts[6 + target_idx*3]) * h_orig
                            kv = int(float(parts[7 + target_idx*3]))
                            kps_4.append((kx, ky, kv))
                    else:
                        for _ in range(16):
                            kps.append((0.0, 0.0, 0))
                    kpts_list.append(kps)
                        for _ in range(4):
                            kps_4.append((0.0, 0.0, 0))
                    kpts_4_list.append(kps_4)

        flat_kpts_xy = []
        kpt_visibilities = []
        if len(kpts_list) > 0:
            for inst_kps in kpts_list:
        if len(kpts_4_list) > 0:
            for inst_kps in kpts_4_list:
                for (kx, ky, kv) in inst_kps:
                    flat_kpts_xy.append((kx, ky))
                    kpt_visibilities.append(kv)
        else:
            flat_kpts_xy = [(0.0, 0.0)] * 16
            kpt_visibilities = [0] * 16
            flat_kpts_xy = [(0.0, 0.0)] * 4
            kpt_visibilities = [0] * 4

        try:
            transformed = self.transform(
                image=img,
                bboxes=boxes_yolo,
                category_ids=category_ids,
                keypoints=flat_kpts_xy
            )
            img_tensor = transformed['image']
            boxes_aug = transformed['bboxes']
            cats_aug = transformed['category_ids']
            kpts_aug_xy = transformed['keypoints']
        except Exception:
            img_resized = cv2.resize(img, (self.img_size, self.img_size)).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1)
            boxes_aug = boxes_yolo
            cats_aug = category_ids
            kpts_aug_xy = [(k[0]*self.img_size/w_orig, k[1]*self.img_size/h_orig) for k in flat_kpts_xy]

        num_anchors = len(self.anchors)
        target_cls = torch.zeros((num_anchors, 2), dtype=torch.float32)
        target_box = torch.zeros((num_anchors, 4), dtype=torch.float32)
        target_kpt = torch.zeros((num_anchors, 16 * 3), dtype=torch.float32)
        target_kpt = torch.zeros((num_anchors, 4 * 3), dtype=torch.float32) # 4 Keypoints * 3 = 12 channels
        target_kpt = torch.zeros((num_anchors, 4 * 3), dtype=torch.float32)
        target_mask = torch.zeros(num_anchors, dtype=torch.bool)

        if len(boxes_aug) > 0:
            for inst_idx, (b_box, c_id) in enumerate(zip(boxes_aug, cats_aug)):
                cx, cy, bw, bh = b_box
                anchor_centers = self.anchors[:, :2]
                dists = torch.norm(anchor_centers - torch.tensor([cx, cy]), dim=1)
                
                topk = 6
                closest_anchor_idxs = torch.topk(dists, topk, largest=False).indices

                inst_kpts_norm = []
                inst_offset = inst_idx * 16
                for ki in range(16):
                inst_offset = inst_idx * 4
                for ki in range(4):
                    k_idx = inst_offset + ki
                    if k_idx < len(kpts_aug_xy):
                        kx_px, ky_px = kpts_aug_xy[k_idx]
                        kv = kpt_visibilities[k_idx] if k_idx < len(kpt_visibilities) else 0
                        kx_norm = min(max(kx_px / self.img_size, 0.0), 1.0)
                        ky_norm = min(max(ky_px / self.img_size, 0.0), 1.0)
                        inst_kpts_norm.extend([kx_norm, ky_norm, float(kv)])
                    else:
                        inst_kpts_norm.extend([0.0, 0.0, 0.0])

                inst_kpts_tensor = torch.tensor(inst_kpts_norm, dtype=torch.float32)

                for a_idx in closest_anchor_idxs:
                    target_mask[a_idx] = True
                    target_cls[a_idx, int(c_id)] = 1.0
                    target_box[a_idx] = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
                    target_kpt[a_idx] = inst_kpts_tensor

        return {
            "image": img_tensor,
            "target_cls": target_cls,
            "target_box": target_box,
            "target_kpt": target_kpt,
            "target_mask": target_mask
        }


def get_blazefoot_loaders(data_root: str = "data/shuffled_v3", batch_size: int = 16, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    train_ds = BlazeFootDataset(data_root, split="train", img_size=320, is_train=True)
    val_ds   = BlazeFootDataset(data_root, split="valid", img_size=320, is_train=False)
def get_blazefoot_loaders(
    data_root: str = "data/shuffled_v3",
    batch_size: int = 64,
    num_workers: int = 4,
    cache_ram: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_ds = BlazeFootDataset(data_root, split="train", img_size=320, is_train=True, cache_ram=cache_ram)
    val_ds   = BlazeFootDataset(data_root, split="valid", img_size=320, is_train=False, cache_ram=cache_ram)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    val_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        train_kwargs["persistent_workers"] = True
        train_kwargs["prefetch_factor"] = 2
        val_kwargs["persistent_workers"] = True
        val_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader   = DataLoader(val_ds, **val_kwargs)

    return train_loader, val_loader

