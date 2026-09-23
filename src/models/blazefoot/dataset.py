"""
dataset.py
──────────
Ultra High-Performance PyTorch Dataset and Anchor Matching for BlazeFoot 4-Keypoint Training.

Optimizations:
  1. Full RAM Caching: Images and labels are parsed into RAM once at startup (Zero disk I/O during training).
  2. Precomputed Validation Tensors: Validation dataset is 100% pre-transformed and tensorized in memory for instant validation (<1ms).
  3. Audited 4-Keypoint Indices: [11: toe_tip, 1: heel_back, 3: ball_medial, 4: ball_lateral].
  4. Keypoint-Safe Geometric Augmentations (Rotation, scaling, color jitter).
"""

import os
import glob
from typing import List, Tuple, Dict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Audited 4 Coarse Keypoint Indices from 16-KP Frozen Contract:
# [toe_tip=11, heel_back=1, ball_medial=3, ball_lateral=4]
COARSE_KP_INDICES = [11, 1, 3, 4]


def generate_blaze_anchors(img_size: int = 320) -> torch.Tensor:
    """
    Generates anchor grid centers [1050, 4] (cx, cy, base_w, base_h) normalized to [0, 1].
    Scales:
      P3: 20x20, stride 16 (anchor base scales: 0.12, 0.22) -> 800 anchors
      P4: 10x10, stride 32 (anchor base scales: 0.35, 0.50) -> 200 anchors
      P5:  5x5,  stride 64 (anchor base scales: 0.65, 0.85) -> 50 anchors
    Total Anchors = 800 + 200 + 50 = 1050 anchors.
    """
    grid_configs = [
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
    def __init__(self, data_root: str, split: str = "train", img_size: int = 320, is_train: bool = True, cache_ram: bool = True):
        self.img_size = img_size
        self.is_train = is_train
        self.cache_ram = cache_ram

        # Smart directory resolution
        resolved_root = data_root
        candidate_img_dir = os.path.join(resolved_root, split, "images")
        if not os.path.exists(candidate_img_dir) or len(glob.glob(os.path.join(candidate_img_dir, "*.*"))) == 0:
            for alt_sub in ["shuffled_v3", "data/shuffled_v3", os.path.join("..", split, "images")]:
                alt_dir = os.path.join(resolved_root, alt_sub, split, "images") if ".." not in alt_sub else os.path.normpath(os.path.join(resolved_root, alt_sub))
                if os.path.exists(alt_dir) and len(glob.glob(os.path.join(alt_dir, "*.*"))) > 0:
                    candidate_img_dir = alt_dir
                    resolved_root = os.path.dirname(os.path.dirname(alt_dir))
                    break

        self.img_dir = os.path.join(resolved_root, split, "images") if os.path.exists(os.path.join(resolved_root, split, "images")) else candidate_img_dir
        self.lbl_dir = os.path.join(os.path.dirname(self.img_dir), "labels")

        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        self.img_paths = [p for p in self.img_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

        self.anchors = generate_blaze_anchors(img_size)
        self.anchor_centers = self.anchors[:, :2].clone()
        self.num_anchors = len(self.anchors)

        # Setup transforms
        if is_train:
            self.transform = A.Compose([
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4, hue=0.015, p=0.7),
                A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.10, rotate_limit=15, p=0.6, border_mode=cv2.BORDER_CONSTANT),
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

        # Pre-cache all images and annotations into RAM
        self.cached_images = []
        self.cached_labels = []
        self.precomputed_val_tensors = []

        if len(self.img_paths) > 0:
            print(f"  [RAM Cache] Preloading & parsing {len(self.img_paths)} '{split}' images + labels into memory...")
            for idx, p in enumerate(self.img_paths):
                im = cv2.imread(p)
                if im is not None:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                else:
                    im = np.zeros((img_size, img_size, 3), dtype=np.uint8)

                h_orig, w_orig = im.shape[:2]
                self.cached_images.append(im)

                # Parse label once
                lbl_path = os.path.join(self.lbl_dir, os.path.splitext(os.path.basename(p))[0] + ".txt")
                boxes_yolo = []
                category_ids = []
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

                            # 4 Audited Keypoints
                            kps_4 = []
                            if len(parts) >= 5 + 48:
                                for target_idx in COARSE_KP_INDICES:
                                    kx = float(parts[5 + target_idx * 3]) * w_orig
                                    ky = float(parts[6 + target_idx * 3]) * h_orig
                                    kv = int(float(parts[7 + target_idx * 3]))
                                    kps_4.append((kx, ky, kv))
                            else:
                                for _ in range(4):
                                    kps_4.append((0.0, 0.0, 0))
                            kpts_4_list.append(kps_4)

                self.cached_labels.append({
                    "boxes_yolo": boxes_yolo,
                    "category_ids": category_ids,
                    "kpts_4_list": kpts_4_list,
                    "h_orig": h_orig,
                    "w_orig": w_orig
                })

                # If validation dataset, pre-generate all output tensors for zero-cost validation
                if not is_train:
                    item_dict = self._process_sample(im, boxes_yolo, category_ids, kpts_4_list, h_orig, w_orig)
                    self.precomputed_val_tensors.append(item_dict)

    def _process_sample(self, img, boxes_yolo, category_ids, kpts_4_list, h_orig, w_orig) -> Dict[str, torch.Tensor]:
        flat_kpts_xy = []
        kpt_visibilities = []
        if len(kpts_4_list) > 0:
            for inst_kps in kpts_4_list:
                for (kx, ky, kv) in inst_kps:
                    flat_kpts_xy.append((kx, ky))
                    kpt_visibilities.append(kv)
        else:
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
            kpts_aug_xy = [(k[0]*self.img_size/max(w_orig, 1), k[1]*self.img_size/max(h_orig, 1)) for k in flat_kpts_xy]

        target_cls = torch.zeros((self.num_anchors, 2), dtype=torch.float32)
        target_box = torch.zeros((self.num_anchors, 4), dtype=torch.float32)
        target_kpt = torch.zeros((self.num_anchors, 12), dtype=torch.float32)
        target_mask = torch.zeros(self.num_anchors, dtype=torch.bool)

        if len(boxes_aug) > 0:
            for inst_idx, (b_box, c_id) in enumerate(zip(boxes_aug, cats_aug)):
                cx, cy, bw, bh = b_box
                box_ctr = torch.tensor([cx, cy], dtype=torch.float32)
                dists = torch.norm(self.anchor_centers - box_ctr, dim=1)
                
                closest_anchor_idxs = torch.topk(dists, 6, largest=False).indices

                inst_kpts_norm = []
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

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.is_train and len(self.precomputed_val_tensors) > idx:
            return self.precomputed_val_tensors[idx]

        img = self.cached_images[idx]
        lbl = self.cached_labels[idx]

        return self._process_sample(
            img,
            lbl["boxes_yolo"],
            lbl["category_ids"],
            lbl["kpts_4_list"],
            lbl["h_orig"],
            lbl["w_orig"]
        )


def get_blazefoot_loaders(
    data_root: str = "data/shuffled_v3",
    batch_size: int = 128,
    num_workers: int = 4,
    cache_ram: bool = True
) -> Tuple[DataLoader, DataLoader]:
    train_ds = BlazeFootDataset(data_root, split="train", img_size=320, is_train=True, cache_ram=cache_ram)
    val_ds   = BlazeFootDataset(data_root, split="valid", img_size=320, is_train=False, cache_ram=cache_ram)

    train_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    val_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": 0, # Precomputed in RAM
        "pin_memory": torch.cuda.is_available(),
    }

    if num_workers > 0:
        train_kwargs["persistent_workers"] = True
        train_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader   = DataLoader(val_ds, **val_kwargs)

    return train_loader, val_loader
