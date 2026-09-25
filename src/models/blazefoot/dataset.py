"""
dataset.py
──────────
Ultra High-Performance Direct-to-VRAM GPU Dataset Loader for BlazeFoot 4-Keypoint Training.

Optimizations:
  1. Direct-to-VRAM: All training and validation images & targets are preloaded into GPU memory (VRAM).
  2. Zero-Copy Batch Slicing: Batches are indexed directly in GPU VRAM (0 CPU/GPU transfers per epoch).
  3. Scale-Aware Geometrical Anchor Matching: Assigns target objects to anchors matching both location and scale.
  4. Audited 4-Keypoint Indices: [11: toe_tip, 1: heel_back, 3: ball_medial, 4: ball_lateral].
"""

import os
import glob
import math
from typing import Dict, Tuple
import cv2
import numpy as np
import torch

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


def find_split_dirs(data_root: str, split: str) -> Tuple[str, str]:
    """
    Finds (img_dir, lbl_dir) with zero configuration, searching all standard Colab & local paths recursively.
    """
    candidates = [
        os.path.join(data_root, split, "images"),
        os.path.join(data_root, "data", split, "images"),
        os.path.join(data_root, "shuffled_v3", split, "images"),
        os.path.join(data_root, "data", "shuffled_v3", split, "images"),
        os.path.join("data", split, "images"),
        os.path.join("data", "shuffled_v3", split, "images"),
        os.path.join("data", "data", "shuffled_v3", split, "images"),
        os.path.join("/content/Shoes_VTO/data/data/shuffled_v3", split, "images"),
        os.path.join("/content/Shoes_VTO/data/shuffled_v3", split, "images"),
        os.path.join("/content/Shoes_VTO/data", split, "images"),
    ]
    for c in candidates:
        if os.path.exists(c):
            imgs = [p for p in glob.glob(os.path.join(c, "*.*")) if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(imgs) > 0:
                lbl_dir = os.path.join(os.path.dirname(c), "labels")
                return c, lbl_dir

    # Recursive fallback search
    search_roots = [data_root, "data", "."]
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            p_base = os.path.basename(dirpath)
            p_parent = os.path.basename(os.path.dirname(dirpath))
            if p_base == "images" and p_parent in [split, "val" if split == "valid" else "valid" if split == "val" else split]:
                imgs = [p for p in filenames if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(imgs) > 0:
                    lbl_dir = os.path.join(os.path.dirname(dirpath), "labels")
                    return dirpath, lbl_dir

    raise FileNotFoundError(f"Could not locate dataset split '{split}' images anywhere in '{data_root}' or workspace.")


def load_dataset_to_gpu(
    data_root: str,
    split: str = "train",
    img_size: int = 320,
    device: torch.device = torch.device("cuda:0")
) -> Dict[str, any]:
    """
    Loads all images and precomputes anchor targets directly into GPU VRAM.
    """
    img_dir, lbl_dir = find_split_dirs(data_root, split)

    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    img_paths = [p for p in img_paths if p.lower().endswith(('.jpg', '.jpeg', '.png'))]

    N = len(img_paths)
    if N == 0:
        raise FileNotFoundError(f"No images found for split '{split}' in {img_dir}")

    anchors = generate_blaze_anchors(img_size)
    num_anchors = len(anchors)
    anchor_centers = anchors[:, :2].clone()
    anchor_scales  = anchors[:, 2].clone()

    print(f"  [⚡ Direct-VRAM] Loading {N} '{split}' images directly into GPU memory ({device})...")

    images_list = []
    target_cls_list = []
    target_box_list = []
    target_kpt_list = []
    target_mask_list = []
    raw_targets = []

    for p in img_paths:
        im = cv2.imread(p)
        if im is None:
            im = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        h_orig, w_orig = im.shape[:2]
        im_resized = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        im_rgb = cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        im_tensor = torch.from_numpy(im_rgb).permute(2, 0, 1) # [3, H, W]
        images_list.append(im_tensor)

        # Parse labels
        lbl_path = os.path.join(lbl_dir, os.path.splitext(os.path.basename(p))[0] + ".txt")
        t_cls = torch.zeros((num_anchors, 2), dtype=torch.float32)
        t_box = torch.zeros((num_anchors, 4), dtype=torch.float32)
        t_kpt = torch.zeros((num_anchors, 12), dtype=torch.float32)
        t_mask = torch.zeros(num_anchors, dtype=torch.bool)

        raw_boxes = []
        raw_classes = []
        raw_kpts = []

        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_id = int(float(parts[0]))
                    cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                    cx = min(max(cx, 0.0), 1.0)
                    cy = min(max(cy, 0.0), 1.0)
                    bw = min(max(bw, 0.001), 1.0)
                    bh = min(max(bh, 0.001), 1.0)

                    # Pixel coordinates for mAP eval
                    x1_px = (cx - bw / 2.0) * img_size
                    y1_px = (cy - bh / 2.0) * img_size
                    x2_px = (cx + bw / 2.0) * img_size
                    y2_px = (cy + bh / 2.0) * img_size
                    raw_boxes.append([x1_px, y1_px, x2_px, y2_px])
                    raw_classes.append(cls_id)

                    # 4 Audited Keypoints: [11: toe_tip, 1: heel_back, 3: ball_medial, 4: ball_lateral]
                    kpts_norm = []
                    kpts_px = []
                    if len(parts) >= 5 + 48:
                        for target_idx in COARSE_KP_INDICES:
                            kx_n = float(parts[5 + target_idx * 3])
                            ky_n = float(parts[6 + target_idx * 3])
                            kv   = float(parts[7 + target_idx * 3])
                            kpts_norm.extend([kx_n, ky_n, kv])
                            kpts_px.append([kx_n * img_size, ky_n * img_size, kv])
                    else:
                        kpts_norm = [0.0] * 12
                        kpts_px = [[0.0, 0.0, 0.0]] * 4

                    raw_kpts.append(kpts_px)

                    kpts_tensor = torch.tensor(kpts_norm, dtype=torch.float32)
                    box_ctr = torch.tensor([cx, cy], dtype=torch.float32)
                    s_foot = math.sqrt(bw * bh)

                    # Smart Scale-Aware & Geometric Center Matching
                    dists = torch.norm(anchor_centers - box_ctr, dim=1) # [1050]
                    scale_diffs = torch.abs(torch.log(anchor_scales / (s_foot + 1e-6))) # [1050]

                    # Anchors geometrically inside the bounding box
                    in_box = (
                        (anchors[:, 0] >= cx - bw * 0.55) & (anchors[:, 0] <= cx + bw * 0.55) &
                        (anchors[:, 1] >= cy - bh * 0.55) & (anchors[:, 1] <= cy + bh * 0.55)
                    )

                    cost = dists / (s_foot + 1e-4) + 0.75 * scale_diffs
                    cost = torch.where(in_box, cost, cost + 5.0)

                    # Match top 3 best matching anchors for this foot (sharp localization)
                    matched_anchors = torch.topk(cost, 3, largest=False).indices

                    for a_idx in matched_anchors:
                        t_mask[a_idx] = True
                        t_cls[a_idx, cls_id] = 1.0
                        t_box[a_idx] = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
                        t_kpt[a_idx] = kpts_tensor

        target_cls_list.append(t_cls)
        target_box_list.append(t_box)
        target_kpt_list.append(t_kpt)
        target_mask_list.append(t_mask)

        raw_targets.append({
            "boxes": torch.tensor(raw_boxes, dtype=torch.float32, device=device) if raw_boxes else torch.zeros((0, 4), dtype=torch.float32, device=device),
            "classes": torch.tensor(raw_classes, dtype=torch.int64, device=device) if raw_classes else torch.zeros((0,), dtype=torch.int64, device=device),
            "keypoints": torch.tensor(raw_kpts, dtype=torch.float32, device=device) if raw_kpts else torch.zeros((0, 4, 3), dtype=torch.float32, device=device)
        })

    # Stack into contiguous GPU tensors
    all_images = torch.stack(images_list, dim=0).to(device)
    all_cls    = torch.stack(target_cls_list, dim=0).to(device)
    all_box    = torch.stack(target_box_list, dim=0).to(device)
    all_kpt    = torch.stack(target_kpt_list, dim=0).to(device)
    all_mask   = torch.stack(target_mask_list, dim=0).to(device)

    vram_mb = (all_images.element_size() * all_images.nelement() +
               all_cls.element_size() * all_cls.nelement() +
               all_box.element_size() * all_box.nelement() +
               all_kpt.element_size() * all_kpt.nelement() +
               all_mask.element_size() * all_mask.nelement()) / (1024 * 1024)

    print(f"  [✅ Direct-VRAM Ready] {N} samples resident in VRAM ({vram_mb:.1f} MB)\n")

    return {
        "images": all_images,
        "target_cls": all_cls,
        "target_box": all_box,
        "target_kpt": all_kpt,
        "target_mask": all_mask,
        "raw_targets": raw_targets,
        "anchors": anchors.to(device),
        "count": N
    }
