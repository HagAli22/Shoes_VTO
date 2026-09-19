"""
foot_dataset.py
───────────────
PyTorch Dataset for foot keypoint detection.

Schema: 18 keypoints (9 per foot, left then right).
Keypoint order per foot:
  0  big_toe
  1  little_toe
  2  near_little_toe
  3  near_big_toe
  4  far_big_toe
  5  far_little_toe
  6  dorsum
  7  heel
  8  upper_heel

Visibility codes (COCO convention):
  0 = not visible / not labeled
  1 = occluded but estimated
  2 = fully visible

Output per sample:
  image      : torch.Tensor [3, H, W]  (256×192, normalised)
  heatmaps   : torch.Tensor [18, H/4, W/4]  (64×48 Gaussian blobs)
  vis_mask   : torch.Tensor [18]  float (0 or 1, used to zero-loss on vis=0)
  keypoints  : torch.Tensor [18, 2]  (x, y) in heatmap coordinate space
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
NUM_KEYPOINTS = 18
INPUT_H, INPUT_W = 256, 192          # model input size
HEATMAP_H, HEATMAP_W = 64, 48       # heatmap size (INPUT / 4)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def generate_gaussian_heatmap(
    heatmap: np.ndarray,
    center_x: float,
    center_y: float,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Draw a 2-D Gaussian blob at (center_x, center_y) into *heatmap* in-place.

    Parameters
    ----------
    heatmap  : np.ndarray shape (H, W), dtype float32 — modified in place.
    center_x : x coordinate in heatmap space.
    center_y : y coordinate in heatmap space.
    sigma    : Gaussian standard deviation in pixels.

    Returns
    -------
    heatmap  : same array, modified in place.
    """
    H, W = heatmap.shape
    size = int(6 * sigma + 1)         # kernel half-width = 3σ
    x0 = int(round(center_x))
    y0 = int(round(center_y))

    # Compute bounding box for the Gaussian kernel
    x_lo = max(0, x0 - size // 2)
    x_hi = min(W, x0 + size // 2 + 1)
    y_lo = max(0, y0 - size // 2)
    y_hi = min(H, y0 + size // 2 + 1)

    if x_lo >= x_hi or y_lo >= y_hi:
        return heatmap  # point is outside heatmap

    # Create meshgrid in local region
    xs = np.arange(x_lo, x_hi) - center_x
    ys = np.arange(y_lo, y_hi) - center_y
    xx, yy = np.meshgrid(xs, ys)
    gaussian = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)

    heatmap[y_lo:y_hi, x_lo:x_hi] = np.maximum(
        heatmap[y_lo:y_hi, x_lo:x_hi], gaussian
    )
    return heatmap


def build_heatmaps(
    keypoints: np.ndarray,       # [18, 3]  (x_img, y_img, vis)
    bbox: Tuple[float, float, float, float],  # (x, y, w, h) in original image
    sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw keypoints to heatmaps after cropping to bbox.

    Returns
    -------
    heatmaps  : np.ndarray [18, HEATMAP_H, HEATMAP_W]
    vis_mask  : np.ndarray [18]  (1.0 if vis > 0 else 0.0)
    kp_hm     : np.ndarray [18, 2]  keypoints in heatmap coordinate space
    """
    bx, by, bw, bh = bbox

    # Scale factors from bbox → input image size → heatmap size
    scale_x = INPUT_W / max(bw, 1.0)
    scale_y = INPUT_H / max(bh, 1.0)
    hm_scale_x = HEATMAP_W / INPUT_W
    hm_scale_y = HEATMAP_H / INPUT_H

    heatmaps = np.zeros((NUM_KEYPOINTS, HEATMAP_H, HEATMAP_W), dtype=np.float32)
    vis_mask = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
    kp_hm    = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float32)

    for k in range(NUM_KEYPOINTS):
        x_img, y_img, vis = keypoints[k]
        if vis == 0:
            continue

        # Translate to bbox-local, scale to input, then to heatmap
        x_local = (x_img - bx) * scale_x * hm_scale_x
        y_local = (y_img - by) * scale_y * hm_scale_y

        kp_hm[k] = [x_local, y_local]
        vis_mask[k] = 1.0

        if 0 <= x_local < HEATMAP_W and 0 <= y_local < HEATMAP_H:
            generate_gaussian_heatmap(heatmaps[k], x_local, y_local, sigma)

    return heatmaps, vis_mask, kp_hm


def crop_and_resize(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    target_h: int = INPUT_H,
    target_w: int = INPUT_W,
) -> np.ndarray:
    """
    Crop the image to bbox (with padding) and resize to (target_h, target_w).

    Parameters
    ----------
    image    : BGR image, np.ndarray [H, W, 3].
    bbox     : (x, y, w, h) in pixel coordinates.
    """
    ih, iw = image.shape[:2]
    bx, by, bw, bh = [int(v) for v in bbox]

    # Clamp to image bounds
    x1 = max(0, bx)
    y1 = max(0, by)
    x2 = min(iw, bx + bw)
    y2 = min(ih, by + bh)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return crop


def normalise(image: np.ndarray) -> np.ndarray:
    """Normalise an RGB float32 image with ImageNet mean/std."""
    return (image.astype(np.float32) / 255.0 - MEAN) / STD


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class FootKeypointDataset(Dataset):
    """
    Dataset for 2-D foot keypoint estimation.

    Loads COCO-format JSON with 18 keypoints per annotation:
      [x0,y0,v0, x1,y1,v1, ..., x17,y17,v17]
    where indices 0–8 are the left foot and 9–17 the right foot.

    Parameters
    ----------
    ann_file  : Path to COCO JSON annotation file.
    image_dir : Root directory of images.
    transforms: Optional callable applied to the dict sample.
    sigma     : Gaussian sigma for heatmap generation.
    use_gt_bbox: If True, use GT bbox from annotation; if False, use whole image.
    """

    def __init__(
        self,
        ann_file: str,
        image_dir: str,
        transforms=None,
        sigma: float = 2.0,
        use_gt_bbox: bool = True,
    ) -> None:
        super().__init__()
        self.image_dir    = image_dir
        self.transforms   = transforms
        self.sigma        = sigma
        self.use_gt_bbox  = use_gt_bbox

        with open(ann_file, "r") as f:
            coco = json.load(f)

        # Build id → filename map
        self._id2file: Dict[int, str] = {
            img["id"]: img["file_name"] for img in coco["images"]
        }
        self._id2size: Dict[int, Tuple[int, int]] = {
            img["id"]: (img["height"], img["width"]) for img in coco["images"]
        }

        # Keep only annotations with at least 1 visible keypoint
        self._anns: List[dict] = [
            ann for ann in coco["annotations"]
            if ann.get("num_keypoints", 0) > 0
        ]

    def __len__(self) -> int:
        return len(self._anns)

    def __getitem__(self, idx: int) -> Dict:
        ann = self._anns[idx]
        image_id = ann["image_id"]

        # ── Load image ──────────────────────────────────────────────────────
        fname = self._id2file[image_id]
        img_path = os.path.join(self.image_dir, fname)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        ih, iw = image_rgb.shape[:2]

        # ── Bounding box ────────────────────────────────────────────────────
        if self.use_gt_bbox and "bbox" in ann:
            bbox = tuple(ann["bbox"])          # (x, y, w, h)
        else:
            bbox = (0.0, 0.0, float(iw), float(ih))

        # ── Keypoints ───────────────────────────────────────────────────────
        raw_kps = ann["keypoints"]             # flat list len = 18*3 = 54
        kps = np.array(raw_kps, dtype=np.float32).reshape(NUM_KEYPOINTS, 3)

        # ── Crop & resize ───────────────────────────────────────────────────
        crop = crop_and_resize(image_rgb, bbox)

        # ── Heatmaps ────────────────────────────────────────────────────────
        heatmaps, vis_mask, kp_hm = build_heatmaps(kps, bbox, sigma=self.sigma)

        sample = {
            "image":      crop,               # np.ndarray [256, 192, 3]  uint8
            "heatmaps":   heatmaps,           # np.ndarray [18, 64, 48]
            "vis_mask":   vis_mask,           # np.ndarray [18]
            "keypoints":  kp_hm,             # np.ndarray [18, 2]  heatmap coords
            "bbox":       np.array(bbox, dtype=np.float32),
            "image_id":   image_id,
            "ann_id":     ann["id"],
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        # ── To tensors ──────────────────────────────────────────────────────
        img_norm = normalise(sample["image"])                              # [H,W,3]
        sample["image"]     = torch.from_numpy(img_norm.transpose(2, 0, 1))  # [3,H,W]
        sample["heatmaps"]  = torch.from_numpy(sample["heatmaps"])
        sample["vis_mask"]  = torch.from_numpy(sample["vis_mask"])
        sample["keypoints"] = torch.from_numpy(sample["keypoints"])

        return sample


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ann",   required=True, help="Path to annotation JSON")
    parser.add_argument("--imgs",  required=True, help="Path to image directory")
    args = parser.parse_args()

    ds = FootKeypointDataset(args.ann, args.imgs)
    sample = ds[0]
    print("image shape   :", sample["image"].shape)       # [3, 256, 192]
    print("heatmaps shape:", sample["heatmaps"].shape)    # [18, 64, 48]
    print("vis_mask      :", sample["vis_mask"])
    print("keypoints     :", sample["keypoints"])

