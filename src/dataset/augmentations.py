"""
augmentations.py
────────────────
Albumentations-based augmentation pipeline for foot keypoint training.

Key design decisions:
  - Horizontal flip is FOOT-AWARE: it swaps left↔right keypoint labels so the
    model always gets correct supervision after mirroring.
  - All geometric transforms operate on both the image and the raw keypoints
    (before heatmap generation) to maintain exact ground truth.
  - Separate pipelines for train (heavy aug) and val (resize only).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import albumentations as A
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Keypoint swap indices for horizontal flip
# ──────────────────────────────────────────────────────────────────────────────
# Left foot = indices 0–8,  Right foot = indices 9–17
# After a horizontal flip, left↔right roles exchange.
FLIP_SWAP_PAIRS: List[Tuple[int, int]] = [
    (0, 9),   # left_big_toe        ↔ right_big_toe
    (1, 10),  # left_little_toe     ↔ right_little_toe
    (2, 11),  # left_near_little    ↔ right_near_little
    (3, 12),  # left_near_big       ↔ right_near_big
    (4, 13),  # left_far_big        ↔ right_far_big
    (5, 14),  # left_far_little     ↔ right_far_little
    (6, 15),  # left_dorsum         ↔ right_dorsum
    (7, 16),  # left_heel           ↔ right_heel
    (8, 17),  # left_upper_heel     ↔ right_upper_heel
]

NUM_KEYPOINTS = 18


# ──────────────────────────────────────────────────────────────────────────────
# Foot-aware horizontal flip helper
# ──────────────────────────────────────────────────────────────────────────────

def apply_foot_aware_flip(sample: Dict) -> Dict:
    """
    Flip image horizontally and swap left↔right keypoint labels.

    Works on the raw numpy sample dict (before tensor conversion):
      sample["image"]     : np.ndarray [H, W, 3]  uint8
      sample["heatmaps"]  : np.ndarray [18, Hm, Wm]
      sample["vis_mask"]  : np.ndarray [18]
      sample["keypoints"] : np.ndarray [18, 2]  (x, y) in heatmap coords
    """
    img = sample["image"]
    H, W = img.shape[:2]

    # Flip image
    sample["image"] = np.fliplr(img).copy()

    # Flip heatmaps horizontally and swap L↔R channels
    hm = sample["heatmaps"]              # [18, Hm, Wm]
    hm_flipped = hm[:, :, ::-1].copy()  # flip each heatmap
    for l_idx, r_idx in FLIP_SWAP_PAIRS:
        hm_flipped[l_idx], hm_flipped[r_idx] = hm_flipped[r_idx].copy(), hm_flipped[l_idx].copy()
    sample["heatmaps"] = hm_flipped

    # Swap vis_mask
    vm = sample["vis_mask"].copy()
    for l_idx, r_idx in FLIP_SWAP_PAIRS:
        vm[l_idx], vm[r_idx] = vm[r_idx], vm[l_idx]
    sample["vis_mask"] = vm

    # Flip keypoint x-coordinates and swap L↔R
    kp = sample["keypoints"].copy()       # [18, 2]
    Wm = hm.shape[2]
    kp[:, 0] = Wm - 1 - kp[:, 0]
    for l_idx, r_idx in FLIP_SWAP_PAIRS:
        kp[[l_idx, r_idx]] = kp[[r_idx, l_idx]]
    sample["keypoints"] = kp

    return sample


# ──────────────────────────────────────────────────────────────────────────────
# Pixel-level (image-only) transforms via Albumentations
# ──────────────────────────────────────────────────────────────────────────────

def get_pixel_train_transforms(
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    gaussian_noise_std: float = 0.01,
) -> A.Compose:
    """
    Pixel-level augmentations that do NOT change keypoint positions.
    Safe to apply after heatmaps are already generated.
    """
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.6,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.4,
        ),
        A.GaussNoise(
            var_limit=(0.0, (gaussian_noise_std * 255) ** 2),
            p=0.3,
        ),
        A.MotionBlur(blur_limit=3, p=0.15),
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.2),
    ])


def get_pixel_val_transforms() -> Optional[A.Compose]:
    """No pixel augmentation for validation."""
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Combined sample-level transform callables
# ──────────────────────────────────────────────────────────────────────────────

class TrainTransform:
    """
    Full training augmentation pipeline.

    Applies to the raw sample dict BEFORE heatmap generation.
    Works directly on:
      sample["image"]    : np.ndarray [256, 192, 3]
      sample["heatmaps"] : np.ndarray [18, 64, 48]
      sample["vis_mask"] : np.ndarray [18]
      sample["keypoints"]: np.ndarray [18, 2]

    Parameters
    ----------
    flip_prob           : Probability of horizontal flip.
    brightness_limit    : ±fraction for brightness jitter.
    contrast_limit      : ±fraction for contrast jitter.
    gaussian_noise_std  : Std of additive Gaussian noise (0–1 range).
    rotation_limit      : Max rotation in degrees (±).
    scale_range         : (min_scale, max_scale) relative to bbox.
    """

    def __init__(
        self,
        flip_prob: float = 0.5,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        gaussian_noise_std: float = 0.01,
        rotation_limit: float = 15.0,
        scale_range: Tuple[float, float] = (0.8, 1.2),
    ) -> None:
        self.flip_prob = flip_prob
        self._pixel_aug = get_pixel_train_transforms(
            brightness_limit, contrast_limit, gaussian_noise_std
        )

    def __call__(self, sample: Dict) -> Dict:
        # 1. Foot-aware horizontal flip
        if np.random.random() < self.flip_prob:
            sample = apply_foot_aware_flip(sample)

        # 2. Pixel-level augmentations (image only — heatmaps unaffected)
        img = sample["image"]
        aug_result = self._pixel_aug(image=img)
        sample["image"] = aug_result["image"]

        return sample


class ValTransform:
    """No-op transform for validation (identity)."""

    def __call__(self, sample: Dict) -> Dict:
        return sample


# ──────────────────────────────────────────────────────────────────────────────
# Factory functions (used in training script)
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transforms(**kwargs) -> TrainTransform:
    """Return the training transform pipeline."""
    return TrainTransform(**kwargs)


def get_val_transforms() -> ValTransform:
    """Return the validation transform pipeline."""
    return ValTransform()

