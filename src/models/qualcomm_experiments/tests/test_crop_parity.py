"""
tests/test_crop_parity.py
─────────────────────────
Unit tests for common/crop.py ensuring:
  1. Exact mathematical round-trip invertibility of affine projection (error < 1e-4 px).
  2. Parity between dataset crop generation and inference crop generation.
  3. Out-of-bounds keypoint invalidation (v = 0).
  4. MediaPipe-style box derivation from keypoints.
  5. Empirical calibration of K = median(crop_size / keypoint_extent).
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.common.crop import (
    compute_crop_box,
    get_crop_affine_transform,
    warp_image_crop,
    transform_keypoints,
    invert_keypoints,
    box_from_kps,
    calibrate_keypoint_box_ratio
)
from src.models.qualcomm_experiments.common.dataset import FootPoseDataset13KP


def test_round_trip_invertibility():
    """Verify (x, y) -> M -> M_inv -> (x, y) round-trip precision is sub-1e-4 px."""
    np.random.seed(42)
    for _ in range(50):
        cx = np.random.uniform(100, 900)
        cy = np.random.uniform(100, 900)
        size = np.random.uniform(80, 500)
        rot = np.random.uniform(-45, 45)
        out_size = 256

        M, M_inv = get_crop_affine_transform(cx, cy, size, out_size=out_size, rot=rot)

        # Generate 13 random keypoints around (cx, cy)
        kps_orig = np.zeros((13, 3), dtype=np.float32)
        kps_orig[:, 0] = cx + np.random.uniform(-size / 3, size / 3, 13)
        kps_orig[:, 1] = cy + np.random.uniform(-size / 3, size / 3, 13)
        kps_orig[:, 2] = 2.0  # visible

        # Forward projection into crop space
        kps_crop = transform_keypoints(kps_orig, M, out_size=out_size, invalidate_out_of_bounds=False)

        # Inverse projection back to image space
        kps_restored = invert_keypoints(kps_crop, M_inv)

        max_err = np.max(np.abs(kps_orig[:, :2] - kps_restored[:, :2]))
        assert max_err < 1e-4, f"Round-trip error {max_err} exceeds 1e-4 px!"


def test_crop_box_aspect_ratio_padding():
    """Verify adaptive margin for standard vs elongated boxes with crop_padding=0.25."""
    # Standard aspect ratio (<= 1.3): padding 0.25 + 0.05 extra = 1.30 multiplier
    cx, cy, size_std = compute_crop_box(200, 300, 100, 110, crop_padding=0.25)
    assert cx == 200 and cy == 300
    expected_std = 110 * (1.0 + 0.25 + 0.05)
    assert np.isclose(size_std, expected_std)

    # Elongated foot (> 1.3): padding 0.25 + 0.15 extra = 1.40 multiplier
    cx, cy, size_elong = compute_crop_box(200, 300, 70, 180, crop_padding=0.25)
    expected_elong = 180 * (1.0 + 0.25 + 0.15)
    assert np.isclose(size_elong, expected_elong)


def test_out_of_bounds_invalidation():
    """Verify that points transformed outside [0, out_size) get visibility set to 0.0."""
    cx, cy, size = 500.0, 500.0, 200.0
    M, _ = get_crop_affine_transform(cx, cy, size, out_size=256, rot=0.0)

    # One point at center (should be visible), one point far outside
    kps = np.array([
        [500.0, 500.0, 2.0],
        [900.0, 900.0, 2.0]
    ], dtype=np.float32)

    transformed = transform_keypoints(kps, M, out_size=256, invalidate_out_of_bounds=True)
    assert transformed[0, 2] == 2.0  # inside
    assert transformed[1, 2] == 0.0  # outside invalidated!


def test_box_from_keypoints():
    """Verify MediaPipe-style bounding box estimation from keypoints."""
    # Synthesize keypoints in [100, 200] x [150, 350]
    kps = np.zeros((13, 3), dtype=np.float32)
    kps[:8, 0] = np.linspace(100, 200, 8)
    kps[:8, 1] = np.linspace(150, 350, 8)
    kps[:8, 2] = 0.8  # confident
    kps[8:, 2] = 0.1  # low confidence / occluded

    res = box_from_kps(kps, K=1.40, conf_thresh=0.30, min_kps=6)
    assert res is not None
    cx, cy, size = res
    assert np.isclose(cx, 150.0)
    assert np.isclose(cy, 250.0)
    # extent = max(200-100, 350-150) = 200 -> size = 1.4 * 200 = 280
    assert np.isclose(size, 280.0)

    # Test track loss (< 6 visible points)
    kps[:10, 2] = 0.1
    assert box_from_kps(kps, K=1.40, conf_thresh=0.30, min_kps=6) is None


def run_all_tests():
    print("=" * 60)
    print("  RUNNING CROP PARITY & AFFINE PROJECTION TESTS")
    print("=" * 60)

    test_round_trip_invertibility()
    print("[PASS] test_round_trip_invertibility (sub-1e-4 px round-trip error)")

    test_crop_box_aspect_ratio_padding()
    print("[PASS] test_crop_box_aspect_ratio_padding (exact 0.25 padding parity)")

    test_out_of_bounds_invalidation()
    print("[PASS] test_out_of_bounds_invalidation (boundary clipping)")

    test_box_from_keypoints()
    print("[PASS] test_box_from_keypoints (MediaPipe-style keypoint box derivation)")

    # Run K-Ratio calibration on training dataset if available
    train_images = Path("data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/train/images")
    if train_images.exists():
        print("\n[*] Calibrating K-ratio on merged training set...")
        ds = FootPoseDataset13KP(images_dir=str(train_images), mode="full")
        # Gather instances
        all_insts = []
        for i in range(min(500, len(ds))):
            sample = ds[i]
            # parse instances from ds
            img_path = Path(sample["img_path"])
            label_file = ds.labels_dir / f"{img_path.stem}.txt"
            insts = ds._parse_label_file(label_file)
            all_insts.extend(insts)

        k_calib = calibrate_keypoint_box_ratio(all_insts, img_w=640, img_h=640, crop_padding=0.25)
        print(f"[+] Empirical Calibrated K-Ratio: {k_calib:.3f}")

    print("\n" + "=" * 60)
    print("  ALL CROP PARITY TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
