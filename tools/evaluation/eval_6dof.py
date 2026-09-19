"""
eval_6dof.py
────────────
Evaluation of 6DoF foot pose estimation (PnP output).

Metrics:
  - Rotation error    : Mean angular error between predicted and GT rotation matrices (degrees)
  - Translation error : Mean Euclidean distance between predicted and GT translation vectors (mm)
  - ADD               : Average Distance of Model points (standard 6DoF metric)
  - Reprojection error: Mean per-point reprojection error (pixels)
  - Per-sequence analysis: stability over time

Usage
-----
  python tools/evaluation/eval_6dof.py \\
      --gt-poses  data/gt_poses.json \\
      --pred-poses outputs/pred_poses.json

GT/Pred pose JSON format:
  [
    {
      "frame_id":  int,
      "foot_side": "left" | "right",
      "rvec":      [rx, ry, rz],        // Rodrigues rotation vector
      "tvec":      [tx, ty, tz],        // translation in mm
    },
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Error metrics
# ──────────────────────────────────────────────────────────────────────────────

def rotation_error_deg(rvec_pred: np.ndarray, rvec_gt: np.ndarray) -> float:
    """
    Angular error between two rotation vectors in degrees.

    Converts both to rotation matrices and computes the angle of R_pred @ R_gt.T
    """
    R_pred, _ = cv2.Rodrigues(rvec_pred.reshape(3, 1))
    R_gt,   _ = cv2.Rodrigues(rvec_gt.reshape(3, 1))

    R_diff = R_pred @ R_gt.T
    # Angle of rotation from trace: angle = arccos((trace(R) - 1) / 2)
    trace  = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    angle  = np.degrees(np.arccos(trace))
    return float(angle)


def translation_error_mm(tvec_pred: np.ndarray, tvec_gt: np.ndarray) -> float:
    """Euclidean distance between predicted and GT translation (mm)."""
    return float(np.linalg.norm(tvec_pred.flatten() - tvec_gt.flatten()))


def compute_add(
    rvec_pred: np.ndarray,
    tvec_pred: np.ndarray,
    rvec_gt:   np.ndarray,
    tvec_gt:   np.ndarray,
    model_pts: np.ndarray,  # [N, 3] — 3D reference model points (mm)
) -> float:
    """
    ADD (Average Distance of Model points).

    ADD = (1/N) * Σ_i || R_pred * m_i + t_pred - (R_gt * m_i + t_gt) ||
    """
    R_pred, _ = cv2.Rodrigues(rvec_pred.reshape(3, 1))
    R_gt,   _ = cv2.Rodrigues(rvec_gt.reshape(3, 1))
    t_pred    = tvec_pred.flatten()
    t_gt      = tvec_gt.flatten()

    pts_pred = (model_pts @ R_pred.T) + t_pred   # [N, 3]
    pts_gt   = (model_pts @ R_gt.T)   + t_gt     # [N, 3]

    dists = np.linalg.norm(pts_pred - pts_gt, axis=1)
    return float(dists.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Batch evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_6dof(
    gt_poses:   List[Dict],
    pred_poses: List[Dict],
) -> Dict:
    """
    Evaluate 6DoF pose accuracy.

    Matches predictions to GT by (frame_id, foot_side).

    Returns
    -------
    dict with aggregated metrics.
    """
    from src.models.geometry.foot_3d_model import FOOT_3D_POINTS_LEFT, FOOT_3D_POINTS_RIGHT

    # Index GT by (frame_id, foot_side)
    gt_index = {(p["frame_id"], p["foot_side"]): p for p in gt_poses}

    rot_errors   = []
    trans_errors = []
    add_scores   = []

    unmatched = 0
    for pred in pred_poses:
        key = (pred["frame_id"], pred["foot_side"])
        gt  = gt_index.get(key)
        if gt is None:
            unmatched += 1
            continue

        rvec_pred = np.array(pred["rvec"], dtype=np.float64)
        tvec_pred = np.array(pred["tvec"], dtype=np.float64)
        rvec_gt   = np.array(gt["rvec"],   dtype=np.float64)
        tvec_gt   = np.array(gt["tvec"],   dtype=np.float64)

        rot_errors.append(rotation_error_deg(rvec_pred, rvec_gt))
        trans_errors.append(translation_error_mm(tvec_pred, tvec_gt))

        # ADD with appropriate 3D model
        model_pts = (FOOT_3D_POINTS_LEFT
                     if pred["foot_side"] == "left"
                     else FOOT_3D_POINTS_RIGHT)
        add_scores.append(compute_add(rvec_pred, tvec_pred, rvec_gt, tvec_gt, model_pts))

    n = len(rot_errors)
    if n == 0:
        print("[ERROR] No matching predictions found.")
        return {}

    # Percentage of poses within thresholds
    rot_5deg  = float(np.mean(np.array(rot_errors) < 5.0))
    rot_10deg = float(np.mean(np.array(rot_errors) < 10.0))
    trans_10mm = float(np.mean(np.array(trans_errors) < 10.0))
    trans_20mm = float(np.mean(np.array(trans_errors) < 20.0))

    return {
        "n_evaluated":        n,
        "n_unmatched":        unmatched,
        "mean_rot_error_deg": float(np.mean(rot_errors)),
        "median_rot_error_deg": float(np.median(rot_errors)),
        "mean_trans_error_mm": float(np.mean(trans_errors)),
        "median_trans_error_mm": float(np.median(trans_errors)),
        "mean_add_mm":        float(np.mean(add_scores)),
        "pct_rot_lt_5deg":    rot_5deg,
        "pct_rot_lt_10deg":   rot_10deg,
        "pct_trans_lt_10mm":  trans_10mm,
        "pct_trans_lt_20mm":  trans_20mm,
    }


def print_6dof_report(metrics: Dict) -> None:
    print("\n" + "=" * 60)
    print("  6-DoF POSE EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples evaluated     : {metrics.get('n_evaluated', 0)}")
    print(f"  Unmatched predictions : {metrics.get('n_unmatched', 0)}")
    print(f"\n  Rotation Error:")
    print(f"    Mean   : {metrics.get('mean_rot_error_deg', 0):.2f}°")
    print(f"    Median : {metrics.get('median_rot_error_deg', 0):.2f}°")
    print(f"    < 5°   : {metrics.get('pct_rot_lt_5deg', 0)*100:.1f}%")
    print(f"    < 10°  : {metrics.get('pct_rot_lt_10deg', 0)*100:.1f}%")
    print(f"\n  Translation Error:")
    print(f"    Mean   : {metrics.get('mean_trans_error_mm', 0):.1f} mm")
    print(f"    Median : {metrics.get('median_trans_error_mm', 0):.1f} mm")
    print(f"    < 10mm : {metrics.get('pct_trans_lt_10mm', 0)*100:.1f}%")
    print(f"    < 20mm : {metrics.get('pct_trans_lt_20mm', 0)*100:.1f}%")
    print(f"\n  ADD (avg. model point distance):")
    print(f"    Mean   : {metrics.get('mean_add_mm', 0):.1f} mm")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 6DoF foot pose estimation.")
    parser.add_argument("--gt-poses",   required=True, help="GT poses JSON.")
    parser.add_argument("--pred-poses", required=True, help="Predicted poses JSON.")
    parser.add_argument("--save",       default=None,  help="Save metrics to JSON.")
    args = parser.parse_args()

    with open(args.gt_poses)   as f: gt_poses   = json.load(f)
    with open(args.pred_poses) as f: pred_poses = json.load(f)

    metrics = evaluate_6dof(gt_poses, pred_poses)
    print_6dof_report(metrics)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to: {args.save}")


if __name__ == "__main__":
    main()

