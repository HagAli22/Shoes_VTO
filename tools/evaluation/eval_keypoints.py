"""
eval_keypoints.py
─────────────────
Evaluation script for foot keypoint model.

Computes:
  - OKS  (Object Keypoint Similarity) — COCO standard metric
  - PCK@5px  and  PCK@10px  — percentage of correct keypoints within threshold
  - Per-keypoint breakdown  — to identify which of the 9 points are hardest
  - Per-category breakdown  — by foot side (left/right), visibility, etc.

Usage
-----
  # Run evaluation against ground truth and prediction files:
  python tools/evaluation/eval_keypoints.py \\
      --gt   data/annotations/test_annotations.json \\
      --pred outputs/predictions.json \\
      [--threshold 5.0]

  # Or evaluate directly using the ONNX model on test images:
  python tools/evaluation/eval_keypoints.py \\
      --gt        data/annotations/test_annotations.json \\
      --model     outputs/exports/rtmpose_tiny_18kp.onnx \\
      --image-dir data/processed/

Prediction JSON format:
  [
    {
      "image_id":  int,
      "ann_id":    int,
      "keypoints": [x0,y0,c0, x1,y1,c1, ..., x17,y17,c17]   // 54 values
    },
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NUM_KEYPOINTS = 18

KEYPOINT_NAMES = [
    "L_big_toe",        "L_little_toe",     "L_near_little",
    "L_near_big",       "L_far_big",        "L_far_little",
    "L_dorsum",         "L_heel",           "L_upper_heel",
    "R_big_toe",        "R_little_toe",     "R_near_little",
    "R_near_big",       "R_far_big",        "R_far_little",
    "R_dorsum",         "R_heel",           "R_upper_heel",
]

# OKS sigmas (same as training_config.yaml)
OKS_SIGMAS = np.array([
    0.025, 0.025, 0.035, 0.035, 0.040, 0.040, 0.040, 0.030, 0.035,  # left foot
    0.025, 0.025, 0.035, 0.035, 0.040, 0.040, 0.040, 0.030, 0.035,  # right foot
], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# OKS computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_oks(
    pred_kps: np.ndarray,   # [18, 3]  (x, y, conf)
    gt_kps:   np.ndarray,   # [18, 3]  (x, y, vis)
    area:     float,        # bounding box area (pixels²) — for scale normalisation
    sigmas:   np.ndarray = OKS_SIGMAS,
) -> float:
    """
    Compute OKS (Object Keypoint Similarity) for a single annotation.

    OKS = Σ_i [exp(-d_i² / (2 * s² * σ_i²)) * δ(v_i > 0)] / Σ_i δ(v_i > 0)

    where:
      d_i   = Euclidean distance between predicted and GT keypoint i
      s     = object scale = sqrt(area)
      σ_i   = per-keypoint sigma
      v_i   = GT visibility

    Returns
    -------
    float in [0, 1] — OKS score for this annotation (1.0 = perfect match).
    """
    s = np.sqrt(max(area, 1.0))
    visible_mask = gt_kps[:, 2] > 0          # bool [18]

    if not visible_mask.any():
        return 0.0

    gt_xy   = gt_kps[:, :2]                  # [18, 2]
    pred_xy = pred_kps[:, :2]                # [18, 2]

    d_sq = np.sum((pred_xy - gt_xy) ** 2, axis=1)   # [18]
    var  = (2 * s * sigmas) ** 2                      # [18]

    exp_val = np.exp(-d_sq / (var + 1e-9))            # [18]
    exp_val[~visible_mask] = 0.0                       # zero out invisible

    oks = exp_val.sum() / visible_mask.sum()
    return float(oks)


def compute_pck(
    pred_kps:  np.ndarray,   # [18, 3]
    gt_kps:    np.ndarray,   # [18, 3]
    threshold: float = 5.0,  # pixels
) -> Tuple[float, int, int]:
    """
    Compute PCK@threshold for one annotation.

    Returns
    -------
    (pck_rate, n_correct, n_visible)
    """
    visible_mask = gt_kps[:, 2] > 0
    n_visible    = int(visible_mask.sum())
    if n_visible == 0:
        return 0.0, 0, 0

    dist    = np.linalg.norm(pred_kps[:, :2] - gt_kps[:, :2], axis=1)  # [18]
    correct = (dist < threshold) & visible_mask
    n_correct = int(correct.sum())
    return n_correct / n_visible, n_correct, n_visible


# ──────────────────────────────────────────────────────────────────────────────
# Generate predictions from ONNX model
# ──────────────────────────────────────────────────────────────────────────────

def run_model_predictions(
    gt_json:   str,
    model_path: str,
    image_dir:  str,
) -> List[Dict]:
    """
    Run the ONNX keypoint model on all test images and return predictions.
    """
    import cv2
    from src.models.keypoint.keypoint_model import FootKeypointEstimator

    estimator = FootKeypointEstimator(model_path=model_path)

    with open(gt_json) as f:
        coco = json.load(f)

    id2file = {img["id"]: img["file_name"] for img in coco["images"]}
    predictions = []

    for ann in tqdm(coco["annotations"], desc="Predicting"):
        img_id   = ann["image_id"]
        fname    = id2file.get(img_id)
        if fname is None:
            continue

        img_path = os.path.join(image_dir, fname)
        frame    = cv2.imread(img_path)
        if frame is None:
            continue

        bbox = ann.get("bbox", [0, 0, frame.shape[1], frame.shape[0]])
        kps  = estimator.predict(frame, tuple(bbox))   # [18, 3]

        predictions.append({
            "image_id":  img_id,
            "ann_id":    ann["id"],
            "keypoints": kps.flatten().tolist(),
        })

    return predictions


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    gt_json:      str,
    pred_list:    List[Dict],
    pck_threshold: float = 5.0,
) -> Dict:
    """
    Compute all evaluation metrics.

    Returns
    -------
    dict with keys:
      mean_oks, pck_5px, pck_10px,
      per_keypoint_pck  [18],
      per_keypoint_oks  [18]
    """
    with open(gt_json) as f:
        coco = json.load(f)

    gt_map = {ann["id"]: ann for ann in coco["annotations"]}

    oks_scores:      List[float]       = []
    pck_5_scores:    List[float]       = []
    pck_10_scores:   List[float]       = []
    per_kp_correct:  np.ndarray = np.zeros(NUM_KEYPOINTS)
    per_kp_visible:  np.ndarray = np.zeros(NUM_KEYPOINTS)
    per_kp_oks_sum:  np.ndarray = np.zeros(NUM_KEYPOINTS)
    per_kp_oks_cnt:  np.ndarray = np.zeros(NUM_KEYPOINTS)

    for pred in tqdm(pred_list, desc="Evaluating"):
        ann_id  = pred["ann_id"]
        gt_ann  = gt_map.get(ann_id)
        if gt_ann is None:
            continue

        gt_kps   = np.array(gt_ann["keypoints"]).reshape(NUM_KEYPOINTS, 3)
        pred_kps = np.array(pred["keypoints"]).reshape(NUM_KEYPOINTS, 3)
        area     = float(gt_ann.get("area", 1.0))
        if area <= 0:
            bbox = gt_ann.get("bbox", [0, 0, 1, 1])
            area = bbox[2] * bbox[3]

        # OKS
        oks = compute_oks(pred_kps, gt_kps, area)
        oks_scores.append(oks)

        # PCK
        pck5,  nc5,  nv = compute_pck(pred_kps, gt_kps, threshold=5.0)
        pck10, nc10, _  = compute_pck(pred_kps, gt_kps, threshold=10.0)
        pck_5_scores.append(pck5)
        pck_10_scores.append(pck10)

        # Per-keypoint
        for k in range(NUM_KEYPOINTS):
            if gt_kps[k, 2] > 0:
                d = float(np.linalg.norm(pred_kps[k, :2] - gt_kps[k, :2]))
                per_kp_visible[k]  += 1
                per_kp_correct[k]  += int(d < pck_threshold)
                per_kp_oks_sum[k]  += float(np.exp(-d**2 / ((2 * np.sqrt(area) * OKS_SIGMAS[k])**2 + 1e-9)))
                per_kp_oks_cnt[k]  += 1

    # Aggregate
    mean_oks   = float(np.mean(oks_scores))         if oks_scores  else 0.0
    mean_pck5  = float(np.mean(pck_5_scores))       if pck_5_scores else 0.0
    mean_pck10 = float(np.mean(pck_10_scores))      if pck_10_scores else 0.0

    per_kp_pck = np.where(
        per_kp_visible > 0, per_kp_correct / per_kp_visible, 0.0
    )
    per_kp_oks = np.where(
        per_kp_oks_cnt > 0, per_kp_oks_sum / per_kp_oks_cnt, 0.0
    )

    return {
        "mean_oks":       mean_oks,
        "pck_5px":        mean_pck5,
        "pck_10px":       mean_pck10,
        "n_samples":      len(oks_scores),
        "per_keypoint_pck": per_kp_pck.tolist(),
        "per_keypoint_oks": per_kp_oks.tolist(),
    }


def print_report(metrics: Dict) -> None:
    """Print a clean evaluation report to stdout."""
    print("\n" + "=" * 60)
    print("  FOOT KEYPOINT EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples evaluated : {metrics['n_samples']}")
    print(f"  Mean OKS          : {metrics['mean_oks']:.4f}")
    print(f"  PCK @ 5 px        : {metrics['pck_5px']:.4f}  ({metrics['pck_5px']*100:.1f}%)")
    print(f"  PCK @ 10 px       : {metrics['pck_10px']:.4f}  ({metrics['pck_10px']*100:.1f}%)")
    print("-" * 60)
    print("  Per-Keypoint Breakdown:")
    print(f"  {'Keypoint':<22} {'PCK@5px':>8}  {'OKS':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}")
    per_pck = metrics["per_keypoint_pck"]
    per_oks = metrics["per_keypoint_oks"]
    for i, name in enumerate(KEYPOINT_NAMES):
        print(f"  {name:<22} {per_pck[i]:>7.1%}   {per_oks[i]:>7.4f}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate foot keypoint model.")
    parser.add_argument("--gt",        required=True, help="Path to GT COCO JSON.")
    parser.add_argument("--pred",      default=None,  help="Path to predictions JSON.")
    parser.add_argument("--model",     default=None,  help="ONNX model path (runs inference).")
    parser.add_argument("--image-dir", default=None,  help="Image directory (needed with --model).")
    parser.add_argument("--threshold", type=float, default=5.0, help="PCK threshold (pixels).")
    parser.add_argument("--save",      default=None,  help="Save results to JSON path.")
    args = parser.parse_args()

    # ── Get predictions ────────────────────────────────────────────────────
    if args.pred:
        with open(args.pred) as f:
            pred_list = json.load(f)
    elif args.model and args.image_dir:
        pred_list = run_model_predictions(args.gt, args.model, args.image_dir)
    else:
        parser.error("Provide either --pred or both --model and --image-dir.")

    # ── Evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate(args.gt, pred_list, pck_threshold=args.threshold)
    print_report(metrics)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to: {args.save}")


if __name__ == "__main__":
    main()

