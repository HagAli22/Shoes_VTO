"""
metrics.py
──────────
YOLO-Style Fast Validation Metrics for Google BlazeFoot Detector.

Computes COCO-Standard & YOLOv8-Style:
  1. Box Metrics: Precision(B), Recall(B), mAP50(B), mAP50-95(B)
  2. Pose/Keypoint Metrics: Precision(P), Recall(P), mAP50(P), mAP50-95(P) (via OKS)
  3. Class Breakdown: Per-class (left_foot, right_foot) and overall 'all'.
"""

import math
from typing import Dict, List, Tuple
import numpy as np
import torch
import torchvision.ops as ops


# Foot-scale keypoint sigmas relative to foot bbox
FOOT_KP_SIGMAS = np.array([0.10, 0.10, 0.10, 0.10], dtype=np.float32)


def box_iou_batch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes pairwise IoU between boxes1 [N, 4] and boxes2 [M, 4] in (x1, y1, x2, y2).
    """
    return ops.box_iou(boxes1, boxes2)


def compute_oks_matrix(
    pred_kpts: torch.Tensor, # [N, 4, 3] (x, y, conf) in pixel scale [0..320]
    tgt_kpts: torch.Tensor,  # [M, 4, 3] (x, y, vis) in pixel scale [0..320]
    tgt_boxes: torch.Tensor, # [M, 4] (x1, y1, x2, y2) in pixel scale
    sigmas: torch.Tensor     # [4]
) -> torch.Tensor:
    """
    Computes Pairwise Object Keypoint Similarity (OKS) matrix [N, M].
    OKS = sum(exp(-d_i^2 / (2 * s^2 * sigma_i^2)) * v_i) / sum(v_i)
    """
    N = pred_kpts.shape[0]
    M = tgt_kpts.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=pred_kpts.device)

    # Scale s = sqrt(w * h)
    tgt_w = (tgt_boxes[:, 2] - tgt_boxes[:, 0]).clamp(min=1.0)
    tgt_h = (tgt_boxes[:, 3] - tgt_boxes[:, 1]).clamp(min=1.0)
    s = torch.sqrt(tgt_w * tgt_h) # [M]
    scale_factor = 2.0 * (s[:, None] ** 2) * (sigmas[None, :] ** 2) # [M, 4]

    oks_mat = torch.zeros((N, M), device=pred_kpts.device)

    # pred_kpts: [N, 4, 2], tgt_kpts: [M, 4, 2]
    p_xy = pred_kpts[:, :, :2] # [N, 4, 2]
    t_xy = tgt_kpts[:, :, :2]  # [M, 4, 2]
    t_vis = (tgt_kpts[:, :, 2] > 0).float() # [M, 4]

    for n in range(N):
        # dx, dy: [M, 4]
        diff = p_xy[n:n+1, :, :] - t_xy # [M, 4, 2]
        d_sq = (diff ** 2).sum(dim=-1)   # [M, 4]
        e = d_sq / (scale_factor + 1e-7) # [M, 4]
        exp_term = torch.exp(-e) * t_vis # [M, 4]
        vis_sum = t_vis.sum(dim=-1).clamp(min=1.0) # [M]
        oks_mat[n] = exp_term.sum(dim=-1) / vis_sum

    return oks_mat


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Computes Average Precision (AP) using COCO 101-point interpolation.
    """
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 101-point interpolation
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)
    return float(ap)


def evaluate_detections(
    predictions: List[Dict], # Per-image list of predictions: [{"boxes", "scores", "classes", "keypoints"}]
    targets: List[Dict],     # Per-image list of targets: [{"boxes", "classes", "keypoints"}]
    iou_thresholds: np.ndarray = np.linspace(0.50, 0.95, 10),
    num_classes: int = 2
) -> Dict[str, float]:
    """
    Evaluates predictions vs targets across all validation images.
    Returns YOLO-style metrics for boxes and pose keypoints.
    """
    sigmas = torch.tensor(FOOT_KP_SIGMAS, dtype=torch.float32)

    # Class accumulators
    # For each class: list of (correct_box_matrix [N, 10], correct_kpt_matrix [N, 10], confidences [N])
    stats = {c: {"box_tp": [], "kpt_tp": [], "conf": [], "n_gt": 0} for c in range(num_classes)}

    for pred, tgt in zip(predictions, targets):
        p_boxes = pred["boxes"]      # [N, 4] (x1, y1, x2, y2)
        p_scores = pred["scores"]    # [N]
        p_classes = pred["classes"]  # [N]
        p_kpts = pred["keypoints"]   # [N, 4, 3]

        t_boxes = tgt["boxes"]      # [M, 4]
        t_classes = tgt["classes"]  # [M]
        t_kpts = tgt["keypoints"]   # [M, 4, 3]

        device = p_boxes.device
        sigmas = sigmas.to(device)

        for c in range(num_classes):
            p_idx = (p_classes == c).nonzero(as_tuple=True)[0]
            t_idx = (t_classes == c).nonzero(as_tuple=True)[0]

            n_p = len(p_idx)
            n_t = len(t_idx)
            stats[c]["n_gt"] += n_t

            if n_p == 0:
                continue

            c_p_boxes = p_boxes[p_idx]
            c_p_scores = p_scores[p_idx]
            c_p_kpts = p_kpts[p_idx]

            # Sort predictions by score descending
            sort_ord = torch.argsort(c_p_scores, descending=True)
            c_p_boxes = c_p_boxes[sort_ord]
            c_p_scores = c_p_scores[sort_ord]
            c_p_kpts = c_p_kpts[sort_ord]

            stats[c]["conf"].append(c_p_scores.cpu().numpy())

            if n_t == 0:
                stats[c]["box_tp"].append(np.zeros((n_p, len(iou_thresholds)), dtype=bool))
                stats[c]["kpt_tp"].append(np.zeros((n_p, len(iou_thresholds)), dtype=bool))
                continue

            c_t_boxes = t_boxes[t_idx]
            c_t_kpts = t_kpts[t_idx]

            # Box IoU Match
            iou_mat = box_iou_batch(c_p_boxes, c_t_boxes) # [N_p, N_t]
            # OKS Match
            oks_mat = compute_oks_matrix(c_p_kpts, c_t_kpts, c_t_boxes, sigmas) # [N_p, N_t]

            box_tp = np.zeros((n_p, len(iou_thresholds)), dtype=bool)
            kpt_tp = np.zeros((n_p, len(iou_thresholds)), dtype=bool)

            # Match at each threshold
            for t_i, thresh in enumerate(iou_thresholds):
                # Box matching
                matched_gt_b = set()
                for p_i in range(n_p):
                    best_iou, best_t = 0.0, -1
                    for t_j in range(n_t):
                        if t_j in matched_gt_b:
                            continue
                        iou_val = float(iou_mat[p_i, t_j])
                        if iou_val >= thresh and iou_val > best_iou:
                            best_iou = iou_val
                            best_t = t_j
                    if best_t >= 0:
                        matched_gt_b.add(best_t)
                        box_tp[p_i, t_i] = True

                # Pose / Keypoint matching (OKS)
                matched_gt_k = set()
                for p_i in range(n_p):
                    best_oks, best_t = 0.0, -1
                    for t_j in range(n_t):
                        if t_j in matched_gt_k:
                            continue
                        oks_val = float(oks_mat[p_i, t_j])
                        if oks_val >= thresh and oks_val > best_oks:
                            best_oks = oks_val
                            best_t = t_j
                    if best_t >= 0:
                        matched_gt_k.add(best_t)
                        kpt_tp[p_i, t_i] = True

            stats[c]["box_tp"].append(box_tp)
            stats[c]["kpt_tp"].append(kpt_tp)

    # Compute AP and summary metrics per class and overall
    class_results = {}
    all_box_ap50, all_box_ap, all_kpt_ap50, all_kpt_ap = [], [], [], []
    all_p_b, all_r_b, all_p_k, all_r_k = [], [], [], []
    class_names = ["left_foot", "right_foot"]

    for c in range(num_classes):
        n_gt = stats[c]["n_gt"]
        if len(stats[c]["conf"]) == 0 or n_gt == 0:
            class_results[class_names[c]] = {
                "instances": n_gt, "p_box": 0.0, "r_box": 0.0, "map50_box": 0.0, "map_box": 0.0,
                "p_kpt": 0.0, "r_kpt": 0.0, "map50_kpt": 0.0, "map_kpt": 0.0
            }
            continue

        conf = np.concatenate(stats[c]["conf"])
        box_tp = np.concatenate(stats[c]["box_tp"], axis=0)
        kpt_tp = np.concatenate(stats[c]["kpt_tp"], axis=0)

        # Sort by confidence
        sort_ord = np.argsort(-conf)
        box_tp = box_tp[sort_ord]
        kpt_tp = kpt_tp[sort_ord]

        # Box Metrics
        b_fp = 1.0 - box_tp
        b_tp_cum = np.cumsum(box_tp, axis=0)
        b_fp_cum = np.cumsum(b_fp, axis=0)

        b_rec = b_tp_cum / max(n_gt, 1)
        b_prec = b_tp_cum / np.maximum(b_tp_cum + b_fp_cum, np.finfo(np.float64).eps)

        ap_box_th = [compute_ap(b_rec[:, i], b_prec[:, i]) for i in range(len(iou_thresholds))]
        map50_box = ap_box_th[0]
        map_box = float(np.mean(ap_box_th))
        p_box = float(b_prec[-1, 0]) if len(b_prec) > 0 else 0.0
        r_box = float(b_rec[-1, 0]) if len(b_rec) > 0 else 0.0

        # Keypoint Metrics
        k_fp = 1.0 - kpt_tp
        k_tp_cum = np.cumsum(kpt_tp, axis=0)
        k_fp_cum = np.cumsum(k_fp, axis=0)

        k_rec = k_tp_cum / max(n_gt, 1)
        k_prec = k_tp_cum / np.maximum(k_tp_cum + k_fp_cum, np.finfo(np.float64).eps)

        ap_kpt_th = [compute_ap(k_rec[:, i], k_prec[:, i]) for i in range(len(iou_thresholds))]
        map50_kpt = ap_kpt_th[0]
        map_kpt = float(np.mean(ap_kpt_th))
        p_kpt = float(k_prec[-1, 0]) if len(k_prec) > 0 else 0.0
        r_kpt = float(k_rec[-1, 0]) if len(k_rec) > 0 else 0.0

        class_results[class_names[c]] = {
            "instances": n_gt,
            "p_box": p_box, "r_box": r_box, "map50_box": map50_box, "map_box": map_box,
            "p_kpt": p_kpt, "r_kpt": r_kpt, "map50_kpt": map50_kpt, "map_kpt": map_kpt
        }

        all_box_ap50.append(map50_box)
        all_box_ap.append(map_box)
        all_kpt_ap50.append(map50_kpt)
        all_kpt_ap.append(map_kpt)
        all_p_b.append(p_box)
        all_r_b.append(r_box)
        all_p_k.append(p_kpt)
        all_r_k.append(r_kpt)

    # Overall 'all'
    total_instances = sum(res["instances"] for res in class_results.values())
    overall = {
        "instances": total_instances,
        "p_box": float(np.mean(all_p_b)) if all_p_b else 0.0,
        "r_box": float(np.mean(all_r_b)) if all_r_b else 0.0,
        "map50_box": float(np.mean(all_box_ap50)) if all_box_ap50 else 0.0,
        "map_box": float(np.mean(all_box_ap)) if all_box_ap else 0.0,
        "p_kpt": float(np.mean(all_p_k)) if all_p_k else 0.0,
        "r_kpt": float(np.mean(all_r_k)) if all_r_k else 0.0,
        "map50_kpt": float(np.mean(all_kpt_ap50)) if all_kpt_ap50 else 0.0,
        "map_kpt": float(np.mean(all_kpt_ap)) if all_kpt_ap else 0.0,
        "classes": class_results
    }

    return overall


def print_yolo_metrics_table(metrics: Dict, title: str = "Validation Results"):
    """
    Renders an authentic Ultralytics YOLO-style summary table in terminal / stdout.
    """
    print("\n" + "=" * 88)
    print(f"  📊 {title.upper()}")
    print("=" * 88)
    print(f"  {'Class':<12} {'Images':>7} {'Instances':>10} | {'Box(P':>7} {'R':>6} {'mAP50':>7} {'mAP50-95)':>10} | {'Pose(P':>7} {'R':>6} {'mAP50':>7} {'mAP50-95)':>10}")
    print("  " + "-" * 84)

    # 'all' row
    print(f"  {'all':<12} {'--':>7} {metrics['instances']:>10d} | {metrics['p_box']:>7.3f} {metrics['r_box']:>6.3f} {metrics['map50_box']:>7.3f} {metrics['map_box']:>10.3f} | {metrics['p_kpt']:>7.3f} {metrics['r_kpt']:>6.3f} {metrics['map50_kpt']:>7.3f} {metrics['map_kpt']:>10.3f}")

    # Per-class rows
    for cname, cdata in metrics.get("classes", {}).items():
        print(f"  {cname:<12} {'--':>7} {cdata['instances']:>10d} | {cdata['p_box']:>7.3f} {cdata['r_box']:>6.3f} {cdata['map50_box']:>7.3f} {cdata['map_box']:>10.3f} | {cdata['p_kpt']:>7.3f} {cdata['r_kpt']:>6.3f} {cdata['map50_kpt']:>7.3f} {cdata['map_kpt']:>10.3f}")

    print("=" * 88 + "\n")

