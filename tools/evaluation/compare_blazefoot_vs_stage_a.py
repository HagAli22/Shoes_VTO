"""
compare_blazefoot_vs_stage_a.py
───────────────────────────────
Automated Comparative Evaluation Tool:
Benchmarks any BlazeFoot Model against the Frozen Stage A Teacher Ground Truth
(cached in outputs/evaluation/teacher_video_groundtruth.json).

Computes:
  1. Box mIoU (Mean Intersection-over-Union with Teacher Bounding Boxes).
  2. 4-Keypoint Euclidean Pixel Error (Per-keypoint: toe_tip, heel_back, ball_medial, ball_lateral).
  3. PCK@10px and PCK@20px (Percentage of Correct Keypoints).
  4. Chirality Accuracy (Left vs Right match rate).
  5. Precision, Recall & False Positive count.
  6. Side-by-Side Visual Comparison Video (Teacher Ground Truth vs BlazeFoot).

Usage:
    python tools/evaluation/compare_blazefoot_vs_stage_a.py \
      --model deliverables/stage_a/stage-a-320-blazefoot-fp32.onnx \
      --video deliverables/stage_a_16kp/test_video.mp4 \
      --gt outputs/evaluation/teacher_video_groundtruth.json \
      --output outputs/evaluation/comparison_result.mp4
"""

import os
import sys
import json
import time
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.evaluation.eval_blazefoot_video import BlazeFootVideoEvaluator, KP_NAMES, KP_COLORS, SKELETON_4KP


def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-7)


def render_side_by_side(frame, gt_dets, pred_dets, frame_idx, avg_iou, avg_kpt_err):
    h, w = frame.shape[:2]
    
    # Left Canvas: Teacher (Ground Truth)
    c_left = frame.copy()
    for det in gt_dets:
        bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
        col = (0, 230, 70) if det["class_id"] == 0 else (0, 160, 255)
        cv2.rectangle(c_left, (bx1, by1), (bx2, by2), col, 2, cv2.LINE_AA)
        
        # 4 KPs
        kpts = det["keypoints_4"]
        for p1_i, p2_i in SKELETON_4KP:
            pt1 = (int(kpts[p1_i]["x"]), int(kpts[p1_i]["y"]))
            pt2 = (int(kpts[p2_i]["x"]), int(kpts[p2_i]["y"]))
            cv2.line(c_left, pt1, pt2, (220, 220, 220), 2, cv2.LINE_AA)
        for k in kpts:
            cv2.circle(c_left, (int(k["x"]), int(k["y"])), 5, (0, 255, 255), -1, cv2.LINE_AA)

    # Right Canvas: Student (BlazeFoot)
    c_right = frame.copy()
    for det in pred_dets:
        bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
        col = (0, 230, 70) if det["class_id"] == 0 else (0, 160, 255)
        cv2.rectangle(c_right, (bx1, by1), (bx2, by2), col, 2, cv2.LINE_AA)
        
        kpts = det["keypoints"]
        for p1_i, p2_i in SKELETON_4KP:
            pt1 = (int(kpts[p1_i]["x"]), int(kpts[p1_i]["y"]))
            pt2 = (int(kpts[p2_i]["x"]), int(kpts[p2_i]["y"]))
            cv2.line(c_right, pt1, pt2, (220, 220, 220), 2, cv2.LINE_AA)
        for k in kpts:
            cv2.circle(c_right, (int(k["x"]), int(k["y"])), 5, k["color"], -1, cv2.LINE_AA)

    # Side-by-side stitch
    combined = np.hstack([c_left, c_right])
    cw = combined.shape[1]

    # Top Header Banner
    cv2.rectangle(combined, (0, 0), (cw, 60), (15, 15, 15), -1)
    cv2.rectangle(combined, (0, 58), (cw, 60), (60, 60, 60), -1)

    # Left Label
    cv2.putText(combined, "TEACHER 16-KP (Ground Truth Baseline)", (20, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 1, cv2.LINE_AA)

    # Right Label
    cv2.putText(combined, f"BLAZEFOOT 4-KP (Student Model) | Frame: {frame_idx:03d}", (w + 20, 38),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 117), 1, cv2.LINE_AA)

    return combined


def compare_models(
    model_path: str = "deliverables/stage_a/stage-a-320-blazefoot-fp32.onnx",
    gt_json_path: str = "outputs/evaluation/teacher_video_groundtruth.json",
    video_path: str = "deliverables/stage_a_16kp/test_video.mp4",
    output_video_path: str = "outputs/evaluation/blazefoot_vs_stage_a_comparison.mp4",
    conf_thresh: float = 0.20,
    iou_match_thresh: float = 0.30,
    save_video: bool = True
):
    if not os.path.exists(gt_json_path):
        raise FileNotFoundError(f"Teacher Ground Truth JSON not found: {gt_json_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print("=" * 75)
    print("  BLAZEFOOT vs STAGE A TEACHER COMPARATIVE EVALUATION")
    print(f"  Model Under Test : {model_path}")
    print(f"  Golden GT File   : {gt_json_path}")
    print(f"  Video Path       : {video_path}")
    print(f"  Match IoU Thresh : {iou_match_thresh} | Confidence: {conf_thresh}")
    print("=" * 75)

    with open(gt_json_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_frames = {item["frame_idx"]: item for item in gt_data["frames"]}
    total_frames = len(gt_frames)

    evaluator = BlazeFootVideoEvaluator(model_path, device_str="cpu")

    cap = cv2.VideoCapture(video_path)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_vid, (orig_w * 2, orig_h)) if save_video else None

    # Metrics Accumulators
    total_gt_feet = 0
    total_pred_feet = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    ious_list = []
    kpt_errors = {0: [], 1: [], 2: [], 3: []} # 0: toe_tip, 1: heel_back, 2: ball_med, 3: ball_lat
    chirality_matches = 0

    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Evaluating Frames")

    while cap.isOpened() and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gt_record = gt_frames.get(frame_idx, {"detections": []})
        gt_dets = gt_record["detections"]

        # Run student prediction
        pred_dets = evaluator.predict(frame, conf_thresh=conf_thresh)

        total_gt_feet += len(gt_dets)
        total_pred_feet += len(pred_dets)

        # Match predictions to Ground Truth via Hungarian/Greedy IoU
        matched_gt = set()
        matched_pred = set()

        for p_i, p_det in enumerate(pred_dets):
            best_iou = 0.0
            best_g_i = -1
            for g_i, g_det in enumerate(gt_dets):
                if g_i in matched_gt:
                    continue
                iou = compute_iou(p_det["bbox"], g_det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_g_i = g_i

            if best_iou >= iou_match_thresh and best_g_i >= 0:
                matched_gt.add(best_g_i)
                matched_pred.add(p_i)
                ious_list.append(best_iou)

                g_det = gt_dets[best_g_i]
                if p_det["class_id"] == g_det["class_id"]:
                    chirality_matches += 1

                # Calculate Keypoint Euclidean Distances (4 Landmarks)
                p_kpts = p_det["keypoints"]
                g_kpts = g_det["keypoints_4"]
                for k in range(4):
                    dist = np.sqrt((p_kpts[k]["x"] - g_kpts[k]["x"])**2 + (p_kpts[k]["y"] - g_kpts[k]["y"])**2)
                    kpt_errors[k].append(dist)

        true_positives += len(matched_gt)
        false_negatives += (len(gt_dets) - len(matched_gt))
        false_positives += (len(pred_dets) - len(matched_pred))

        if writer is not None:
            side_by_side = render_side_by_side(frame, gt_dets, pred_dets, frame_idx,
                                               np.mean(ious_list) if ious_list else 0.0,
                                               np.mean([np.mean(v) for v in kpt_errors.values() if v]) if any(kpt_errors.values()) else 0.0)
            writer.write(side_by_side)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    # Aggregate Statistics
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(total_gt_feet, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 1e-7)
    mean_iou = float(np.mean(ious_list)) if ious_list else 0.0
    chirality_acc = chirality_matches / max(true_positives, 1)

    all_errs = []
    for k in range(4):
        all_errs.extend(kpt_errors[k])

    overall_mean_err = float(np.mean(all_errs)) if all_errs else 999.0
    pck_10 = float(np.mean([e <= 10.0 for e in all_errs])) * 100 if all_errs else 0.0
    pck_20 = float(np.mean([e <= 20.0 for e in all_errs])) * 100 if all_errs else 0.0

    print("\n" + "=" * 75)
    print("  [BENCHMARK] BLAZEFOOT vs STAGE A GOLDEN BENCHMARK RESULTS")
    print("=" * 75)
    print(f"  Total Video Frames Evaluated : {frame_idx}")
    print(f"  Total Ground Truth Feet (GT) : {total_gt_feet}")
    print(f"  Total BlazeFoot Detections   : {total_pred_feet}")
    print(f"  Matched True Positives (TP)  : {true_positives}")
    print(f"  False Positives (FP)         : {false_positives}")
    print(f"  False Negatives (Missed)     : {false_negatives}")
    print("-" * 75)
    print(f"  Detection Precision          : {precision:.1%}")
    print(f"  Detection Recall             : {recall:.1%}")
    print(f"  F1 Score                     : {f1_score:.1%}")
    print(f"  Bounding Box Mean IoU        : {mean_iou:.1%}")
    print(f"  Foot Chirality (L/R) Match   : {chirality_acc:.1%}")
    print("-" * 75)
    print(f"  Overall 4-KP Mean Error      : {overall_mean_err:.2f} px")
    print(f"  PCK @ 10 pixels              : {pck_10:.1f}%")
    print(f"  PCK @ 20 pixels              : {pck_20:.1f}%")
    print("-" * 75)
    print("  Per-Keypoint Mean Pixel Error:")
    for k in range(4):
        name = KP_NAMES[k]
        err_k = float(np.mean(kpt_errors[k])) if kpt_errors[k] else 0.0
        pck_k = float(np.mean([e <= 15.0 for e in kpt_errors[k]])) * 100 if kpt_errors[k] else 0.0
        print(f"    - {name:<14}: {err_k:5.2f} px  (PCK@15px: {pck_k:5.1f}%)")
    print("=" * 75)
    if save_video:
        print(f"  Side-by-Side Comparison Video: {output_video_path}")
    print("=" * 75 + "\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mean_iou": mean_iou,
        "chirality_acc": chirality_acc,
        "mean_kpt_error_px": overall_mean_err,
        "pck_10": pck_10,
        "pck_20": pck_20,
        "per_kpt_error": {KP_NAMES[k]: float(np.mean(kpt_errors[k])) if kpt_errors[k] else 0.0 for k in range(4)}
    }


def main():
    parser = argparse.ArgumentParser(description="Compare BlazeFoot vs Teacher Ground Truth.")
    parser.add_argument("--model", default="deliverables/stage_a/stage-a-320-blazefoot-fp32.onnx", help="Path to BlazeFoot model")
    parser.add_argument("--gt", default="outputs/evaluation/teacher_video_groundtruth.json", help="Path to Teacher Ground Truth JSON")
    parser.add_argument("--video", default="deliverables/stage_a_16kp/test_video.mp4", help="Path to test video")
    parser.add_argument("--output", default="outputs/evaluation/blazefoot_vs_stage_a_comparison.mp4", help="Path to output comparison video")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    parser.add_argument("--no_video", action="store_true", help="Skip saving video to save time")
    args = parser.parse_args()

    compare_models(
        model_path=args.model,
        gt_json_path=args.gt,
        video_path=args.video,
        output_video_path=args.output,
        conf_thresh=args.conf,
        save_video=not args.no_video
    )


if __name__ == "__main__":
    main()
