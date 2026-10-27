"""
qualcomm_experiments/benchmark_three_experiments.py
────────────────────────────────────────────────────
Rigorous Validation Benchmark comparing 3 Architectures:
  1. NanoDet-Plus (2-Class, 320 ONNX) + RTMPose-13KP (256 ONNX)
  2. Original YOLO Stage-A (2-Class, 256 ONNX) + RTMPose-13KP (256 ONNX)
  3. Single-Stage RTMPose SimCC (320 ONNX) [RoI-Align + SimCC one forward pass]

Dataset:
  data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid (827 images)

Metrics:
  - Box F1(50), Box F1(80), Box F1(90)
  - Pose PCK(0.05), Pose PCK(0.10), Pose PCK(0.20)
  - End-to-End Latency (ms) & Parameters (M)
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torchvision import ops

from src.models.qualcomm_experiments.common.dataset import safe_imread, safe_path_str
from src.models.qualcomm_experiments.rtmpose.decoder import decode_simcc
from src.models.qualcomm_experiments.common.crop import (
    compute_crop_box,
    get_crop_affine_transform,
    warp_image_crop,
    invert_keypoints,
)


def box_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Computes IoU matrix [N, M] between boxes1 [N, 4] and boxes2 [M, 4]."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = np.clip(rb - lt, a_min=0.0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, a_min=1e-7, a_max=None)


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.20) -> List[int]:
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def parse_yolo_labels(label_path: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parses YOLO label format:
    cls cx cy w h kp0_x kp0_y kp0_v ... kp12_x kp12_y kp12_v
    Returns:
      boxes: [N, 4] (x1, y1, x2, y2)
      classes: [N]
      keypoints: [N, 13, 3] (x, y, v)
    """
    p_str = safe_path_str(label_path)
    if not Path(p_str).exists() or Path(p_str).stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0, 13, 3), dtype=np.float32)

    boxes = []
    classes = []
    kpts = []

    with open(p_str, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = (cx - bw / 2.0) * img_w
            y1 = (cy - bh / 2.0) * img_h
            x2 = (cx + bw / 2.0) * img_w
            y2 = (cy + bh / 2.0) * img_h

            kpt_arr = np.zeros((13, 3), dtype=np.float32)
            if len(parts) >= 44:
                vals = [float(x) for x in parts[5:44]]
                for i in range(13):
                    kpt_arr[i, 0] = vals[i * 3] * img_w
                    kpt_arr[i, 1] = vals[i * 3 + 1] * img_h
                    kpt_arr[i, 2] = vals[i * 3 + 2]

            boxes.append([x1, y1, x2, y2])
            classes.append(cls_id)
            kpts.append(kpt_arr)

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0, 13, 3), dtype=np.float32)

    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int64), np.array(kpts, dtype=np.float32)


# ==============================================================================
# Model Inference Routines
# ==============================================================================

def infer_nanodet_rtmpose(
    img: np.ndarray,
    nanodet_sess: ort.InferenceSession,
    rtmpose_sess: ort.InferenceSession,
    conf_thresh: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns boxes [N, 4], classes [N], keypoints [N, 13, 3] in original image space."""
    h0, w0 = img.shape[:2]
    det_size = 320

    # Step 1: NanoDet
    frame_resized = cv2.resize(img, (det_size, det_size))
    inp = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]

    det_outs = nanodet_sess.run(None, {nanodet_sess.get_inputs()[0].name: inp})
    boxes_raw = det_outs[0][0]   # [N, 4]
    scores_raw = det_outs[1][0]  # [N, 2]

    max_scores = np.max(scores_raw, axis=-1)
    cls_ids = np.argmax(scores_raw, axis=-1)
    mask = max_scores >= conf_thresh

    if not np.any(mask):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0, 13, 3), dtype=np.float32)

    f_boxes = boxes_raw[mask]
    f_scores = max_scores[mask]
    f_cls = cls_ids[mask]

    keep = nms_numpy(f_boxes, f_scores, iou_thresh=0.20)
    sx = w0 / float(det_size)
    sy = h0 / float(det_size)

    pred_boxes = []
    pred_cls = []
    pred_kpts = []

    for idx in keep:
        bx = f_boxes[idx]
        scaled_box = np.array([bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy], dtype=np.float32)
        x1, y1, x2, y2 = scaled_box
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy, size = compute_crop_box((x1 + x2) / 2.0, (y1 + y2) / 2.0, bw, bh, crop_padding=0.25)

        M, M_inv = get_crop_affine_transform(cx, cy, size, out_size=256, rot=0.0)
        crop = warp_image_crop(img, M, out_size=256, border_value=(114, 114, 114))
        crop_inp = crop.astype(np.float32) / 255.0
        crop_inp = np.transpose(crop_inp, (2, 0, 1))[None, ...]

        rtm_outs = rtmpose_sess.run(None, {rtmpose_sess.get_inputs()[0].name: crop_inp})
        simcc_x_t = torch.from_numpy(rtm_outs[0])
        simcc_y_t = torch.from_numpy(rtm_outs[1])
        vis_t = torch.from_numpy(rtm_outs[3]) if len(rtm_outs) > 3 else None

        preds_dict = {"simcc_x": simcc_x_t, "simcc_y": simcc_y_t, "vis": vis_t}
        simcc_kps = decode_simcc(preds_dict, simcc_split_ratio=2.0, win=12)[0]
        frame_kps = invert_keypoints(simcc_kps, M_inv)

        pred_boxes.append(scaled_box)
        pred_cls.append(int(f_cls[idx]))
        pred_kpts.append(frame_kps)

    return np.array(pred_boxes, dtype=np.float32), np.array(pred_cls, dtype=np.int64), np.array(pred_kpts, dtype=np.float32)


def infer_yolo_rtmpose(
    img: np.ndarray,
    yolo_sess: ort.InferenceSession,
    rtmpose_sess: ort.InferenceSession,
    conf_thresh: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns boxes [N, 4], classes [N], keypoints [N, 13, 3] in original image space."""
    h0, w0 = img.shape[:2]
    det_size = 256

    frame_resized = cv2.resize(img, (det_size, det_size))
    inp = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]

    det_outs = yolo_sess.run(None, {yolo_sess.get_inputs()[0].name: inp})
    preds = det_outs[0][0].T  # [1344, 45]
    cls0 = preds[:, 4]
    cls1 = preds[:, 5]
    max_scores = np.maximum(cls0, cls1)
    cls_ids = np.where(cls0 >= cls1, 0, 1)

    mask = max_scores >= conf_thresh
    if not np.any(mask):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0, 13, 3), dtype=np.float32)

    f_preds = preds[mask]
    f_scores = max_scores[mask]
    f_cls = cls_ids[mask]

    cx, cy, bw, bh = f_preds[:, 0], f_preds[:, 1], f_preds[:, 2], f_preds[:, 3]
    f_boxes = np.stack([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], axis=-1)

    keep = nms_numpy(f_boxes, f_scores, iou_thresh=0.20)
    sx = w0 / float(det_size)
    sy = h0 / float(det_size)

    pred_boxes = []
    pred_cls = []
    pred_kpts = []

    for idx in keep:
        bx = f_boxes[idx]
        scaled_box = np.array([bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy], dtype=np.float32)
        x1, y1, x2, y2 = scaled_box
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy, size = compute_crop_box((x1 + x2) / 2.0, (y1 + y2) / 2.0, bw, bh, crop_padding=0.25)

        M, M_inv = get_crop_affine_transform(cx, cy, size, out_size=256, rot=0.0)
        crop = warp_image_crop(img, M, out_size=256, border_value=(114, 114, 114))
        crop_inp = crop.astype(np.float32) / 255.0
        crop_inp = np.transpose(crop_inp, (2, 0, 1))[None, ...]

        rtm_outs = rtmpose_sess.run(None, {rtmpose_sess.get_inputs()[0].name: crop_inp})
        simcc_x_t = torch.from_numpy(rtm_outs[0])
        simcc_y_t = torch.from_numpy(rtm_outs[1])
        vis_t = torch.from_numpy(rtm_outs[3]) if len(rtm_outs) > 3 else None

        preds_dict = {"simcc_x": simcc_x_t, "simcc_y": simcc_y_t, "vis": vis_t}
        simcc_kps = decode_simcc(preds_dict, simcc_split_ratio=2.0, win=12)[0]
        frame_kps = invert_keypoints(simcc_kps, M_inv)

        pred_boxes.append(scaled_box)
        pred_cls.append(int(f_cls[idx]))
        pred_kpts.append(frame_kps)

    return np.array(pred_boxes, dtype=np.float32), np.array(pred_cls, dtype=np.int64), np.array(pred_kpts, dtype=np.float32)


def decode_rois_simcc_numpy(simcc_x: np.ndarray, simcc_y: np.ndarray, vis_logits: np.ndarray, rois: np.ndarray) -> np.ndarray:
    """
    Continuous sub-pixel expectation decoding in numpy.
    simcc_x: [K, 13, 512], simcc_y: [K, 13, 512], vis_logits: [K, 13, 1], rois: [K, 4]
    """
    e_x = np.exp(simcc_x - np.max(simcc_x, axis=-1, keepdims=True))
    prob_x = e_x / np.sum(e_x, axis=-1, keepdims=True)
    e_y = np.exp(simcc_y - np.max(simcc_y, axis=-1, keepdims=True))
    prob_y = e_y / np.sum(e_y, axis=-1, keepdims=True)

    bin_coords = np.arange(512, dtype=np.float32)
    rel_u = np.sum(prob_x * bin_coords, axis=-1) / 511.0  # [K, 13]
    rel_v = np.sum(prob_y * bin_coords, axis=-1) / 511.0  # [K, 13]

    bx1 = rois[:, 0:1]
    by1 = rois[:, 1:2]
    bw = np.maximum(1.0, rois[:, 2:3] - rois[:, 0:1])
    bh = np.maximum(1.0, rois[:, 3:4] - rois[:, 1:2])

    abs_x = bx1 + rel_u * bw
    abs_y = by1 + rel_v * bh
    conf = 1.0 / (1.0 + np.exp(-vis_logits.squeeze(-1)))
    return np.stack([abs_x, abs_y, conf], axis=-1)


def infer_single_stage_onnx(
    img: np.ndarray,
    det_sess: ort.InferenceSession,
    roi_sess: ort.InferenceSession,
    conf_thresh: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single forward pass with internal RoI-Align + SimCC continuous expectation."""
    h0, w0 = img.shape[:2]
    det_size = 320

    frame_resized = cv2.resize(img, (det_size, det_size))
    inp = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]

    boxes_raw, scores_raw, p2 = det_sess.run(None, {det_sess.get_inputs()[0].name: inp})
    boxes = boxes_raw[0]   # [2100, 4]
    scores = scores_raw[0] # [2100, 2]

    max_scores = np.max(scores, axis=-1)
    cls_ids = np.argmax(scores, axis=-1)

    mask = max_scores >= conf_thresh
    if not np.any(mask):
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0, 13, 3), dtype=np.float32)

    f_boxes = boxes[mask]
    f_scores = max_scores[mask]
    f_cls = cls_ids[mask]

    keep = nms_numpy(f_boxes, f_scores, iou_thresh=0.20)
    if len(keep) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0, 13, 3), dtype=np.float32)

    kept_boxes = f_boxes[keep]
    kept_cls = f_cls[keep]

    # RoI Align via RoI ONNX session (batch index 0)
    rois = np.zeros((len(keep), 5), dtype=np.float32)
    rois[:, 1:] = kept_boxes

    sx, sy, vl = roi_sess.run(None, {"p2": p2, "rois": rois})
    kpts_320 = decode_rois_simcc_numpy(sx, sy, vl, kept_boxes)

    # Scale back to original image
    sx_scale = w0 / float(det_size)
    sy_scale = h0 / float(det_size)

    pred_boxes = kept_boxes.copy()
    pred_boxes[:, [0, 2]] *= sx_scale
    pred_boxes[:, [1, 3]] *= sy_scale

    pred_kpts = kpts_320.copy()
    pred_kpts[:, :, 0] *= sx_scale
    pred_kpts[:, :, 1] *= sy_scale

    return pred_boxes, kept_cls, pred_kpts


# ==============================================================================
# Metric Evaluator Engine
# ==============================================================================

class BenchmarkAccumulator:
    def __init__(self):
        self.box_thresholds = [0.50, 0.80, 0.90]
        self.pck_thresholds = [0.05, 0.10, 0.20]

        # For each IoU thresh: [TP, FP, FN]
        self.box_stats = {t: [0, 0, 0] for t in self.box_thresholds}

        # For PCK: correct count per thresh, and total visible keypoints
        self.pck_correct = {t: 0 for t in self.pck_thresholds}
        self.pck_total_kpts = 0
        self.latencies = []

    def update(
        self,
        pred_boxes: np.ndarray,
        pred_classes: np.ndarray,
        pred_kpts: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
        gt_kpts: np.ndarray,
        latency_ms: float
    ):
        self.latencies.append(latency_ms)

        num_preds = len(pred_boxes)
        num_gt = len(gt_boxes)

        # 1. Update Box Metrics across thresholds
        if num_preds == 0 and num_gt == 0:
            return  # Clean true negative image

        iou_mat = box_iou_matrix(pred_boxes, gt_boxes) if (num_preds > 0 and num_gt > 0) else np.zeros((num_preds, num_gt))

        for t in self.box_thresholds:
            tp, fp, fn = 0, 0, 0
            matched_gt = set()
            matched_pred = set()

            if num_preds > 0 and num_gt > 0:
                cur_iou = iou_mat.copy()
                while True:
                    if cur_iou.size == 0 or cur_iou.max() < t:
                        break
                    r, c = np.unravel_index(cur_iou.argmax(), cur_iou.shape)
                    if pred_classes[r] == gt_classes[c]:
                        tp += 1
                        matched_pred.add(r)
                        matched_gt.add(c)
                    cur_iou[r, :] = -1.0
                    cur_iou[:, c] = -1.0

            fp = num_preds - len(matched_pred)
            fn = num_gt - len(matched_gt)

            self.box_stats[t][0] += tp
            self.box_stats[t][1] += fp
            self.box_stats[t][2] += fn

        # 2. Update PCK Metrics (IoU >= 0.50 match)
        # For all GT feet, accumulate visible keypoints
        for g_idx in range(num_gt):
            g_kps = gt_kpts[g_idx]
            vis_mask = (g_kps[:, 2] > 0)
            n_vis = int(vis_mask.sum())
            if n_vis == 0:
                continue

            self.pck_total_kpts += n_vis
            gt_w = gt_boxes[g_idx, 2] - gt_boxes[g_idx, 0]
            gt_h = gt_boxes[g_idx, 3] - gt_boxes[g_idx, 1]
            scale = max(1.0, max(gt_w, gt_h))

            # Find matching pred with highest IoU >= 0.50
            best_p_idx = -1
            best_iou = 0.50
            if num_preds > 0:
                for p_idx in range(num_preds):
                    if pred_classes[p_idx] == gt_classes[g_idx] and iou_mat[p_idx, g_idx] >= best_iou:
                        best_iou = iou_mat[p_idx, g_idx]
                        best_p_idx = p_idx

            if best_p_idx >= 0:
                p_kps = pred_kpts[best_p_idx]
                dists = np.linalg.norm(p_kps[vis_mask, :2] - g_kps[vis_mask, :2], axis=-1)
                for alpha in self.pck_thresholds:
                    self.pck_correct[alpha] += int((dists <= alpha * scale).sum())
            else:
                # Missed foot detection: 0 correct keypoints for this GT
                pass

    def compute_summary(self) -> Dict[str, float]:
        res = {}
        for t in self.box_thresholds:
            tp, fp, fn = self.box_stats[t]
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            f1 = 2.0 * prec * rec / max(1e-7, prec + rec)
            res[f"Box F1({int(t*100)})"] = f1 * 100.0

        for alpha in self.pck_thresholds:
            pck = self.pck_correct[alpha] / max(1, self.pck_total_kpts)
            res[f"Pose PCK({alpha:.2f})"] = pck * 100.0

        res["Avg Latency (ms)"] = float(np.mean(self.latencies)) if self.latencies else 0.0
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_images", type=str, default="data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid/images")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of validation images for quick test (0 = full 827)")
    args = parser.parse_args()

    val_dir = Path(args.val_images)
    lbl_dir = val_dir.parent / "labels"
    image_paths = sorted([p for p in val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])

    if args.limit > 0:
        image_paths = image_paths[:args.limit]

    print("=" * 75)
    print("  RIGOROUS BENCHMARK EVALUATION ACROSS 3 EXPERIMENTS (ONNX ENGINES)")
    print(f"  Validation Dataset: {val_dir} ({len(image_paths)} images)")
    print(f"  Metrics:            Box F1(50, 80, 90) | Pose PCK(0.05, 0.10, 0.20)")
    print("=" * 75)

    # 1. Load ONNX Models
    providers = ["CPUExecutionProvider"]
    nanodet_onnx = "deliverables/qualcomm_experiments/models/nanodet-plus-foot-2class-320.onnx"
    yolo_onnx = "deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.onnx"
    rtmpose_onnx = "deliverables/qualcomm_experiments/models/rtmpose-13kp-full-champion-256.onnx"
    single_stage_det_onnx = "deliverables/qualcomm_experiments/models/single-stage-rtmpose-simcc-det.onnx"
    single_stage_roi_onnx = "deliverables/qualcomm_experiments/models/single-stage-rtmpose-simcc-roi.onnx"

    print("[*] Loading ONNX Sessions...")
    sess_nanodet = ort.InferenceSession(nanodet_onnx, providers=providers)
    sess_rtmpose = ort.InferenceSession(rtmpose_onnx, providers=providers)
    sess_yolo = ort.InferenceSession(yolo_onnx, providers=providers)
    sess_single_det = ort.InferenceSession(single_stage_det_onnx, providers=providers)
    sess_single_roi = ort.InferenceSession(single_stage_roi_onnx, providers=providers)
    print("[+] All ONNX Sessions loaded successfully!\n")

    experiments = [
        ("Exp 1: NanoDet-Plus (2-Class) + RTMPose-13KP", "nanodet_rtmpose"),
        ("Exp 2: Original YOLO Stage-A + RTMPose-13KP", "yolo_rtmpose"),
        ("Exp 3: Single-Stage RTMPose SimCC 320 [ONNX]", "single_stage_simcc"),
    ]

    all_results = {}

    for exp_name, exp_mode in experiments:
        print("-" * 75)
        print(f"  Running: {exp_name} on {len(image_paths)} validation images...")
        print("-" * 75)

        accum = BenchmarkAccumulator()
        t0 = time.time()

        for idx, img_p in enumerate(image_paths, 1):
            img = safe_imread(img_p)
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            gt_b, gt_c, gt_k = parse_yolo_labels(lbl_dir / f"{img_p.stem}.txt", w0, h0)

            t_infer0 = time.perf_counter()
            if exp_mode == "nanodet_rtmpose":
                p_b, p_c, p_k = infer_nanodet_rtmpose(img, sess_nanodet, sess_rtmpose, conf_thresh=0.25)
            elif exp_mode == "yolo_rtmpose":
                p_b, p_c, p_k = infer_yolo_rtmpose(img, sess_yolo, sess_rtmpose, conf_thresh=0.25)
            else:  # single_stage_simcc
                p_b, p_c, p_k = infer_single_stage_onnx(img, sess_single_det, sess_single_roi, conf_thresh=0.25)
            t_infer = (time.perf_counter() - t_infer0) * 1000.0

            accum.update(p_b, p_c, p_k, gt_b, gt_c, gt_k, t_infer)

            if idx % 100 == 0 or idx == len(image_paths):
                print(f"  [{idx}/{len(image_paths)}] Processed (Avg: {np.mean(accum.latencies):.1f} ms/img)")

        dur = time.time() - t0
        summary = accum.compute_summary()
        summary["Total Benchmark Time (s)"] = dur
        all_results[exp_name] = summary

        print(f"\n  Results for {exp_name}:")
        for k, v in summary.items():
            if "Time" in k:
                print(f"    {k:<20}: {v:.1f}s")
            elif "Latency" in k:
                print(f"    {k:<20}: {v:.2f} ms")
            else:
                print(f"    {k:<20}: {v:.2f}%")
        print()

    # Final Comparative Table
    print("\n" + "=" * 80)
    print("  FINAL COMPARATIVE BENCHMARK MATRIX (VAL DATASET: 827 IMAGES)")
    print("=" * 80)
    header = f"{'Metric':<22} | {'Exp 1: NanoDet + RTM':<22} | {'Exp 2: YOLO + RTM':<20} | {'Exp 3: Single-Stage SimCC':<22}"
    print(header)
    print("-" * len(header))

    metrics_list = [
        "Box F1(50)", "Box F1(80)", "Box F1(90)",
        "Pose PCK(0.05)", "Pose PCK(0.10)", "Pose PCK(0.20)",
        "Avg Latency (ms)"
    ]

    for m in metrics_list:
        v1 = all_results["Exp 1: NanoDet-Plus (2-Class) + RTMPose-13KP"].get(m, 0.0)
        v2 = all_results["Exp 2: Original YOLO Stage-A + RTMPose-13KP"].get(m, 0.0)
        v3 = all_results["Exp 3: Single-Stage RTMPose SimCC 320 [ONNX]"].get(m, 0.0)

        unit = "ms" if "Latency" in m else "%"
        print(f"{m:<22} | {v1:>19.2f}{unit} | {v2:>17.2f}{unit} | {v3:>19.2f}{unit}")
    print("=" * 80)


if __name__ == "__main__":
    main()
