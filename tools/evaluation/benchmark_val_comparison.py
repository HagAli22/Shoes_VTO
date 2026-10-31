#!/usr/bin/env python3
"""
tools/evaluation/benchmark_val_comparison.py
────────────────────────────────────────────
Comprehensive Validation Benchmark Script comparing:
  1. Two-Stage Qualcomm Pipeline:
     - NanoDet-Plus improved4 (ONNX, opset=12)
     - RTMPose-13KP with Hand Head (ONNX, opset=12)
  2. Single-Stage Champion Pipeline:
     - YOLOv8n-pose v7 merged-negatives (ONNX, opset=12)

Evaluates on the exact 827 images in:
  data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid

Metrics Computed:
  - Detection: Precision, Recall, F1@0.50, F1@0.75, mIoU, Negative False Alarm %
  - Pose: PCK@0.20, PCK@0.10, PCK@0.05, Mean Absolute Error (px), Chirality Accuracy
"""

import os
import sys
import time
import math
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import onnxruntime as ort

# Add PyTorch CUDA DLLs for onnxruntime CUDA execution
try:
    import torch
    _tlib = Path(torch.__file__).parent / "lib"
    if _tlib.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(_tlib))
except Exception:
    pass


# ── Geometric & Crop Transformation Helpers ─────────────────────────────────────

def compute_crop_box(cx: float, cy: float, bw: float, bh: float, crop_padding: float = 0.25) -> Tuple[float, float, float]:
    pad_w = bw * (1.0 + crop_padding * 2.0)
    pad_h = bh * (1.0 + crop_padding * 2.0)
    size = max(pad_w, pad_h, 32.0)
    return cx, cy, size


def get_crop_affine_transform(center_x: float, center_y: float, size: float, out_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    half_size = size / 2.0
    src = np.array([
        [center_x, center_y],
        [center_x, center_y - half_size],
        [center_x + half_size, center_y]
    ], dtype=np.float32)

    half_out = out_size / 2.0
    dst = np.array([
        [half_out, half_out],
        [half_out, 0.0],
        [out_size, half_out]
    ], dtype=np.float32)

    M = cv2.getAffineTransform(src, dst)
    M_inv = cv2.getAffineTransform(dst, src)
    return M, M_inv


def warp_image_crop(image: np.ndarray, M: np.ndarray, out_size: int = 256, border_value=(114, 114, 114)) -> np.ndarray:
    return cv2.warpAffine(image, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def invert_keypoints(kpts: np.ndarray, M_inv: np.ndarray) -> np.ndarray:
    n = kpts.shape[0]
    pts = np.ones((n, 3), dtype=np.float32)
    pts[:, :2] = kpts[:, :2]
    orig_xy = (M_inv @ pts.T).T
    out = np.zeros_like(kpts)
    out[:, :2] = orig_xy
    out[:, 2] = kpts[:, 2]
    return out


def decode_simcc_onnx(simcc_x: np.ndarray, simcc_y: np.ndarray, simcc_split_ratio: float = 2.0) -> np.ndarray:
    """Decodes 1D SimCC coordinate representations to [K, 2]."""
    # simcc_x: [13, W * split_ratio]
    # simcc_y: [13, H * split_ratio]
    kpts = np.zeros((simcc_x.shape[0], 2), dtype=np.float32)
    x_argmax = np.argmax(simcc_x, axis=-1)
    y_argmax = np.argmax(simcc_y, axis=-1)
    kpts[:, 0] = x_argmax / simcc_split_ratio
    kpts[:, 1] = y_argmax / simcc_split_ratio
    return kpts


def compute_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1e-4, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-4, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


# ── Pipeline 1: Two-Stage NanoDet-Plus + RTMPose-13KP ──────────────────────────

class TwoStageNanoDetRTMPoseONNX:
    def __init__(
        self,
        nanodet_path: str,
        rtmpose_path: str,
        conf_thresh: float = 0.35,
        nms_thresh: float = 0.45
    ):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print(f"[*] Initializing NanoDet Stage A ONNX: {nanodet_path}")
        self.det_sess = ort.InferenceSession(nanodet_path, providers=providers)
        self.det_in_name = self.det_sess.get_inputs()[0].name
        print(f"    NanoDet active provider: {self.det_sess.get_providers()[0]}")

        print(f"[*] Initializing RTMPose Stage B ONNX: {rtmpose_path}")
        self.pose_sess = ort.InferenceSession(rtmpose_path, providers=providers)
        self.pose_in_name = self.pose_sess.get_inputs()[0].name
        print(f"    RTMPose active provider: {self.pose_sess.get_providers()[0]}")

    def predict(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        orig_h, orig_w = img_bgr.shape[:2]

        # 1. NanoDet Stage A Detection
        det_img = cv2.resize(img_bgr, (256, 256))
        det_blob = (det_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        boxes_out, scores_out = self.det_sess.run(None, {self.det_in_name: det_blob})
        # boxes_out: [1, N, 4], scores_out: [1, N, 1]
        raw_boxes = boxes_out[0]
        raw_scores = scores_out[0, :, 0]

        mask = raw_scores >= self.conf_thresh
        if not np.any(mask):
            return []

        filt_boxes = raw_boxes[mask]
        filt_scores = raw_scores[mask]
        keep = nms_boxes(filt_boxes, filt_scores, iou_thresh=self.nms_thresh)
        if len(keep) == 0:
            return []

        scale_x = orig_w / 256.0
        scale_y = orig_h / 256.0

        candidates = []
        for idx in keep:
            bx = filt_boxes[idx]
            det_score = float(filt_scores[idx])
            x1 = float(np.clip(bx[0] * scale_x, 0, orig_w))
            y1 = float(np.clip(bx[1] * scale_y, 0, orig_h))
            x2 = float(np.clip(bx[2] * scale_x, 0, orig_w))
            y2 = float(np.clip(bx[3] * scale_y, 0, orig_h))
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                continue

            # 2. RTMPose Stage B Estimation
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = x2 - x1, y2 - y1
            crop_cx, crop_cy, size = compute_crop_box(cx, cy, bw, bh, crop_padding=0.25)
            M, M_inv = get_crop_affine_transform(crop_cx, crop_cy, size, out_size=256)
            crop = warp_image_crop(img_bgr, M, out_size=256)
            crop_blob = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

            pose_outs = self.pose_sess.run(None, {self.pose_in_name: crop_blob})
            # Outputs: simcc_x, simcc_y, vis_logits, vis, [hand_logits, hand]
            simcc_x = pose_outs[0][0]
            simcc_y = pose_outs[1][0]
            vis = np.squeeze(pose_outs[3])
            hand_prob = float(np.squeeze(pose_outs[5])) if len(pose_outs) >= 6 else 0.5

            kpts_crop = decode_simcc_onnx(simcc_x, simcc_y, simcc_split_ratio=2.0)
            kpts_with_vis = np.column_stack([kpts_crop, vis])
            frame_kps = invert_keypoints(kpts_with_vis, M_inv)

            mean_conf = float(vis.mean())
            valid_kpts = int((vis >= 0.20).sum())
            if mean_conf < 0.15 or valid_kpts < 3:
                continue

            candidates.append({
                "bbox": [x1, y1, x2, y2],
                "score": det_score,
                "keypoints": frame_kps,
                "hand_prob": hand_prob,
                "class_id": 1 if hand_prob >= 0.50 else 0,  # 0: left, 1: right
                "cx": cx
            })

        if len(candidates) == 0:
            return []

        # Dual-Foot Chirality Gate: At most 1 Left Foot & at most 1 Right Foot
        left_cands = [c for c in candidates if c["class_id"] == 0]
        right_cands = [c for c in candidates if c["class_id"] == 1]

        if len(left_cands) >= 2 and len(right_cands) == 0:
            left_cands.sort(key=lambda c: c["cx"])
            if abs(left_cands[1]["cx"] - left_cands[0]["cx"]) > 40:
                right_cands.append(left_cands.pop(-1))
        elif len(right_cands) >= 2 and len(left_cands) == 0:
            right_cands.sort(key=lambda c: c["cx"])
            if abs(right_cands[1]["cx"] - right_cands[0]["cx"]) > 40:
                left_cands.append(right_cands.pop(0))

        final_dets = []
        if left_cands:
            best_left = max(left_cands, key=lambda c: c["score"])
            best_left["class_id"] = 0
            final_dets.append(best_left)
        if right_cands:
            best_right = max(right_cands, key=lambda c: c["score"])
            best_right["class_id"] = 1
            final_dets.append(best_right)

        return final_dets


# ── Pipeline 2: Single-Stage YOLOv8n-Pose v7 ONNX ──────────────────────────────

class YOLOv8PoseONNX:
    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.25,
        nms_thresh: float = 0.45
    ):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print(f"[*] Initializing YOLOv8-Pose ONNX: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"    YOLO active provider: {self.session.get_providers()[0]}")

    def letterbox(self, img: np.ndarray, new_shape=(256, 256), color=(114, 114, 114)):
        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def predict(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        orig_h, orig_w = img_bgr.shape[:2]
        lb_img, scale, (dw, dh) = self.letterbox(img_bgr, (256, 256))
        blob = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        preds = self.session.run(None, {self.input_name: blob})[0][0]  # [45, 1344]

        candidates = []
        for col in range(preds.shape[1]):
            s_left = 1.0 / (1.0 + math.exp(-preds[4, col]))
            s_right = 1.0 / (1.0 + math.exp(-preds[5, col]))
            max_score = max(s_left, s_right)
            if max_score < self.conf_thresh:
                continue

            cls_id = 0 if s_left >= s_right else 1
            cx, cy, w, h = preds[0:4, col]
            x1 = (cx - w / 2 - dw) / scale
            y1 = (cy - h / 2 - dh) / scale
            x2 = (cx + w / 2 - dw) / scale
            y2 = (cy + h / 2 - dh) / scale
            x1 = max(0.0, min(float(x1), float(orig_w)))
            y1 = max(0.0, min(float(y1), float(orig_h)))
            x2 = max(0.0, min(float(x2), float(orig_w)))
            y2 = max(0.0, min(float(y2), float(orig_h)))

            kpts = []
            for k in range(13):
                kx = (preds[6 + k * 3, col] - dw) / scale
                ky = (preds[7 + k * 3, col] - dh) / scale
                kv = 1.0 / (1.0 + math.exp(-preds[8 + k * 3, col]))
                kpts.append([float(kx), float(ky), float(kv)])

            candidates.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(max_score),
                "class_id": cls_id,
                "keypoints": np.array(kpts, dtype=np.float32)
            })

        if not candidates:
            return []

        # Deduplicate per class with NMS
        final = []
        for c_id in [0, 1]:
            cls_cands = [c for c in candidates if c["class_id"] == c_id]
            if not cls_cands:
                continue
            boxes = np.array([c["bbox"] for c in cls_cands], dtype=np.float32)
            scores = np.array([c["score"] for c in cls_cands], dtype=np.float32)
            keep = nms_boxes(boxes, scores, iou_thresh=self.nms_thresh)
            for idx in keep:
                final.append(cls_cands[idx])

        # Suppress duplicate overlapping feet (IoU > 0.60)
        final.sort(key=lambda x: x["score"], reverse=True)
        kept = []
        for cand in final:
            dup = False
            for ex in kept:
                if compute_iou(cand["bbox"], ex["bbox"]) > 0.60:
                    dup = True
                    break
            if not dup:
                kept.append(cand)

        return kept


# ── Ground Truth Label Parser ──────────────────────────────────────────────────

def parse_gt_label(label_path: Path, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    if not label_path.exists():
        return []
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    gt_objects = []
    for line in lines:
        parts = [float(v) for v in line.split()]
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = parts[1:5]
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h

        # 13 keypoints: [x, y, v]
        kpts = []
        if len(parts) >= 5 + 13 * 3:
            for k in range(13):
                kx = parts[5 + k * 3] * img_w
                ky = parts[6 + k * 3] * img_h
                kv = parts[7 + k * 3]
                kpts.append([kx, ky, kv])
        else:
            kpts = [[0.0, 0.0, 0.0] for _ in range(13)]

        diag = math.sqrt(max(1.0, (x2 - x1) ** 2 + (y2 - y1) ** 2))
        max_dim = max(x2 - x1, y2 - y1, 1.0)

        gt_objects.append({
            "class_id": cls_id,
            "bbox": [x1, y1, x2, y2],
            "keypoints": np.array(kpts, dtype=np.float32),
            "scale_diag": diag,
            "scale_max": max_dim
        })

    return gt_objects


# ── Benchmark Evaluation Harness ───────────────────────────────────────────────

def run_benchmark():
    dataset_root = Path("data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid")
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"

    img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    total_imgs = len(img_files)
    print(f"[*] Found {total_imgs} validation images in {images_dir}")

    # Initialize both models
    two_stage = TwoStageNanoDetRTMPoseONNX(
        nanodet_path="deliverables/qualcomm_experiments/models/nanodet_stage_a_improved4_opset12.onnx",
        rtmpose_path="deliverables/qualcomm_experiments/models/rtmpose_13kp_hand_opset12.onnx",
        conf_thresh=0.35,
        nms_thresh=0.45
    )

    yolo_model = YOLOv8PoseONNX(
        model_path="deliverables/stage_a_13kp/models/stage-a-256-13kp-v7-merged-negatives.onnx",
        conf_thresh=0.25,
        nms_thresh=0.45
    )

    models = {
        "Two-Stage (NanoDet improved4 + RTMPose-13KP)": two_stage,
        "Single-Stage YOLOv8n-Pose (v7 merged-negatives)": yolo_model
    }

    results = {}
    for name in models:
        results[name] = {
            "tp_50": 0, "fp_50": 0, "fn_50": 0,
            "tp_75": 0, "fp_75": 0, "fn_75": 0,
            "ious": [],
            "neg_total": 0, "neg_fp": 0,
            "kpt_vis_total": 0,
            "pck20_hits": 0, "pck10_hits": 0, "pck05_hits": 0,
            "pck20_max_hits": 0, "pck10_max_hits": 0, "pck05_max_hits": 0,
            "mae_sum": 0.0,
            "correct_chirality": 0, "total_matched_feet": 0,
            "total_inference_time": 0.0
        }

    print("\n" + "=" * 78)
    print(f"  RUNNING SYSTEM COMPARISON BENCHMARK ON {total_imgs} VALIDATION IMAGES")
    print("=" * 78)

    t0 = time.time()
    for idx, img_path in enumerate(img_files):
        try:
            img_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img_bgr = None
        if img_bgr is None:
            img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        label_path = labels_dir / (img_path.stem + ".txt")
        gt_objs = parse_gt_label(label_path, w, h)
        is_negative = (len(gt_objs) == 0)

        for name, model in models.items():
            res = results[name]
            if is_negative:
                res["neg_total"] += 1

            t_pred_start = time.perf_counter()
            preds = model.predict(img_bgr)
            t_pred = time.perf_counter() - t_pred_start
            res["total_inference_time"] += t_pred

            if is_negative:
                if len(preds) > 0:
                    res["neg_fp"] += 1
                res["fp_50"] += len(preds)
                res["fp_75"] += len(preds)
                continue

            # Bipartite matching for detection
            n_gt = len(gt_objs)
            n_pred = len(preds)

            matched_gt_50 = set()
            matched_pred_50 = set()
            matched_gt_75 = set()
            matched_pred_75 = set()

            # Pairwise IoU matrix
            iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
            for p_i, p in enumerate(preds):
                for g_i, g in enumerate(gt_objs):
                    iou_mat[p_i, g_i] = compute_iou(p["bbox"], g["bbox"])

            # Greedy match for IoU 0.50
            if n_pred > 0 and n_gt > 0:
                flat_order = np.argsort(-iou_mat.flatten())
                for flat_idx in flat_order:
                    p_i = flat_idx // n_gt
                    g_i = flat_idx % n_gt
                    if p_i in matched_pred_50 or g_i in matched_gt_50:
                        continue
                    if iou_mat[p_i, g_i] >= 0.50:
                        matched_pred_50.add(p_i)
                        matched_gt_50.add(g_i)
                        res["tp_50"] += 1
                        res["ious"].append(float(iou_mat[p_i, g_i]))

                        # Keypoint Evaluation on matched detection
                        g = gt_objs[g_i]
                        p = preds[p_i]
                        gt_kpts = g["keypoints"]
                        pred_kpts = p["keypoints"]
                        diag = g["scale_diag"]
                        scale_max = g["scale_max"]

                        res["total_matched_feet"] += 1
                        if p["class_id"] == g["class_id"]:
                            res["correct_chirality"] += 1

                        for k in range(13):
                            if gt_kpts[k, 2] > 0:  # visible keypoint
                                res["kpt_vis_total"] += 1
                                dist = math.sqrt((pred_kpts[k, 0] - gt_kpts[k, 0]) ** 2 +
                                                 (pred_kpts[k, 1] - gt_kpts[k, 1]) ** 2)
                                res["mae_sum"] += dist

                                # PCK vs diagonal
                                if dist <= 0.20 * diag:
                                    res["pck20_hits"] += 1
                                if dist <= 0.10 * diag:
                                    res["pck10_hits"] += 1
                                if dist <= 0.05 * diag:
                                    res["pck05_hits"] += 1

                                # PCK vs max(w, h)
                                if dist <= 0.20 * scale_max:
                                    res["pck20_max_hits"] += 1
                                if dist <= 0.10 * scale_max:
                                    res["pck10_max_hits"] += 1
                                if dist <= 0.05 * scale_max:
                                    res["pck05_max_hits"] += 1

            # Greedy match for IoU 0.75
            if n_pred > 0 and n_gt > 0:
                flat_order = np.argsort(-iou_mat.flatten())
                for flat_idx in flat_order:
                    p_i = flat_idx // n_gt
                    g_i = flat_idx % n_gt
                    if p_i in matched_pred_75 or g_i in matched_gt_75:
                        continue
                    if iou_mat[p_i, g_i] >= 0.75:
                        matched_pred_75.add(p_i)
                        matched_gt_75.add(g_i)
                        res["tp_75"] += 1

            res["fp_50"] += (n_pred - len(matched_pred_50))
            res["fn_50"] += (n_gt - len(matched_gt_50))
            res["fp_75"] += (n_pred - len(matched_pred_75))
            res["fn_75"] += (n_gt - len(matched_gt_75))

        if (idx + 1) % 100 == 0 or (idx + 1) == total_imgs:
            elapsed = time.time() - t0
            print(f"[*] Processed {idx + 1:03d}/{total_imgs} images ({elapsed:.1f}s)...")

    # ── Compute Final Consolidated Metrics ─────────────────────────────────────
    print("\n" + "=" * 92)
    print("  FINAL COMPARATIVE BENCHMARK REPORT (827 VALIDATION IMAGES)")
    print("=" * 92)

    summary_data = {}
    for name, res in results.items():
        # Detection @ 0.50
        prec_50 = res["tp_50"] / max(1e-9, res["tp_50"] + res["fp_50"])
        rec_50 = res["tp_50"] / max(1e-9, res["tp_50"] + res["fn_50"])
        f1_50 = 2 * prec_50 * rec_50 / max(1e-9, prec_50 + rec_50)

        # Detection @ 0.75
        prec_75 = res["tp_75"] / max(1e-9, res["tp_75"] + res["fp_75"])
        rec_75 = res["tp_75"] / max(1e-9, res["tp_75"] + res["fn_75"])
        f1_75 = 2 * prec_75 * rec_75 / max(1e-9, prec_75 + rec_75)

        miou = float(np.mean(res["ious"])) if res["ious"] else 0.0
        neg_fa = (res["neg_fp"] / max(1, res["neg_total"])) * 100.0

        # Pose Metrics
        n_kpts = max(1, res["kpt_vis_total"])
        pck20 = (res["pck20_hits"] / n_kpts) * 100.0
        pck10 = (res["pck10_hits"] / n_kpts) * 100.0
        pck05 = (res["pck05_hits"] / n_kpts) * 100.0

        pck20_max = (res["pck20_max_hits"] / n_kpts) * 100.0
        pck10_max = (res["pck10_max_hits"] / n_kpts) * 100.0
        pck05_max = (res["pck05_max_hits"] / n_kpts) * 100.0

        mae = res["mae_sum"] / n_kpts
        chir_acc = (res["correct_chirality"] / max(1, res["total_matched_feet"])) * 100.0
        avg_lat_ms = (res["total_inference_time"] / total_imgs) * 1000.0

        summary_data[name] = {
            "F1@0.50": f1_50 * 100.0,
            "Precision@0.50": prec_50 * 100.0,
            "Recall@0.50": rec_50 * 100.0,
            "F1@0.75": f1_75 * 100.0,
            "Precision@0.75": prec_75 * 100.0,
            "Recall@0.75": rec_75 * 100.0,
            "mIoU": miou * 100.0,
            "Neg_FA_Rate": neg_fa,
            "PCK@0.20_diag": pck20,
            "PCK@0.10_diag": pck10,
            "PCK@0.05_diag": pck05,
            "PCK@0.20_max": pck20_max,
            "PCK@0.10_max": pck10_max,
            "PCK@0.05_max": pck05_max,
            "MAE_pixels": mae,
            "Chirality_Acc": chir_acc,
            "Avg_Latency_ms": avg_lat_ms
        }

    # Save summary to JSON first
    out_json = Path("deliverables/qualcomm_experiments/benchmark_val_summary.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[+] Benchmark summary saved to: {out_json}")

    for name, s in summary_data.items():
        print(f"\n[{name}]")
        print(f"  -- Detection Metrics --")
        print(f"     F1@0.50:          {s['F1@0.50']:.2f}%  (Precision: {s['Precision@0.50']:.2f}%, Recall: {s['Recall@0.50']:.2f}%)")
        print(f"     F1@0.75:          {s['F1@0.75']:.2f}%  (Precision: {s['Precision@0.75']:.2f}%, Recall: {s['Recall@0.75']:.2f}%)")
        print(f"     Mean BBox IoU:    {s['mIoU']:.2f}%")
        print(f"     Neg False Alarm:  {s['Neg_FA_Rate']:.2f}%")
        print(f"  -- Pose Metrics (PCK) --")
        print(f"     PCK@0.20 (diag):  {s['PCK@0.20_diag']:.2f}%  | PCK@0.20 (max_dim): {s['PCK@0.20_max']:.2f}%")
        print(f"     PCK@0.10 (diag):  {s['PCK@0.10_diag']:.2f}%  | PCK@0.10 (max_dim): {s['PCK@0.10_max']:.2f}%")
        print(f"     PCK@0.05 (diag):  {s['PCK@0.05_diag']:.2f}%  | PCK@0.05 (max_dim): {s['PCK@0.05_max']:.2f}%")
        print(f"     Mean Pixel Error: {s['MAE_pixels']:.2f} px")
        print(f"     Chirality Acc:    {s['Chirality_Acc']:.2f}%")
        print(f"  -- Speed / Latency --")
        print(f"     Avg Image Latency:{s['Avg_Latency_ms']:.2f} ms ({1000.0/max(1.0, s['Avg_Latency_ms']):.1f} FPS)")


if __name__ == "__main__":
    run_benchmark()
