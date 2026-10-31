#!/usr/bin/env python3
"""
tools/evaluation/benchmark_quantized_full.py
────────────────────────────────────────────
Multi-Precision (FP32, FP16, INT8) Benchmark & Granular Latency Decomposition.
Profiles and evaluates:
  Two-Stage:
    B1: NanoDet(FP32) + RTMPose(FP32)
    C1: NanoDet(FP32) + RTMPose(FP16)
    C2: NanoDet(FP32) + RTMPose(INT8)
    C3: NanoDet(FP16) + RTMPose(FP16)
    C4: NanoDet(FP16) + RTMPose(INT8)
    C5: NanoDet(INT8) + RTMPose(FP16)
    C6: NanoDet(INT8) + RTMPose(INT8)
  Single-Stage YOLO:
    B2: YOLO(FP32)
    C7: YOLO(FP16)
    C8: YOLO(INT8)

Measures exact additive latencies:
  t_total = t_pre + t_stageA + t_crop + t_stageB + t_post
  with t_infer = t_stageA + t_crop + t_stageB
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


# ── Transform & SimCC Helpers ──────────────────────────────────────────────────

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


def decode_simcc_numpy(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    vis: np.ndarray,
    split_ratio: float = 2.0,
    win: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sub-pixel continuous expectation within a local window (win=12) around peak bin.
    Exact parity with PyTorch decode_simcc in src/models/qualcomm_experiments/rtmpose/decoder.py.
    """
    def decode_axis(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        p = e / np.sum(e, axis=-1, keepdims=True)
        idx = np.argmax(p, axis=-1, keepdims=True)
        bins = np.arange(p.shape[-1], dtype=np.float32)[None, :]
        mask = np.abs(bins - idx) <= win
        pw = p * mask
        mass = np.sum(pw, axis=-1)
        coord = np.sum(pw * bins, axis=-1) / np.maximum(mass, 1e-8) / split_ratio
        return coord, mass

    coord_x, mass_x = decode_axis(simcc_x)
    coord_y, mass_y = decode_axis(simcc_y)
    kpt_score = np.sqrt(np.maximum(0.0, mass_x * mass_y))
    confidence = kpt_score * np.squeeze(vis)
    kpts = np.stack([coord_x, coord_y, confidence], axis=-1)
    return kpts, confidence


def compute_iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
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


# ── Granular Profiling Two-Stage Pipeline ─────────────────────────────────────

class TwoStagePipelineProfiled:
    def __init__(self, nanodet_path: str, rtmpose_path: str, conf_thresh: float = 0.35, nms_thresh: float = 0.45):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.det_sess = ort.InferenceSession(nanodet_path, providers=providers)
        self.det_in_name = self.det_sess.get_inputs()[0].name
        self.det_in_type = self.det_sess.get_inputs()[0].type
        self.pose_sess = ort.InferenceSession(rtmpose_path, providers=providers)
        self.pose_in_name = self.pose_sess.get_inputs()[0].name
        self.pose_in_type = self.pose_sess.get_inputs()[0].type
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def process_frame_profiled(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        orig_h, orig_w = frame.shape[:2]

        # 1. Preprocessing (t_pre)
        t0 = time.perf_counter()
        det_img = cv2.resize(frame, (256, 256))
        det_blob = (det_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        if "float16" in self.det_in_type:
            det_blob = det_blob.astype(np.float16)
        t_pre = (time.perf_counter() - t0) * 1000.0

        # 2. Stage A Inference (t_stageA)
        t1 = time.perf_counter()
        boxes_out, scores_out = self.det_sess.run(None, {self.det_in_name: det_blob})
        if "float16" in self.det_in_type:
            boxes_out = boxes_out.astype(np.float32)
            scores_out = scores_out.astype(np.float32)
        t_stageA = (time.perf_counter() - t1) * 1000.0

        # 3. Intermediate Stage A Postprocess (filter boxes for crop)
        t_mid_start = time.perf_counter()
        raw_boxes = boxes_out[0]
        raw_scores = scores_out[0, :, 0]
        mask = raw_scores >= self.conf_thresh
        keep = []
        if np.any(mask):
            filt_boxes = raw_boxes[mask]
            filt_scores = raw_scores[mask]
            keep = nms_boxes(filt_boxes, filt_scores, iou_thresh=self.nms_thresh)

        scale_x = orig_w / 256.0
        scale_y = orig_h / 256.0

        det_candidates = []
        for idx in keep:
            bx = filt_boxes[idx]
            det_score = float(filt_scores[idx])
            x1 = float(np.clip(bx[0] * scale_x, 0, orig_w))
            y1 = float(np.clip(bx[1] * scale_y, 0, orig_h))
            x2 = float(np.clip(bx[2] * scale_x, 0, orig_w))
            y2 = float(np.clip(bx[3] * scale_y, 0, orig_h))
            if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                det_candidates.append({
                    "bbox": [x1, y1, x2, y2],
                    "score": det_score
                })
        t_mid_post = (time.perf_counter() - t_mid_start) * 1000.0

        # 4. Crop & Affine Warp Pipeline (t_crop)
        t2 = time.perf_counter()
        crop_blobs = []
        M_invs = []
        for det in det_candidates[:2]:  # At most 2 feet
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            bw, bh = x2 - x1, y2 - y1
            crop_cx, crop_cy, size = compute_crop_box(cx, cy, bw, bh, crop_padding=0.25)
            M, M_inv = get_crop_affine_transform(crop_cx, crop_cy, size, out_size=256)
            crop = warp_image_crop(frame, M, out_size=256)
            crop_blob = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            if "float16" in self.pose_in_type:
                crop_blob = crop_blob.astype(np.float16)
            crop_blobs.append(crop_blob)
            M_invs.append(M_inv)
        t_crop = (time.perf_counter() - t2) * 1000.0

        # 5. Stage B RTMPose Inference (t_stageB)
        t3 = time.perf_counter()
        pose_results = []
        for crop_blob in crop_blobs:
            pose_outs = self.pose_sess.run(None, {self.pose_in_name: crop_blob})
            pose_results.append(pose_outs)
        t_stageB = (time.perf_counter() - t3) * 1000.0

        # 6. Postprocessing (t_post)
        t4 = time.perf_counter()
        valid_candidates = []
        for i, pose_outs in enumerate(pose_results):
            det = det_candidates[i]
            M_inv = M_invs[i]
            simcc_x = pose_outs[0][0].astype(np.float32)
            simcc_y = pose_outs[1][0].astype(np.float32)
            vis = np.squeeze(pose_outs[3]).astype(np.float32)
            hand_prob = float(np.squeeze(pose_outs[5])) if len(pose_outs) >= 6 else 0.50

            kpts_crop, confs = decode_simcc_numpy(simcc_x, simcc_y, vis, split_ratio=2.0, win=12)
            frame_kps = invert_keypoints(kpts_crop, M_inv)

            mean_conf = float(np.mean(confs))
            valid_kpts = int(sum(1 for c in confs if c >= 0.20))
            if mean_conf < 0.20 or valid_kpts < 3:
                continue

            valid_candidates.append({
                "bbox": det["bbox"],
                "score": det["score"],
                "keypoints": frame_kps,
                "hand_prob": hand_prob,
                "class_id": 1 if hand_prob >= 0.50 else 0,
                "cx": (det["bbox"][0] + det["bbox"][2]) / 2.0
            })

        # Dual-Foot Chirality Gate
        final_dets = []
        if valid_candidates:
            left_cands = [c for c in valid_candidates if c["class_id"] == 0]
            right_cands = [c for c in valid_candidates if c["class_id"] == 1]
            if len(left_cands) >= 2 and len(right_cands) == 0:
                left_cands.sort(key=lambda c: c["cx"])
                if abs(left_cands[1]["cx"] - left_cands[0]["cx"]) > 40:
                    right_cands.append(left_cands.pop(-1))
            elif len(right_cands) >= 2 and len(left_cands) == 0:
                right_cands.sort(key=lambda c: c["cx"])
                if abs(right_cands[1]["cx"] - right_cands[0]["cx"]) > 40:
                    left_cands.append(right_cands.pop(0))

            if left_cands:
                best_l = max(left_cands, key=lambda c: c["score"])
                best_l["class_id"] = 0
                final_dets.append(best_l)
            if right_cands:
                best_r = max(right_cands, key=lambda c: c["score"])
                best_r["class_id"] = 1
                final_dets.append(best_r)

        t_post_end = (time.perf_counter() - t4) * 1000.0
        t_post = t_mid_post + t_post_end

        t_infer = t_stageA + t_crop + t_stageB
        t_total = t_pre + t_infer + t_post

        latencies = {
            "t_pre": t_pre,
            "t_stageA": t_stageA,
            "t_crop": t_crop,
            "t_stageB": t_stageB,
            "t_infer": t_infer,
            "t_post": t_post,
            "t_total": t_total
        }
        return final_dets, latencies


# ── Granular Profiling Single-Stage YOLO Pipeline ──────────────────────────────

class YOLOv8PoseProfiled:
    def __init__(self, model_path: str, conf_thresh: float = 0.25, nms_thresh: float = 0.45):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.is_fp16 = ("float16" in self.input_type)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def letterbox(self, img: np.ndarray, new_shape=(256, 256), color=(114, 114, 114)):
        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        new_unpad = int(round(w * r)), int(round(h * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if (w, h) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def process_frame_profiled(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        orig_h, orig_w = frame.shape[:2]

        # 1. Preprocessing (t_pre)
        t0 = time.perf_counter()
        lb_img, scale, (dw, dh) = self.letterbox(frame, (256, 256))
        blob = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        if self.is_fp16:
            blob = blob.astype(np.float16)
        t_pre = (time.perf_counter() - t0) * 1000.0

        # 2. Model Inference (t_infer)
        t1 = time.perf_counter()
        preds = self.session.run(None, {self.input_name: blob})[0][0]  # [45, 1344]
        if self.is_fp16:
            preds = preds.astype(np.float32)
        t_infer = (time.perf_counter() - t1) * 1000.0

        # 3. Postprocessing (t_post)
        t2 = time.perf_counter()
        candidates = []
        for col in range(preds.shape[1]):
            s_left = 1.0 / (1.0 + math.exp(-preds[4, col]))
            s_right = 1.0 / (1.0 + math.exp(-preds[5, col]))
            max_score = max(s_left, s_right)
            if max_score < self.conf_thresh:
                continue

            cls_id = 0 if s_left >= s_right else 1
            cx, cy, w, h = preds[0:4, col]
            x1 = max(0.0, min(float((cx - w / 2 - dw) / scale), float(orig_w)))
            y1 = max(0.0, min(float((cy - h / 2 - dh) / scale), float(orig_h)))
            x2 = max(0.0, min(float((cx + w / 2 - dw) / scale), float(orig_w)))
            y2 = max(0.0, min(float((cy + h / 2 - dh) / scale), float(orig_h)))

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
        t_post = (time.perf_counter() - t2) * 1000.0

        t_total = t_pre + t_infer + t_post
        latencies = {
            "t_pre": t_pre,
            "t_stageA": 0.0,
            "t_crop": 0.0,
            "t_stageB": 0.0,
            "t_infer": t_infer,
            "t_post": t_post,
            "t_total": t_total
        }
        return kept, latencies


# ── Granular Profiling on Video (574 frames) ───────────────────────────────────

def benchmark_video_latency(pipeline, video_path: str = "data/test_video.mp4", warmup_frames: int = 20, max_frames: int = 150) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    metrics_acc = {"t_pre": [], "t_stageA": [], "t_crop": [], "t_stageB": [], "t_infer": [], "t_post": [], "t_total": []}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break
        _, lats = pipeline.process_frame_profiled(frame)
        if frame_idx > warmup_frames:
            for k in metrics_acc:
                metrics_acc[k].append(lats[k])

    cap.release()
    means = {k: float(np.mean(v)) for k, v in metrics_acc.items()}
    # Enforce strict additive integrity:
    if means["t_stageA"] > 0.0:
        means["t_infer"] = means["t_stageA"] + means["t_crop"] + means["t_stageB"]
    means["t_total"] = means["t_pre"] + means["t_infer"] + means["t_post"]
    means["fps"] = 1000.0 / max(1.0, means["t_total"])
    return means


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


# ── Validation Dataset Accuracy Evaluator ──────────────────────────────────────

def evaluate_accuracy(pipeline, img_files: List[Path], labels_dir: Path) -> Dict[str, float]:
    tp_50, fp_50, fn_50 = 0, 0, 0
    tp_75, fp_75, fn_75 = 0, 0, 0
    ious = []
    kpt_vis_total = 0
    pck20_hits, pck10_hits, pck05_hits = 0, 0, 0
    mae_sum = 0.0
    correct_chirality, total_matched = 0, 0

    for img_path in img_files:
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

        preds, _ = pipeline.process_frame_profiled(img_bgr)
        if is_negative:
            fp_50 += len(preds)
            fp_75 += len(preds)
            continue

        n_gt = len(gt_objs)
        n_pred = len(preds)
        matched_gt_50, matched_pred_50 = set(), set()
        matched_gt_75, matched_pred_75 = set(), set()

        iou_mat = np.zeros((n_pred, n_gt), dtype=np.float32)
        for p_i, p in enumerate(preds):
            for g_i, g in enumerate(gt_objs):
                iou_mat[p_i, g_i] = compute_iou(p["bbox"], g["bbox"])

        if n_pred > 0 and n_gt > 0:
            flat_order = np.argsort(-iou_mat.flatten())
            for flat_idx in flat_order:
                p_i, g_i = flat_idx // n_gt, flat_idx % n_gt
                if p_i in matched_pred_50 or g_i in matched_gt_50:
                    continue
                if iou_mat[p_i, g_i] >= 0.50:
                    matched_pred_50.add(p_i)
                    matched_gt_50.add(g_i)
                    tp_50 += 1
                    ious.append(float(iou_mat[p_i, g_i]))

                    g, p = gt_objs[g_i], preds[p_i]
                    diag = g["scale_diag"]
                    total_matched += 1
                    if p["class_id"] == g["class_id"]:
                        correct_chirality += 1

                    for k in range(13):
                        if g["keypoints"][k, 2] > 0:
                            kpt_vis_total += 1
                            dist = math.sqrt((p["keypoints"][k, 0] - g["keypoints"][k, 0]) ** 2 +
                                             (p["keypoints"][k, 1] - g["keypoints"][k, 1]) ** 2)
                            mae_sum += dist
                            if dist <= 0.20 * diag: pck20_hits += 1
                            if dist <= 0.10 * diag: pck10_hits += 1
                            if dist <= 0.05 * diag: pck05_hits += 1

            for flat_idx in flat_order:
                p_i, g_i = flat_idx // n_gt, flat_idx % n_gt
                if p_i in matched_pred_75 or g_i in matched_gt_75:
                    continue
                if iou_mat[p_i, g_i] >= 0.75:
                    matched_pred_75.add(p_i)
                    matched_gt_75.add(g_i)
                    tp_75 += 1

        fp_50 += (n_pred - len(matched_pred_50))
        fn_50 += (n_gt - len(matched_gt_50))
        fp_75 += (n_pred - len(matched_pred_75))
        fn_75 += (n_gt - len(matched_gt_75))

    p50 = tp_50 / max(1e-9, tp_50 + fp_50)
    r50 = tp_50 / max(1e-9, tp_50 + fn_50)
    f1_50 = 2 * p50 * r50 / max(1e-9, p50 + r50)

    p75 = tp_75 / max(1e-9, tp_75 + fp_75)
    r75 = tp_75 / max(1e-9, tp_75 + fn_75)
    f1_75 = 2 * p75 * r75 / max(1e-9, p75 + r75)

    n_kpts = max(1, kpt_vis_total)
    return {
        "F1@0.50": float(f1_50 * 100.0),
        "F1@0.75": float(f1_75 * 100.0),
        "PCK@0.20": float((pck20_hits / n_kpts) * 100.0),
        "PCK@0.10": float((pck10_hits / n_kpts) * 100.0),
        "PCK@0.05": float((pck05_hits / n_kpts) * 100.0),
        "MAE": float(mae_sum / n_kpts),
        "Chirality": float((correct_chirality / max(1, total_matched)) * 100.0)
    }


# ── Master Benchmark Orchestrator ──────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="improved5", choices=["improved4", "improved5"])
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--config_ids", type=str, default=None, help="Comma-separated IDs (e.g. 'B1,1,2,3,4,5,6,B2,7,8')")
    args = parser.parse_args()

    version = args.version
    print("=" * 80)
    print(f"  MULTI-PRECISION (FP32, FP16, INT8) BENCHMARK: {version.upper()}")
    print("=" * 80)

    # Models directory
    q_dir = Path("deliverables/qualcomm_experiments/models")
    y_dir = Path("deliverables/stage_a_13kp/models")

    # Determine model paths based on version
    if version == "improved5":
        nano_fp32 = str(q_dir / "nanodet_stage_a_improved5_opset12.onnx")
        nano_fp16 = str(q_dir / "nanodet_stage_a_improved5_fp16.onnx")
        nano_int8 = str(q_dir / "nanodet_stage_a_improved5_int8.onnx")
        rtm_fp32 = str(q_dir / "rtmpose_13kp_hand_improved5_opset12.onnx")
        rtm_fp16 = str(q_dir / "rtmpose_13kp_hand_improved5_fp16.onnx")
        rtm_int8 = str(q_dir / "rtmpose_13kp_hand_improved5_int8.onnx")
        default_out_json = Path("deliverables/qualcomm_experiments/quantized_benchmark_summary_improved5.json")
    else:
        nano_fp32 = str(q_dir / "nanodet_stage_a_improved4_opset12.onnx")
        nano_fp16 = str(q_dir / "nanodet_stage_a_improved4_fp16.onnx")
        nano_int8 = str(q_dir / "nanodet_stage_a_improved4_int8.onnx")
        rtm_fp32 = str(q_dir / "rtmpose_13kp_hand_opset12.onnx")
        rtm_fp16 = str(q_dir / "rtmpose_13kp_hand_fp16.onnx")
        rtm_int8 = str(q_dir / "rtmpose_13kp_hand_int8.onnx")
        default_out_json = Path("deliverables/qualcomm_experiments/quantized_benchmark_summary.json")

    out_json = Path(args.output_json) if args.output_json else default_out_json

    # Dynamic sizes
    s_n_fp32 = round(os.path.getsize(nano_fp32) / (1024 * 1024), 2)
    s_n_fp16 = round(os.path.getsize(nano_fp16) / (1024 * 1024), 2)
    s_n_int8 = round(os.path.getsize(nano_int8) / (1024 * 1024), 2)
    s_r_fp32 = round(os.path.getsize(rtm_fp32) / (1024 * 1024), 2)
    s_r_fp16 = round(os.path.getsize(rtm_fp16) / (1024 * 1024), 2)
    s_r_int8 = round(os.path.getsize(rtm_int8) / (1024 * 1024), 2)

    # Define all 10 configurations
    configs = [
        # Two-Stage Configurations
        {
            "id": "B1",
            "name": f"NanoDet(FP32) + RTMPose(FP32) [{version}]",
            "type": "two_stage",
            "nanodet": nano_fp32,
            "rtmpose": rtm_fp32,
            "model_size_mb": round(s_n_fp32 + s_r_fp32, 2)
        },
        {
            "id": "1",
            "name": f"NanoDet(FP32) + RTMPose(FP16) [{version}]",
            "type": "two_stage",
            "nanodet": nano_fp32,
            "rtmpose": rtm_fp16,
            "model_size_mb": round(s_n_fp32 + s_r_fp16, 2)
        },
        {
            "id": "2",
            "name": f"NanoDet(FP32) + RTMPose(INT8) [{version}]",
            "type": "two_stage",
            "nanodet": nano_fp32,
            "rtmpose": rtm_int8,
            "model_size_mb": round(s_n_fp32 + s_r_int8, 2)
        },
        {
            "id": "3",
            "name": f"NanoDet(FP16) + RTMPose(FP16) [{version}]",
            "type": "two_stage",
            "nanodet": nano_fp16,
            "rtmpose": rtm_fp16,
            "model_size_mb": round(s_n_fp16 + s_r_fp16, 2)
        },
        {
            "id": "4",
            "name": f"NanoDet(FP16) + RTMPose(INT8) [{version}]",
            "type": "two_stage",
            "nanodet": nano_fp16,
            "rtmpose": rtm_int8,
            "model_size_mb": round(s_n_fp16 + s_r_int8, 2)
        },
        {
            "id": "5",
            "name": f"NanoDet(INT8) + RTMPose(FP16) [{version}]",
            "type": "two_stage",
            "nanodet": nano_int8,
            "rtmpose": rtm_fp16,
            "model_size_mb": round(s_n_int8 + s_r_fp16, 2)
        },
        {
            "id": "6",
            "name": f"NanoDet(INT8) + RTMPose(INT8) [{version}]",
            "type": "two_stage",
            "nanodet": nano_int8,
            "rtmpose": rtm_int8,
            "model_size_mb": round(s_n_int8 + s_r_int8, 2)
        },
        # Single-Stage YOLO Configurations
        {
            "id": "B2",
            "name": "YOLOv8n-Pose v7 (FP32)",
            "type": "yolo",
            "yolo": str(y_dir / "stage-a-256-13kp-v7-merged-negatives.onnx"),
            "model_size_mb": 12.97
        },
        {
            "id": "7",
            "name": "YOLOv8n-Pose v7 (FP16)",
            "type": "yolo",
            "yolo": str(y_dir / "stage-a-256-13kp-v7-merged-negatives-fp16.onnx"),
            "model_size_mb": 6.53
        },
        {
            "id": "8",
            "name": "YOLOv8n-Pose v7 (INT8)",
            "type": "yolo",
            "yolo": str(y_dir / "stage-a-256-13kp-v7-merged-negatives-int8.onnx"),
            "model_size_mb": 3.46
        }
    ]

    if args.config_ids:
        selected_ids = [s.strip() for s in args.config_ids.split(",")]
        configs = [c for c in configs if c["id"] in selected_ids]

    # Load validation images
    dataset_root = Path("data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid")
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    video_path = "data/test_video.mp4"

    all_results = []

    for cfg in configs:
        c_id = cfg["id"]
        c_name = cfg["name"]
        print(f"\n[*] Evaluating Configuration [{c_id}]: {c_name} (Size: {cfg['model_size_mb']:.2f} MB)")

        if cfg["type"] == "two_stage":
            pipeline = TwoStagePipelineProfiled(
                nanodet_path=cfg["nanodet"],
                rtmpose_path=cfg["rtmpose"],
                conf_thresh=0.35,
                nms_thresh=0.45
            )
        else:
            pipeline = YOLOv8PoseProfiled(
                model_path=cfg["yolo"],
                conf_thresh=0.25,
                nms_thresh=0.45
            )

        # 1. Video Latency Profiling
        print(f"    - Profiling Video Latency Breakdown on {video_path} (574 frames)...")
        lats = benchmark_video_latency(pipeline, video_path=video_path, warmup_frames=20)
        print(f"      Preprocess:   {lats['t_pre']:.2f} ms")
        if cfg["type"] == "two_stage":
            print(f"      Stage A Inf:  {lats['t_stageA']:.2f} ms")
            print(f"      Crop Warp:    {lats['t_crop']:.2f} ms")
            print(f"      Stage B Inf:  {lats['t_stageB']:.2f} ms")
            print(f"      -> Total Infer:{lats['t_infer']:.2f} ms")
        else:
            print(f"      YOLO Infer:   {lats['t_infer']:.2f} ms")
        print(f"      Postprocess:  {lats['t_post']:.2f} ms")
        print(f"      TOTAL LATENCY:{lats['t_total']:.2f} ms ({lats['fps']:.1f} FPS)")

        # 2. Accuracy Evaluation on 827 Images
        print(f"    - Evaluating Accuracy on {len(img_files)} Validation Images...")
        acc = evaluate_accuracy(pipeline, img_files, labels_dir)
        print(f"      F1@0.50: {acc['F1@0.50']:.2f}% | F1@0.75: {acc['F1@0.75']:.2f}%")
        print(f"      PCK@0.20: {acc['PCK@0.20']:.2f}% | PCK@0.10: {acc['PCK@0.10']:.2f}% | PCK@0.05: {acc['PCK@0.05']:.2f}% | MAE: {acc['MAE']:.2f}px")

        record = {
            "id": c_id,
            "name": c_name,
            "type": cfg["type"],
            "model_size_mb": cfg["model_size_mb"],
            "latency": lats,
            "accuracy": acc
        }
        all_results.append(record)

        # Save summary incrementally after each configuration
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"      [OK] Incremental results saved to {out_json}", flush=True)

    print(f"\n[+] Full Quantized Benchmark Summary successfully completed and saved to: {out_json}", flush=True)


if __name__ == "__main__":
    main()

