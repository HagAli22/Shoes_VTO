#!/usr/bin/env python3
"""
tools/evaluation/generate_quantized_videos.py
─────────────────────────────────────────────
Generates full visual demonstration videos for all 10 multi-precision benchmark configurations
(FP32, FP16, INT8 for Two-Stage Qualcomm Pipeline and Single-Stage YOLOv8n-pose Baseline).

Pipeline & Post-Processing Parity:
  • Two-Stage: NanoDet ONNX + RTMPose ONNX + Anatomical Clutter Gate + Dual-Foot Chirality Gate
               + Vectorized 1-Euro Temporal Filter + Tracking Extrapolation.
  • Single-Stage: YOLOv8n-pose ONNX + cv2.dnn.NMSBoxes + 1-Euro Temporal Filter.
  • Real-time Telemetry HUD with configuration name, model size, frame counter, latency & rolling FPS.
"""

import sys
import os
import time
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.qualcomm_experiments.common.crop import (
    compute_crop_box,
    get_crop_affine_transform,
    warp_image_crop,
    invert_keypoints
)

KEYPOINT_NAMES = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "instep_top", "arch_medial", "midfoot_lateral",
    "malleolus_medial", "malleolus_lateral", "toe_tip", "throat", "achilles"
]

SKELETON_PAIRS = [
    # Sole / Foot contour
    (2, 1),   # heel_ground -> heel_back
    (2, 6),   # heel_ground -> arch_medial
    (6, 3),   # arch_medial -> ball_medial
    (3, 10),  # ball_medial -> toe_tip
    (10, 0),  # toe_tip -> toe_ground
    (0, 4),   # toe_ground -> ball_lateral
    (4, 7),   # ball_lateral -> midfoot_lateral
    (7, 2),   # midfoot_lateral -> heel_ground
    # Dorsum / Instep arch
    (10, 5),  # toe_tip -> instep_top
    (5, 11),  # instep_top -> throat
    # Ankle & Achilles
    (8, 11),  # malleolus_medial -> throat
    (9, 11),  # malleolus_lateral -> throat
    (8, 12),  # malleolus_medial -> achilles
    (9, 12),  # malleolus_lateral -> achilles
    (1, 12),  # heel_back -> achilles
]

COLOR_LEFT = {
    "bbox": (255, 191, 0),       # Cyan / Sky Blue (BGR)
    "skeleton": (255, 140, 0),
    "kpt_vis": (0, 255, 255),    # Yellow
    "kpt_occ": (0, 180, 180),
    "label": "Left Foot"
}

COLOR_RIGHT = {
    "bbox": (0, 140, 255),       # Orange
    "skeleton": (50, 205, 50),   # Lime Green
    "kpt_vis": (0, 255, 0),      # Bright Green
    "kpt_occ": (0, 180, 0),
    "label": "Right Foot"
}


# ── SimCC NumPy Decoder (Exact Parity with rtmpose/decoder.py) ──────────────────

def decode_simcc_numpy(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    vis: np.ndarray,
    split_ratio: float = 2.0,
    win: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sub-pixel continuous expectation within a local window (win=12) around peak bin.
    Exact 100% mathematical parity with PyTorch decode_simcc in src/models/qualcomm_experiments/rtmpose/decoder.py.
    """
    def decode_axis(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Stable numerical Softmax
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        p = e / np.sum(e, axis=-1, keepdims=True)  # [13, 512]
        idx = np.argmax(p, axis=-1, keepdims=True)  # [13, 1]
        bins = np.arange(p.shape[-1], dtype=np.float32)[None, :]  # [1, 512]
        mask = np.abs(bins - idx) <= win
        pw = p * mask
        mass = np.sum(pw, axis=-1)  # [13]
        coord = np.sum(pw * bins, axis=-1) / np.maximum(mass, 1e-8) / split_ratio
        return coord, mass

    coord_x, mass_x = decode_axis(simcc_x)
    coord_y, mass_y = decode_axis(simcc_y)
    kpt_score = np.sqrt(np.maximum(0.0, mass_x * mass_y))
    confidence = kpt_score * np.squeeze(vis)
    kpts = np.stack([coord_x, coord_y, confidence], axis=-1)  # [13, 3]
    return kpts, confidence


# ── One-Euro Filter ────────────────────────────────────────────────────────────

class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.s: Optional[np.ndarray] = None

    def filter_with_alpha(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        self.alpha = alpha
        if self.s is None:
            self.s = np.array(x, dtype=np.float64).copy()
        else:
            self.s = alpha * np.array(x, dtype=np.float64) + (1.0 - alpha) * self.s
        return self.s


class OneEuroFilter:
    def __init__(self, t0: float, x0: np.ndarray, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.t_prev = float(t0)
        self.x_filt.filter_with_alpha(x0, 1.0)
        self.dx_filt.filter_with_alpha(np.zeros_like(x0, dtype=np.float64), 1.0)

    def _alpha(self, cutoff: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, t: float, x: np.ndarray) -> np.ndarray:
        dt = max(1e-4, float(t - self.t_prev))
        self.t_prev = float(t)
        x_curr = np.array(x, dtype=np.float64)
        dx = (x_curr - self.x_filt.s) / dt
        edx = self.dx_filt.filter_with_alpha(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(cutoff, dt)
        return self.x_filt.filter_with_alpha(x_curr, alpha)


class FootTrack:
    def __init__(self, track_id: int, t0: float, raw_bbox: np.ndarray, initial_hand_prob: float):
        self.track_id = track_id
        self.box_filter = OneEuroFilter(t0, np.array(raw_bbox, dtype=np.float64), min_cutoff=1.5, beta=0.015, d_cutoff=1.0)
        self.kpt_filter: Optional[OneEuroFilter] = None
        self.conf_filter: Optional[OneEuroFilter] = None
        self.hand_prob = float(initial_hand_prob)
        self.lost_count = 0
        self.age = 1
        self.last_kps: Optional[np.ndarray] = None
        self.smoothed_bbox = tuple(float(v) for v in raw_bbox)

    def update_box(self, t: float, raw_bbox: np.ndarray) -> Tuple[float, float, float, float]:
        smoothed = self.box_filter.filter(t, np.array(raw_bbox, dtype=np.float64))
        self.smoothed_bbox = (float(smoothed[0]), float(smoothed[1]), float(smoothed[2]), float(smoothed[3]))
        return self.smoothed_bbox

    def update_keypoints(self, t: float, kps: np.ndarray, hand_prob: float) -> np.ndarray:
        coords = kps[:, :2]
        confs = kps[:, 2]
        if self.kpt_filter is None:
            self.kpt_filter = OneEuroFilter(t, coords, min_cutoff=0.8, beta=0.01, d_cutoff=1.0)
            self.conf_filter = OneEuroFilter(t, confs, min_cutoff=2.0, beta=0.005, d_cutoff=1.0)
            smoothed_coords = coords
            smoothed_confs = confs
        else:
            smoothed_coords = self.kpt_filter.filter(t, coords)
            smoothed_confs = self.conf_filter.filter(t, confs)

        alpha_hand = 0.20
        self.hand_prob = (1.0 - alpha_hand) * self.hand_prob + alpha_hand * float(hand_prob)
        out = np.zeros_like(kps)
        out[:, :2] = smoothed_coords
        out[:, 2] = np.clip(smoothed_confs, 0.0, 1.0)
        self.last_kps = out
        self.lost_count = 0
        self.age += 1
        return out


def draw_skeleton(frame: np.ndarray, kpts: np.ndarray, is_left: bool = True):
    colors = COLOR_LEFT if is_left else COLOR_RIGHT
    for p1, p2 in SKELETON_PAIRS:
        x1, y1, v1 = kpts[p1]
        x2, y2, v2 = kpts[p2]
        if v1 > 0.20 and v2 > 0.20:
            cv2.line(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), colors["skeleton"], 2, cv2.LINE_AA)

    for i in range(13):
        x, y, v = kpts[i]
        ix, iy = int(round(x)), int(round(y))
        if v >= 0.35:
            cv2.circle(frame, (ix, iy), 5, colors["kpt_vis"], -1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 6, (0, 0, 0), 1, cv2.LINE_AA)
        elif v >= 0.15:
            cv2.circle(frame, (ix, iy), 4, colors["kpt_occ"], 1, cv2.LINE_AA)


def draw_telemetry_hud(
    frame: np.ndarray,
    config_name: str,
    model_size_mb: float,
    frame_idx: int,
    total_frames: int,
    latency_ms: float,
    fps: float,
    active_feet: int,
    arch_type: str = "Two-Stage Qualcomm"
):
    # HUD panel top-left
    panel_w, panel_h = 580, 115
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), (0, 255, 255), 1)

    cv2.putText(frame, f"SHOES VTO BENCHMARK: {config_name}", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Architecture: {arch_type} | Combined Footprint: {model_size_mb:.2f} MB", (25, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | Latency: {latency_ms:.1f}ms ({fps:.1f} FPS)", (25, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active Foot Tracks: {active_feet} (Strict Max: 2 Feet)", (25, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)


# ── Two-Stage Video Processor ──────────────────────────────────────────────────

def process_video_two_stage(
    cfg: Dict[str, Any],
    video_path: Path,
    output_path: Path,
    conf_thresh: float = 0.40,
    nms_thresh: float = 0.45
):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    det_sess = ort.InferenceSession(cfg["nanodet"], providers=providers)
    det_in_name = det_sess.get_inputs()[0].name
    det_is_fp16 = "float16" in det_sess.get_inputs()[0].type

    pose_sess = ort.InferenceSession(cfg["rtmpose"], providers=providers)
    pose_in_name = pose_sess.get_inputs()[0].name
    pose_is_fp16 = "float16" in pose_sess.get_inputs()[0].type

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))

    tracks: Dict[str, FootTrack] = {}
    latencies = []
    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_current = frame_idx / video_fps
        t_frame_start = time.perf_counter()

        # 1. NanoDet Stage A Detection
        det_img = cv2.resize(frame, (256, 256))
        det_blob = (det_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        if det_is_fp16:
            det_blob = det_blob.astype(np.float16)

        boxes_out, scores_out = det_sess.run(None, {det_in_name: det_blob})
        if det_is_fp16:
            boxes_out, scores_out = boxes_out.astype(np.float32), scores_out.astype(np.float32)

        raw_boxes = boxes_out[0]
        raw_scores = scores_out[0, :, 0]
        mask = raw_scores >= conf_thresh

        detected_feet = []
        if np.any(mask):
            f_boxes = raw_boxes[mask]
            f_scores = raw_scores[mask]
            # Convert xyxy to format for cv2.dnn.NMSBoxes
            boxes_w = f_boxes[:, 2] - f_boxes[:, 0]
            boxes_h = f_boxes[:, 3] - f_boxes[:, 1]
            xywh = np.stack([f_boxes[:, 0], f_boxes[:, 1], boxes_w, boxes_h], axis=1).tolist()
            indices = cv2.dnn.NMSBoxes(xywh, f_scores.tolist(), conf_thresh, nms_thresh)
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                scale_x, scale_y = width / 256.0, height / 256.0
                for idx in indices:
                    b = f_boxes[idx]
                    sc = float(f_scores[idx])
                    scaled_box = np.array([b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y], dtype=np.float32)
                    detected_feet.append({"bbox": scaled_box, "score": sc})

        # 2. RTMPose Stage B Inference & Anatomical Gate
        valid_candidates = []
        for det in detected_feet:
            x1, y1, x2, y2 = det["bbox"]
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            crop_cx, crop_cy, size = compute_crop_box(cx, cy, bw, bh, crop_padding=0.25)
            M, M_inv = get_crop_affine_transform(crop_cx, crop_cy, size, out_size=256, rot=0.0)
            crop = warp_image_crop(frame, M, out_size=256, border_value=(114, 114, 114))

            pose_blob = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            if pose_is_fp16:
                pose_blob = pose_blob.astype(np.float16)

            pose_outs = pose_sess.run(None, {pose_in_name: pose_blob})
            simcc_x = pose_outs[0].astype(np.float32)[0]  # [13, 512]
            simcc_y = pose_outs[1].astype(np.float32)[0]  # [13, 512]
            vis = np.squeeze(pose_outs[3]).astype(np.float32)  # [13]
            # Output index 5 is 'hand' (P(Right Foot) in [0, 1])
            hand_prob = float(np.squeeze(pose_outs[5])) if len(pose_outs) >= 6 else 0.50

            # Decode SimCC using Sub-Pixel Continuous Expectation & Softmax Mass
            kpts_crop, confs = decode_simcc_numpy(simcc_x, simcc_y, vis, split_ratio=2.0, win=12)
            frame_kps = invert_keypoints(kpts_crop, M_inv)
            mean_conf = float(np.mean(confs))
            valid_kpts = int(sum(1 for c in confs if c >= 0.20))

            # Anatomical Clutter Gate: Suppress carpet textures, shadows, floor patterns
            if mean_conf < 0.20 or valid_kpts < 3:
                continue

            valid_candidates.append({
                "bbox": det["bbox"],
                "score": det["score"],
                "kps": frame_kps,
                "hand_prob": hand_prob,
                "joint_score": det["score"] * (0.5 + 0.5 * mean_conf),
                "cx": cx
            })

        # 3. Dual-Foot Chirality Partitioning (Strictly max 1 Left, max 1 Right)
        left_cands = [c for c in valid_candidates if c["hand_prob"] < 0.50]
        right_cands = [c for c in valid_candidates if c["hand_prob"] >= 0.50]

        if len(left_cands) >= 2 and len(right_cands) == 0:
            left_cands.sort(key=lambda c: c["cx"])
            if abs(left_cands[1]["cx"] - left_cands[0]["cx"]) > 60:
                right_cands.append(left_cands.pop(-1))
        elif len(right_cands) >= 2 and len(left_cands) == 0:
            right_cands.sort(key=lambda c: c["cx"])
            if abs(right_cands[1]["cx"] - right_cands[0]["cx"]) > 60:
                left_cands.append(right_cands.pop(0))

        best_left = max(left_cands, key=lambda c: c["joint_score"]) if left_cands else None
        best_right = max(right_cands, key=lambda c: c["joint_score"]) if right_cands else None

        active_rendered = 0
        for side_name, best_det, track_id in [("left", best_left, 1), ("right", best_right, 2)]:
            is_left = (side_name == "left")
            colors = COLOR_LEFT if is_left else COLOR_RIGHT

            if best_det is not None:
                if side_name not in tracks:
                    tracks[side_name] = FootTrack(track_id=track_id, t0=t_current, raw_bbox=best_det["bbox"], initial_hand_prob=0.0 if is_left else 1.0)
                s_bbox = tracks[side_name].update_box(t_current, best_det["bbox"])
                smooth_kps = tracks[side_name].update_keypoints(t_current, best_det["kps"], best_det["hand_prob"])
                tracks[side_name].lost_count = 0

                bx1, by1, bx2, by2 = [int(round(v)) for v in s_bbox]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), colors["bbox"], 2)
                cv2.putText(frame, f"ID:{track_id} {colors['label']} ({best_det['score']:.2f})", (bx1, max(25, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 2)
                draw_skeleton(frame, smooth_kps, is_left=is_left)
                active_rendered += 1
            else:
                if side_name in tracks:
                    tracks[side_name].lost_count += 1
                    if tracks[side_name].lost_count <= 6 and tracks[side_name].last_kps is not None:
                        s_bbox = tracks[side_name].smoothed_bbox
                        bx1, by1, bx2, by2 = [int(round(v)) for v in s_bbox]
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), colors["bbox"], 1)
                        cv2.putText(frame, f"ID:{track_id} {colors['label']} (Tracked)", (bx1, max(25, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 1)
                        draw_skeleton(frame, tracks[side_name].last_kps, is_left=is_left)
                        active_rendered += 1
                    elif tracks[side_name].lost_count > 6:
                        del tracks[side_name]

        elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0
        latencies.append(elapsed_ms)
        rolling_fps = 1000.0 / max(1.0, np.mean(latencies[-30:]))

        draw_telemetry_hud(
            frame,
            config_name=cfg["name"],
            model_size_mb=cfg["model_size_mb"],
            frame_idx=frame_idx,
            total_frames=total_frames,
            latency_ms=elapsed_ms,
            fps=rolling_fps,
            active_feet=active_rendered,
            arch_type="Two-Stage Qualcomm (Apache-2.0)"
        )

        writer.write(frame)
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            print(f"    [{cfg['id']}] Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | Latency: {elapsed_ms:.1f}ms ({rolling_fps:.1f} FPS)")

    cap.release()
    writer.release()
    tot_time = time.time() - t_start
    print(f"  [+] Finished [{cfg['id']}]: {output_path.name} in {tot_time:.1f}s ({frame_idx/tot_time:.1f} FPS, Avg Latency: {np.mean(latencies):.1f}ms)")


# ── Single-Stage YOLO Video Processor ──────────────────────────────────────────

def process_video_yolo(
    cfg: Dict[str, Any],
    video_path: Path,
    output_path: Path,
    conf_thresh: float = 0.25,
    nms_thresh: float = 0.45
):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(cfg["yolo"], providers=providers)
    in_name = session.get_inputs()[0].name
    is_fp16 = "float16" in session.get_inputs()[0].type

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, video_fps, (width, height))

    # Stateful track filter
    filters: Dict[int, OneEuroFilter] = {}
    latencies = []
    frame_idx = 0
    t_start = time.time()

    # Precalculate letterbox geometry (256x256)
    r = min(256.0 / height, 256.0 / width)
    unpad_w, unpad_h = int(round(width * r)), int(round(height * r))
    dw, dh = (256.0 - unpad_w) / 2.0, (256.0 - unpad_h) / 2.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_current = frame_idx / video_fps
        t_frame_start = time.perf_counter()

        # Letterbox 256x256
        resized = cv2.resize(frame, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        lb_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        blob = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        if is_fp16:
            blob = blob.astype(np.float16)

        out0 = session.run(None, {in_name: blob})[0]
        if out0.shape[1] == 45:
            preds = out0[0].T  # [1344, 45]
        else:
            preds = out0[0]

        if is_fp16:
            preds = preds.astype(np.float32)

        # Vectorized parsing with fast cv2.dnn.NMSBoxes
        boxes_xywh = preds[:, :4]
        cls_scores = preds[:, 4:6]
        max_scores = np.max(cls_scores, axis=1)
        cls_ids = np.argmax(cls_scores, axis=1)

        mask = max_scores >= conf_thresh
        rendered_count = 0

        if np.any(mask):
            f_xywh = boxes_xywh[mask]
            f_scores = max_scores[mask]
            f_cls = cls_ids[mask]
            f_kpts = preds[mask, 6:]

            # Convert to xywh top-left
            tl_x = f_xywh[:, 0] - f_xywh[:, 2] / 2.0
            tl_y = f_xywh[:, 1] - f_xywh[:, 3] / 2.0
            dnn_boxes = np.stack([tl_x, tl_y, f_xywh[:, 2], f_xywh[:, 3]], axis=1).tolist()

            indices = cv2.dnn.NMSBoxes(dnn_boxes, f_scores.tolist(), conf_thresh, nms_thresh)
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                for idx in indices:
                    sc = float(f_scores[idx])
                    cl = int(f_cls[idx])
                    is_left = (cl == 0)
                    colors = COLOR_LEFT if is_left else COLOR_RIGHT

                    bx, by, bw, bh = dnn_boxes[idx]
                    rx1 = float(np.clip((bx - dw) / r, 0, width))
                    ry1 = float(np.clip((by - dh) / r, 0, height))
                    rx2 = float(np.clip((bx + bw - dw) / r, 0, width))
                    ry2 = float(np.clip((by + bh - dh) / r, 0, height))

                    kp_raw = f_kpts[idx].reshape(-1, 3)
                    kpts_frame = []
                    for k_i in range(13):
                        kx, ky, kv = kp_raw[k_i]
                        vis = float(kv) if (0.0 <= kv <= 1.0) else float(1.0 / (1.0 + math.exp(-kv)))
                        fx = float(np.clip((kx - dw) / r, 0, width))
                        fy = float(np.clip((ky - dh) / r, 0, height))
                        kpts_frame.append([fx, fy, vis])

                    kpts_arr = np.array(kpts_frame, dtype=np.float32)

                    # Smoothing
                    if cl not in filters:
                        filters[cl] = OneEuroFilter(t_current, kpts_arr[:, :2], min_cutoff=0.8, beta=0.01)
                        smooth_coords = kpts_arr[:, :2]
                    else:
                        smooth_coords = filters[cl].filter(t_current, kpts_arr[:, :2])
                    kpts_arr[:, :2] = smooth_coords

                    box_x1, box_y1, box_x2, box_y2 = int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2))
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), colors["bbox"], 2)
                    cv2.putText(frame, f"{colors['label']} ({sc:.2f})", (box_x1, max(25, box_y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 2)
                    draw_skeleton(frame, kpts_arr, is_left=is_left)
                    rendered_count += 1

        elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0
        latencies.append(elapsed_ms)
        rolling_fps = 1000.0 / max(1.0, np.mean(latencies[-30:]))

        draw_telemetry_hud(
            frame,
            config_name=cfg["name"],
            model_size_mb=cfg["model_size_mb"],
            frame_idx=frame_idx,
            total_frames=total_frames,
            latency_ms=elapsed_ms,
            fps=rolling_fps,
            active_feet=rendered_count,
            arch_type="Single-Stage YOLOv8n (AGPLv3)"
        )

        writer.write(frame)
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            print(f"    [{cfg['id']}] Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | Latency: {elapsed_ms:.1f}ms ({rolling_fps:.1f} FPS)")

    cap.release()
    writer.release()
    tot_time = time.time() - t_start
    print(f"  [+] Finished [{cfg['id']}]: {output_path.name} in {tot_time:.1f}s ({frame_idx/tot_time:.1f} FPS, Avg Latency: {np.mean(latencies):.1f}ms)")


# ── Main Orchestrator ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-Precision Video Demo Generator")
    parser.add_argument("--video", type=str, default="data/test_video.mp4", help="Input video path")
    parser.add_argument("--output_dir", type=str, default="deliverables/qualcomm_experiments/demo_videos2", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold (default: 0.40)")
    parser.add_argument("--config_ids", type=str, default=None, help="Comma-separated IDs to run, e.g. B1,1,2,3,4,5,6")
    args = parser.parse_args()

    print("=" * 80)
    print("  MULTI-PRECISION VIDEO DEMO GENERATOR (10 EXPERIMENTAL CONFIGURATIONS)")
    print(f"  Input Video:  {args.video}")
    print(f"  Confidence:   {args.conf}")
    print(f"  Selected IDs: {args.config_ids or 'ALL'}")
    print("=" * 80)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[!] Error: {video_path} does not exist!")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_dir = Path("deliverables/qualcomm_experiments/models")
    y_dir = Path("deliverables/stage_a_13kp/models")

    configs = [
        # Two-Stage Configurations
        {
            "id": "B1",
            "name": "NanoDet(FP32) + RTMPose(FP32)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved4_opset12.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_opset12.onnx"),
            "model_size_mb": 4.16 + 14.93,
            "filename": "B1_twostage_nanodet_fp32_rtmpose_fp32.mp4"
        },
        {
            "id": "1",
            "name": "NanoDet(FP32) + RTMPose(FP16)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved4_opset12.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_fp16.onnx"),
            "model_size_mb": 4.16 + 7.50,
            "filename": "exp1_twostage_nanodet_fp32_rtmpose_fp16.mp4"
        },
        {
            "id": "2",
            "name": "NanoDet(FP32) + RTMPose(INT8)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved4_opset12.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_int8.onnx"),
            "model_size_mb": 4.16 + 3.88,
            "filename": "exp2_twostage_nanodet_fp32_rtmpose_int8.mp4"
        },
        {
            "id": "3",
            "name": "NanoDet(FP16) + RTMPose(FP16)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved4_fp16.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_fp16.onnx"),
            "model_size_mb": 2.25 + 7.50,
            "filename": "exp3_twostage_nanodet_fp16_rtmpose_fp16.mp4"
        },
        {
            "id": "4",
            "name": "NanoDet(FP16) + RTMPose(INT8)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved4_fp16.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_int8.onnx"),
            "model_size_mb": 2.25 + 3.88,
            "filename": "exp4_twostage_nanodet_fp16_rtmpose_int8.mp4"
        },
        {
            "id": "5",
            "name": "NanoDet(INT8) + RTMPose(FP16)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved4_int8.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_fp16.onnx"),
            "model_size_mb": 1.44 + 7.50,
            "filename": "exp5_twostage_nanodet_int8_rtmpose_fp16.mp4"
        },
        {
            "id": "6",
            "name": "NanoDet(INT8) + RTMPose(INT8)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved4_int8.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_int8.onnx"),
            "model_size_mb": 1.44 + 3.88,
            "filename": "exp6_twostage_nanodet_int8_rtmpose_int8.mp4"
        },
        # Single-Stage YOLO Configurations
        {
            "id": "B2",
            "name": "YOLOv8n-Pose v7 (FP32)",
            "type": "yolo",
            "yolo": str(y_dir / "stage-a-256-13kp-v7-merged-negatives.onnx"),
            "model_size_mb": 12.97,
            "filename": "B2_yolov8n_fp32.mp4"
        },
        {
            "id": "7",
            "name": "YOLOv8n-Pose v7 (FP16)",
            "type": "yolo",
            "yolo": str(y_dir / "stage-a-256-13kp-v7-merged-negatives-fp16.onnx"),
            "model_size_mb": 6.53,
            "filename": "exp7_yolov8n_fp16.mp4"
        },
        {
            "id": "8",
            "name": "YOLOv8n-Pose v7 (INT8)",
            "type": "yolo",
            "yolo": str(y_dir / "stage-a-256-13kp-v7-merged-negatives-int8.onnx"),
            "model_size_mb": 3.46,
            "filename": "exp8_yolov8n_int8.mp4"
        }
    ]

    if args.config_ids:
        target_ids = [s.strip() for s in args.config_ids.split(",")]
        configs = [c for c in configs if c["id"] in target_ids]

    for cfg in configs:
        out_file = output_dir / cfg["filename"]
        print(f"\n[*] Generating Video Demo for [{cfg['id']}]: {cfg['name']} -> {out_file.name}")
        if cfg["type"] == "two_stage":
            process_video_two_stage(cfg, video_path=video_path, output_path=out_file, conf_thresh=args.conf)
        else:
            process_video_yolo(cfg, video_path=video_path, output_path=out_file)

    print("\n" + "=" * 80)
    print(f"  ALL 10 MULTI-PRECISION DEMO VIDEOS SUCCESSFULLY GENERATED!")
    print(f"  Directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

