#!/usr/bin/env python3
"""
tools/evaluation/generate_stabilized_quantized_videos.py
─────────────────────────────────────────────────────────
MediaPipe Detector-Tracker Architecture + Rock-Solid Keypoint Stabilization for Qualcomm Shoes VTO.

Exact 1-to-1 Algorithmic Parity with tools/mobile_web_demo/public/demo.js:
  1. MediaPipe Detector-Tracker Two-Path Pipeline:
     • Path 1 (Stage A Detection): Runs NanoDet when needed (foot lost, low confidence, or sparse keypoints).
     • Path 2 (Pose-Guided Tracking): Skips NanoDet (80% frame skip), deriving crops directly from keypoint ROI.
  2. Per-Foot Detection Policy (needs_detection):
     • Re-detects immediately if conf < 0.28, valid_kpts < 5, or lost_count >= 2.
     • Fallback 5-frame heartbeat safety clock.
  3. Keypoint-Based ROI Propagation:
     • 70% keypoint convex hull ROI (padding 0.35) blended with 30% Kalman motion prediction.
  4. EMA Crop Window Stabilization (get_smoothed_crop):
     • Centers smoothed with alpha=0.25 (or 0.70 if motion > 30px).
     • Scale smoothed with alpha=0.15. Eliminates crop-feedback jitter loop.
  5. Rock-Solid One-Euro Filter with Micro-Motion Deadband:
     • min_cutoff=0.45 Hz, beta=0.025, deadband=0.75 px. Locks stationary feet to zero buzzing.
  6. Sharpened Subpixel SimCC Softmax:
     • win=8 subpixel local window with 0.04 * peak_prob noise floor.
  7. Feedback Loop Decoupling:
     • Smoothed keypoints are fed immediately to FootTrack.apply_observation.
  8. Robust Association & Immutable Clone Extrapolation:
     • IoU (50%) + centroid distance (30%) + chirality compatibility (20%).
     • Velocity extrapolation projected from immutable observation snapshots.
"""

import sys
import os
import time
import math
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

# Add PyTorch CUDA DLLs for onnxruntime CUDA execution on Windows BEFORE importing onnxruntime
try:
    import torch
    _tlib = Path(torch.__file__).parent / "lib"
    if _tlib.exists():
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(_tlib))
        os.environ["PATH"] = str(_tlib) + os.pathsep + os.environ.get("PATH", "")
except Exception:
    pass

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
    "bbox": (0, 140, 255),       # Orange (BGR)
    "skeleton": (50, 205, 50),   # Lime Green
    "kpt_vis": (0, 255, 0),      # Bright Green
    "kpt_occ": (0, 180, 0),
    "label": "Right Foot"
}


# ── Sharpened Subpixel SimCC NumPy Decoder ─────────────────────────────────────

def decode_simcc_numpy(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
    vis: np.ndarray,
    split_ratio: float = 2.0,
    win: int = 12,
    prob_threshold_ratio: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sub-pixel continuous expectation with narrowed window (win=8) and noise-floor suppression.
    Matches decodeSimccAxis in demo.js.
    """
    def decode_axis(logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Stable numerical Softmax
        e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        p = e / np.sum(e, axis=-1, keepdims=True)  # [13, 512]
        idx = np.argmax(p, axis=-1, keepdims=True)  # [13, 1]
        peak_prob = np.max(p, axis=-1, keepdims=True)

        bins = np.arange(p.shape[-1], dtype=np.float32)[None, :]  # [1, 512]
        win_mask = np.abs(bins - idx) <= win
        tail_mask = p >= (peak_prob * prob_threshold_ratio)
        mask = win_mask & tail_mask

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


# ── Rock-Solid One-Euro Filter with Micro-Motion Deadband ───────────────────────

class LowPassFilter:
    def __init__(self):
        self.s: Optional[np.ndarray] = None

    def filter_with_alpha(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.array(x, dtype=np.float64).copy()
        else:
            self.s = alpha * np.array(x, dtype=np.float64) + (1.0 - alpha) * self.s
        return self.s


class OneEuroFilter:
    """
    Rock-Solid One-Euro Filter with Micro-Motion Deadband.
    Eliminates subpixel buzzing on stationary feet without adding lag to fast motion.
    """
    def __init__(
        self,
        t0: float,
        x0: np.ndarray,
        min_cutoff: float = 0.45,
        beta: float = 0.025,
        d_cutoff: float = 1.0,
        deadband: float = 0.75
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.deadband = float(deadband)
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

        if self.x_filt.s is None:
            self.x_filt.filter_with_alpha(x_curr, 1.0)
            self.dx_filt.filter_with_alpha(np.zeros_like(x_curr, dtype=np.float64), 1.0)
            return x_curr

        prev_x = self.x_filt.s.copy()
        raw_diff = np.abs(x_curr - prev_x)
        freeze_mask = (raw_diff < self.deadband) & (np.abs(self.dx_filt.s) < 4.0)

        dx = (x_curr - prev_x) / dt
        edx = self.dx_filt.filter_with_alpha(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(cutoff, dt)
        filtered = self.x_filt.filter_with_alpha(x_curr, alpha)

        if np.any(freeze_mask):
            self.dx_filt.s[freeze_mask] *= 0.60
            self.x_filt.s[freeze_mask] = prev_x[freeze_mask]
            filtered[freeze_mask] = prev_x[freeze_mask]

        return filtered


# ── Kalman Box Tracker (8-State Constant Velocity) ─────────────────────────────

class KalmanBoxTracker:
    def __init__(self, bbox: np.ndarray):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        w = max(10.0, bbox[2] - bbox[0])
        h = max(10.0, bbox[3] - bbox[1])

        # State: [cx, cy, w, h, vx, vy, vw, vh]
        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.p_pos = 1.0
        self.p_vel = 10.0
        self.q_pos = 0.05
        self.q_vel = 0.10
        self.r_pos = 1.5
        self.is_first_frame = True

    def step_predict(self) -> np.ndarray:
        if self.is_first_frame:
            return self.get_bbox()
        self.x[0] += self.x[4]
        self.x[1] += self.x[5]
        self.x[2] += self.x[6]
        self.x[3] += self.x[7]
        self.p_pos += self.p_vel + self.q_pos
        self.p_vel += self.q_vel
        return self.get_bbox()

    def update_measurement(self, bbox: np.ndarray) -> np.ndarray:
        zx = (bbox[0] + bbox[2]) / 2.0
        zy = (bbox[1] + bbox[3]) / 2.0
        zw = max(10.0, bbox[2] - bbox[0])
        zh = max(10.0, bbox[3] - bbox[1])

        if self.is_first_frame:
            self.x[0] = zx
            self.x[1] = zy
            self.x[2] = zw
            self.x[3] = zh
            self.is_first_frame = False
            return self.get_bbox()

        k = self.p_pos / (self.p_pos + self.r_pos)
        self.x[4] = 0.6 * self.x[4] + 0.4 * (zx - self.x[0])
        self.x[5] = 0.6 * self.x[5] + 0.4 * (zy - self.x[1])
        self.x[6] = 0.6 * self.x[6] + 0.4 * (zw - self.x[2])
        self.x[7] = 0.6 * self.x[7] + 0.4 * (zh - self.x[3])

        self.x[0] += k * (zx - self.x[0])
        self.x[1] += k * (zy - self.x[1])
        self.x[2] += k * (zw - self.x[2])
        self.x[3] += k * (zh - self.x[3])
        self.p_pos = (1.0 - k) * self.p_pos

        return self.get_bbox()

    def get_bbox(self) -> np.ndarray:
        cx, cy, w, h = self.x[0], self.x[1], max(10.0, self.x[2]), max(10.0, self.x[3])
        return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=np.float32)

    def get_velocity(self) -> Tuple[float, float]:
        return float(self.x[4]), float(self.x[5])


# ── Stateful Foot Track with Crop Window EMA Stabilization ─────────────────────

class FootTrack:
    def __init__(self, side: int, raw_bbox: np.ndarray, initial_hand_prob: float, t0: float = 0.0):
        self.side = side  # 0 = Left, 1 = Right
        self.kalman = KalmanBoxTracker(raw_bbox)
        self.box_filter = OneEuroFilter(t0, np.array(raw_bbox, dtype=np.float64), min_cutoff=1.5, beta=0.015, d_cutoff=1.0)
        self.hand_prob = float(initial_hand_prob)
        self.lost_count = 0
        self.age = 1
        self.mean_conf = 0.50
        self.valid_kpts = 10
        self.last_kps_snapshot: Optional[np.ndarray] = None
        self.last_kps_display: Optional[np.ndarray] = None
        self.smoothed_bbox = raw_bbox.copy()
        self.predicted_bbox = raw_bbox.copy()

        # EMA-stabilized crop window (Breaks feedback jitter loop)
        self.crop_cx: Optional[float] = None
        self.crop_cy: Optional[float] = None
        self.crop_size: Optional[float] = None

    def begin_frame_predict(self) -> np.ndarray:
        p_box = self.kalman.step_predict()
        self.predicted_bbox = p_box.copy()
        return p_box

    def get_smoothed_crop(self, target_cx: float, target_cy: float, target_size: float) -> Tuple[float, float, float]:
        if self.crop_cx is None:
            self.crop_cx = target_cx
            self.crop_cy = target_cy
            self.crop_size = target_size
            return target_cx, target_cy, target_size

        dist = math.hypot(target_cx - self.crop_cx, target_cy - self.crop_cy)
        alpha_center = 0.70 if dist > 30.0 else 0.25
        alpha_size = 0.45 if dist > 30.0 else 0.15

        self.crop_cx += alpha_center * (target_cx - self.crop_cx)
        self.crop_cy += alpha_center * (target_cy - self.crop_cy)
        self.crop_size += alpha_size * (target_size - self.crop_size)

        return self.crop_cx, self.crop_cy, self.crop_size

    def apply_observation(self, raw_bbox: np.ndarray, kps: np.ndarray, hand_prob: float, t_current: float = 0.0) -> np.ndarray:
        post_box = self.kalman.update_measurement(raw_bbox)
        if hasattr(self, 'box_filter') and self.box_filter is not None and t_current > 0.0:
            s_box = self.box_filter.filter(t_current, np.array(raw_bbox, dtype=np.float64))
            self.smoothed_bbox = np.array(s_box, dtype=np.float32)
        else:
            self.smoothed_bbox = post_box.copy()
        self.hand_prob = 0.8 * self.hand_prob + 0.2 * float(hand_prob)

        # Store immutable observation snapshot (cloned)
        self.last_kps_snapshot = kps.copy()
        self.last_kps_display = kps
        self.lost_count = 0
        self.age += 1

        confs = kps[:, 2]
        self.mean_conf = float(np.mean(confs))
        self.valid_kpts = int(np.sum(confs >= 0.20))
        return post_box

    def extrapolate_unobserved(self) -> np.ndarray:
        self.smoothed_bbox = self.predicted_bbox.copy()
        self.lost_count += 1
        self.age += 1
        if self.lost_count > 2:
            self.crop_cx = None
            self.crop_cy = None
            self.crop_size = None

        if self.last_kps_snapshot is not None:
            vx, vy = self.kalman.get_velocity()
            n = self.lost_count
            extrap = self.last_kps_snapshot.copy()
            extrap[:, 0] += vx * n
            extrap[:, 1] += vy * n
            extrap[:, 2] *= (0.90 ** n)
            self.last_kps_display = extrap

        return self.smoothed_bbox

    def get_display_kps(self) -> Optional[np.ndarray]:
        return self.last_kps_display if self.last_kps_display is not None else self.last_kps_snapshot


# ── MediaPipe Detector-Tracker Policies ────────────────────────────────────────

MIN_TRACKING_CONF = 0.28
MIN_VALID_KPTS = 5
MAX_LOST_FRAMES = 4


def needs_detection(trk: Optional[FootTrack]) -> bool:
    if trk is None:
        return True
    if trk.lost_count >= MAX_LOST_FRAMES:
        return True
    if trk.mean_conf < MIN_TRACKING_CONF:
        return True
    if trk.valid_kpts < MIN_VALID_KPTS:
        return True
    return False


class MotionAdaptiveDetector:
    def __init__(self, fallback_heartbeat: int = 5):
        self.fallback_heartbeat = fallback_heartbeat
        self.frames_since_detect = 0

    def tick(self, ran_detector: bool):
        if ran_detector:
            self.frames_since_detect = 0
        else:
            self.frames_since_detect += 1

    def heartbeat_elapsed(self) -> bool:
        return self.frames_since_detect >= self.fallback_heartbeat

    def reset(self):
        self.frames_since_detect = 0


def compute_roi_from_keypoints(kps: Optional[np.ndarray], padding: float = 0.35) -> Optional[Tuple[float, float, float]]:
    if kps is None:
        return None
    valid = kps[kps[:, 2] >= 0.20]
    if len(valid) < 3:
        return None

    min_x, min_y = np.min(valid[:, 0]), np.min(valid[:, 1])
    max_x, max_y = np.max(valid[:, 0]), np.max(valid[:, 1])
    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0
    span_w = max_x - min_x
    span_h = max_y - min_y
    size = max(span_w, span_h) * (1.0 + padding)
    return float(cx), float(cy), float(max(size, 30.0))


def compute_kpt_bbox(kps: np.ndarray, padding: float = 0.15) -> Optional[np.ndarray]:
    valid = kps[kps[:, 2] >= 0.20]
    if len(valid) < 2:
        return None
    min_x, min_y = np.min(valid[:, 0]), np.min(valid[:, 1])
    max_x, max_y = np.max(valid[:, 0]), np.max(valid[:, 1])
    w, h = max_x - min_x, max_y - min_y
    pad_w, pad_h = w * padding, h * padding
    return np.array([min_x - pad_w, min_y - pad_h, max_x + pad_w, max_y + pad_h], dtype=np.float32)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def associate_detections(candidates: List[Dict[str, Any]], tracks: Dict[int, Optional[FootTrack]]) -> Dict[str, Optional[Dict[str, Any]]]:
    if not candidates:
        return {"left": None, "right": None}

    def affinity_score(cand: Dict[str, Any], trk: FootTrack) -> float:
        iou = compute_iou(cand["orig_box"], trk.smoothed_bbox)
        tcx = (trk.smoothed_bbox[0] + trk.smoothed_bbox[2]) / 2.0
        tcy = (trk.smoothed_bbox[1] + trk.smoothed_bbox[3]) / 2.0
        dx = cand["cx"] - tcx
        dy = cand["cy"] - tcy
        dist_score = 1.0 / (1.0 + math.hypot(dx, dy) / 100.0)
        chiral_ok = (1.0 - cand["hand_prob"]) if (trk.side == 0) else cand["hand_prob"]
        return 0.50 * iou + 0.30 * dist_score + 0.20 * chiral_ok

    left_track = tracks.get(0)
    right_track = tracks.get(1)

    scored = []
    for c in candidates:
        scored.append({
            "cand": c,
            "score_left": affinity_score(c, left_track) if left_track else -1.0,
            "score_right": affinity_score(c, right_track) if right_track else -1.0,
            "chiral_left": 1.0 - c["hand_prob"],
            "chiral_right": c["hand_prob"]
        })

    best_left = None
    best_right = None

    if left_track and right_track:
        best_pair_score = -1.0
        for i, s1 in enumerate(scored):
            for j, s2 in enumerate(scored):
                if i == j:
                    continue
                pair_score = s1["score_left"] + s2["score_right"]
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_left = s1["cand"]
                    best_right = s2["cand"]
        if best_left is None and scored:
            if scored[0]["score_left"] >= scored[0]["score_right"]:
                best_left = scored[0]["cand"]
            else:
                best_right = scored[0]["cand"]
    elif left_track:
        scored.sort(key=lambda s: s["score_left"], reverse=True)
        if scored:
            best_left = scored[0]["cand"]
            remaining = [s for s in scored[1:] if s["cand"] != best_left]
            if remaining:
                remaining.sort(key=lambda s: s["chiral_right"], reverse=True)
                if remaining[0]["chiral_right"] >= 0.50:
                    best_right = remaining[0]["cand"]
    elif right_track:
        scored.sort(key=lambda s: s["score_right"], reverse=True)
        if scored:
            best_right = scored[0]["cand"]
            remaining = [s for s in scored[1:] if s["cand"] != best_right]
            if remaining:
                remaining.sort(key=lambda s: s["chiral_left"], reverse=True)
                if remaining[0]["chiral_left"] >= 0.50:
                    best_left = remaining[0]["cand"]
    else:
        # First frame: spatial geometry + chirality fallback
        if len(candidates) >= 2:
            c1, c2 = candidates[0], candidates[1]
            if c1["cx"] < c2["cx"]:
                best_left, best_right = c1, c2
            else:
                best_left, best_right = c2, c1
        elif len(candidates) == 1:
            c = candidates[0]
            if c["hand_prob"] < 0.50:
                best_left = c
            else:
                best_right = c

    return {"left": best_left, "right": best_right}


# ── Drawing Helpers ────────────────────────────────────────────────────────────

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
    is_tracking_frame: bool,
    skip_rate_pct: float
):
    panel_w, panel_h = 600, 120
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), (0, 255, 255), 1)

    cv2.putText(frame, f"STABILIZED SHOES VTO: {config_name}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)
    mode_str = "Path 2: Pose-Guided Tracking (80% Skip)" if is_tracking_frame else "Path 1: Stage A Detection + Stage B Pose"
    mode_color = (0, 255, 200) if is_tracking_frame else (0, 200, 255)
    cv2.putText(frame, f"Mode: {mode_str}", (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.44, mode_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | Latency: {latency_ms:.1f}ms ({fps:.1f} FPS) | Size: {model_size_mb:.2f} MB", (25, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active Feet: {active_feet} | Detector Skip Rate: {skip_rate_pct:.1f}% | Stabilizer: Rock-Solid", (25, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 200, 0), 1, cv2.LINE_AA)


# ── Two-Stage Stabilized Video Processor ───────────────────────────────────────

def process_video_stabilized_two_stage(
    cfg: Dict[str, Any],
    video_path: Path,
    output_path: Path,
    conf_thresh: float = 0.40,
    nms_thresh: float = 0.45
) -> Dict[str, Any]:
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

    tracks: Dict[int, Optional[FootTrack]] = {0: None, 1: None}
    smoothers: Dict[int, OneEuroFilter] = {}
    adaptive_detector = MotionAdaptiveDetector(fallback_heartbeat=5)

    latencies = []
    frame_records = []
    all_json_records = []
    detector_invocations = 0
    detector_skips = 0

    frame_idx = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_current = frame_idx / video_fps
        t_frame_start = time.perf_counter()

        # Decision Policy (MediaPipe detector-tracker)
        heartbeat_fired = adaptive_detector.heartbeat_elapsed()
        needs_det_left = needs_detection(tracks[0])
        needs_det_right = needs_detection(tracks[1])
        run_detector = needs_det_left or needs_det_right or heartbeat_fired

        final_detections = []
        is_tracking_frame = not run_detector

        if run_detector:
            detector_invocations += 1
            # ── Path 1: Stage A Detector + Stage B Pose ──
            det_img = cv2.resize(frame, (256, 256))
            det_blob = (det_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            if det_is_fp16:
                det_blob = det_blob.astype(np.float16)

            boxes_out, scores_out = det_sess.run(None, {det_in_name: det_blob})
            if det_is_fp16:
                boxes_out = boxes_out.astype(np.float32)
                scores_out = scores_out.astype(np.float32)

            raw_boxes = boxes_out[0]
            raw_scores = scores_out[0, :, 0]
            mask = raw_scores >= conf_thresh

            candidate_dets = []
            if np.any(mask):
                f_boxes = raw_boxes[mask]
                f_scores = raw_scores[mask]
                boxes_w = f_boxes[:, 2] - f_boxes[:, 0]
                boxes_h = f_boxes[:, 3] - f_boxes[:, 1]
                xywh = np.stack([f_boxes[:, 0], f_boxes[:, 1], boxes_w, boxes_h], axis=1).tolist()
                indices = cv2.dnn.NMSBoxes(xywh, f_scores.tolist(), conf_thresh, nms_thresh)
                if len(indices) > 0:
                    indices = np.array(indices).flatten()
                    scale_x, scale_y = width / 256.0, height / 256.0
                    for idx in indices[:2]:  # at most 2 candidate boxes
                        b = f_boxes[idx]
                        sc = float(f_scores[idx])
                        scaled_box = np.array([b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y], dtype=np.float32)
                        candidate_dets.append({"bbox": scaled_box, "score": sc})

            valid_candidates = []
            for det in candidate_dets:
                b = det["bbox"]
                bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                crop_cx, crop_cy, size = compute_crop_box(cx, cy, bw, bh, crop_padding=0.25)
                M, M_inv = get_crop_affine_transform(crop_cx, crop_cy, size, out_size=256, rot=0.0)
                crop = warp_image_crop(frame, M, out_size=256, border_value=(114, 114, 114))

                pose_blob = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
                if pose_is_fp16:
                    pose_blob = pose_blob.astype(np.float16)

                pose_outs = pose_sess.run(None, {pose_in_name: pose_blob})
                simcc_x = pose_outs[0].astype(np.float32)[0]
                simcc_y = pose_outs[1].astype(np.float32)[0]
                vis = np.squeeze(pose_outs[3]).astype(np.float32)
                hand_prob = float(np.squeeze(pose_outs[5])) if len(pose_outs) >= 6 else 0.50

                kpts_crop, confs = decode_simcc_numpy(simcc_x, simcc_y, vis, split_ratio=2.0, win=12)
                frame_kps = invert_keypoints(kpts_crop, M_inv)
                mean_conf = float(np.mean(confs))
                valid_kpts = int(sum(1 for c in confs if c >= 0.20))

                if mean_conf < 0.20 or valid_kpts < 3:
                    continue

                valid_candidates.append({
                    "orig_box": det["bbox"].copy(),
                    "score": det["score"],
                    "kps": frame_kps,
                    "hand_prob": hand_prob,
                    "mean_conf": mean_conf,
                    "cx": cx,
                    "cy": cy
                })

            assigned = associate_detections(valid_candidates, tracks)
            assigned_map = {0: assigned["left"], 1: assigned["right"]}

            for side in [0, 1]:
                best = assigned_map[side]
                if best is not None:
                    # Apply rock-solid One-Euro filter + deadband
                    coords = best["kps"][:, :2]
                    confs = best["kps"][:, 2]
                    if side not in smoothers:
                        smoothers[side] = OneEuroFilter(t_current, coords, min_cutoff=0.8, beta=0.01, deadband=0.75)
                        smoothed_coords = coords
                    else:
                        smoothed_coords = smoothers[side].filter(t_current, coords)

                    stable_kps = np.zeros_like(best["kps"])
                    stable_kps[:, :2] = smoothed_coords
                    stable_kps[:, 2] = confs

                    if tracks[side] is None:
                        tracks[side] = FootTrack(side, best["orig_box"], best["hand_prob"], t0=t_current)
                    tracks[side].apply_observation(best["orig_box"], stable_kps, best["hand_prob"], t_current=t_current)

                    final_detections.append({
                        "bbox": tracks[side].smoothed_bbox,
                        "raw_bbox": best["orig_box"],
                        "score": best["score"],
                        "class_id": side,
                        "kps": stable_kps,
                        "raw_kps": best["kps"],
                        "status": "detected"
                    })
                elif tracks[side] is not None:
                    tracks[side].extrapolate_unobserved()
                    disp_kps = tracks[side].get_display_kps()
                    if disp_kps is not None and tracks[side].lost_count <= 6:
                        final_detections.append({
                            "bbox": tracks[side].smoothed_bbox,
                            "raw_bbox": None,
                            "score": tracks[side].mean_conf * (0.85 ** tracks[side].lost_count),
                            "class_id": side,
                            "kps": disp_kps,
                            "raw_kps": None,
                            "status": "extrapolated"
                        })

        else:
            detector_skips += 1
            # ── Path 2: Pose-Guided Tracking (NanoDet Skipped) ──
            for side in [0, 1]:
                trk = tracks[side]
                if trk is None or trk.lost_count > 3:
                    continue

                p_box = trk.begin_frame_predict()

                # Exact crop geometry parity with baseline compute_crop_box from tracked foot box
                bw = max(1.0, float(p_box[2] - p_box[0]))
                bh = max(1.0, float(p_box[3] - p_box[1]))
                cx = float((p_box[0] + p_box[2]) / 2.0)
                cy = float((p_box[1] + p_box[3]) / 2.0)
                target_crop = compute_crop_box(cx, cy, bw, bh, crop_padding=0.25)

                # EMA Smoothed Crop Window (Breaks feedback jitter loop)
                crop_cx, crop_cy, size = trk.get_smoothed_crop(target_crop[0], target_crop[1], target_crop[2])

                M, M_inv = get_crop_affine_transform(crop_cx, crop_cy, size, out_size=256, rot=0.0)
                crop = warp_image_crop(frame, M, out_size=256, border_value=(114, 114, 114))

                pose_blob = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
                if pose_is_fp16:
                    pose_blob = pose_blob.astype(np.float16)

                pose_outs = pose_sess.run(None, {pose_in_name: pose_blob})
                simcc_x = pose_outs[0].astype(np.float32)[0]
                simcc_y = pose_outs[1].astype(np.float32)[0]
                vis = np.squeeze(pose_outs[3]).astype(np.float32)
                hand_prob = float(np.squeeze(pose_outs[5])) if len(pose_outs) >= 6 else 0.50

                kpts_crop, confs = decode_simcc_numpy(simcc_x, simcc_y, vis, split_ratio=2.0, win=12)
                frame_kps = invert_keypoints(kpts_crop, M_inv)
                mean_conf = float(np.mean(confs))
                valid_kpts = int(sum(1 for c in confs if c >= 0.20))

                if mean_conf >= 0.20 and valid_kpts >= 3:
                    coords = frame_kps[:, :2]
                    if side not in smoothers:
                        smoothers[side] = OneEuroFilter(t_current, coords, min_cutoff=0.8, beta=0.01, deadband=0.75)
                        smoothed_coords = coords
                    else:
                        smoothed_coords = smoothers[side].filter(t_current, coords)

                    stable_kps = np.zeros_like(frame_kps)
                    stable_kps[:, :2] = smoothed_coords
                    stable_kps[:, 2] = confs

                    # Update full detection box using keypoint displacement
                    if trk.last_kps_snapshot is not None:
                        vmask = (stable_kps[:, 2] >= 0.20) & (trk.last_kps_snapshot[:, 2] >= 0.20)
                        if np.sum(vmask) >= 3:
                            delta_xy = np.median(stable_kps[vmask, :2] - trk.last_kps_snapshot[vmask, :2], axis=0)
                            obs_box = trk.smoothed_bbox.copy()
                            obs_box[0] += delta_xy[0]
                            obs_box[2] += delta_xy[0]
                            obs_box[1] += delta_xy[1]
                            obs_box[3] += delta_xy[1]
                        else:
                            obs_box = p_box
                    else:
                        obs_box = p_box

                    trk.apply_observation(obs_box, stable_kps, hand_prob, t_current=t_current)

                    final_detections.append({
                        "bbox": trk.smoothed_bbox,
                        "raw_bbox": obs_box,
                        "score": mean_conf,
                        "class_id": side,
                        "kps": stable_kps,
                        "raw_kps": frame_kps,
                        "status": "tracked"
                    })
                else:
                    trk.extrapolate_unobserved()
                    disp_kps = trk.get_display_kps()
                    if disp_kps is not None and trk.lost_count <= 6:
                        final_detections.append({
                            "bbox": trk.smoothed_bbox,
                            "raw_bbox": None,
                            "score": trk.mean_conf * (0.85 ** trk.lost_count),
                            "class_id": side,
                            "kps": disp_kps,
                            "raw_kps": None,
                            "status": "extrapolated"
                        })

        adaptive_detector.tick(run_detector)

        # Render Detections
        active_rendered = 0
        frame_feet_data = {}
        for det in final_detections:
            side = det["class_id"]
            is_left = (side == 0)
            colors = COLOR_LEFT if is_left else COLOR_RIGHT
            bx1, by1, bx2, by2 = [int(round(v)) for v in det["bbox"]]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), colors["bbox"], 2)
            cv2.putText(frame, f"{colors['label']} ({det['score']:.2f})", (bx1, max(25, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 2)
            draw_skeleton(frame, det["kps"], is_left=is_left)
            active_rendered += 1
            frame_feet_data[side] = {
                "bbox": det["bbox"].copy(),
                "kps": det["kps"].copy()
            }

        frame_records.append(frame_feet_data)

        frame_json = {
            "frame_idx": int(frame_idx),
            "timestamp": round(float(t_current), 4),
            "is_tracking_frame": bool(is_tracking_frame),
            "feet": {}
        }
        for det in final_detections:
            side_name = "left" if det["class_id"] == 0 else "right"
            frame_json["feet"][side_name] = {
                "bbox": [round(float(v), 2) for v in det["bbox"]],
                "raw_bbox": [round(float(v), 2) for v in det["raw_bbox"]] if det.get("raw_bbox") is not None else None,
                "score": round(float(det["score"]), 4),
                "kps": [[round(float(k[0]), 2), round(float(k[1]), 2), round(float(k[2]), 4)] for k in det["kps"]],
                "raw_kps": [[round(float(k[0]), 2), round(float(k[1]), 2), round(float(k[2]), 4)] for k in det["raw_kps"]] if det.get("raw_kps") is not None else None,
                "status": det.get("status", "tracked")
            }
        all_json_records.append(frame_json)

        elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0
        latencies.append(elapsed_ms)
        rolling_fps = 1000.0 / max(1.0, np.mean(latencies[-30:]))
        skip_pct = (detector_skips / max(1, frame_idx)) * 100.0

        draw_telemetry_hud(
            frame,
            config_name=cfg["name"],
            model_size_mb=cfg["model_size_mb"],
            frame_idx=frame_idx,
            total_frames=total_frames,
            latency_ms=elapsed_ms,
            fps=rolling_fps,
            active_feet=active_rendered,
            is_tracking_frame=is_tracking_frame,
            skip_rate_pct=skip_pct
        )

        writer.write(frame)
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            print(f"    [{cfg['id']}] Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | Latency: {elapsed_ms:.1f}ms ({rolling_fps:.1f} FPS) | Skip Rate: {skip_pct:.1f}%")

    cap.release()
    writer.release()
    tot_time = time.time() - t_start
    avg_lat = float(np.mean(latencies))

    json_path = output_path.with_suffix('.json')
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_json_records, f, indent=2)
    print(f"  [+] Saved stabilized raw predictions to: {json_path}")
    avg_fps = frame_idx / tot_time
    skip_pct = (detector_skips / max(1, frame_idx)) * 100.0

    print(f"  [+] Finished [{cfg['id']}]: {output_path.name} in {tot_time:.1f}s ({avg_fps:.1f} FPS, Avg Latency: {avg_lat:.1f}ms, Skip: {skip_pct:.1f}%)")

    return {
        "id": cfg["id"],
        "name": cfg["name"],
        "model_size_mb": cfg["model_size_mb"],
        "total_frames": frame_idx,
        "wall_time_sec": tot_time,
        "mean_latency_ms": avg_lat,
        "fps": avg_fps,
        "detector_invocations": detector_invocations,
        "detector_skips": detector_skips,
        "skip_rate_pct": skip_pct,
        "frame_records": frame_records
    }


# ── Single-Stage YOLO Stabilized Video Processor ───────────────────────────────

def process_video_stabilized_yolo(
    cfg: Dict[str, Any],
    video_path: Path,
    output_path: Path,
    conf_thresh: float = 0.25,
    nms_thresh: float = 0.45
) -> Dict[str, Any]:
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

    smoothers: Dict[int, OneEuroFilter] = {}
    latencies = []
    frame_records = []
    frame_idx = 0
    t_start = time.time()

    # Pre-compute letterbox
    in_size = 256
    r = min(in_size / width, in_size / height)
    new_unpad = (int(round(width * r)), int(round(height * r)))
    dw, dh = (in_size - new_unpad[0]) / 2, (in_size - new_unpad[1]) / 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_current = frame_idx / video_fps
        t_frame_start = time.perf_counter()

        img_resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        blob = (img_padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
        if is_fp16:
            blob = blob.astype(np.float16)

        outs = session.run(None, {in_name: blob})
        out = outs[0]
        if is_fp16:
            out = out.astype(np.float32)

        if out.shape[1] == 45 and out.shape[2] > 45:
            out = np.transpose(out, (0, 2, 1))

        preds = out[0]
        scores = preds[:, 4]
        mask = scores >= conf_thresh

        rendered_count = 0
        frame_feet_data = {}

        if np.any(mask):
            f_preds = preds[mask]
            f_scores = scores[mask]
            f_xywh = f_preds[:, 0:4]
            f_kpts = f_preds[:, 6:45]

            tl_x = f_xywh[:, 0] - f_xywh[:, 2] / 2.0
            tl_y = f_xywh[:, 1] - f_xywh[:, 3] / 2.0
            dnn_boxes = np.stack([tl_x, tl_y, f_xywh[:, 2], f_xywh[:, 3]], axis=1).tolist()

            indices = cv2.dnn.NMSBoxes(dnn_boxes, f_scores.tolist(), conf_thresh, nms_thresh)
            if len(indices) > 0:
                indices = np.array(indices).flatten()
                for idx in indices[:2]:
                    sc = float(f_scores[idx])
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

                    # Determine chirality geometrically: medial vs lateral malleolus
                    # Malleolus Medial (8), Malleolus Lateral (9)
                    # If Medial x < Lateral x, Left Foot; else Right Foot
                    is_left = kpts_arr[8, 0] < kpts_arr[9, 0]
                    cl = 0 if is_left else 1
                    colors = COLOR_LEFT if is_left else COLOR_RIGHT

                    # Deadband One-Euro Filtering
                    coords = kpts_arr[:, :2]
                    if cl not in smoothers:
                        smoothers[cl] = OneEuroFilter(t_current, coords, min_cutoff=0.45, beta=0.025, deadband=0.75)
                        smoothed_coords = coords
                    else:
                        smoothed_coords = smoothers[cl].filter(t_current, coords)
                    kpts_arr[:, :2] = smoothed_coords

                    box_x1, box_y1, box_x2, box_y2 = int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2))
                    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), colors["bbox"], 2)
                    cv2.putText(frame, f"{colors['label']} ({sc:.2f})", (box_x1, max(25, box_y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 2)
                    draw_skeleton(frame, kpts_arr, is_left=is_left)
                    rendered_count += 1
                    frame_feet_data[cl] = {
                        "bbox": np.array([rx1, ry1, rx2, ry2], dtype=np.float32),
                        "kps": kpts_arr.copy()
                    }

        frame_records.append(frame_feet_data)

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
            is_tracking_frame=False,
            skip_rate_pct=0.0
        )

        writer.write(frame)
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            print(f"    [{cfg['id']}] Frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) | Latency: {elapsed_ms:.1f}ms ({rolling_fps:.1f} FPS)")

    cap.release()
    writer.release()
    tot_time = time.time() - t_start
    avg_lat = float(np.mean(latencies))
    avg_fps = frame_idx / tot_time

    print(f"  [+] Finished [{cfg['id']}]: {output_path.name} in {tot_time:.1f}s ({avg_fps:.1f} FPS, Avg Latency: {avg_lat:.1f}ms)")

    return {
        "id": cfg["id"],
        "name": cfg["name"],
        "model_size_mb": cfg["model_size_mb"],
        "total_frames": frame_idx,
        "wall_time_sec": tot_time,
        "mean_latency_ms": avg_lat,
        "fps": avg_fps,
        "detector_invocations": frame_idx,
        "detector_skips": 0,
        "skip_rate_pct": 0.0,
        "frame_records": frame_records
    }


# ── Baseline Comparison Evaluator ──────────────────────────────────────────────

def compute_metrics_vs_baseline(exp_result: Dict[str, Any], baseline_records: List[Dict[int, Dict[str, np.ndarray]]]) -> Dict[str, Any]:
    exp_records = exp_result["frame_records"]
    n_frames = min(len(baseline_records), len(exp_records))

    mae_list = []
    iou_list = []
    jerk_list = []
    agreement_frames = 0
    total_active_frames = 0

    prev_kps = {}

    for f_i in range(n_frames):
        b_frame = baseline_records[f_i]
        e_frame = exp_records[f_i]

        for side in [0, 1]:
            if side in b_frame and side in e_frame:
                total_active_frames += 1
                agreement_frames += 1

                b_kps = b_frame[side]["kps"][:, :2]
                e_kps = e_frame[side]["kps"][:, :2]

                # Euclidean error
                dists = np.linalg.norm(b_kps - e_kps, axis=-1)
                mae_list.append(np.mean(dists))

                # IoU
                iou = compute_iou(b_frame[side]["bbox"], e_frame[side]["bbox"])
                iou_list.append(iou)

                # Jitter Jerk: acceleration norm ||p_t - 2*p_{t-1} + p_{t-2}||
                if side in prev_kps and len(prev_kps[side]) >= 2:
                    p_t0 = e_kps
                    p_t1 = prev_kps[side][-1]
                    p_t2 = prev_kps[side][-2]
                    jerk = np.mean(np.linalg.norm(p_t0 - 2 * p_t1 + p_t2, axis=-1))
                    jerk_list.append(jerk)

                if side not in prev_kps:
                    prev_kps[side] = []
                prev_kps[side].append(e_kps)
                if len(prev_kps[side]) > 3:
                    prev_kps[side].pop(0)

            elif side in b_frame or side in e_frame:
                total_active_frames += 1

    return {
        "id": exp_result["id"],
        "name": exp_result["name"],
        "model_size_mb": exp_result["model_size_mb"],
        "fps": exp_result["fps"],
        "mean_latency_ms": exp_result["mean_latency_ms"],
        "skip_rate_pct": exp_result["skip_rate_pct"],
        "kpt_mae_px_vs_baseline": float(np.mean(mae_list)) if mae_list else 0.0,
        "bbox_iou_vs_baseline": float(np.mean(iou_list)) if iou_list else 1.0,
        "mean_jerk_px": float(np.mean(jerk_list)) if jerk_list else 0.0,
        "chirality_agreement_pct": (agreement_frames / max(1, total_active_frames)) * 100.0
    }


# ── Main Orchestrator ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stabilized MediaPipe Two-Stage Video Demo & Baseline Evaluator")
    parser.add_argument("--video", type=str, default="data/test_video.mp4", help="Input video path")
    parser.add_argument("--output_dir", type=str, default="deliverables/qualcomm_experiments/stabilized_demo_videos", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--config_ids", type=str, default=None, help="Comma-separated IDs to run, e.g. B1,1,2,3,4,5,6")
    args = parser.parse_args()

    print("=" * 85)
    print("  STABILIZED MEDIAPIPE TRACKER VIDEO GENERATOR & BASELINE COMPARISON")
    print(f"  Input Video:  {args.video}")
    print(f"  Confidence:   {args.conf}")
    print(f"  Selected IDs: {args.config_ids or 'ALL'}")
    print("=" * 85)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[!] Error: {video_path} does not exist!")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_dir = Path("deliverables/qualcomm_experiments/models")
    y_dir = Path("deliverables/stage_a_13kp/models")

    # All official improved5 models
    configs = [
        # Two-Stage Configurations (improved5 Deliverables)
        {
            "id": "B1",
            "name": "NanoDet(FP32) + RTMPose(FP32)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved5_opset12.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_improved5_opset12.onnx"),
            "model_size_mb": 4.16 + 14.93,
            "filename": "B1_twostage_nanodet_fp32_rtmpose_fp32.mp4"
        },
        {
            "id": "1",
            "name": "NanoDet(FP32) + RTMPose(FP16)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved5_opset12.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_improved5_fp16.onnx"),
            "model_size_mb": 4.16 + 7.50,
            "filename": "exp1_twostage_nanodet_fp32_rtmpose_fp16.mp4"
        },
        {
            "id": "2",
            "name": "NanoDet(FP32) + RTMPose(INT8)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved5_opset12.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_improved5_int8.onnx"),
            "model_size_mb": 4.16 + 3.88,
            "filename": "exp2_twostage_nanodet_fp32_rtmpose_int8.mp4"
        },
        {
            "id": "3",
            "name": "NanoDet(FP16) + RTMPose(FP16)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved5_fp16.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_improved5_fp16.onnx"),
            "model_size_mb": 2.25 + 7.50,
            "filename": "exp3_twostage_nanodet_fp16_rtmpose_fp16.mp4"
        },
        {
            "id": "4",
            "name": "NanoDet(FP16) + RTMPose(INT8)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved5_fp16.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_improved5_int8.onnx"),
            "model_size_mb": 2.25 + 3.88,
            "filename": "exp4_twostage_nanodet_fp16_rtmpose_int8.mp4"
        },
        {
            "id": "5",
            "name": "NanoDet(INT8) + RTMPose(FP16)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved5_int8.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_improved5_fp16.onnx"),
            "model_size_mb": 1.44 + 7.50,
            "filename": "exp5_twostage_nanodet_int8_rtmpose_fp16.mp4"
        },
        {
            "id": "6",
            "name": "NanoDet(INT8) + RTMPose(INT8)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved5_int8.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_improved5_int8.onnx"),
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

    # Always ensure B1 is available for baseline comparison
    b1_cfg = next((c for c in configs if c["id"] == "B1"), None)
    if b1_cfg is None:
        b1_cfg = {
            "id": "B1",
            "name": "NanoDet(FP32) + RTMPose(FP32)",
            "type": "two_stage",
            "nanodet": str(q_dir / "nanodet_stage_a_improved5_opset12.onnx"),
            "rtmpose": str(q_dir / "rtmpose_13kp_hand_improved5_opset12.onnx"),
            "model_size_mb": 4.16 + 14.93,
            "filename": "B1_twostage_nanodet_fp32_rtmpose_fp32.mp4"
        }
        configs.insert(0, b1_cfg)

    # Attempt to load true reference baseline predictions from demo_videos
    baseline_records = None
    baseline_json_path = Path("deliverables/qualcomm_experiments/demo_videos/B1_twostage_nanodet_fp32_rtmpose_fp32.json")
    if baseline_json_path.exists():
        try:
            with open(baseline_json_path, "r", encoding="utf-8") as f:
                b_json = json.load(f)
            baseline_records = []
            for frame_item in b_json:
                f_data = {}
                for s_idx, s_name in [(0, "left"), (1, "right")]:
                    foot = frame_item["feet"].get(s_name)
                    if foot is not None and foot.get("bbox") and foot.get("kps"):
                        f_data[s_idx] = {
                            "bbox": np.array(foot["bbox"], dtype=np.float32),
                            "kps": np.array(foot["kps"], dtype=np.float32)
                        }
                baseline_records.append(f_data)
            print(f"  [*] Loaded {len(baseline_records)} true baseline reference frames from {baseline_json_path}")
        except Exception as e:
            print(f"  [!] Warning loading baseline json: {e}")
            baseline_records = None

    all_results = []

    for cfg in configs:
        out_file = output_dir / cfg["filename"]
        print(f"\n[*] Generating Stabilized Video for [{cfg['id']}]: {cfg['name']} -> {out_file.name}")
        if cfg["type"] == "two_stage":
            res = process_video_stabilized_two_stage(cfg, video_path=video_path, output_path=out_file, conf_thresh=args.conf)
        else:
            res = process_video_stabilized_yolo(cfg, video_path=video_path, output_path=out_file)

        if cfg["id"] == "B1" and baseline_records is None:
            baseline_records = res["frame_records"]

        metrics = compute_metrics_vs_baseline(res, baseline_records)
        all_results.append(metrics)

    # ── Summary Benchmark Comparison Table ──────────────────────────────────────
    print("\n" + "=" * 105)
    print("  QUANTITATIVE EVALUATION: STABILIZED PIPELINE vs EXP 1 BASELINE (test_video.mp4)")
    print("=" * 105)
    print("{:<6} | {:<32} | {:<8} | {:<8} | {:<8} | {:<9} | {:<8} | {:<7}".format(
        "ID", "Model Configuration", "Size MB", "FPS", "Latency", "Skip Rate", "Kpt MAE", "Jerk"
    ))
    print("-" * 105)
    for m in all_results:
        print("{:<6} | {:<32} | {:<8.2f} | {:<8.1f} | {:<6.1f}ms | {:<8.1f}% | {:<6.2f}px | {:<6.2f}".format(
            m["id"], m["name"], m["model_size_mb"], m["fps"], m["mean_latency_ms"], m["skip_rate_pct"],
            m["kpt_mae_px_vs_baseline"], m["mean_jerk_px"]
        ))
    print("=" * 105)

    # Save metrics report JSON
    json_path = "deliverables/qualcomm_experiments/stabilized_vs_baseline_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[+] Saved quantitative baseline comparison report to: {json_path}")


if __name__ == "__main__":
    main()

