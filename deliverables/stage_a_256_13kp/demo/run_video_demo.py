#!/usr/bin/env python3
"""
run_video_demo.py
─────────────────
Production Video Inference & Advanced Post-Processing for Shoes VTO Stage A (256 Mobile 13-KP Model).
Supports loading both ONNX (.onnx) and PyTorch (.pt) models.

Key Features & Post-Processing:
  1. 13-Keypoint Schema (excision of ankle_center, ball_top, shin_mid).
  2. Anatomical Geometric Chirality (Deterministically resolves Left vs Right using signed 2D cross-product with heel_ground/ankle fallback).
  3. Occluded Keypoint Rendering: Solid circles for visible points; hollow rings for occluded/inferred points.
  4. Physical Foot Duplicate Resolver: Suppresses phantom duplicate boxes (IoU > 0.60) on the same physical foot.
  5. One-Euro Adaptive Temporal Smoothing: Eliminates jitter while preserving instant dynamic response.
  6. Real-time Telemetry HUD: FPS, latency (ms), frame count, active feet count.
  7. Optional 15-Keypoint Expansion (--expand_15kp): Deterministic derivation of ball_top and shin_mid.

Usage:
  python deliverables/stage_a_256_13kp/demo/run_video_demo.py
  python deliverables/stage_a_256_13kp/demo/run_video_demo.py --weights deliverables/stage_a_256_13kp/models/stage-a-256-13kp-fp32.onnx --video data/test_video.mp4
  python deliverables/stage_a_256_13kp/demo/run_video_demo.py --expand_15kp --output demo_15kp.mp4
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np
import cv2


# ── 15-Keypoint Anatomical Mapping (Derived) ──────────────────────────────────
KEYPOINT_NAMES_15KP = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "ball_top", "instep_top", "arch_medial", "midfoot_lateral",
    "malleolus_medial", "malleolus_lateral", "toe_tip", "throat", "achilles", "shin_mid"
]

def derive_secondary_keypoints_15kp(keypoints_13, bbox=None):
    """Deterministically derives ball_top (5) and shin_mid (14) from 13-KP detections."""
    k = {item["index"]: item for item in keypoints_13}
    p_bm = np.array([k[3]["x"], k[3]["y"]], dtype=np.float32)
    p_bl = np.array([k[4]["x"], k[4]["y"]], dtype=np.float32)
    p_instep = np.array([k[5]["x"], k[5]["y"]], dtype=np.float32)
    m_ball = 0.5 * (p_bm + p_bl)
    p_ball_top = m_ball + 0.28 * (p_instep - m_ball)
    v_ball_top = min(k[3]["visibility"], k[4]["visibility"], k[5]["visibility"])

    p_throat = np.array([k[11]["x"], k[11]["y"]], dtype=np.float32)
    p_ach = np.array([k[12]["x"], k[12]["y"]], dtype=np.float32)
    p_toe = np.array([k[10]["x"], k[10]["y"]], dtype=np.float32)
    p_hg = np.array([k[2]["x"], k[2]["y"]], dtype=np.float32)
    p_mm = np.array([k[8]["x"], k[8]["y"]], dtype=np.float32)
    p_ml = np.array([k[9]["x"], k[9]["y"]], dtype=np.float32)

    v_toe, v_hg, v_ach, v_thr = k[10]["visibility"], k[2]["visibility"], k[12]["visibility"], k[11]["visibility"]
    if bbox is not None:
        ref_len = float(np.sqrt((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2)) * 0.40
    else:
        xs = [kp["x"] for kp in keypoints_13]
        ys = [kp["y"] for kp in keypoints_13]
        ref_len = float(np.sqrt((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)) * 0.40

    if v_toe < 0.30 and (v_hg > 0.40 or v_ach > 0.50):
        v_leg = p_ach - p_hg
        n_leg = float(np.linalg.norm(v_leg))
        u_leg = (v_leg / n_leg) if n_leg > 1e-4 else np.array([0.0, -1.0], dtype=np.float32)
        p_shin = p_throat + 1.20 * ref_len * u_leg
        v_shin = min(v_thr, v_ach, max(v_hg, 0.4)) * 0.90
    else:
        v_dorsal = p_throat - p_toe
        n_dorsal = float(np.linalg.norm(v_dorsal))
        v_malleoli = p_ml - p_mm
        perp = np.array([-v_malleoli[1], v_malleoli[0]], dtype=np.float32)
        if np.dot(perp, v_dorsal) < 0:
            perp = -perp
        n_perp = float(np.linalg.norm(perp))
        u_perp = (perp / n_perp) if n_perp > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)
        u_dorsal = (v_dorsal / n_dorsal) if n_dorsal > 1e-4 else u_perp
        u_leg = 0.65 * u_dorsal + 0.35 * u_perp
        n_u = float(np.linalg.norm(u_leg))
        u_leg = (u_leg / n_u) if n_u > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)
        p_shin = p_throat + 1.10 * ref_len * u_leg
        v_shin = min(v_thr, max(v_toe, 0.5)) * 0.92

    return [
        {"index": 0, "name": KEYPOINT_NAMES_15KP[0], "x": k[0]["x"], "y": k[0]["y"], "visibility": k[0]["visibility"]},
        {"index": 1, "name": KEYPOINT_NAMES_15KP[1], "x": k[1]["x"], "y": k[1]["y"], "visibility": k[1]["visibility"]},
        {"index": 2, "name": KEYPOINT_NAMES_15KP[2], "x": k[2]["x"], "y": k[2]["y"], "visibility": k[2]["visibility"]},
        {"index": 3, "name": KEYPOINT_NAMES_15KP[3], "x": k[3]["x"], "y": k[3]["y"], "visibility": k[3]["visibility"]},
        {"index": 4, "name": KEYPOINT_NAMES_15KP[4], "x": k[4]["x"], "y": k[4]["y"], "visibility": k[4]["visibility"]},
        {"index": 5, "name": KEYPOINT_NAMES_15KP[5], "x": float(p_ball_top[0]), "y": float(p_ball_top[1]), "visibility": float(v_ball_top)},
        {"index": 6, "name": KEYPOINT_NAMES_15KP[6], "x": k[5]["x"], "y": k[5]["y"], "visibility": k[5]["visibility"]},
        {"index": 7, "name": KEYPOINT_NAMES_15KP[7], "x": k[6]["x"], "y": k[6]["y"], "visibility": k[6]["visibility"]},
        {"index": 8, "name": KEYPOINT_NAMES_15KP[8], "x": k[7]["x"], "y": k[7]["y"], "visibility": k[7]["visibility"]},
        {"index": 9, "name": KEYPOINT_NAMES_15KP[9], "x": k[8]["x"], "y": k[8]["y"], "visibility": k[8]["visibility"]},
        {"index": 10, "name": KEYPOINT_NAMES_15KP[10], "x": k[9]["x"], "y": k[9]["y"], "visibility": k[9]["visibility"]},
        {"index": 11, "name": KEYPOINT_NAMES_15KP[11], "x": k[10]["x"], "y": k[10]["y"], "visibility": k[10]["visibility"]},
        {"index": 12, "name": KEYPOINT_NAMES_15KP[12], "x": k[11]["x"], "y": k[11]["y"], "visibility": k[11]["visibility"]},
        {"index": 13, "name": KEYPOINT_NAMES_15KP[13], "x": k[12]["x"], "y": k[12]["y"], "visibility": k[12]["visibility"]},
        {"index": 14, "name": KEYPOINT_NAMES_15KP[14], "x": float(p_shin[0]), "y": float(p_shin[1]), "visibility": float(v_shin)},
    ]

# ── 13-Keypoint Anatomical Mapping ─────────────────────────────────────────────
KEYPOINT_NAMES = [
    "toe_ground",        # Index 0  (sole)
    "heel_back",         # Index 1  (sole)
    "heel_ground",       # Index 2  (sole - origin)
    "ball_medial",       # Index 3  (ball)
    "ball_lateral",      # Index 4  (ball)
    "instep_top",        # Index 5  (mid)
    "arch_medial",       # Index 6  (mid)
    "midfoot_lateral",   # Index 7  (mid)
    "malleolus_medial",  # Index 8  (ankle)
    "malleolus_lateral", # Index 9  (ankle)
    "toe_tip",           # Index 10 (sole)
    "throat",            # Index 11 (ankle)
    "achilles"           # Index 12 (ankle)
]

# Anatomical skeleton connections: pairs of (kp_idx_1, kp_idx_2)
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

# Anatomical skeleton connections for 15-KP (Derived ball_top & shin_mid)
SKELETON_PAIRS_15KP = [
    # Sole / Foot contour
    (2, 1),   # heel_ground -> heel_back
    (2, 7),   # heel_ground -> arch_medial
    (7, 3),   # arch_medial -> ball_medial
    (3, 11),  # ball_medial -> toe_tip
    (11, 0),  # toe_tip -> toe_ground
    (0, 4),   # toe_ground -> ball_lateral
    (4, 8),   # ball_lateral -> midfoot_lateral
    (8, 2),   # midfoot_lateral -> heel_ground
    # Dorsum / Instep arch with ball_top
    (3, 5),   # ball_medial -> ball_top
    (4, 5),   # ball_lateral -> ball_top
    (11, 5),  # toe_tip -> ball_top
    (5, 6),   # ball_top -> instep_top
    (6, 12),  # instep_top -> throat
    # Ankle, Achilles & Shin
    (9, 12),  # malleolus_medial -> throat
    (10, 12), # malleolus_lateral -> throat
    (9, 13),  # malleolus_medial -> achilles
    (10, 13), # malleolus_lateral -> achilles
    (1, 13),  # heel_back -> achilles
    (12, 14), # throat -> shin_mid
]

COLOR_LEFT = {
    "bbox": (255, 191, 0),      # Deep Sky Blue / Cyan (BGR)
    "skeleton": (255, 140, 0),
    "kpt_vis": (0, 255, 255),   # Solid Yellow
    "kpt_occ": (0, 200, 200),   # Hollow Yellow ring
    "name": "Left Foot"
}
COLOR_RIGHT = {
    "bbox": (180, 105, 255),    # Hot Pink / Magenta (BGR)
    "skeleton": (147, 20, 255),
    "kpt_vis": (0, 255, 128),   # Solid Bright Green
    "kpt_occ": (0, 180, 90),    # Hollow Green ring
    "name": "Right Foot"
}


def determine_geometric_chirality_13kp(keypoints):
    """
    Deterministically determines Left vs Right foot in the 13-KP schema
    using signed difference between lateral and medial cross products,
    with heel_ground / ankle center fallback as required by SDK team.
    
    Returns:
        0 for 'left_foot', 1 for 'right_foot'
    """
    toe = keypoints[10]       # toe_tip
    ball_lat = keypoints[4]   # ball_lateral
    ball_med = keypoints[3]   # ball_medial

    # Fallback rear anchor: heel_ground is preferred over heel_back top-down
    if keypoints[2]["visibility"] > 0.20:
        heel = keypoints[2]
    elif keypoints[1]["visibility"] > 0.20:
        heel = keypoints[1]
    else:
        heel = {
            "x": 0.5 * (keypoints[8]["x"] + keypoints[9]["x"]),
            "y": 0.5 * (keypoints[8]["y"] + keypoints[9]["y"]),
            "visibility": 0.5 * (keypoints[8]["visibility"] + keypoints[9]["visibility"])
        }

    v_axis_x = toe["x"] - heel["x"]
    v_axis_y = toe["y"] - heel["y"]

    v_lat_x = ball_lat["x"] - heel["x"]
    v_lat_y = ball_lat["y"] - heel["y"]
    cross_lat = v_axis_x * v_lat_y - v_axis_y * v_lat_x

    v_med_x = ball_med["x"] - heel["x"]
    v_med_y = ball_med["y"] - heel["y"]
    cross_med = v_axis_x * v_med_y - v_axis_y * v_med_x

    diff = cross_lat - cross_med
    return 0 if diff < 0 else 1


class OneEuroFilter:
    """Adaptive low-pass filter for smooth, lag-free keypoint tracking."""
    def __init__(self, min_cutoff=1.2, beta=0.008, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def alpha(self, rate, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate if rate > 0 else 0.033
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t
        rate = 1.0 / dt

        dx = (x - self.x_prev) * rate
        edx = self.dx_prev + (self.alpha(rate, self.d_cutoff) * (dx - self.dx_prev))
        self.dx_prev = edx

        cutoff = self.min_cutoff + self.beta * abs(edx)
        a = self.alpha(rate, cutoff)
        x_filtered = self.x_prev + a * (x - self.x_prev)
        self.x_prev = x_filtered
        return x_filtered


class KeypointSmoother:
    """Manages One-Euro filters for each keypoint of a tracked foot."""
    def __init__(self, num_keypoints=13):
        self.filters_x = [OneEuroFilter() for _ in range(num_keypoints)]
        self.filters_y = [OneEuroFilter() for _ in range(num_keypoints)]

    def smooth(self, keypoints, t):
        smoothed = []
        for i, kp in enumerate(keypoints):
            fx = self.filters_x[i].filter(kp["x"], t)
            fy = self.filters_y[i].filter(kp["y"], t)
            smoothed.append({
                "index": kp["index"],
                "name": kp["name"],
                "x": fx,
                "y": fy,
                "visibility": kp["visibility"]
            })
        return smoothed


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def deduplicate_physical_feet(detections, iou_thresh=0.60):
    if len(detections) <= 1:
        return detections
    sorted_dets = sorted(detections, key=lambda d: d["score"], reverse=True)
    kept = []
    for cand in sorted_dets:
        overlap = False
        for k in kept:
            if compute_iou(cand["bbox"], k["bbox"]) > iou_thresh:
                overlap = True
                break
        if not overlap:
            kept.append(cand)
    return kept


class VideoInferencePipeline:
    def __init__(self, model_path, input_size=256, conf_thresh=0.25, nms_thresh=0.45):
        self.model_path = str(model_path)
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.is_onnx = self.model_path.endswith(".onnx")

        print(f"[*] Initializing model: {self.model_path}")
        if self.is_onnx:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_type = self.session.get_inputs()[0].type
            print(f"    ONNX Session initialized using: {self.session.get_providers()[0]} (Type: {self.input_type})")
        else:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print("    Ultralytics PyTorch YOLO model initialized.")

    def preprocess_letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.input_size / h, self.input_size / w)
        rw, rh = int(round(w * r)), int(round(h * r))
        dw, dh = (self.input_size - rw) / 2, (self.input_size - rh) / 2

        if (w, h) != (rw, rh):
            img_resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        canvas = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return canvas, r, left, top

    def infer(self, frame):
        orig_h, orig_w = frame.shape[:2]

        if not self.is_onnx:
            # PyTorch pipeline
            t0 = time.perf_counter()
            results = self.model.predict(
                frame,
                imgsz=self.input_size,
                conf=self.conf_thresh,
                iou=self.nms_thresh,
                verbose=False
            )[0]
            lat_ms = (time.perf_counter() - t0) * 1000.0

            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                kpts_data = results.keypoints.data.cpu().numpy()

                for b, s, c, kpts in zip(boxes, scores, classes, kpts_data):
                    kpt_list = []
                    for idx, (kx, ky, kv) in enumerate(kpts):
                        kpt_list.append({
                            "index": idx,
                            "name": KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kp_{idx}",
                            "x": float(kx),
                            "y": float(ky),
                            "visibility": float(kv)
                        })
                    detections.append({
                        "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                        "score": float(s),
                        "class_id": int(c),
                        "keypoints": kpt_list
                    })
            return detections, lat_ms

        # ONNX Pipeline
        t0 = time.perf_counter()
        img_lb, ratio, pad_x, pad_y = self.preprocess_letterbox(frame)
        blob = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]
        if hasattr(self, 'input_type') and 'float16' in self.input_type:
            blob = blob.astype(np.float16)

        outputs = self.session.run(None, {self.input_name: blob})
        out0 = outputs[0]  # [1, 45, N]
        if out0.shape[1] == 45:
            preds = out0[0].T  # [N, 45]
        else:
            preds = out0[0]

        lat_ms = (time.perf_counter() - t0) * 1000.0

        # Parse detections
        boxes_xywh = preds[:, :4]
        cls_scores = preds[:, 4:6]
        max_cls_score = np.max(cls_scores, axis=1)
        pred_cls_id = np.argmax(cls_scores, axis=1)

        mask = max_cls_score >= self.conf_thresh
        if not np.any(mask):
            return [], lat_ms

        sel_boxes = boxes_xywh[mask]
        sel_scores = max_cls_score[mask]
        sel_cls = pred_cls_id[mask]
        sel_kpts = preds[mask, 6:]

        # Convert xywh to xyxy
        cx, cy, w, h = sel_boxes[:, 0], sel_boxes[:, 1], sel_boxes[:, 2], sel_boxes[:, 3]
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        boxes_px = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_px.tolist(),
            sel_scores.tolist(),
            self.conf_thresh,
            self.nms_thresh
        )

        detections = []
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            for idx in indices:
                bx = boxes_px[idx]
                sc = sel_scores[idx]
                cl = sel_cls[idx]
                kp_raw = sel_kpts[idx].reshape(-1, 3)

                # Rescale box back to original image
                orig_box = [
                    float(np.clip((bx[0] - pad_x) / ratio, 0, orig_w)),
                    float(np.clip((bx[1] - pad_y) / ratio, 0, orig_h)),
                    float(np.clip((bx[2] - pad_x) / ratio, 0, orig_w)),
                    float(np.clip((bx[3] - pad_y) / ratio, 0, orig_h))
                ]

                # Rescale keypoints
                kpt_list = []
                for k_i in range(min(13, len(kp_raw))):
                    kx, ky, kv = kp_raw[k_i]
                    # Check if already probability
                    vis = float(kv) if (0.0 <= kv <= 1.0) else float(1.0 / (1.0 + np.exp(-kv)))
                    rx = float((kx - pad_x) / ratio)
                    ry = float((ky - pad_y) / ratio)
                    kpt_list.append({
                        "index": k_i,
                        "name": KEYPOINT_NAMES[k_i],
                        "x": rx,
                        "y": ry,
                        "visibility": vis
                    })

                detections.append({
                    "bbox": orig_box,
                    "score": float(sc),
                    "class_id": int(cl),
                    "keypoints": kpt_list
                })

        return detections, lat_ms


def draw_visuals(frame, detections, show_indices=False, show_labels=False):
    canvas = frame.copy()

    for det in detections:
        cls_id = det["class_id"]
        cfg = COLOR_LEFT if cls_id == 0 else COLOR_RIGHT
        bbox = [int(v) for v in det["bbox"]]
        kpts = det["keypoints"]
        num_kpts = len(kpts)

        # Bounding Box
        cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), cfg["bbox"], 2)

        # Class & Score Badge
        badge_text = f"{cfg['name']} {det['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(canvas, (bbox[0], bbox[1] - th - 8), (bbox[0] + tw + 8, bbox[1]), cfg["bbox"], -1)
        cv2.putText(canvas, badge_text, (bbox[0] + 4, bbox[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Skeleton connections
        pairs = SKELETON_PAIRS_15KP if num_kpts >= 15 else SKELETON_PAIRS
        for i1, i2 in pairs:
            if i1 < num_kpts and i2 < num_kpts:
                p1, p2 = kpts[i1], kpts[i2]
                v1, v2 = p1["visibility"], p2["visibility"]
                if v1 >= 0.15 and v2 >= 0.15:
                    pt1 = (int(round(p1["x"])), int(round(p1["y"])))
                    pt2 = (int(round(p2["x"])), int(round(p2["y"])))
                    thickness = 2 if (v1 >= 0.35 and v2 >= 0.35) else 1
                    cv2.line(canvas, pt1, pt2, cfg["skeleton"], thickness, cv2.LINE_AA)

        # Keypoints
        for kp in kpts:
            vis = kp["visibility"]
            if vis < 0.15:
                continue
            pt = (int(round(kp["x"])), int(round(kp["y"])))

            if vis >= 0.35:
                # Solid circle
                cv2.circle(canvas, pt, 4, cfg["kpt_vis"], -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 4, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                # Hollow ring
                cv2.circle(canvas, pt, 3, cfg["kpt_occ"], 1, cv2.LINE_AA)

            if show_indices or show_labels:
                lbl = str(kp["index"]) if show_indices else kp["name"]
                cv2.putText(canvas, lbl, (pt[0] + 5, pt[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def draw_hud(frame, frame_idx, fps, lat_ms, active_feet, schema_name="13-KP Mobile"):
    h, w = frame.shape[:2]
    # HUD panel top-left
    panel_w, panel_h = 320, 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (15, 15), (15 + panel_w, 15 + panel_h), (80, 80, 80), 1)

    cv2.putText(frame, f"STAGE A: {schema_name}", (25, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f} ({lat_ms:.1f} ms)", (25, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active Feet: {active_feet} | Frame: {frame_idx}", (25, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, "One-Euro Filter: ACTIVE", (25, 106),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 128), 1, cv2.LINE_AA)


def run_demo(video_path, weights_path, output_path, conf_thresh=0.25, nms_thresh=0.45,
             max_frames=0, show_indices=False, show_labels=False, apply_smoothing=True,
             expand_15kp=False):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps_in, (w, h))

    pipeline = VideoInferencePipeline(
        model_path=weights_path,
        input_size=256,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh
    )

    smoothers = {
        0: KeypointSmoother(num_keypoints=13),
        1: KeypointSmoother(num_keypoints=13)
    }

    schema_label = "15-KP Derived (256)" if expand_15kp else "13-KP Mobile (256)"
    print(f"[*] Processing: {video_path} ({total_frames} frames) -> {output_path}")

    frame_idx = 0
    rolling_fps = fps_in
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if 0 < max_frames < frame_idx:
            break

        now = frame_idx / fps_in

        # 1. Infer detections
        dets, lat_ms = pipeline.infer(frame)

        # 2. De-duplicate physical feet
        dets = deduplicate_physical_feet(dets, iou_thresh=0.60)

        # 3. Geometric chirality & temporal smoothing
        processed_dets = []
        for det in dets:
            kpts = det["keypoints"]
            if len(kpts) >= 11:
                geom_cls = determine_geometric_chirality_13kp(kpts)
                det["class_id"] = geom_cls

                if apply_smoothing:
                    smoothed_kpts = smoothers[geom_cls].smooth(kpts, now)
                    det["keypoints"] = smoothed_kpts

                if expand_15kp:
                    det["keypoints"] = derive_secondary_keypoints_15kp(det["keypoints"], bbox=det.get("bbox"))

            processed_dets.append(det)

        # 4. Render visuals & HUD
        annotated = draw_visuals(
            frame,
            processed_dets,
            show_indices=show_indices,
            show_labels=show_labels
        )

        instant_fps = 1000.0 / max(lat_ms, 1.0)
        rolling_fps = 0.9 * rolling_fps + 0.1 * instant_fps
        draw_hud(annotated, frame_idx, rolling_fps, lat_ms, len(processed_dets), schema_name=schema_label)

        out.write(annotated)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"  Frame [{frame_idx:04d}/{total_frames}] | Latency: {lat_ms:.1f}ms | Active Feet: {len(processed_dets)}")

    cap.release()
    out.release()
    elapsed = time.time() - t_start

    print("=" * 75)
    print(f"[OK] Demo video successfully generated in {elapsed:.1f}s!")
    print(f"     Saved to: {output_path}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Stage A 256 Mobile Video Demo Generator")
    parser.add_argument("--video", default="data/test_video.mp4", help="Input video file path")
    parser.add_argument("--weights", default="",
                        help="Trained PyTorch weights path (.pt) or ONNX model path (.onnx)")
    parser.add_argument("--output", default="deliverables/stage_a_256_13kp/demo/annotated_output_256.mp4",
                        help="Rendered video output path")
    parser.add_argument("--conf", type=float, default=0.25, help="Foot confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--show_indices", action="store_true", help="Display 0..12/0..14 keypoint index numbers")
    parser.add_argument("--show_labels", action="store_true", help="Display keypoint names")
    parser.add_argument("--no_smooth", action="store_true", help="Disable One-Euro temporal smoothing")
    parser.add_argument("--expand_15kp", action="store_true",
                        help="Derive secondary keypoints (ball_top, shin_mid) via anatomical post-processing to yield 15-KP")
    args = parser.parse_args()

    weights_path = args.weights
    if not weights_path:
        for cand in [
            "deliverables/stage_a_256_13kp/models/stage-a-256-13kp-fp32.onnx",
            "deliverables/stage_a_256_13kp/models/stage-a-256-13kp-v7.pt",
            "deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.pt",
            "outputs/stage_a/run_v7_256_13kp/weights/best.pt"
        ]:
            if Path(cand).exists():
                weights_path = cand
                break
        if not weights_path:
            weights_path = "deliverables/stage_a_256_13kp/models/stage-a-256-13kp-fp32.onnx"

    run_demo(
        video_path=args.video,
        weights_path=weights_path,
        output_path=args.output,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        max_frames=args.max_frames,
        show_indices=args.show_indices,
        show_labels=args.show_labels,
        apply_smoothing=not args.no_smooth,
        expand_15kp=args.expand_15kp
    )


if __name__ == "__main__":
    main()
