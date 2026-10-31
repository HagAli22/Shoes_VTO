#!/usr/bin/env python3
"""
run_video_demo.py
─────────────────
Production Video Inference & Advanced Post-Processing for Shoes VTO Stage A (13-KP Model).
Supports loading both PyTorch (.pt) and ONNX (.onnx) models.

Key Features & Post-Processing:
  1. 13-Keypoint Schema (excision of ankle_center, ball_top, shin_mid).
  2. Anatomical Geometric Chirality (Deterministically resolves Left vs Right using 2D cross-product).
  3. Occluded Keypoint Rendering: Solid circles for visible points; hollow rings for occluded/inferred points.
  4. Physical Foot Duplicate Resolver: Suppresses phantom duplicate boxes (IoU > 0.60) on the same physical foot.
  5. One-Euro Adaptive Temporal Smoothing: Eliminates jitter while preserving instant dynamic response.
  6. Real-time Telemetry HUD: FPS, latency (ms), frame count, active feet count.

Usage:
  python deliverables/stage_a_13kp/demo/run_video_demo.py
  python deliverables/stage_a_13kp/demo/run_video_demo.py --weights outputs/stage_a/run_v4_13kp/weights/best.pt --video data/test_video.mp4
  python deliverables/stage_a_13kp/demo/run_video_demo.py --model outputs/stage_a/run_v4_13kp/weights/best.onnx --output output_13kp.mp4
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import numpy as np
import cv2

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.stage_a.postprocess import derive_secondary_keypoints_15kp, KEYPOINT_NAMES_15KP

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
    Deterministically determines Left vs Right foot in the 13-KP schema:
      toe:      keypoints[10] (toe_tip)
      heel:     keypoints[1]  (heel_back)
      ball_lat: keypoints[4]  (ball_lateral)
    
    Returns:
        0 for 'left_foot', 1 for 'right_foot'
    """
    toe = keypoints[10]
    heel = keypoints[1]
    ball_lat = keypoints[4]

    v_axis_x = toe["x"] - heel["x"]
    v_axis_y = toe["y"] - heel["y"]

    v_lat_x = ball_lat["x"] - heel["x"]
    v_lat_y = ball_lat["y"] - heel["y"]

    cross_lat = v_axis_x * v_lat_y - v_axis_y * v_lat_x
    return 0 if cross_lat < 0 else 1


class OneEuroFilter:
    """Adaptive low-pass filter for smooth, lag-free keypoint tracking."""
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
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
        self.filters_x = [OneEuroFilter(min_cutoff=1.2, beta=0.008) for _ in range(num_keypoints)]
        self.filters_y = [OneEuroFilter(min_cutoff=1.2, beta=0.008) for _ in range(num_keypoints)]

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


def nms_per_class(candidates, iou_threshold=0.45):
    """Executes NMS independently per class to preserve crossing feet."""
    classes = set(c["class_id"] for c in candidates)
    final = []
    for cls in classes:
        cls_cands = [c for c in candidates if c["class_id"] == cls]
        cls_cands.sort(key=lambda x: x["confidence"], reverse=True)
        keep = []
        for c in cls_cands:
            dup = False
            for k in keep:
                if compute_iou(c["bbox"], k["bbox"]) > iou_threshold:
                    dup = True
                    break
            if not dup:
                keep.append(c)
        final.extend(keep)
    return final


def deduplicate_physical_feet(detections, iou_threshold=0.60):
    """Suppresses duplicate conflicting boxes on the exact same foot."""
    if len(detections) <= 1:
        return detections
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    kept = []
    for cand in detections:
        is_dup = False
        for ex in kept:
            if compute_iou(cand["bbox"], ex["bbox"]) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(cand)
    return kept


class FootPoseDetector13KP:
    """Wrapper that supports running inference using either Ultralytics (.pt) or ONNX."""
    def __init__(self, model_path: str, imgsz=0, conf_thresh=0.25, nms_thresh=0.45):
        self.model_path = model_path
        self.imgsz = imgsz if imgsz > 0 else 320
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.is_onnx = model_path.endswith(".onnx")

        if self.is_onnx:
            # Enable CUDA libraries if running under Windows PyTorch environment
            try:
                import torch
                tlib = Path(torch.__file__).parent / "lib"
                if tlib.exists() and hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(tlib))
            except Exception:
                pass
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            # Auto-detect resolution from model input dimensions
            inp_shape = self.session.get_inputs()[0].shape
            if imgsz <= 0 and len(inp_shape) == 4 and isinstance(inp_shape[2], int):
                self.imgsz = inp_shape[2]
            print(f"[*] Loaded ONNX model with providers: {self.session.get_providers()} | Resolution: {self.imgsz}x{self.imgsz}")
        else:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"[*] Loaded PyTorch YOLO model: {model_path} | Resolution: {self.imgsz}x{self.imgsz}")

    def letterbox(self, img, new_shape=(320, 320), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def infer(self, frame):
        orig_h, orig_w = frame.shape[:2]
        t0 = time.perf_counter()

        if not self.is_onnx:
            # PyTorch inference
            results = self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf_thresh,
                iou=self.nms_thresh,
                verbose=False
            )[0]
            lat_ms = (time.perf_counter() - t0) * 1000.0

            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                kpts_data = results.keypoints.data.cpu().numpy() if results.keypoints is not None else None

                for i in range(len(boxes)):
                    c_id = int(classes[i])
                    kpts = []
                    if kpts_data is not None and i < len(kpts_data):
                        for k in range(13):
                            kx, ky, kv = kpts_data[i][k]
                            kpts.append({
                                "index": k,
                                "name": KEYPOINT_NAMES[k],
                                "x": float(kx),
                                "y": float(ky),
                                "visibility": float(kv)
                            })
                    detections.append({
                        "bbox": [float(b) for b in boxes[i]],
                        "class_id": c_id,
                        "confidence": float(confs[i]),
                        "keypoints": kpts
                    })
            return detections, lat_ms

        else:
            # ONNX inference
            lb_img, scale, (dw, dh) = self.letterbox(frame, (self.imgsz, self.imgsz))
            blob = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(blob, (2, 0, 1))[None, ...]

            preds = self.session.run(None, {self.input_name: blob})[0]
            lat_ms = (time.perf_counter() - t0) * 1000.0

            # Parse [1, 45, 2100]
            preds = preds[0]  # [45, 2100]
            candidates = []
            for col in range(preds.shape[1]):
                cx, cy, w, h = preds[0:4, col]
                s_left = 1.0 / (1.0 + math.exp(-preds[4, col]))
                s_right = 1.0 / (1.0 + math.exp(-preds[5, col]))

                max_score = max(s_left, s_right)
                if max_score < self.conf_thresh:
                    continue

                cls_id = 0 if s_left >= s_right else 1

                # Letterbox unpadding
                x1 = (cx - w / 2 - dw) / scale
                y1 = (cy - h / 2 - dh) / scale
                x2 = (cx + w / 2 - dw) / scale
                y2 = (cy + h / 2 - dh) / scale

                # 13 keypoints (rows 6..44)
                kpts = []
                for k in range(13):
                    kx = (preds[6 + k * 3, col] - dw) / scale
                    ky = (preds[6 + k * 3 + 1, col] - dh) / scale
                    kv = 1.0 / (1.0 + math.exp(-preds[6 + k * 3 + 2, col]))
                    kpts.append({
                        "index": k,
                        "name": KEYPOINT_NAMES[k],
                        "x": float(kx),
                        "y": float(ky),
                        "visibility": float(kv)
                    })

                candidates.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "confidence": max_score,
                    "keypoints": kpts
                })

            detections = nms_per_class(candidates, self.nms_thresh)
            return detections, lat_ms


def draw_visuals(frame, detections, show_indices=False, show_labels=False, kpt_vis_thresh=0.35, kpt_occ_thresh=0.15):
    overlay = frame.copy()

    for det in detections:
        cls_id = det["class_id"]
        col = COLOR_LEFT if cls_id == 0 else COLOR_RIGHT
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        # Bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col["bbox"], 2)

        # Label tag
        tag = f"{col['name']}: {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 8), (x1 + tw + 6, y1), col["bbox"], -1)
        cv2.putText(overlay, tag, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        kpts = det["keypoints"]
        if not kpts:
            continue

        # Draw anatomical skeleton
        skeleton = SKELETON_PAIRS_15KP if len(kpts) == 15 else SKELETON_PAIRS
        for k1, k2 in skeleton:
            kp1 = kpts[k1]
            kp2 = kpts[k2]
            if kp1["visibility"] >= kpt_occ_thresh and kp2["visibility"] >= kpt_occ_thresh:
                pt1 = (int(kp1["x"]), int(kp1["y"]))
                pt2 = (int(kp2["x"]), int(kp2["y"]))
                # If both visible -> solid line; if one occluded -> thinner line
                thickness = 2 if (kp1["visibility"] >= kpt_vis_thresh and kp2["visibility"] >= kpt_vis_thresh) else 1
                cv2.line(overlay, pt1, pt2, col["skeleton"], thickness, cv2.LINE_AA)

        # Draw keypoints (solid dots for visible, hollow rings for occluded)
        for kp in kpts:
            v = kp["visibility"]
            px, py = int(kp["x"]), int(kp["y"])

            if v >= kpt_vis_thresh:
                # Solid dot
                cv2.circle(overlay, (px, py), 4, col["kpt_vis"], -1, cv2.LINE_AA)
                cv2.circle(overlay, (px, py), 4, (0, 0, 0), 1, cv2.LINE_AA)
            elif v >= kpt_occ_thresh:
                # Hollow ring
                cv2.circle(overlay, (px, py), 3, col["kpt_occ"], 1, cv2.LINE_AA)

            # Optional annotations
            if show_indices:
                cv2.putText(overlay, str(kp["index"]), (px + 5, py - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            elif show_labels:
                cv2.putText(overlay, f"{kp['index']}:{kp['name']}", (px + 5, py - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def draw_hud(frame, frame_idx, fps, lat_ms, feet_count, schema_name="13-KP"):
    h, w = frame.shape[:2]
    # HUD banner background
    hud_bg = frame.copy()
    cv2.rectangle(hud_bg, (0, 0), (w, 42), (20, 20, 20), -1)
    cv2.addWeighted(hud_bg, 0.85, frame, 0.15, 0, frame)

    text = f"Shoes VTO ({schema_name}) | Frame: {frame_idx:04d} | Latency: {lat_ms:.1f}ms | FPS: {fps:.1f} | Active Feet: {feet_count}"
    cv2.putText(frame, text, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 1, cv2.LINE_AA)


def run_demo(
    video_path: str,
    weights_path: str,
    output_path: str,
    conf_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    imgsz: int = 0,
    max_frames: int = 0,
    show_indices: bool = False,
    show_labels: bool = False,
    apply_smoothing: bool = True,
    expand_15kp: bool = False
):
    schema_label = "15-KP (Derived)" if expand_15kp else "13-KP"
    print("=" * 75)
    print(f"  STAGE A {schema_label} VIDEO INFERENCE & DEMO GENERATOR                     ")
    print(f"  Model:       {weights_path}")
    print(f"  Input Video: {video_path}")
    print(f"  Output:      {output_path}")
    print(f"  Expand 15KP: {expand_15kp}")
    print("=" * 75)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    detector = FootPoseDetector13KP(
        model_path=weights_path,
        imgsz=imgsz,
        conf_thresh=conf_thresh,
        nms_thresh=nms_thresh
    )

    smoothers = {0: KeypointSmoother(13), 1: KeypointSmoother(13)}

    frame_idx = 0
    t_start = time.time()
    rolling_fps = fps_in

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if max_frames > 0 and frame_idx > max_frames:
            break

        now = time.time()
        detections, lat_ms = detector.infer(frame)

        # 1. Deduplicate conflicting physical feet (IoU > 0.60)
        detections = deduplicate_physical_feet(detections, iou_threshold=0.60)

        # 2. Geometric Chirality disambiguation & temporal smoothing
        processed_dets = []
        for det in detections:
            kpts = det["keypoints"]
            if len(kpts) == 13:
                # Disambiguate class ID geometrically
                geom_cls = determine_geometric_chirality_13kp(kpts)
                det["class_id"] = geom_cls

                # Smooth keypoints temporally
                if apply_smoothing:
                    smoothed_kpts = smoothers[geom_cls].smooth(kpts, now)
                    det["keypoints"] = smoothed_kpts

                # Derive 15-KP if requested (ball_top, shin_mid)
                if expand_15kp:
                    det["keypoints"] = derive_secondary_keypoints_15kp(det["keypoints"], bbox=det.get("bbox"))

            processed_dets.append(det)

        # 3. Render visuals & HUD
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
    parser = argparse.ArgumentParser(description="Stage A 13-KP Video Demo Generator")
    parser.add_argument("--video", default="data/test_video.mp4", help="Input video file path")
    parser.add_argument("--weights", default="",
                        help="Trained PyTorch weights path (.pt) or ONNX model path (.onnx)")
    parser.add_argument("--output", default="deliverables/stage_a_13kp/demo/annotated_output_13kp.mp4",
                        help="Rendered video output path")
    parser.add_argument("--conf", type=float, default=0.25, help="Foot confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=0, help="Inference resolution (0 = auto-detect from model)")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--show_indices", action="store_true", help="Display 0..12/0..14 keypoint index numbers")
    parser.add_argument("--show_labels", action="store_true", help="Display keypoint names")
    parser.add_argument("--no_smooth", action="store_true", help="Disable One-Euro temporal smoothing")
    parser.add_argument("--expand_15kp", action="store_true",
                        help="Derive secondary keypoints (ball_top, shin_mid) via anatomical post-processing to yield 15-KP")
    args = parser.parse_args()

    # Smart default weights resolution
    weights_path = args.weights
    if not weights_path:
        for cand in [
            "deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt",
            "deliverables/stage_a_13kp/models/stage-a-320-13kp.pt",
            "outputs/stage_a/run_v4_13kp/weights/best.pt",
            "deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.onnx",
            "deliverables/stage_a_13kp/models/stage-a-320-13kp.onnx",
            "outputs/stage_a/run_v4_13kp/weights/best.onnx"
        ]:
            if Path(cand).exists():
                weights_path = cand
                break
        if not weights_path:
            weights_path = "deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt"

    run_demo(
        video_path=args.video,
        weights_path=weights_path,
        output_path=args.output,
        conf_thresh=args.conf,
        nms_thresh=args.nms,
        imgsz=args.imgsz,
        max_frames=args.max_frames,
        show_indices=args.show_indices,
        show_labels=args.show_labels,
        apply_smoothing=not args.no_smooth,
        expand_15kp=args.expand_15kp
    )


if __name__ == "__main__":
    main()

