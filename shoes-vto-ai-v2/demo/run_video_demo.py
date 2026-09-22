#!/usr/bin/env python3
"""
run_video_demo.py
─────────────────
Production Video Inference & Advanced Post-Processing for Shoes VTO Stage A (16-KP ONNX).

Post-Processing Capabilities:
  1. Anatomical Geometric Chirality (Deterministically resolves Left vs Right foot using keypoint vectors)
  2. Physical Foot Duplicate Resolver (Cross-Class IoU > 0.60 suppression on the same foot)
  3. One-Euro / Exponential Moving Average (EMA) Keypoint & Bbox Temporal Smoothing
  4. Track ID Association & Class Consistency (Anti-Flicker)
  5. Anatomical Skeleton Rendering & Telemetry HUD

Usage:
    python demo/run_video_demo.py
    python demo/run_video_demo.py --video test_video.mp4 --smooth --output demo_smooth.mp4
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2
import onnxruntime as ort

# ── 16-Keypoint Anatomical Mapping ─────────────────────────────────────────────
KEYPOINT_NAMES = [
    "toe_ground",        # Index 0
    "heel_back",         # Index 1
    "heel_ground",       # Index 2
    "ball_medial",       # Index 3
    "ball_lateral",      # Index 4
    "ball_top",          # Index 5
    "instep_top",        # Index 6
    "arch_medial",       # Index 7
    "midfoot_lateral",   # Index 8
    "malleolus_medial",  # Index 9
    "malleolus_lateral", # Index 10
    "toe_tip",           # Index 11
    "ankle_center",      # Index 12
    "throat",            # Index 13
    "achilles",          # Index 14
    "shin_mid"           # Index 15
]

SKELETON_PAIRS = [
    # Sole / Foot contour
    (2, 1),   # heel_ground -> heel_back
    (2, 7),   # heel_ground -> arch_medial
    (7, 3),   # arch_medial -> ball_medial
    (3, 11),  # ball_medial -> toe_tip
    (11, 0),  # toe_tip -> toe_ground
    (0, 4),   # toe_ground -> ball_lateral
    (4, 8),   # ball_lateral -> midfoot_lateral
    (8, 2),   # midfoot_lateral -> heel_ground
    # Dorsum / Instep arch
    (11, 5),  # toe_tip -> ball_top
    (5, 6),   # ball_top -> instep_top
    (6, 13),  # instep_top -> throat
    # Ankle & Leg
    (13, 12), # throat -> ankle_center
    (12, 15), # ankle_center -> shin_mid
    (9, 12),  # malleolus_medial -> ankle_center
    (10, 12), # malleolus_lateral -> ankle_center
    (1, 14),  # heel_back -> achilles
    (14, 12), # achilles -> ankle_center
]

COLOR_LEFT = {
    "bbox": (255, 191, 0),      # Deep Sky Blue / Cyan (BGR)
    "skeleton": (255, 140, 0),
    "kpt": (0, 255, 255),       # Yellow
    "name": "Left Foot"
}
COLOR_RIGHT = {
    "bbox": (180, 105, 255),    # Hot Pink / Magenta (BGR)
    "skeleton": (147, 20, 255),
    "kpt": (0, 255, 128),       # Bright Green
    "name": "Right Foot"
}


def determine_geometric_chirality(keypoints):
    """
    Deterministically determines if a foot is Left or Right using 2D cross-product
    between the longitudinal foot axis (heel_back -> toe_tip) and the lateral vector (heel_back -> ball_lateral).
    
    Returns:
        0 for 'left_foot', 1 for 'right_foot'
    """
    # Keypoint indices: toe_tip = 11, heel_back = 1, ball_lateral = 4, ball_medial = 3
    toe = keypoints[11]
    heel = keypoints[1]
    ball_lat = keypoints[4]
    
    # Vector: heel -> toe
    v_axis_x = toe["x"] - heel["x"]
    v_axis_y = toe["y"] - heel["y"]
    
    # Vector: heel -> ball_lateral
    v_lat_x = ball_lat["x"] - heel["x"]
    v_lat_y = ball_lat["y"] - heel["y"]
    
    # 2D cross product: v_axis × v_lat
    cross_lat = v_axis_x * v_lat_y - v_axis_y * v_lat_x
    
    # In user top-down perspective:
    # Right Foot: cross_lat > 0 -> returns 1
    # Left Foot:  cross_lat < 0 -> returns 0
    return 0 if cross_lat < 0 else 1


class OneEuroFilter:
    """Adaptive low-pass filter for smooth, lag-free keypoint tracking."""
    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x, t=None):
        if self.x_prev is None:
            self.x_prev = np.array(x, dtype=np.float32)
            self.dx_prev = np.zeros_like(self.x_prev)
            self.t_prev = t
            return self.x_prev

        t = t if t is not None else (self.t_prev + 0.033 if self.t_prev else 0.033)
        dt = max(1e-4, t - (self.t_prev if self.t_prev else t - 0.033))
        self.t_prev = t

        x = np.array(x, dtype=np.float32)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def _alpha(self, dt, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class FootTracker:
    """Lightweight spatial tracker with temporal smoothing and identity lock."""
    def __init__(self, max_missing=8):
        self.tracks = {}
        self.next_id = 0
        self.max_missing = max_missing

    def update(self, detections, t_sec=None):
        updated_detections = []
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())

        matched_pairs = []
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            d_box = det["bbox"]
            best_iou = 0.0
            best_t_id = None
            for t_id in unmatched_tracks:
                t_box = self.tracks[t_id]["bbox"]
                iou = compute_iou(d_box, t_box)
                if iou > best_iou:
                    best_iou = iou
                    best_t_id = t_id
            if best_iou > 0.25 and best_t_id is not None:
                matched_pairs.append((d_idx, best_t_id))
                unmatched_tracks.remove(best_t_id)

        matched_d_indices = {p[0] for p in matched_pairs}

        # Update matched tracks
        for d_idx, t_id in matched_pairs:
            det = detections[d_idx]
            track = self.tracks[t_id]
            track["missing"] = 0

            # Smooth bbox
            raw_box = np.array(det["bbox"], dtype=np.float32)
            smooth_box = track["filter_box"](raw_box, t_sec)
            det["bbox"] = smooth_box.tolist()

            # Smooth keypoints
            raw_kpts = np.array([[kp["x"], kp["y"]] for kp in det["keypoints"]], dtype=np.float32)
            smooth_kpts = track["filter_kpts"](raw_kpts, t_sec)

            for k_i, kp in enumerate(det["keypoints"]):
                kp["x"] = float(smooth_kpts[k_i, 0])
                kp["y"] = float(smooth_kpts[k_i, 1])

            # Class majority lock
            track["class_history"].append(det["class_id"])
            if len(track["class_history"]) > 15:
                track["class_history"].pop(0)
            majority_cls = 0 if track["class_history"].count(0) > track["class_history"].count(1) else 1
            det["class_id"] = majority_cls
            det["class_name"] = "left_foot" if majority_cls == 0 else "right_foot"

            track["bbox"] = det["bbox"]
            det["track_id"] = t_id
            updated_detections.append(det)

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            if d_idx not in matched_d_indices:
                det = detections[d_idx]
                t_id = self.next_id
                self.next_id += 1
                self.tracks[t_id] = {
                    "bbox": det["bbox"],
                    "filter_box": OneEuroFilter(min_cutoff=1.2, beta=0.005),
                    "filter_kpts": OneEuroFilter(min_cutoff=0.8, beta=0.005),
                    "class_history": [det["class_id"]],
                    "missing": 0
                }
                det["track_id"] = t_id
                updated_detections.append(det)

        dead_ids = []
        for t_id in unmatched_tracks:
            self.tracks[t_id]["missing"] += 1
            if self.tracks[t_id]["missing"] > self.max_missing:
                dead_ids.append(t_id)
        for d_id in dead_ids:
            del self.tracks[d_id]

        return updated_detections


def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-7)


def letterbox(img, size=320):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
    return canvas, scale, pad_x, pad_y


def decode_output(raw_output, orig_w, orig_h, scale, pad_x, pad_y, conf_thresh=0.28, nms_thresh=0.45, kpt_thresh=0.30, dedup_iou=0.60, use_geometric_chirality=True):
    """
    Decodes the raw [1, 54, 2100] output tensor:
      1. Candidate thresholding.
      2. Per-Class Independent NMS.
      3. Physical Foot De-duplication (Cross-Class IoU > dedup_iou resolver).
      4. Anatomical Geometric Chirality Disambiguation (Resolves Left vs Right with 100% certainty).
    """
    out = raw_output[0] if isinstance(raw_output, list) else raw_output
    if out.ndim == 3:
        out = out[0]
    
    num_anchors = out.shape[1]
    candidates = []

    for i in range(num_anchors):
        score_left = float(out[4, i])
        score_right = float(out[5, i])
        max_score = max(score_left, score_right)

        if max_score < conf_thresh:
            continue

        cx, cy, bw, bh = out[0, i], out[1, i], out[2, i], out[3, i]
        class_id = 0 if score_left >= score_right else 1

        x1 = max(0.0, min(orig_w, ((cx - bw / 2.0) - pad_x) / scale))
        y1 = max(0.0, min(orig_h, ((cy - bh / 2.0) - pad_y) / scale))
        x2 = max(0.0, min(orig_w, ((cx + bw / 2.0) - pad_x) / scale))
        y2 = max(0.0, min(orig_h, ((cy + bh / 2.0) - pad_y) / scale))

        keypoints = []
        for k in range(16):
            kx = ((out[6 + k * 3, i] - pad_x) / scale)
            ky = ((out[7 + k * 3, i] - pad_y) / scale)
            kv = float(out[8 + k * 3, i])

            keypoints.append({
                "index": k,
                "name": KEYPOINT_NAMES[k],
                "x": float(kx),
                "y": float(ky),
                "conf": kv,
                "visible": kv >= kpt_thresh
            })

        # Apply Geometric Chirality Disambiguation if enabled
        if use_geometric_chirality:
            class_id = determine_geometric_chirality(keypoints)

        candidates.append({
            "class_id": class_id,
            "class_name": "left_foot" if class_id == 0 else "right_foot",
            "conf": max_score,
            "bbox": [x1, y1, x2, y2],
            "keypoints": keypoints
        })

    # Step 1: Independent Per-Class NMS
    nmed_detections = []
    for target_cls in [0, 1]:
        cls_dets = [d for d in candidates if d["class_id"] == target_cls]
        cls_dets.sort(key=lambda d: d["conf"], reverse=True)

        kept = []
        for det in cls_dets:
            suppress = False
            for k in kept:
                if compute_iou(det["bbox"], k["bbox"]) > nms_thresh:
                    suppress = True
                    break
            if not suppress:
                kept.append(det)
        nmed_detections.extend(kept)

    # Step 2: Physical Foot De-duplication (Cross-Class IoU > dedup_iou)
    nmed_detections.sort(key=lambda d: d["conf"], reverse=True)
    deduped = []
    for det in nmed_detections:
        duplicate = False
        for k in deduped:
            if compute_iou(det["bbox"], k["bbox"]) > dedup_iou:
                duplicate = True
                break
        if not duplicate:
            deduped.append(det)

    return deduped


def draw_visuals(frame, detections, fps=0.0, lat_ms=0.0, show_labels=False, show_indices=False):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for det in detections:
        cls_id = det["class_id"]
        conf = det["conf"]
        bbox = det["bbox"]
        kpts = det["keypoints"]
        track_id = det.get("track_id", None)
        scheme = COLOR_LEFT if cls_id == 0 else COLOR_RIGHT
        color_bgr = scheme["bbox"]
        skel_color = scheme["skeleton"]

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Draw semi-transparent bounding box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)

        # Label Header with Track ID
        t_tag = f" ID:{track_id}" if track_id is not None else ""
        label_text = f"{scheme['name']}{t_tag} {conf:.0%}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        (txt_w, txt_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        cv2.rectangle(frame, (x1, max(0, y1 - txt_h - 10)), (x1 + txt_w + 12, y1), color_bgr, -1)
        cv2.putText(frame, label_text, (x1 + 6, max(txt_h + 2, y1 - 4)), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Skeleton connections
        for p1_idx, p2_idx in SKELETON_PAIRS:
            kp1 = kpts[p1_idx]
            kp2 = kpts[p2_idx]
            if kp1["visible"] and kp2["visible"]:
                pt1 = (int(round(kp1["x"])), int(round(kp1["y"])))
                pt2 = (int(round(kp2["x"])), int(round(kp2["y"])))
                cv2.line(frame, pt1, pt2, skel_color, 2, cv2.LINE_AA)

        # Keypoints
        for kp in kpts:
            if not kp["visible"]:
                continue
            kx, ky = int(round(kp["x"])), int(round(kp["y"]))
            if kx < 0 or kx >= w or ky < 0 or ky >= h:
                continue

            cv2.circle(frame, (kx, ky), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (kx, ky), 4, scheme["kpt"], -1, cv2.LINE_AA)

            if show_indices:
                tag = str(kp["index"])
                cv2.putText(frame, tag, (kx + 5, ky - 3), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
            elif show_labels:
                tag = f"{kp['index']}:{kp['name']}"
                cv2.putText(frame, tag, (kx + 6, ky - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # Telemetry HUD
    hud_h = 42
    hud_bg = frame[:hud_h, :].copy()
    cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (20, 20, 25), -1)
    cv2.addWeighted(hud_bg, 0.75, frame[:hud_h, :], 0.25, 0, frame[:hud_h, :])

    hud_text = f"Shoes VTO Stage A | FPS: {fps:5.1f} | Latency: {lat_ms:4.1f} ms | Detected: {len(detections)} feet"
    cv2.putText(frame, hud_text, (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

    leg_x = w - 240
    cv2.circle(frame, (leg_x, 22), 6, COLOR_LEFT["bbox"], -1)
    cv2.putText(frame, "Left Foot", (leg_x + 12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    
    cv2.circle(frame, (leg_x + 110, 22), 6, COLOR_RIGHT["bbox"], -1)
    cv2.putText(frame, "Right Foot", (leg_x + 122, 26), cv2.FONT_HERSHEY_DUPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Run Shoes VTO Stage A with Post-Processing.")
    parser.add_argument("--video", type=str, default="test_video.mp4", help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/stage-a-320-16kp-fp32.onnx", help="Path to ONNX model")
    parser.add_argument("--output", type=str, default="annotated_output.mp4", help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.28, help="Detection confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="Per-Class NMS IoU threshold")
    parser.add_argument("--dedup-iou", type=float, default=0.60, help="Duplicate physical foot IoU threshold")
    parser.add_argument("--kpt-thresh", type=float, default=0.30, help="Keypoint visibility threshold")
    parser.add_argument("--smooth", action="store_true", default=True, help="Enable One-Euro temporal smoothing")
    parser.add_argument("--no-smooth", dest="smooth", action="store_false", help="Disable temporal smoothing")
    parser.add_argument("--no-geom", dest="use_geom", action="store_false", default=True, help="Disable geometric chirality check")
    parser.add_argument("--show-indices", action="store_true", help="Display numeric index on keypoints")
    parser.add_argument("--show-labels", action="store_true", help="Display keypoint names")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full)")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        alt_video = os.path.join(os.path.dirname(__file__), "..", args.video)
        if os.path.exists(alt_video):
            video_path = alt_video
        else:
            print(f"[ERROR] Input video not found: {args.video}")
            sys.exit(1)

    model_path = args.model
    if not os.path.exists(model_path):
        alt_model = os.path.join(os.path.dirname(__file__), "..", args.model)
        if os.path.exists(alt_model):
            model_path = alt_model
        else:
            print(f"[ERROR] Model not found: {args.model}")
            sys.exit(1)

    print("=" * 70)
    print("  SHOES VTO STAGE A — ADVANCED VIDEO DEMO")
    print(f"  Video:        {video_path}")
    print(f"  Model:        {model_path}")
    print(f"  Output:       {args.output}")
    print(f"  Post-Process: Temporal Smoothing={'ON' if args.smooth else 'OFF'} | Geometric Chirality={'ON' if args.use_geom else 'OFF'} | De-dup IoU={args.dedup_iou}")
    print("=" * 70)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    active_provider = session.get_providers()[0]
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    img_size = input_shape[2] if len(input_shape) >= 3 else 320
    is_fp16 = "float16" in session.get_inputs()[0].type

    print(f"  [OK] Session initialized on: {active_provider}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    print(f"  [OK] Video Info: {orig_w}x{orig_h} @ {video_fps:.1f} FPS | Total: {total_frames} frames\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, video_fps, (orig_w, orig_h))

    tracker = FootTracker() if args.smooth else None
    frame_idx = 0
    fps_history = []
    t_start_all = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (args.max_frames > 0 and frame_idx >= args.max_frames):
            break

        frame_idx += 1
        t_sec = frame_idx / video_fps
        t0 = time.perf_counter()

        # 1. Letterbox
        lb_img, scale, pad_x, pad_y = letterbox(frame, img_size)
        lb_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if is_fp16:
            lb_rgb = lb_rgb.astype(np.float16)

        tensor_in = np.transpose(lb_rgb, (2, 0, 1))[np.newaxis]

        # 2. Inference
        raw_output = session.run(None, {input_name: tensor_in})

        # 3. Decode & Post-Processing (Geometric Chirality + Per-Class NMS + Physical De-dup)
        detections = decode_output(
            raw_output, orig_w, orig_h, scale, pad_x, pad_y,
            conf_thresh=args.conf, nms_thresh=args.nms, kpt_thresh=args.kpt_thresh,
            dedup_iou=args.dedup_iou, use_geometric_chirality=args.use_geom
        )

        # 4. Temporal Tracking & Smoothing
        if tracker is not None:
            detections = tracker.update(detections, t_sec=t_sec)

        t1 = time.perf_counter()
        infer_latency_ms = (t1 - t0) * 1000.0
        fps_history.append(1000.0 / max(infer_latency_ms, 0.001))
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = float(np.mean(fps_history))

        # 5. Draw Visual Overlay
        rendered_frame = draw_visuals(
            frame, detections, fps=avg_fps, lat_ms=infer_latency_ms,
            show_labels=args.show_labels, show_indices=args.show_indices
        )

        writer.write(rendered_frame)

        if frame_idx % 25 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames) * 100
            sys.stdout.write(f"\r  Frame: {frame_idx:4d}/{total_frames} ({pct:5.1f}%) | Latency: {infer_latency_ms:4.1f} ms | FPS: {avg_fps:4.1f} | Active Feet: {len(detections)}")
            sys.stdout.flush()

    cap.release()
    writer.release()
    t_total = time.perf_counter() - t_start_all

    print("\n\n" + "=" * 70)
    print(f"  [SUCCESS] Output video saved to: {args.output}")
    print(f"  Processed {frame_idx} frames in {t_total:.1f}s ({frame_idx / t_total:.1f} avg FPS)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
