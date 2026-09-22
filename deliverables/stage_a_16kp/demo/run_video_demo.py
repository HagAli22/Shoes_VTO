#!/usr/bin/env python3
"""
run_video_demo.py
─────────────────
Standalone Video Inference & Visualization for Shoes VTO Stage A (16-Keypoint ONNX Model).

Dependencies:
    pip install onnxruntime opencv-python numpy

Usage:
    # Basic usage (defaults to models/stage-a-320-16kp-fp32.onnx and test_video.mp4):
    python demo/run_video_demo.py

    # Custom options:
    python demo/run_video_demo.py --video test_video.mp4 --output demo_output.mp4 --show-labels
    python demo/run_video_demo.py --model models/stage-a-320-16kp-fp16.onnx --conf 0.30
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

# Anatomical skeleton connections: pairs of (kp_idx_1, kp_idx_2)
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

# Distinct colors per foot class (BGR)
COLOR_LEFT = {
    "bbox": (255, 191, 0),     # Deep Sky Blue / Cyan (BGR)
    "skeleton": (255, 140, 0), # Deep Sky Blue
    "kpt": (0, 255, 255),      # Yellow
    "kpt_outline": (0, 100, 100),
    "name": "Left Foot"
}
COLOR_RIGHT = {
    "bbox": (180, 105, 255),   # Hot Pink / Magenta (BGR)
    "skeleton": (147, 20, 255),# Deep Pink
    "kpt": (0, 255, 128),      # Bright Green
    "kpt_outline": (0, 100, 0),
    "name": "Right Foot"
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(img, size=320):
    """Letterbox resize image preserving aspect ratio with pad 114."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
    return canvas, scale, pad_x, pad_y


def decode_output(raw_output, orig_w, orig_h, scale, pad_x, pad_y, conf_thresh=0.25, nms_thresh=0.45, kpt_thresh=0.30):
    """
    Decodes the raw [1, 54, 2100] output tensor with Per-Class Independent NMS.
    """
    out = raw_output[0] if isinstance(raw_output, list) else raw_output
    if out.ndim == 3:
        out = out[0]  # [54, 2100]
    
    num_anchors = out.shape[1]
    candidates = []

    for i in range(num_anchors):
        cx, cy, bw, bh = out[0, i], out[1, i], out[2, i], out[3, i]
        score_left = float(out[4, i])
        score_right = float(out[5, i])
        max_score = max(score_left, score_right)

        if max_score < conf_thresh:
            continue

        class_id = 0 if score_left >= score_right else 1

        # Transform box back to original coordinates
        x1 = ((cx - bw / 2.0) - pad_x) / scale
        y1 = ((cy - bh / 2.0) - pad_y) / scale
        x2 = ((cx + bw / 2.0) - pad_x) / scale
        y2 = ((cy + bh / 2.0) - pad_y) / scale

        x1 = max(0.0, min(orig_w, x1))
        y1 = max(0.0, min(orig_h, y1))
        x2 = max(0.0, min(orig_w, x2))
        y2 = max(0.0, min(orig_h, y2))

        # Decode 16 keypoints
        keypoints = []
        for k in range(16):
            kx_raw = out[6 + k * 3, i]
            ky_raw = out[7 + k * 3, i]
            kv_raw = out[8 + k * 3, i]

            kx = ((kx_raw - pad_x) / scale)
            ky = ((ky_raw - pad_y) / scale)
            kv = float(kv_raw)

            keypoints.append({
                "index": k,
                "name": KEYPOINT_NAMES[k],
                "x": float(kx),
                "y": float(ky),
                "conf": kv,
                "visible": kv >= kpt_thresh
            })


        candidates.append({
            "class_id": class_id,
            "class_name": "left_foot" if class_id == 0 else "right_foot",
            "conf": max_score,
            "bbox": [x1, y1, x2, y2],
            "keypoints": keypoints
        })

    # Independent Per-Class NMS
    final_detections = []
    for target_cls in [0, 1]:
        cls_dets = [d for d in candidates if d["class_id"] == target_cls]
        cls_dets.sort(key=lambda d: d["conf"], reverse=True)

        kept = []
        for det in cls_dets:
            suppress = False
            b1 = det["bbox"]
            for k in kept:
                b2 = k["bbox"]
                ix1 = max(b1[0], b2[0])
                iy1 = max(b1[1], b2[1])
                ix2 = min(b1[2], b2[2])
                iy2 = min(b1[3], b2[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                iou = inter / (a1 + a2 - inter + 1e-7)
                if iou > nms_thresh:
                    suppress = True
                    break
            if not suppress:
                kept.append(det)
        final_detections.extend(kept)

    return sorted(final_detections, key=lambda d: d["conf"], reverse=True)


def draw_visuals(frame, detections, fps=0.0, lat_ms=0.0, show_labels=False, show_indices=False):
    """Renders sleek bounding boxes, skeleton, keypoints, and HUD overlay."""
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for det in detections:
        cls_id = det["class_id"]
        conf = det["conf"]
        bbox = det["bbox"]
        kpts = det["keypoints"]
        scheme = COLOR_LEFT if cls_id == 0 else COLOR_RIGHT
        color_bgr = scheme["bbox"]
        skel_color = scheme["skeleton"]

        x1, y1, x2, y2 = [int(v) for v in bbox]

        # Draw semi-transparent bounding box fill + clean outline
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)

        # Draw Label Header
        label_text = f"{scheme['name']} {conf:.0%}"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        cv2.rectangle(frame, (x1, max(0, y1 - txt_h - 10)), (x1 + txt_w + 12, y1), color_bgr, -1)
        cv2.putText(frame, label_text, (x1 + 6, max(txt_h + 2, y1 - 4)), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # Draw Skeleton lines
        for p1_idx, p2_idx in SKELETON_PAIRS:
            kp1 = kpts[p1_idx]
            kp2 = kpts[p2_idx]
            if kp1["visible"] and kp2["visible"]:
                pt1 = (int(kp1["x"]), int(kp1["y"]))
                pt2 = (int(kp2["x"]), int(kp2["y"]))
                cv2.line(frame, pt1, pt2, skel_color, 2, cv2.LINE_AA)

        # Draw Keypoint dots & optional tags
        for kp in kpts:
            if not kp["visible"]:
                continue
            kx, ky = int(kp["x"]), int(kp["y"])
            if kx < 0 or kx >= w or ky < 0 or ky >= h:
                continue

            # Outer ring & Inner dot
            cv2.circle(frame, (kx, ky), 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (kx, ky), 4, scheme["kpt"], -1, cv2.LINE_AA)

            if show_indices:
                tag = str(kp["index"])
                cv2.putText(frame, tag, (kx + 5, ky - 3), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 255, 255), 1, cv2.LINE_AA)
            elif show_labels:
                tag = f"{kp['index']}:{kp['name']}"
                cv2.putText(frame, tag, (kx + 6, ky - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend semi-transparent box fill (alpha=0.12)
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    # ── Top HUD (Live Telemetry) ──────────────────────────────────────────────
    hud_h = 42
    hud_bg = frame[:hud_h, :].copy()
    cv2.rectangle(hud_bg, (0, 0), (w, hud_h), (20, 20, 25), -1)
    cv2.addWeighted(hud_bg, 0.75, frame[:hud_h, :], 0.25, 0, frame[:hud_h, :])

    hud_text = f"Shoes VTO Stage A | FPS: {fps:5.1f} | Latency: {lat_ms:4.1f} ms | Detected: {len(detections)} feet"
    cv2.putText(frame, hud_text, (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)

    # Legend on top right
    leg_x = w - 240
    cv2.circle(frame, (leg_x, 22), 6, COLOR_LEFT["bbox"], -1)
    cv2.putText(frame, "Left Foot", (leg_x + 12, 26), cv2.FONT_HERSHEY_DUPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
    
    cv2.circle(frame, (leg_x + 110, 22), 6, COLOR_RIGHT["bbox"], -1)
    cv2.putText(frame, "Right Foot", (leg_x + 122, 26), cv2.FONT_HERSHEY_DUPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Run Shoes VTO Stage A 16-KP Model on Video.")
    parser.add_argument("--video", type=str, default="test_video.mp4", help="Path to input video file")
    parser.add_argument("--model", type=str, default="models/stage-a-320-16kp-fp32.onnx", help="Path to Stage A ONNX model")
    parser.add_argument("--output", type=str, default="annotated_output.mp4", help="Path to save output video")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--kpt-thresh", type=float, default=0.30, help="Keypoint visibility threshold")
    parser.add_argument("--show-indices", action="store_true", help="Display numeric index on each keypoint")
    parser.add_argument("--show-labels", action="store_true", help="Display keypoint names on video")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = full video)")
    args = parser.parse_args()

    # Search paths if running from root or demo folder
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
    print("  SHOES VTO STAGE A — 16-KP VIDEO DEMO")
    print(f"  Video:  {video_path}")
    print(f"  Model:  {model_path}")
    print(f"  Output: {args.output}")
    print("=" * 70)

    # Initialize ONNX session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)
    active_provider = session.get_providers()[0]
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    img_size = input_shape[2] if len(input_shape) >= 3 else 320
    is_fp16 = "float16" in session.get_inputs()[0].type

    print(f"  [OK] Session initialized on: {active_provider}")
    print(f"  [OK] Model input: {input_shape} | dtype: {session.get_inputs()[0].type}")

    # Open Video
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

    # Output Video Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, video_fps, (orig_w, orig_h))

    frame_idx = 0
    fps_history = []
    t_start_all = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (args.max_frames > 0 and frame_idx >= args.max_frames):
            break

        frame_idx += 1
        t0 = time.perf_counter()

        # 1. Letterbox Preprocess
        lb_img, scale, pad_x, pad_y = letterbox(frame, img_size)
        lb_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if is_fp16:
            lb_rgb = lb_rgb.astype(np.float16)

        tensor_in = np.transpose(lb_rgb, (2, 0, 1))[np.newaxis]

        # 2. Model Inference
        raw_output = session.run(None, {input_name: tensor_in})

        # 3. Decode & Per-Class NMS
        detections = decode_output(
            raw_output, orig_w, orig_h, scale, pad_x, pad_y,
            conf_thresh=args.conf, nms_thresh=args.nms, kpt_thresh=args.kpt_thresh
        )

        t1 = time.perf_counter()
        infer_latency_ms = (t1 - t0) * 1000.0
        instant_fps = 1000.0 / max(infer_latency_ms, 0.001)
        fps_history.append(instant_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = float(np.mean(fps_history))

        # 4. Draw Visual Overlay
        rendered_frame = draw_visuals(
            frame, detections, fps=avg_fps, lat_ms=infer_latency_ms,
            show_labels=args.show_labels, show_indices=args.show_indices
        )

        # 5. Write Frame
        writer.write(rendered_frame)

        # Progress bar
        if frame_idx % 25 == 0 or frame_idx == total_frames:
            pct = (frame_idx / total_frames) * 100
            sys.stdout.write(f"\r  Processing Frame: {frame_idx:4d}/{total_frames} ({pct:5.1f}%) | Latency: {infer_latency_ms:4.1f} ms | FPS: {avg_fps:4.1f} | Detections: {len(detections)}")
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
