"""
eval_video_models.py
────────────────────
Benchmark and visual video generation script for Stage A models:
  1. Standard YOLOv8n detector: FP32, FP16, INT8
  2. Compact YOLOv8-Pico detector: FP32, FP16, INT8

Runs on data/test_video.mp4 and saves 6 annotated comparison videos
with inference latency, detection rate, bounding boxes, and coarse keypoint skeletons.
"""

import os
import sys
import time
import math
import cv2
import numpy as np
import onnxruntime as ort


def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = img.shape[:2]  # [h, w]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def nms_boxes(boxes, scores, iou_thresh=0.45):
    """Greedy Non-Maximum Suppression."""
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def decode_18ch(raw_out, r, dw, dh, orig_w, orig_h, conf_thresh=0.25, iou_thresh=0.45):
    """
    Decodes Stage A [1, 18, 2100] output with independent per-class NMS.
    No cross-class suppression: left_foot and right_foot are processed independently.
    """
    raw = raw_out[0]  # [18, 2100]
    cx = raw[0]
    cy = raw[1]
    w  = raw[2]
    h  = raw[3]
    s_left  = raw[4]
    s_right = raw[5]

    # Convert center coordinates to corner coordinates mapped back to original frame
    x1 = np.clip((cx - w / 2.0 - dw) / r, 0, orig_w)
    y1 = np.clip((cy - h / 2.0 - dh) / r, 0, orig_h)
    x2 = np.clip((cx + w / 2.0 - dw) / r, 0, orig_w)
    y2 = np.clip((cy + h / 2.0 - dh) / r, 0, orig_h)
    boxes_all = np.stack([x1, y1, x2, y2], axis=1)

    detections = []
    classes = [("left_foot", s_left, 0), ("right_foot", s_right, 1)]

    for name, scores, cls_id in classes:
        mask = scores >= conf_thresh
        if not np.any(mask):
            continue

        idxs = np.where(mask)[0]
        cls_boxes = boxes_all[idxs]
        cls_scores = scores[idxs]

        keep = nms_boxes(cls_boxes, cls_scores, iou_thresh=iou_thresh)
        if len(keep) > 0:
            # Keep top detection for this foot side
            best_idx = idxs[keep[0]]
            kpts = []
            for k in range(4):
                kx = (raw[6 + k * 3, best_idx] - dw) / r
                ky = (raw[7 + k * 3, best_idx] - dh) / r
                kv = float(raw[8 + k * 3, best_idx])
                kpts.append((float(kx), float(ky), kv))

            detections.append({
                "class_name": name,
                "cls_id": cls_id,
                "confidence": float(scores[best_idx]),
                "box": [float(v) for v in cls_boxes[keep[0]]],
                "kpts": kpts
            })

    return detections


def render_overlay(frame, detections, model_info, frame_idx, total_frames, latency_ms, fps):
    """Renders professional AR-style bounding boxes, keypoints, and HUD overlay."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Colors
    c_left = (0, 230, 70)      # Neon Green for Left Foot
    c_right = (0, 160, 255)    # Bright Orange/Amber for Right Foot
    c_kp = [
        (0, 0, 255),    # 0: toe_tip (Red)
        (255, 50, 50),  # 1: heel_back (Blue)
        (0, 240, 255),  # 2: ball_medial (Yellow)
        (255, 0, 255),  # 3: ball_lateral (Magenta)
    ]
    kp_names = ["Toe Tip", "Heel Back", "Ball Medial", "Ball Lateral"]

    # Draw Detections
    for det in detections:
        color = c_left if det["class_name"] == "left_foot" else c_right
        x1, y1, x2, y2 = [int(v) for v in det["box"]]

        # Bounding box with corner accents
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

        # Corner corner brackets
        corner_len = min(30, (x2 - x1) // 4, (y2 - y1) // 4)
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), (255, 255, 255), 4)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), (255, 255, 255), 4)
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), (255, 255, 255), 4)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), (255, 255, 255), 4)
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), (255, 255, 255), 4)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), (255, 255, 255), 4)
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), (255, 255, 255), 4)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), (255, 255, 255), 4)

        # Box label tag
        tag = f"{det['class_name'].replace('_', ' ').title()}: {det['confidence']*100:.1f}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 12)), (x1 + tw + 16, y1), color, -1)
        cv2.putText(annotated, tag, (x1 + 8, max(th + 4, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        # Skeleton connections: Toe(0) -> Ball Medial(2) -> Heel(1) -> Ball Lateral(3) -> Toe(0)
        kpts = det["kpts"]
        skeleton_pairs = [(0, 2), (2, 1), (1, 3), (3, 0), (2, 3)]
        for p1_i, p2_i in skeleton_pairs:
            p1 = (int(kpts[p1_i][0]), int(kpts[p1_i][1]))
            p2 = (int(kpts[p2_i][0]), int(kpts[p2_i][1]))
            if kpts[p1_i][2] > 0.2 and kpts[p2_i][2] > 0.2:
                cv2.line(annotated, p1, p2, (220, 220, 220), 2, cv2.LINE_AA)

        # Keypoint circles
        for k_idx, (kx, ky, kv) in enumerate(kpts):
            if kv > 0.2:
                pt = (int(kx), int(ky))
                cv2.circle(annotated, pt, 8, c_kp[k_idx], -1)
                cv2.circle(annotated, pt, 10, (255, 255, 255), 2)

    # Top HUD Bar (Semi-transparent dark header)
    hud_h = 130
    sub_img = annotated[0:hud_h, 0:w]
    dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    res = cv2.addWeighted(sub_img, 0.25, dark_rect, 0.75, 1.0)
    annotated[0:hud_h, 0:w] = res
    cv2.line(annotated, (0, hud_h), (w, hud_h), (0, 200, 255), 2)

    # Header Text
    title = f"{model_info['display_name']} ({model_info['precision'].upper()})"
    cv2.putText(annotated, title, (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 240, 255), 3, cv2.LINE_AA)

    # Sub-header details: Model footprint, Latency, FPS, Progress
    sub_text = (
        f"Size: {model_info['size_mb']:.2f} MB ({'<=5MB Compliant' if model_info['budget_ok'] else 'Full Accuracy'}) | "
        f"Latency: {latency_ms:.1f} ms ({fps:.1f} FPS) | "
        f"Frame: {frame_idx + 1}/{total_frames}"
    )
    cv2.putText(annotated, sub_text, (25, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)

    # Detection status indicator
    l_det = next((d for d in detections if d["class_name"] == "left_foot"), None)
    r_det = next((d for d in detections if d["class_name"] == "right_foot"), None)

    l_str = f"LEFT: {l_det['confidence']*100:.1f}%" if l_det else "LEFT: NONE"
    r_str = f"RIGHT: {r_det['confidence']*100:.1f}%" if r_det else "RIGHT: NONE"

    cv2.putText(annotated, l_str, (25, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 70) if l_det else (120, 120, 120), 2, cv2.LINE_AA)
    cv2.putText(annotated, r_str, (300, 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 160, 255) if r_det else (120, 120, 120), 2, cv2.LINE_AA)

    # Legend for keypoints at bottom right
    leg_x = w - 380
    leg_y = h - 140
    leg_w = 360
    leg_h = 120
    sub_leg = annotated[leg_y:leg_y+leg_h, leg_x:leg_x+leg_w]
    dark_leg = np.zeros(sub_leg.shape, dtype=np.uint8)
    annotated[leg_y:leg_y+leg_h, leg_x:leg_x+leg_w] = cv2.addWeighted(sub_leg, 0.3, dark_leg, 0.7, 1.0)
    cv2.rectangle(annotated, (leg_x, leg_y), (leg_x+leg_w, leg_y+leg_h), (100, 100, 100), 1)

    cv2.putText(annotated, "Path A Coarse Keypoints:", (leg_x + 15, leg_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for k_i, (k_name, k_col) in enumerate(zip(kp_names, c_kp)):
        col_x = leg_x + 15 if k_i < 2 else leg_x + 190
        row_y = leg_y + 55 if (k_i % 2 == 0) else leg_y + 90
        cv2.circle(annotated, (col_x + 8, row_y - 5), 6, k_col, -1)
        cv2.putText(annotated, k_name, (col_x + 22, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    return annotated


def process_video_for_model(model_cfg, video_path, output_path, max_frames=None, device="cuda"):
    """Runs full video inference, benchmark, and rendering for one model configuration."""
    model_path = model_cfg["path"]
    model_name = model_cfg["display_name"]
    precision = model_cfg["precision"]

    print(f"\n==================================================================")
    print(f"  Processing: {model_name} ({precision.upper()})")
    print(f"  Model File: {model_path} ({os.path.getsize(model_path)/1e6:.2f} MB)")
    print(f"  Output Video: {output_path}")
    print(f"==================================================================")

    # Select execution provider
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"  [!] Fallback to CPUExecutionProvider: {e}")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total_frames = min(total_frames, max_frames)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output_path, fourcc, orig_fps, (orig_w, orig_h))

    latencies = []
    frames_with_detections = 0
    frames_with_both_feet = 0
    left_scores = []
    right_scores = []

    t_start = time.time()

    for f_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess: letterbox to 320x320
        img, r, (dw, dh) = letterbox(frame, new_shape=(320, 320))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        # Inference with timing
        t0 = time.perf_counter()
        raw_out = session.run(None, {input_name: blob})[0]
        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        latencies.append(dt_ms)
        inst_fps = 1000.0 / max(dt_ms, 1e-4)

        # Independent per-class decoding
        conf_thresh = model_cfg.get("conf_thresh", 0.25)
        detections = decode_18ch(raw_out, r, dw, dh, orig_w, orig_h, conf_thresh=conf_thresh, iou_thresh=0.45)

        has_left = any(d["class_name"] == "left_foot" for d in detections)
        has_right = any(d["class_name"] == "right_foot" for d in detections)

        if has_left or has_right:
            frames_with_detections += 1
        if has_left and has_right:
            frames_with_both_feet += 1

        for d in detections:
            if d["class_name"] == "left_foot":
                left_scores.append(d["confidence"])
            elif d["class_name"] == "right_foot":
                right_scores.append(d["confidence"])

        # Render AR HUD & skeleton overlay
        annotated = render_overlay(frame, detections, model_cfg, f_idx, total_frames, dt_ms, inst_fps)
        out_writer.write(annotated)

        if (f_idx + 1) % 50 == 0 or (f_idx + 1) == total_frames:
            print(f"  Frame {f_idx + 1:3d}/{total_frames} | Latency: {np.mean(latencies[-50:]):.1f} ms | "
                  f"Feet detected: Left={'YES' if has_left else ' NO'} Right={'YES' if has_right else ' NO'}")

    cap.release()
    out_writer.release()

    total_time = time.time() - t_start
    mean_lat = np.mean(latencies)
    mean_fps = 1000.0 / mean_lat
    det_rate_any = (frames_with_detections / total_frames) * 100.0
    det_rate_both = (frames_with_both_feet / total_frames) * 100.0

    print(f"\n--- Summary for {model_name} ({precision.upper()}) ---")
    print(f"  Total Video Duration: {total_frames / orig_fps:.2f} s ({total_frames} frames)")
    print(f"  Processing Time: {total_time:.2f} s ({total_frames / total_time:.1f} overall FPS)")
    print(f"  Inference Latency: {mean_lat:.2f} ms ({mean_fps:.1f} FPS)")
    print(f"  Frames with >=1 Foot: {det_rate_any:.1f}% ({frames_with_detections}/{total_frames})")
    print(f"  Frames with Both Feet: {det_rate_both:.1f}% ({frames_with_both_feet}/{total_frames})")
    if left_scores:
        print(f"  Mean Left Foot Conf: {np.mean(left_scores)*100:.1f}%")
    if right_scores:
        print(f"  Mean Right Foot Conf: {np.mean(right_scores)*100:.1f}%")
    print(f"  Video saved to: {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")

    return {
        "model_name": model_name,
        "precision": precision,
        "size_mb": model_cfg["size_mb"],
        "mean_latency_ms": round(float(mean_lat), 2),
        "mean_fps": round(float(mean_fps), 1),
        "det_rate_any_pct": round(float(det_rate_any), 1),
        "det_rate_both_pct": round(float(det_rate_both), 1),
        "mean_left_conf": round(float(np.mean(left_scores)), 3) if left_scores else 0.0,
        "mean_right_conf": round(float(np.mean(right_scores)), 3) if right_scores else 0.0,
        "output_path": output_path
    }


def main():
    video_path = "data/test_video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: Video not found at {video_path}")
        sys.exit(1)

    out_dir = "data/test_video_results"
    os.makedirs(out_dir, exist_ok=True)

    # 6 Target Model Configurations
    models_to_test = [
        # Standard Models (Full Accuracy, ~13.2 MB)
        {
            "id": "standard_fp32",
            "display_name": "Stage A Standard 320",
            "precision": "fp32",
            "path": "shoes-vto-ai-v2/models/stage-a-320-fp32.onnx",
            "size_mb": 13.23,
            "budget_ok": False,
            "conf_thresh": 0.30,
            "out_filename": "stage_a_320_standard_fp32.mp4"
        },
        {
            "id": "standard_fp16",
            "display_name": "Stage A Standard 320",
            "precision": "fp16",
            "path": "shoes-vto-ai-v2/models/stage-a-320-fp16.onnx",
            "size_mb": 6.64,
            "budget_ok": False,
            "conf_thresh": 0.30,
            "out_filename": "stage_a_320_standard_fp16.mp4"
        },
        {
            "id": "standard_int8",
            "display_name": "Stage A Standard 320",
            "precision": "int8",
            "path": "shoes-vto-ai-v2/models/stage-a-320-int8.onnx",
            "size_mb": 3.55,
            "budget_ok": False,
            "conf_thresh": 0.30,
            "out_filename": "stage_a_320_standard_int8.mp4"
        },
        # Compact Models (Mobile Budget Compliant, <5 MB)
        {
            "id": "compact_fp32",
            "display_name": "Stage A Compact 320",
            "precision": "fp32",
            "path": "shoes-vto-ai-v2/models/stage-a-320-compact-fp32.onnx",
            "size_mb": 4.93,
            "budget_ok": True,
            "conf_thresh": 0.25,
            "out_filename": "stage_a_320_compact_fp32.mp4"
        },
        {
            "id": "compact_fp16",
            "display_name": "Stage A Compact 320",
            "precision": "fp16",
            "path": "shoes-vto-ai-v2/models/stage-a-320-compact-fp16.onnx",
            "size_mb": 2.49,
            "budget_ok": True,
            "conf_thresh": 0.25,
            "out_filename": "stage_a_320_compact_fp16.mp4"
        },
        {
            "id": "compact_int8",
            "display_name": "Stage A Compact 320",
            "precision": "int8",
            "path": "shoes-vto-ai-v2/models/stage-a-320-compact-int8.onnx",
            "size_mb": 1.46,
            "budget_ok": True,
            "conf_thresh": 0.25,
            "out_filename": "stage_a_320_compact_int8.mp4"
        }
    ]

    all_results = []

    for cfg in models_to_test:
        out_path = os.path.join(out_dir, cfg["out_filename"])
        res = process_video_for_model(cfg, video_path, out_path, device="cuda")
        all_results.append(res)

    # Save summary report
    import json
    summary_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("                      ALL 6 VIDEOS GENERATED SUCCESSFULLY                     ")
    print("="*80)
    print(f"{'Model':<24} | {'Prec':<5} | {'Size (MB)':<9} | {'Latency':<9} | {'FPS':<6} | {'Det Rate':<8} | {'Output Video'}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['model_name']:<24} | {r['precision'].upper():<5} | {r['size_mb']:>9.2f} | {r['mean_latency_ms']:>6.2f} ms | {r['mean_fps']:>6.1f} | {r['det_rate_any_pct']:>7.1f}% | {os.path.basename(r['output_path'])}")
    print("="*80)
    print(f"Summary JSON saved to: {summary_path}")


if __name__ == "__main__":
    main()

