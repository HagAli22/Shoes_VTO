"""
eval_blazefoot_video.py
───────────────────────
Comprehensive Video Inference, Benchmark, and Visualizer for Google BlazeFoot (4-KP).

Features:
  - Dual Engine: Runs on PyTorch (.pt) or ONNX (.onnx) via GPU/CPU.
  - Draws 4 Anatomical Keypoints with color-coded labels:
      Index 0: toe_tip (Distal toe apex)
      Index 1: heel_back (Posterior calcaneus)
      Index 2: ball_medial (1st metatarsal head)
      Index 3: ball_lateral (5th metatarsal head)
  - Draws Anatomical Quad Skeleton connecting the 4 foot boundaries.
  - Real-time Telemetry HUD: Live FPS, Latency (ms), and Detection Confidence.
  - Outputs full annotated MP4 video with benchmark report.

Usage:
    python tools/evaluation/eval_blazefoot_video.py \
      --video data/test_video.mp4 \
      --model outputs/stage_a/blazefoot_4kp_colab/weights/best.pt \
      --output outputs/stage_a/blazefoot_annotated_video.mp4
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 4 Coarse Keypoint Names & Colors (BGR)
KP_NAMES = ["toe_tip", "heel_back", "ball_medial", "ball_lateral"]
KP_COLORS = [
    (255, 255, 0),   # 0: toe_tip (Cyan)
    (0, 0, 255),     # 1: heel_back (Red)
    (0, 255, 255),   # 2: ball_medial (Yellow)
    (0, 255, 0),     # 3: ball_lateral (Green)
]

# 4-KP Skeleton Polygon Connections
SKELETON_4KP = [
    (0, 2), # toe_tip -> ball_medial
    (0, 3), # toe_tip -> ball_lateral
    (2, 3), # ball_medial -> ball_lateral (Metatarsal arch)
    (2, 1), # ball_medial -> heel_back
    (3, 1), # ball_lateral -> heel_back
]

COLOR_LEFT = {"box": (0, 230, 70), "name": "Left Foot"}
COLOR_RIGHT = {"box": (0, 160, 255), "name": "Right Foot"}


def letterbox(img, size=320, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size, 3), color[0], dtype=np.uint8)
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
    return canvas, r, pad_x, pad_y


class BlazeFootVideoEvaluator:
    def __init__(self, model_path: str, device_str: str = "0", img_size: int = 320):
        self.model_path = model_path
        self.img_size = img_size
        self.is_onnx = model_path.lower().endswith(".onnx")

        if self.is_onnx:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device_str != "cpu" and torch.cuda.is_available() else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.active_provider = self.session.get_providers()[0]
            self.input_name = self.session.get_inputs()[0].name
            self.is_fp16 = "float16" in self.session.get_inputs()[0].type
            print(f"  [Engine] Loaded ONNX session on: {self.active_provider}")
        else:
            from src.models.blazefoot.blazefoot_detector import BlazeFoot, BlazeFootPathAExport
            self.device = torch.device(f"cuda:{device_str}" if torch.cuda.is_available() and device_str != "cpu" else "cpu")
            checkpoint = torch.load(model_path, map_location=self.device)
            base_model = BlazeFoot(num_classes=2, num_keypoints=4).to(self.device)
            sd = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            base_model.load_state_dict(sd)
            base_model.eval()
            self.export_wrapper = BlazeFootPathAExport(base_model).to(self.device)
            self.export_wrapper.eval()
            print(f"  [Engine] Loaded PyTorch model on: {self.device}")

    def predict(self, frame_bgr, conf_thresh=0.25):
        orig_h, orig_w = frame_bgr.shape[:2]
        lb_img, r, pad_x, pad_y = letterbox(frame_bgr, size=self.img_size)
        lb_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.is_onnx:
            if self.is_fp16:
                lb_rgb = lb_rgb.astype(np.float16)
            tensor_in = np.transpose(lb_rgb, (2, 0, 1))[np.newaxis]
            raw_out = self.session.run(None, {self.input_name: tensor_in})[0]
            if raw_out.ndim == 3:
                raw = raw_out[0]
        else:
            tensor_in = torch.from_numpy(lb_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                raw_tensor = self.export_wrapper(tensor_in)
                raw = raw_tensor[0].cpu().numpy()

        # Output shape: [18, 1050]
        # Rows 0..3: cx, cy, w, h
        # Rows 4..5: score_left, score_right
        # Rows 6..17: 4 Keypoints * 3 (kx, ky, kv)
        cx = raw[0] * float(self.img_size)
        cy = raw[1] * float(self.img_size)
        bw = raw[2] * float(self.img_size)
        bh = raw[3] * float(self.img_size)
        scores_left  = raw[4]
        scores_right = raw[5]

        x1_all = np.clip(((cx - bw / 2.0) - pad_x) / r, 0, orig_w)
        y1_all = np.clip(((cy - bh / 2.0) - pad_y) / r, 0, orig_h)
        x2_all = np.clip(((cx + bw / 2.0) - pad_x) / r, 0, orig_w)
        y2_all = np.clip(((cy + bh / 2.0) - pad_y) / r, 0, orig_h)

        detections = []
        for cls_name, scores, cls_id in [("left_foot", scores_left, 0), ("right_foot", scores_right, 1)]:
            best_anchor_idx = int(np.argmax(scores))
            best_score = float(scores[best_anchor_idx])

            if best_score >= conf_thresh:
                kpts = []
                for k in range(4):
                    kx_px = ((raw[6 + k * 3, best_anchor_idx] * float(self.img_size)) - pad_x) / r
                    ky_px = ((raw[7 + k * 3, best_anchor_idx] * float(self.img_size)) - pad_y) / r
                    kv    = float(raw[8 + k * 3, best_anchor_idx])
                    kpts.append({
                        "name": KP_NAMES[k],
                        "x": float(kx_px),
                        "y": float(ky_px),
                        "conf": kv,
                        "color": KP_COLORS[k]
                    })

                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": best_score,
                    "bbox": [float(x1_all[best_anchor_idx]), float(y1_all[best_anchor_idx]),
                             float(x2_all[best_anchor_idx]), float(y2_all[best_anchor_idx])],
                    "keypoints": kpts
                })

        return detections


def render_visuals(frame, detections, fps=0.0, lat_ms=0.0, show_labels=True):
    annotated = frame.copy()
    h, w = frame.shape[:2]

    for det in detections:
        cls_id = det["class_id"]
        conf = det["confidence"]
        bx1, by1, bx2, by2 = [int(v) for v in det["bbox"]]
        kpts = det["keypoints"]
        scheme = COLOR_LEFT if cls_id == 0 else COLOR_RIGHT
        color_bgr = scheme["box"]

        # Draw semi-transparent BBox
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color_bgr, 2, cv2.LINE_AA)

        # Header Badge
        badge_text = f"{scheme['name']} {conf:.0%}"
        font = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(badge_text, font, 0.55, 1)
        cv2.rectangle(annotated, (bx1, max(0, by1 - th - 8)), (bx1 + tw + 10, by1), color_bgr, -1)
        cv2.putText(annotated, badge_text, (bx1 + 5, max(th + 2, by1 - 4)), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw 4-KP Skeleton
        for p1_i, p2_i in SKELETON_4KP:
            pt1 = (int(kpts[p1_i]["x"]), int(kpts[p1_i]["y"]))
            pt2 = (int(kpts[p2_i]["x"]), int(kpts[p2_i]["y"]))
            cv2.line(annotated, pt1, pt2, (200, 200, 200), 2, cv2.LINE_AA)

        # Draw 4 Keypoint Circles & Labels
        for k_idx, kp in enumerate(kpts):
            px, py = int(kp["x"]), int(kp["y"])
            col = kp["color"]
            cv2.circle(annotated, (px, py), 6, col, -1, cv2.LINE_AA)
            cv2.circle(annotated, (px, py), 7, (0, 0, 0), 1, cv2.LINE_AA)

            if show_labels:
                cv2.putText(annotated, f"{kp['name']}", (px + 8, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # Telemetry HUD in top-left
    hud_bg = (20, 20, 20)
    cv2.rectangle(annotated, (12, 12), (320, 60), hud_bg, -1)
    cv2.rectangle(annotated, (12, 12), (320, 60), (100, 100, 100), 1)
    cv2.putText(annotated, f"Google BlazeFoot 4-KP | Latency: {lat_ms:.1f} ms", (20, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"Inference: {fps:.1f} FPS | Active Feet: {len(detections)}", (20, 50),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 255, 117), 1, cv2.LINE_AA)

    return annotated


def main():
    parser = argparse.ArgumentParser(description="Evaluate BlazeFoot 4-KP on Video.")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="outputs/stage_a/blazefoot_4kp_colab/weights/best.pt",
                        help="Path to PyTorch (.pt) or ONNX (.onnx) model")
    parser.add_argument("--output", default="outputs/stage_a/blazefoot_annotated_video.mp4",
                        help="Path to output annotated video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default="0", help="CUDA device ('0' or 'cpu')")
    parser.add_argument("--max_frames", type=int, default=0, help="Stop after N frames (0 = full)")
    parser.add_argument("--no_labels", action="store_true", help="Hide keypoint name labels")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video file not found: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print("=" * 70)
    print("  GOOGLE BLAZEFOOT (4-KP) VIDEO BENCHMARK & EVALUATION")
    print(f"  Input Video  : {args.video}")
    print(f"  Model Path   : {args.model}")
    print(f"  Output Video : {args.output}")
    print(f"  Confidence   : {args.conf} | Device: {args.device}")
    print("=" * 70)

    evaluator = BlazeFootVideoEvaluator(args.model, device_str=args.device)

    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (orig_w, orig_h))

    latencies = []
    fps_history = []
    frame_idx = 0
    t_start = time.perf_counter()

    pbar = tqdm(total=total_frames, desc="Processing Video")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (args.max_frames > 0 and frame_idx >= args.max_frames):
            break

        frame_idx += 1
        t0 = time.perf_counter()
        detections = evaluator.predict(frame, conf_thresh=args.conf)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat_ms)

        fps_val = 1000.0 / max(lat_ms, 0.001)
        fps_history.append(fps_val)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = float(np.mean(fps_history))

        rendered = render_visuals(frame, detections, fps=avg_fps, lat_ms=lat_ms, show_labels=not args.no_labels)
        writer.write(rendered)
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    t_total = time.perf_counter() - t_start

    mean_latency = float(np.mean(latencies)) if len(latencies) > 0 else 0.0
    throughput_fps = frame_idx / max(t_total, 0.001)

    print("\n" + "=" * 70)
    print("  [SUCCESS] BlazeFoot Video Benchmark Completed!")
    print(f"  Processed Frames   : {frame_idx}/{total_frames}")
    print(f"  Average Latency    : {mean_latency:.2f} ms ({1000.0/max(mean_latency, 0.001):.1f} Inference FPS)")
    print(f"  Overall Throughput : {throughput_fps:.1f} FPS (including decode + video rendering)")
    print(f"  Saved Output Video : {args.output}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
