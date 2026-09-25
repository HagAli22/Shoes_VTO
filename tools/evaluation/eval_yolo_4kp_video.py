"""
eval_yolo_4kp_video.py
──────────────────────
Runs video inference using the trained 4-Keypoint YOLOv8n-Pose detector
(either .pt or .onnx) and produces an annotated output video with:
  - Foot Bounding Boxes (Left: Cyan, Right: Orange)
  - 4 Coarse Keypoint Markers (0: Toe, 1: Heel, 2: Ball Medial, 3: Ball Lateral)
  - Connected Foot Perimeter Skeleton
  - Real-time Latency / FPS Overlay HUD

Usage:
  python tools/evaluation/eval_yolo_4kp_video.py --video data/test_video.mp4 --model outputs/stage1_4kp/finetune_shuffled_v3/weights/best.pt --output outputs/stage1_4kp/annotated_video.mp4
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

KP_NAMES = ["0: Toe", "1: Heel", "2: Medial", "3: Lateral"]
KP_COLORS = [
    (0, 0, 255),    # 0: Toe (Red in BGR)
    (255, 0, 0),    # 1: Heel (Blue in BGR)
    (0, 255, 0),    # 2: Ball Medial (Green in BGR)
    (0, 165, 255),  # 3: Ball Lateral (Orange in BGR)
]
SKELETON_LINKS = [
    (0, 2),  # Toe -> Medial
    (2, 1),  # Medial -> Heel
    (1, 3),  # Heel -> Lateral
    (3, 0),  # Lateral -> Toe
    (2, 3),  # Medial <-> Lateral (Ball width)
]
CLASS_COLORS = {
    0: (255, 200, 0),  # Cyan for left_foot
    1: (0, 140, 255)   # Orange for right_foot
}
CLASS_NAMES = {0: "Left Foot", 1: "Right Foot"}


def process_video(video_path: str, model_path: str, output_path: str, conf_thresh: float = 0.25, device: str = "0"):
    print("=" * 70)
    print("  4-KP YOLOv8 Video Evaluation & HUD Overlay")
    print(f"  Video Input : {video_path}")
    print(f"  Model       : {model_path}")
    print(f"  Video Output: {output_path}")
    print("=" * 70)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model = YOLO(model_path)
    frame_idx = 0
    t_start = time.time()
    latencies = []

    print(f"Processing {total_frames} frames ({width}x{height} @ {fps:.1f} FPS)...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t0 = time.time()

        # Run inference
        results = model.predict(frame, imgsz=320, conf=conf_thresh, device=device, verbose=False)[0]
        infer_ms = (time.time() - t0) * 1000.0
        latencies.append(infer_ms)

        annotated = frame.copy()

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            kpts = results.keypoints.data.cpu().numpy() if results.keypoints is not None else []

            for i in range(len(boxes)):
                box = boxes[i].astype(int)
                cls_id = clss[i]
                conf = confs[i]
                color = CLASS_COLORS.get(cls_id, (0, 255, 0))
                cls_name = CLASS_NAMES.get(cls_id, f"Foot {cls_id}")

                # Draw Bounding Box
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 2)
                lbl = f"{cls_name} {conf:.2f}"
                cv2.putText(annotated, lbl, (box[0], max(20, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw Keypoints and Skeleton
                if len(kpts) > i:
                    inst_kpts = kpts[i]  # shape: [4, 3] or [4, 2]
                    pts_dict = {}

                    for k in range(min(4, len(inst_kpts))):
                        kx, ky = inst_kpts[k][0], inst_kpts[k][1]
                        kv = inst_kpts[k][2] if len(inst_kpts[k]) > 2 else 1.0

                        if kv > 0.3 and kx > 0 and ky > 0:
                            pts_dict[k] = (int(kx), int(ky))
                            cv2.circle(annotated, (int(kx), int(ky)), 5, KP_COLORS[k], -1)
                            cv2.circle(annotated, (int(kx), int(ky)), 7, (255, 255, 255), 1)
                            cv2.putText(annotated, KP_NAMES[k], (int(kx) + 6, int(ky) - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    # Skeleton lines
                    for link in SKELETON_LINKS:
                        if link[0] in pts_dict and link[1] in pts_dict:
                            cv2.line(annotated, pts_dict[link[0]], pts_dict[link[1]], (0, 255, 255), 2)

        # HUD Overlay
        avg_lat = np.mean(latencies[-30:]) if latencies else infer_ms
        fps_est = 1000.0 / avg_lat if avg_lat > 0 else 0
        hud_text = f"Frame: {frame_idx}/{total_frames} | Latency: {infer_ms:.1f} ms | FPS: {fps_est:.1f} | Model: YOLOv8n-Pose (4-KP)"
        
        cv2.rectangle(annotated, (10, 10), (width - 10, 45), (0, 0, 0), -1)
        cv2.putText(annotated, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

        out.write(annotated)

        if frame_idx % 50 == 0:
            print(f"   Frame {frame_idx}/{total_frames} (Latency: {infer_ms:.1f} ms, {fps_est:.1f} FPS)")

    cap.release()
    out.release()

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print("  Video Processing Complete!")
    print(f"  Total Time    : {total_time:.1f}s (Avg {np.mean(latencies):.1f} ms/frame)")
    print(f"  Saved Output  : {output_path}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate 4-KP YOLO model on video.")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--model", type=str, required=True, help="Model weights path (.pt or .onnx)")
    parser.add_argument("--output", type=str, default="outputs/stage1_4kp/annotated_video.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    args = parser.parse_args()

    process_video(args.video, args.model, args.output, args.conf, args.device)


if __name__ == "__main__":
    main()
