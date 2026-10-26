"""
rtmpose/eval_video.py
─────────────────────
Evaluates the trained RTMPose-13KP Foot Pose model on video benchmarks.
Pipeline:
  1. Foot Bounding Box Detection (YOLO Stage A detector or tracker)
  2. Crop foot patch (with margin) -> resize to 256x256
  3. RTMPose-13KP SimCC inference (continuous sub-pixel expectation)
  4. Coordinate re-mapping to video frame
  5. One-Euro temporal filtering + 13-KP anatomical skeleton HUD rendering
  6. Output rendered MP4 video
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP
from src.models.qualcomm_experiments.rtmpose.decoder import decode_simcc
from ultralytics import YOLO

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


def draw_skeleton(frame, kpts, is_left=True):
    colors = COLOR_LEFT if is_left else COLOR_RIGHT
    # Draw bones
    for p1, p2 in SKELETON_PAIRS:
        x1, y1, v1 = kpts[p1]
        x2, y2, v2 = kpts[p2]
        if v1 > 0.20 and v2 > 0.20:
            alpha = min(v1, v2)
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors["skeleton"], 2, cv2.LINE_AA)

    # Draw joints
    for i in range(13):
        x, y, v = kpts[i]
        if v >= 0.35:
            # High confidence: solid circle
            cv2.circle(frame, (int(x), int(y)), 4, colors["kpt_vis"], -1, cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 0), 1, cv2.LINE_AA)
        elif v >= 0.15:
            # Inferred / occluded: hollow ring
            cv2.circle(frame, (int(x), int(y)), 4, colors["kpt_occ"], 1, cv2.LINE_AA)


def run_video_evaluation(
    rtmpose_weights: str,
    detector_weights: str,
    video_path: str,
    output_path: str,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    conf_thresh: float = 0.30
):
    device = torch.device(device_name)
    print("=" * 70)
    print("  QUALCOMM RTMPOSE-13KP VIDEO EVALUATION & VISUALIZATION")
    print(f"  RTMPose Weights:   {rtmpose_weights}")
    print(f"  Detector Weights:  {detector_weights}")
    print(f"  Input Video:       {video_path}")
    print(f"  Output Video:      {output_path}")
    print(f"  Device:            {device}")
    print("=" * 70)

    # 1. Load RTMPose Model
    try:
        ckpt = torch.load(rtmpose_weights, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(rtmpose_weights, map_location=device)
    backbone_name = ckpt.get("backbone_name", "cspnext")
    rtmpose = RTMPose13KP(num_keypoints=13, input_size=(256, 256), simcc_split_ratio=2.0, backbone_name=backbone_name).to(device)
    rtmpose.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    rtmpose.eval()
    print(f"[+] Loaded RTMPose-13KP Checkpoint ({backbone_name}) successfully.")

    # 2. Load Detector
    detector = YOLO(detector_weights)
    print("[+] Loaded Foot Bounding Box Detector successfully.")

    # 3. Open Video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Video Specs: {width}x{height} @ {fps:.1f} FPS, {total_frames} frames.")

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    t_start = time.time()
    latencies = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_frame_start = time.perf_counter()

        # Step A: Detect feet using detector
        det_res = detector(frame, imgsz=256, conf=conf_thresh, verbose=False, device=device)[0]

        detected_feet = []
        if det_res.boxes is not None and len(det_res.boxes) > 0:
            for b in det_res.boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                cls_id = int(b.cls[0].item())
                score = float(b.conf[0].item())
                detected_feet.append({
                    "bbox": xyxy,
                    "class_id": cls_id,
                    "score": score
                })

        # Step B: For each foot, crop with unified affine warp matching training geometry
        for foot in detected_feet:
            x1, y1, x2, y2 = foot["bbox"]
            bw = x2 - x1
            bh = y2 - y1
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Adaptive padding: Elongated feet get higher margin
            aspect_ratio = max(bw / max(1.0, bh), bh / max(1.0, bw))
            size = max(bw, bh) * (1.0 + 0.20 + (0.15 if aspect_ratio > 1.3 else 0.05))

            # Affine warp to 256x256 with constant (114, 114, 114) border
            M = cv2.getRotationMatrix2D((cx, cy), 0.0, 256.0 / size)
            M[0, 2] += 128.0 - cx
            M[1, 2] += 128.0 - cy

            crop = cv2.warpAffine(
                frame, M, (256, 256),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114)
            )

            crop_t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                preds = rtmpose(crop_t)
                simcc_kps = decode_simcc(preds, simcc_split_ratio=2.0, win=12)[0]  # [13, 3]

            # Invert affine warp M to project keypoints back to full video frame coordinates
            M_inv = cv2.invertAffineTransform(M)
            kps_homo = np.stack([simcc_kps[:, 0], simcc_kps[:, 1], np.ones(13)], axis=1)
            orig_xy = kps_homo @ M_inv.T
            frame_kps = simcc_kps.copy()
            frame_kps[:, 0] = orig_xy[:, 0]
            frame_kps[:, 1] = orig_xy[:, 1]
            frame_kps[:, 0] = (frame_kps[:, 0] / 256.0) * cw + cx1
            frame_kps[:, 1] = (frame_kps[:, 1] / 256.0) * ch + cy1

            is_left = (foot["class_id"] == 0)
            colors = COLOR_LEFT if is_left else COLOR_RIGHT

            # Draw box & label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors["bbox"], 2)
            lbl = f"{colors['label']} {foot['score']:.2f} (RTMPose SimCC)"
            cv2.putText(frame, lbl, (int(x1), max(20, int(y1) - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 2)

            # Draw 13-KP anatomical skeleton
            draw_skeleton(frame, frame_kps, is_left=is_left)

        elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0
        latencies.append(elapsed_ms)

        # Draw Telemetry HUD
        rolling_fps = 1000.0 / max(1.0, np.mean(latencies[-30:]))
        cv2.rectangle(frame, (10, 10), (380, 85), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (380, 85), (0, 255, 255), 1)
        cv2.putText(frame, "SHOES VTO: QUALCOMM RTMPOSE-13KP", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | Latency: {elapsed_ms:.1f}ms", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, f"Rolling FPS: {rolling_fps:.1f} | Active Feet: {len(detected_feet)}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        writer.write(frame)

        if frame_idx % 100 == 0 or frame_idx == total_frames:
            print(f"[*] Processed {frame_idx}/{total_frames} frames ({frame_idx/total_frames*100:.1f}%) | Latency: {elapsed_ms:.1f}ms ({rolling_fps:.1f} FPS)")

    cap.release()
    writer.release()
    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE")
    print(f"  Output saved to:    {output_path} ({Path(output_path).stat().st_size / (1024*1024):.2f} MB)")
    print(f"  Total Time:         {total_time:.1f}s ({frame_idx / total_time:.1f} FPS)")
    print(f"  Average Latency:    {np.mean(latencies):.2f} ms")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_weights = "checkpoints/rtmpose_13kp_official_coco/best.pt"
    if not Path(default_weights).exists() and Path("checkpoints/rtmpose_13kp/best.pt").exists():
        default_weights = "checkpoints/rtmpose_13kp/best.pt"
    parser.add_argument("--rtmpose", type=str, default=default_weights)
    parser.add_argument("--detector", type=str, default="deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.pt")
    parser.add_argument("--video", type=str, default="src/models/qualcomm_experiments/tests/test_video.mp4")
    parser.add_argument("--output", type=str, default="deliverables/qualcomm_experiments/rtmpose_eval_demo.mp4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()

    run_video_evaluation(
        rtmpose_weights=args.rtmpose,
        detector_weights=args.detector,
        video_path=args.video,
        output_path=args.output,
        device_name=args.device,
        conf_thresh=args.conf
    )

