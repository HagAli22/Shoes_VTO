"""
single_stage_rtmpose/eval_video.py
──────────────────────────────────
Single-Pass Video Evaluation for Single-Stage RTMPose (Apache-2.0).
Zero cropping stage: produces both Left/Right foot bounding boxes and
13 anatomical foot landmarks in a single forward pass per frame.
Applies 1€ temporal filtering on boxes and keypoints for jitter-free tracking.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.single_stage_rtmpose.model import (
    SingleStageRTMPose,
    postprocess_single_stage_pose
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
    (3, 5),   # ball_medial -> instep_top
    (4, 5),   # ball_lateral -> instep_top
    # Ankle bridge
    (11, 8),  # throat -> malleolus_medial
    (11, 9),  # throat -> malleolus_lateral
    (8, 12),  # malleolus_medial -> achilles
    (9, 12),  # malleolus_lateral -> achilles
    (1, 12),  # heel_back -> achilles
]

COLOR_LEFT = {
    "bbox": (255, 160, 0),       # Cyan / Sky Blue
    "kpt": (255, 220, 0),        # Bright Cyan
    "skeleton": (230, 115, 0),   # Deep Blue
    "label": "Left Foot"
}

COLOR_RIGHT = {
    "bbox": (0, 140, 255),       # Orange
    "kpt": (0, 255, 100),        # Lime Green
    "skeleton": (0, 180, 50),    # Emerald
    "label": "Right Foot"
}


class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.s: Optional[np.ndarray] = None

    def filter(self, x: np.ndarray, alpha: Optional[np.ndarray] = None) -> np.ndarray:
        a = self.alpha if alpha is None else alpha
        if self.s is None:
            self.s = np.array(x, dtype=np.float64).copy()
        else:
            self.s = a * np.array(x, dtype=np.float64) + (1.0 - a) * self.s
        return self.s


class OneEuroFilter:
    """1€ Filter (Casiez et al., CHI 2012) for jitter elimination with zero motion lag."""
    def __init__(
        self,
        t0: float,
        x0: np.ndarray,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.t_prev = float(t0)
        self.x_filt.filter(x0)
        self.dx_filt.filter(np.zeros_like(x0, dtype=np.float64))

    def _alpha(self, cutoff: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, t: float, x: np.ndarray) -> np.ndarray:
        dt = max(1e-4, float(t - self.t_prev))
        self.t_prev = float(t)

        x_curr = np.array(x, dtype=np.float64)
        dx = (x_curr - self.x_filt.s) / dt
        edx = self.dx_filt.filter(dx, self._alpha(self.d_cutoff, dt))

        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(cutoff, dt)
        return self.x_filt.filter(x_curr, alpha)


class FootTrackSingleStage:
    def __init__(self, track_id: int, t0: float, raw_box: List[float], kpts: np.ndarray, class_id: int):
        self.track_id = track_id
        self.class_id = class_id  # 0: left, 1: right
        self.box_filter = OneEuroFilter(t0, np.array(raw_box, dtype=np.float64), min_cutoff=1.5, beta=0.015)
        self.kpt_filter = OneEuroFilter(t0, kpts[:, :2], min_cutoff=0.8, beta=0.01)
        self.smoothed_box = tuple(raw_box)
        self.smoothed_kpts = kpts.copy()
        self.lost_count = 0

    def update(self, t: float, raw_box: List[float], kpts: np.ndarray):
        s_box = self.box_filter.filter(t, np.array(raw_box, dtype=np.float64))
        self.smoothed_box = tuple(float(v) for v in s_box)

        s_coords = self.kpt_filter.filter(t, kpts[:, :2])
        out_kpts = kpts.copy()
        out_kpts[:, :2] = s_coords
        self.smoothed_kpts = out_kpts
        self.lost_count = 0
        return self.smoothed_box, self.smoothed_kpts


def box_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, is_left: bool):
    colors = COLOR_LEFT if is_left else COLOR_RIGHT
    # Bones
    for idx1, idx2 in SKELETON_PAIRS:
        p1 = keypoints[idx1]
        p2 = keypoints[idx2]
        if p1[2] > 0.15 and p2[2] > 0.15:
            pt1 = (int(round(p1[0])), int(round(p1[1])))
            pt2 = (int(round(p2[0])), int(round(p2[1])))
            cv2.line(frame, pt1, pt2, colors["skeleton"], 2, cv2.LINE_AA)

    # Keypoint dots
    for i, kp in enumerate(keypoints):
        if kp[2] > 0.15:
            pt = (int(round(kp[0])), int(round(kp[1])))
            cv2.circle(frame, pt, 4, colors["kpt"], -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (0, 0, 0), 1, cv2.LINE_AA)


def run_single_stage_video_eval(
    model_weights: str,
    video_path: str,
    output_path: str,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    img_size: int = 320,
    conf_thresh: float = 0.30
):
    device = torch.device(device_name)
    print("=" * 70)
    print("  SINGLE-STAGE RTMPOSE VIDEO EVALUATION (ONE-PASS)")
    print(f"  Model Weights:     {model_weights}")
    print(f"  Input Video:       {video_path}")
    print(f"  Output Video:      {output_path}")
    print(f"  Resolution:        {img_size}x{img_size}")
    print(f"  Device:            {device}")
    print("=" * 70)

    # Load model
    ckpt = torch.load(model_weights, map_location=device, weights_only=False)
    state = ckpt.get("ema_state", ckpt.get("model_state", ckpt))
    model = SingleStageRTMPose(
        num_classes=2,
        num_keypoints=13,
        input_size=(img_size, img_size),
        pretrained=False
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[+] Loaded SingleStageRTMPose model (missing: {len(missing)}, unexpected: {len(unexpected)}).")

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

    tracks: Dict[int, FootTrackSingleStage] = {}
    next_track_id = 1
    latencies = []
    frame_idx = 0
    t_start = time.time()

    scale = min(img_size / width, img_size / height)
    nw = int(round(width * scale))
    nh = int(round(height * scale))
    pad_x = (img_size - nw) // 2
    pad_y = (img_size - nh) // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_current = frame_idx / fps
        t_frame_start = time.perf_counter()

        # Preprocessing: letterbox resize
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(canvas_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        # Single forward pass: produces both boxes and 13 keypoints
        with torch.no_grad():
            out_tensor = model(inp, export_mode=True)[0]  # [45, N]
            preds = postprocess_single_stage_pose(out_tensor, conf_thresh=conf_thresh, iou_thresh=0.45)

        # Scale predictions back to original video frame
        detected_feet = []
        for p in preds:
            bx = p["bbox"]
            kps = p["keypoints"].copy()

            # Invert letterbox
            x1 = (bx[0] - pad_x) / scale
            y1 = (bx[1] - pad_y) / scale
            x2 = (bx[2] - pad_x) / scale
            y2 = (bx[3] - pad_y) / scale
            orig_box = [x1, y1, x2, y2]

            kps[:, 0] = (kps[:, 0] - pad_x) / scale
            kps[:, 1] = (kps[:, 1] - pad_y) / scale

            detected_feet.append({
                "bbox": orig_box,
                "class_id": p["class_id"],
                "score": p["score"],
                "keypoints": kps
            })

        # Track association
        active_tids = list(tracks.keys())
        matched_tracks = set()
        matched_dets = set()

        if active_tids and detected_feet:
            iou_mat = np.zeros((len(active_tids), len(detected_feet)), dtype=np.float32)
            for r, tid in enumerate(active_tids):
                for c, det in enumerate(detected_feet):
                    iou_mat[r, c] = box_iou(tracks[tid].smoothed_box, det["bbox"])

            while True:
                max_val = iou_mat.max()
                if max_val < 0.25:
                    break
                r, c = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
                tid = active_tids[r]
                matched_tracks.add(tid)
                matched_dets.add(c)

                det = detected_feet[c]
                s_box, s_kps = tracks[tid].update(t_current, det["bbox"], det["keypoints"])
                is_left = (det["class_id"] == 0)
                colors = COLOR_LEFT if is_left else COLOR_RIGHT

                # Render stabilized tight box & 13 keypoints
                x1_d, y1_d, x2_d, y2_d = map(int, map(round, s_box))
                cv2.rectangle(frame, (x1_d, y1_d), (x2_d, y2_d), colors["bbox"], 2)
                lbl = f"ID:{tid} {colors['label']} ({det['score']:.2f})"
                cv2.putText(frame, lbl, (x1_d, max(20, y1_d - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 2)
                draw_skeleton(frame, s_kps, is_left=is_left)

                iou_mat[r, :] = -1.0
                iou_mat[:, c] = -1.0

        # Spawn new tracks
        for c, det in enumerate(detected_feet):
            if c in matched_dets:
                continue
            tid = next_track_id
            next_track_id += 1
            tracks[tid] = FootTrackSingleStage(tid, t_current, det["bbox"], det["keypoints"], det["class_id"])
            is_left = (det["class_id"] == 0)
            colors = COLOR_LEFT if is_left else COLOR_RIGHT

            x1_d, y1_d, x2_d, y2_d = map(int, map(round, det["bbox"]))
            cv2.rectangle(frame, (x1_d, y1_d), (x2_d, y2_d), colors["bbox"], 2)
            lbl = f"ID:{tid} {colors['label']} (New: {det['score']:.2f})"
            cv2.putText(frame, lbl, (x1_d, max(20, y1_d - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 2)
            draw_skeleton(frame, det["keypoints"], is_left=is_left)

        # Cleanup lost tracks
        for tid in list(tracks.keys()):
            if tid not in matched_tracks and tid < next_track_id - len(detected_feet):
                tracks[tid].lost_count += 1
                if tracks[tid].lost_count > 6:
                    del tracks[tid]

        elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0
        latencies.append(elapsed_ms)
        rolling_fps = 1000.0 / max(1.0, np.mean(latencies[-30:]))

        # Telemetry HUD
        cv2.rectangle(frame, (10, 10), (480, 95), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (480, 95), (0, 255, 255), 1)
        cv2.putText(frame, "SHOES VTO: SINGLE-STAGE RTMPOSE (ONE-PASS)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        cv2.putText(frame, "Pipeline: Full-Image CSPNeXt-PAFPN + 1-Euro Filter", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | Latency: {elapsed_ms:.1f}ms ({rolling_fps:.1f} FPS)", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.putText(frame, f"Active Foot Tracks: {len(tracks)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1)

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
    parser.add_argument("--weights", type=str, default="checkpoints/single_stage_rtmpose_320/best.pt")
    parser.add_argument("--video", type=str, default="data/test_video.mp4")
    parser.add_argument("--output", type=str, default="deliverables/qualcomm_experiments/single_stage_rtmpose_demo.mp4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()

    run_single_stage_video_eval(
        model_weights=args.weights,
        video_path=args.video,
        output_path=args.output,
        device_name=args.device,
        conf_thresh=args.conf
    )

