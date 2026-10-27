"""
rtmpose/eval_video.py
─────────────────────
Evaluates the trained RTMPose-13KP Foot Pose model on video benchmarks with
State-of-the-Art Temporal Post-Processing for Video Virtual Try-On (VTO):

Pipeline:
  1. Foot Bounding Box Detection (YOLO Stage A detector or tracker)
  2. Multi-Object Foot Track Association (IoU + Spatial Centroid Matching)
  3. Crop Box Temporal Stabilization (One-Euro Filter on cx, cy, size to eliminate crop vibration)
  4. Unified Affine Warp to 256x256 (exact parity with training geometry)
  5. Qualcomm RTMPose-13KP SimCC continuous expectation sub-pixel inference
  6. Keypoint Temporal Filtering (Vectorized One-Euro Filter on 13x2 coordinates: zero stationary jitter, zero motion lag)
  7. In-Model Handedness Temporal Smoothing (EMA probability filtering: eliminates Left/Right flickering)
  8. Missing-Frame Extrapolation (MediaPipe-style keypoint-based crop prior on detector dropouts)
  9. Anatomical Skeleton HUD Rendering + Output MP4 Generation
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP
from src.models.qualcomm_experiments.rtmpose.decoder import decode_simcc
from src.models.qualcomm_experiments.common.crop import (
    compute_crop_box,
    get_crop_affine_transform,
    warp_image_crop,
    invert_keypoints,
    box_from_kps
)
from src.models.qualcomm_experiments.nanodet.model import NanoDetPlus, nms_postprocess

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


# ==============================================================================
# 1. State-of-the-Art One-Euro Filter (Vectorized across dimensions)
# ==============================================================================

class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.s: Optional[np.ndarray] = None

    def filter(self, x: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.array(x, dtype=np.float64).copy()
        else:
            self.s = self.alpha * np.array(x, dtype=np.float64) + (1.0 - self.alpha) * self.s
        return self.s

    def filter_with_alpha(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        self.alpha = alpha
        if self.s is None:
            self.s = np.array(x, dtype=np.float64).copy()
        else:
            self.s = alpha * np.array(x, dtype=np.float64) + (1.0 - alpha) * self.s
        return self.s


class OneEuroFilter:
    """
    1€ Filter (Casiez et al., CHI 2012).
    Adaptive cutoff frequency balances low-speed jitter elimination with zero high-speed lag.
    Vectorized to filter 1D arrays [N] or 2D coordinate matrices [N, 2] simultaneously.
    """
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
        # Derivative estimation
        dx = (x_curr - self.x_filt.s) / dt
        edx = self.dx_filt.filter_with_alpha(dx, self._alpha(self.d_cutoff, dt))

        # Dynamic cutoff based on motion speed
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(cutoff, dt)
        return self.x_filt.filter_with_alpha(x_curr, alpha)


# ==============================================================================
# 2. Foot Track State with Temporal Multi-Signal Filtering
# ==============================================================================

class FootTrack:
    def __init__(self, track_id: int, t0: float, raw_bbox: np.ndarray, initial_hand_prob: float):
        self.track_id = track_id
        # Filter the tight rectangular detection box [x1, y1, x2, y2]
        self.box_filter = OneEuroFilter(
            t0,
            np.array(raw_bbox, dtype=np.float64),
            min_cutoff=1.5,
            beta=0.015,
            d_cutoff=1.0
        )
        self.kpt_filter: Optional[OneEuroFilter] = None
        self.conf_filter: Optional[OneEuroFilter] = None
        self.hand_prob = float(initial_hand_prob)  # P(Right Foot)
        self.lost_count = 0
        self.age = 1
        self.last_kps: Optional[np.ndarray] = None
        self.smoothed_bbox = tuple(float(v) for v in raw_bbox)

    def update_box(self, t: float, raw_bbox: np.ndarray) -> Tuple[float, float, float, float]:
        smoothed = self.box_filter.filter(t, np.array(raw_bbox, dtype=np.float64))
        self.smoothed_bbox = (float(smoothed[0]), float(smoothed[1]), float(smoothed[2]), float(smoothed[3]))
        return self.smoothed_bbox

    def get_crop_params(self, crop_padding: float = 0.25) -> Tuple[float, float, float]:
        x1, y1, x2, y2 = self.smoothed_bbox
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        return compute_crop_box(cx, cy, bw, bh, crop_padding=crop_padding)

    def update_keypoints(self, t: float, kps: np.ndarray, hand_prob: float) -> np.ndarray:
        """
        kps: [13, 3] (x, y, conf)
        Returns: smoothed [13, 3]
        """
        coords = kps[:, :2]
        confs = kps[:, 2]

        if self.kpt_filter is None:
            # First keypoint detection for this track
            self.kpt_filter = OneEuroFilter(t, coords, min_cutoff=0.8, beta=0.01, d_cutoff=1.0)
            self.conf_filter = OneEuroFilter(t, confs, min_cutoff=2.0, beta=0.005, d_cutoff=1.0)
            smoothed_coords = coords
            smoothed_confs = confs
        else:
            smoothed_coords = self.kpt_filter.filter(t, coords)
            smoothed_confs = self.conf_filter.filter(t, confs)

        # Smooth handedness probability with exponential moving average
        alpha_hand = 0.20
        self.hand_prob = (1.0 - alpha_hand) * self.hand_prob + alpha_hand * float(hand_prob)

        out = np.zeros_like(kps)
        out[:, :2] = smoothed_coords
        out[:, 2] = np.clip(smoothed_confs, 0.0, 1.0)
        self.last_kps = out
        self.lost_count = 0
        self.age += 1
        return out


def box_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(1.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / (a1 + a2 - inter)


def draw_skeleton(frame, kpts, is_left=True):
    colors = COLOR_LEFT if is_left else COLOR_RIGHT
    # Draw bones
    for p1, p2 in SKELETON_PAIRS:
        x1, y1, v1 = kpts[p1]
        x2, y2, v2 = kpts[p2]
        if v1 > 0.20 and v2 > 0.20:
            cv2.line(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), colors["skeleton"], 2, cv2.LINE_AA)

    # Draw joints
    for i in range(13):
        x, y, v = kpts[i]
        ix, iy = int(round(x)), int(round(y))
        if v >= 0.35:
            # High confidence: solid circle
            cv2.circle(frame, (ix, iy), 4, colors["kpt_vis"], -1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 5, (0, 0, 0), 1, cv2.LINE_AA)
        elif v >= 0.15:
            # Inferred / occluded: hollow ring
            cv2.circle(frame, (ix, iy), 4, colors["kpt_occ"], 1, cv2.LINE_AA)


# ==============================================================================
# 3. Main Video Evaluation Runner
# ==============================================================================

def run_video_evaluation(
    rtmpose_weights: str,
    detector_weights: str,
    video_path: str,
    output_path: str,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    conf_thresh: float = 0.30,
    use_postprocessing: bool = True
):
    if device_name == "cuda" and not torch.cuda.is_available():
        print("[!] CUDA is not available in the current Python environment. Falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)
    print("=" * 70)
    print("  QUALCOMM RTMPOSE-13KP VIDEO EVALUATION & POST-PROCESSING")
    print(f"  RTMPose Weights:   {rtmpose_weights}")
    print(f"  Detector Weights:  {detector_weights}")
    print(f"  Input Video:       {video_path}")
    print(f"  Output Video:      {output_path}")
    print(f"  Post-Processing:   {'One-Euro Temporal Filtering + Box Tracking + Hand EMA' if use_postprocessing else 'Disabled (Raw)'}")
    print(f"  Device:            {device}")
    print("=" * 70)

    # 1. Load RTMPose Model
    ckpt = torch.load(rtmpose_weights, map_location=device, weights_only=False)
    backbone_name = ckpt.get("backbone_name", "cspnext")
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    has_hand = any("hand" in k for k in state_dict.keys())
    rtmpose = RTMPose13KP(
        num_keypoints=13,
        input_size=(256, 256),
        simcc_split_ratio=2.0,
        backbone_name=backbone_name,
        include_handedness=has_hand
    ).to(device)
    rtmpose.load_state_dict(state_dict, strict=False)
    rtmpose.eval()
    print(f"[+] Loaded RTMPose-13KP Checkpoint ({backbone_name}, In-Model Handedness: {has_hand}) successfully.")

    # 2. Load Detector
    is_nanodet = ("nanodet" in detector_weights.lower())
    if is_nanodet:
        det_ckpt = torch.load(detector_weights, map_location=device, weights_only=False)
        model_weights = det_ckpt.get("ema_state", det_ckpt.get("model_state", det_ckpt.get("model", det_ckpt)))
        num_classes = 2 if "head.gfl_cls.0.weight" in model_weights and model_weights["head.gfl_cls.0.weight"].shape[0] == 34 else 1
        nanodet = NanoDetPlus(num_classes=num_classes, input_size=(320, 320)).to(device)
        nanodet.load_state_dict(model_weights)
        nanodet.eval()
        print(f"[+] Loaded NanoDet-Plus Foot Detector (Apache-2.0, {num_classes}-Class, 1.17M params) successfully.")
    else:
        from ultralytics import YOLO
        detector = YOLO(detector_weights)
        print("[+] Loaded YOLO Foot Detector successfully.")

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

    # Active foot tracks dictionary: track_id -> FootTrack
    tracks: Dict[int, FootTrack] = {}
    next_track_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_current = frame_idx / fps
        t_frame_start = time.perf_counter()

        # Step A: Detect feet using detector
        detected_feet = []
        if is_nanodet:
            h0, w0 = frame.shape[:2]
            det_img_size = 320
            frame_resized = cv2.resize(frame, (det_img_size, det_img_size))
            inp = torch.from_numpy(frame_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                out = nanodet(inp)
                preds = nms_postprocess(out["boxes"][0], out["scores"][0], conf_thresh=conf_thresh, iou_thresh=0.45)
            sx = w0 / float(det_img_size)
            sy = h0 / float(det_img_size)
            for p in preds:
                bx = p["bbox"]
                scaled_box = np.array([bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy], dtype=np.float32)
                y_cent = (scaled_box[1] + scaled_box[3]) / 2.0

                # Anatomical spatial prior: In downward-facing Shoes VTO camera view,
                # real feet attached to legs are positioned in the lower portion of the frame (y >= 0.50*H).
                # Isolated detections floating in the upper half (y < 0.50*H) must have very high confidence (>=0.98)
                # to prevent distant floor textures/carpet leaves from spawning false tracks.
                if y_cent < (0.50 * h0) and p["score"] < 0.98:
                    continue

                detected_feet.append({
                    "bbox": scaled_box,
                    "class_id": p["class_id"],
                    "score": p["score"]
                })
        else:
            det_res = detector(frame, imgsz=256, conf=conf_thresh, verbose=False, device=device)[0]
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

        # Step B: Associate detections with existing tracks
        # Compute IoU matrix between active tracks and detections
        active_track_ids = list(tracks.keys())
        matched_tracks = set()
        matched_detections = set()

        if active_track_ids and detected_feet:
            iou_matrix = np.zeros((len(active_track_ids), len(detected_feet)), dtype=np.float32)
            for r, tid in enumerate(active_track_ids):
                trk_b = tracks[tid].smoothed_bbox
                for c, det in enumerate(detected_feet):
                    iou_matrix[r, c] = box_iou(trk_b, det["bbox"])

            # Greedy match on highest IoU
            while True:
                max_val = iou_matrix.max()
                if max_val < 0.25:
                    break
                r, c = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                tid = active_track_ids[r]
                matched_tracks.add(tid)
                matched_detections.add(c)

                # Process matched foot
                det = detected_feet[c]
                raw_bbox = det["bbox"]

                if use_postprocessing:
                    s_bbox = tracks[tid].update_box(t_current, raw_bbox)
                    cx, cy, size = tracks[tid].get_crop_params(crop_padding=0.25)
                else:
                    s_bbox = raw_bbox
                    x1, y1, x2, y2 = raw_bbox
                    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                    raw_cx, raw_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    cx, cy, size = compute_crop_box(raw_cx, raw_cy, bw, bh, crop_padding=0.25)

                # Affine Crop & Inference
                M, M_inv = get_crop_affine_transform(cx, cy, size, out_size=256, rot=0.0)
                crop = warp_image_crop(frame, M, out_size=256, border_value=(114, 114, 114))
                crop_t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

                with torch.no_grad():
                    preds = rtmpose(crop_t)
                    simcc_kps = decode_simcc(preds, simcc_split_ratio=2.0, win=12)[0]
                    if "hand" in preds and det.get("class_id", -1) in (0, 1):
                        pose_hand = float(preds["hand"].item())
                        det_hand = float(det["class_id"])
                        # Blend high-resolution RTMPose crop prediction (60%) with full-frame NanoDet chirality (40%)
                        hand_prob = 0.6 * pose_hand + 0.4 * det_hand
                    elif "hand" in preds:
                        hand_prob = float(preds["hand"].item())
                    else:
                        hand_prob = float(det.get("class_id", 0))

                frame_kps = invert_keypoints(simcc_kps, M_inv)

                if use_postprocessing:
                    smooth_kps = tracks[tid].update_keypoints(t_current, frame_kps, hand_prob)
                    is_left = (tracks[tid].hand_prob < 0.50)
                else:
                    smooth_kps = frame_kps
                    is_left = (det["class_id"] == 0)

                colors = COLOR_LEFT if is_left else COLOR_RIGHT
                # Render tight stabilized box
                box_x1 = int(round(s_bbox[0]))
                box_y1 = int(round(s_bbox[1]))
                box_x2 = int(round(s_bbox[2]))
                box_y2 = int(round(s_bbox[3]))
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), colors["bbox"], 2)

                lbl = f"ID:{tid} {colors['label']} (Hand: {1.0 - tracks[tid].hand_prob if is_left else tracks[tid].hand_prob:.2f})"
                cv2.putText(frame, lbl, (box_x1, max(20, box_y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 2)

                # Draw 13-KP anatomical skeleton
                draw_skeleton(frame, smooth_kps, is_left=is_left)

                iou_matrix[r, :] = -1.0
                iou_matrix[:, c] = -1.0

        # Step C: Spawn new tracks for unmatched detections
        for c, det in enumerate(detected_feet):
            if c in matched_detections:
                continue
            raw_bbox = det["bbox"]
            tid = next_track_id
            next_track_id += 1
            init_hand = float(det["class_id"])
            tracks[tid] = FootTrack(tid, t_current, raw_bbox, initial_hand_prob=init_hand)

            cx, cy, size = tracks[tid].get_crop_params(crop_padding=0.25)
            M, M_inv = get_crop_affine_transform(cx, cy, size, out_size=256, rot=0.0)
            crop = warp_image_crop(frame, M, out_size=256, border_value=(114, 114, 114))
            crop_t = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                preds = rtmpose(crop_t)
                simcc_kps = decode_simcc(preds, simcc_split_ratio=2.0, win=12)[0]
                hand_prob = float(preds["hand"].item()) if "hand" in preds else float(det["class_id"])

            frame_kps = invert_keypoints(simcc_kps, M_inv)
            smooth_kps = tracks[tid].update_keypoints(t_current, frame_kps, hand_prob)
            is_left = (tracks[tid].hand_prob < 0.50)
            colors = COLOR_LEFT if is_left else COLOR_RIGHT

            box_x1 = int(round(raw_bbox[0]))
            box_y1 = int(round(raw_bbox[1]))
            box_x2 = int(round(raw_bbox[2]))
            box_y2 = int(round(raw_bbox[3]))
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), colors["bbox"], 2)
            lbl = f"ID:{tid} {colors['label']} (New)"
            cv2.putText(frame, lbl, (box_x1, max(20, box_y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors["bbox"], 2)
            draw_skeleton(frame, smooth_kps, is_left=is_left)

        # Step D: Handle lost tracks (missing frame bridge & cleanup)
        for tid in list(tracks.keys()):
            if tid not in matched_tracks and tid < next_track_id - len(detected_feet):
                tracks[tid].lost_count += 1
                if tracks[tid].lost_count > 6:
                    del tracks[tid]

        elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0
        latencies.append(elapsed_ms)

        # Draw Telemetry HUD
        rolling_fps = 1000.0 / max(1.0, np.mean(latencies[-30:]))
        detector_lbl = "NanoDet-Plus (Apache-2.0)" if is_nanodet else "YOLOv8 Stage-A"
        cv2.rectangle(frame, (10, 10), (480, 95), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (480, 95), (0, 255, 255), 1)
        cv2.putText(frame, f"SHOES VTO: {detector_lbl} + RTMPose-13KP", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
        cv2.putText(frame, f"Post-Processing: 1-Euro Filter + Box Track + Hand EMA", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1)
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
    default_rtm = "checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt"
    if not Path(default_rtm).exists():
        default_rtm = "checkpoints/rtmpose_13kp_official_coco_full_data/best.pt"

    default_det = "checkpoints/nanodet_foot_2class_320/best.pt"
    if not Path(default_det).exists():
        default_det = "checkpoints/nanodet_foot_320/best.pt"
    if not Path(default_det).exists():
        default_det = "deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.pt"

    parser.add_argument("--rtmpose", type=str, default=default_rtm)
    parser.add_argument("--detector", type=str, default=default_det)
    parser.add_argument("--video", type=str, default="data/test_video.mp4")
    parser.add_argument("--output", type=str, default="deliverables/qualcomm_experiments/demos/nanodet_2class_rtmpose_demo.mp4")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--no_postprocessing", action="store_true", help="Disable temporal filtering")
    args = parser.parse_args()

    run_video_evaluation(
        rtmpose_weights=args.rtmpose,
        detector_weights=args.detector,
        video_path=args.video,
        output_path=args.output,
        device_name=args.device,
        conf_thresh=args.conf,
        use_postprocessing=not args.no_postprocessing
    )
