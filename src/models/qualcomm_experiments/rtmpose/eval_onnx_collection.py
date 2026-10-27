"""
rtmpose/eval_onnx_collection.py
─────────────────────────────────
Batch Video Evaluation using Pure ONNX Models (CUDA / CPU):
  - Detector A: NanoDet-Plus (2-Class, Apache-2.0, 320x320 ONNX)
  - Detector B: YOLO Stage-A (2-Class, 256x256 ONNX)
  - Pose Estimator: Qualcomm RTMPose-13KP + Handedness (256x256 ONNX)
  - Tracker: DualFootTracker (1-Euro Filter + Max 2 Physical Feet + Temporal Confirmation)
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import onnxruntime as ort
import torch

from src.models.qualcomm_experiments.rtmpose.decoder import decode_simcc
from src.models.qualcomm_experiments.common.crop import (
    compute_crop_box,
    get_crop_affine_transform,
    warp_image_crop,
    invert_keypoints,
)

# Colors: BGR
COLOR_LEFT = {
    "bbox": (255, 128, 0),       # Cyan / Blue
    "kpt_vis": (0, 255, 255),    # Yellow
    "kpt_occ": (0, 140, 255),    # Orange
    "skeleton": (255, 200, 0),
    "label": "Left Foot"
}

COLOR_RIGHT = {
    "bbox": (0, 165, 255),       # Orange / Amber
    "kpt_vis": (0, 255, 100),    # Bright Green
    "kpt_occ": (50, 150, 50),    # Olive
    "skeleton": (0, 220, 100),
    "label": "Right Foot"
}

SKELETON_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11),
    (0, 12), (11, 12),
]


class LowPassFilter:
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        self.s: Optional[np.ndarray] = None

    def filter_with_alpha(self, x: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        if self.s is None:
            self.s = np.array(x, dtype=np.float64).copy()
        else:
            self.s = alpha * np.array(x, dtype=np.float64) + (1.0 - alpha) * self.s
        return self.s


class OneEuroFilter:
    def __init__(
        self,
        t0: float,
        x0: np.ndarray,
        min_cutoff: float = 1.5,
        beta: float = 0.015,
        d_cutoff: float = 1.0
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filt = LowPassFilter()
        self.dx_filt = LowPassFilter()
        self.t_prev = float(t0)
        self.x_filt.filter_with_alpha(x0, np.ones_like(x0, dtype=np.float64))
        self.dx_filt.filter_with_alpha(np.zeros_like(x0, dtype=np.float64), np.ones_like(x0, dtype=np.float64))

    def _alpha(self, cutoff: np.ndarray, dt: float) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, t: float, x: np.ndarray) -> np.ndarray:
        dt = max(1e-4, float(t - self.t_prev))
        self.t_prev = float(t)
        x_curr = np.array(x, dtype=np.float64)
        dx = (x_curr - self.x_filt.s) / dt
        edx = self.dx_filt.filter_with_alpha(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * np.abs(edx)
        alpha = self._alpha(cutoff, dt)
        return self.x_filt.filter_with_alpha(x_curr, alpha)


class FootTrack:
    def __init__(self, track_id: int, t0: float, raw_bbox: np.ndarray, initial_hand_prob: float):
        self.track_id = track_id
        self.hits = 1
        self.box_filter = OneEuroFilter(t0, np.array(raw_bbox, dtype=np.float64), min_cutoff=1.5, beta=0.015)
        self.kpt_filter: Optional[OneEuroFilter] = None
        self.conf_filter: Optional[OneEuroFilter] = None
        self.hand_prob = float(initial_hand_prob)
        self.lost_count = 0
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
        coords = kps[:, :2]
        confs = kps[:, 2]
        if self.kpt_filter is None:
            self.kpt_filter = OneEuroFilter(t, coords, min_cutoff=0.8, beta=0.01)
            self.conf_filter = OneEuroFilter(t, confs, min_cutoff=2.0, beta=0.005)
            smoothed_coords = coords
            smoothed_confs = confs
        else:
            smoothed_coords = self.kpt_filter.filter(t, coords)
            smoothed_confs = self.conf_filter.filter(t, confs)

        alpha_hand = 0.20
        self.hand_prob = (1.0 - alpha_hand) * self.hand_prob + alpha_hand * float(hand_prob)

        out = np.zeros_like(kps)
        out[:, :2] = smoothed_coords
        out[:, 2] = np.clip(smoothed_confs, 0.0, 1.0)
        self.last_kps = out
        self.lost_count = 0
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


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.20) -> List[int]:
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
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


class DualFootTracker:
    def __init__(self, min_hits: int = 2, max_lost: int = 4):
        self.min_hits = min_hits
        self.max_lost = max_lost
        self.slots: Dict[str, Optional[FootTrack]] = {"left": None, "right": None}
        self.pending: Dict[str, Optional[Dict[str, Any]]] = {"left": None, "right": None}
        self.slot_ids = {"left": 1, "right": 2}

    def update(
        self,
        t_current: float,
        frame_idx: int,
        candidates: List[Dict[str, Any]],
        rtmpose_sess: ort.InferenceSession,
        frame: np.ndarray,
        use_postprocessing: bool = True
    ) -> List[Dict[str, Any]]:
        best_candidate: Dict[str, Optional[Dict[str, Any]]] = {"left": None, "right": None}

        if candidates:
            left_cands = [c for c in candidates if c["class_id"] == 0]
            right_cands = [c for c in candidates if c["class_id"] == 1]

            if len(candidates) >= 2 and (len(left_cands) == 0 or len(right_cands) == 0):
                sorted_by_x = sorted(candidates, key=lambda c: (c["bbox"][0] + c["bbox"][2]) / 2.0)
                left_cands = [sorted_by_x[0]]
                right_cands = [sorted_by_x[-1]]

            if left_cands:
                trk = self.slots["left"]
                if trk is not None:
                    left_cands.sort(key=lambda c: box_iou(trk.smoothed_bbox, c["bbox"]), reverse=True)
                else:
                    left_cands.sort(key=lambda c: c["score"], reverse=True)
                best_candidate["left"] = left_cands[0]

            if right_cands:
                trk = self.slots["right"]
                if trk is not None:
                    right_cands.sort(key=lambda c: box_iou(trk.smoothed_bbox, c["bbox"]), reverse=True)
                else:
                    right_cands.sort(key=lambda c: c["score"], reverse=True)
                best_candidate["right"] = right_cands[0]

        for slot in ["left", "right"]:
            cand = best_candidate[slot]
            track = self.slots[slot]

            if track is not None:
                matched = False
                if cand is not None:
                    iou = box_iou(track.smoothed_bbox, cand["bbox"])
                    c_trk = ((track.smoothed_bbox[0] + track.smoothed_bbox[2]) / 2.0, (track.smoothed_bbox[1] + track.smoothed_bbox[3]) / 2.0)
                    c_det = ((cand["bbox"][0] + cand["bbox"][2]) / 2.0, (cand["bbox"][1] + cand["bbox"][3]) / 2.0)
                    dist = np.hypot(c_trk[0] - c_det[0], c_trk[1] - c_det[1])
                    if iou >= 0.15 or dist < 120.0:
                        matched = True

                if matched and cand is not None:
                    track.hits += 1
                    track.lost_count = 0
                    raw_bbox = cand["bbox"]
                    if use_postprocessing:
                        s_bbox = track.update_box(t_current, raw_bbox)
                        cx, cy, size = track.get_crop_params(crop_padding=0.25)
                    else:
                        s_bbox = raw_bbox
                        x1, y1, x2, y2 = raw_bbox
                        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                        cx, cy, size = compute_crop_box((x1 + x2) / 2.0, (y1 + y2) / 2.0, bw, bh, crop_padding=0.25)

                    M, M_inv = get_crop_affine_transform(cx, cy, size, out_size=256, rot=0.0)
                    crop = warp_image_crop(frame, M, out_size=256, border_value=(114, 114, 114))
                    crop_inp = crop.astype(np.float32) / 255.0
                    crop_inp = np.transpose(crop_inp, (2, 0, 1))[None, ...]

                    # RTMPose ONNX Inference
                    rtm_outputs = rtmpose_sess.run(None, {rtmpose_sess.get_inputs()[0].name: crop_inp})
                    simcc_x_t = torch.from_numpy(rtm_outputs[0])
                    simcc_y_t = torch.from_numpy(rtm_outputs[1])
                    vis_t = torch.from_numpy(rtm_outputs[3]) if len(rtm_outputs) > 3 else None
                    pose_hand = float(rtm_outputs[5][0]) if len(rtm_outputs) > 5 else float(cand["class_id"])

                    preds_dict = {"simcc_x": simcc_x_t, "simcc_y": simcc_y_t, "vis": vis_t}
                    simcc_kps = decode_simcc(preds_dict, simcc_split_ratio=2.0, win=12)[0]
                    det_hand = float(cand["class_id"])
                    hand_prob = 0.6 * pose_hand + 0.4 * det_hand

                    frame_kps = invert_keypoints(simcc_kps, M_inv)
                    if use_postprocessing:
                        track.update_keypoints(t_current, frame_kps, hand_prob)
                    else:
                        track.last_kps = frame_kps
                        track.hand_prob = hand_prob
                else:
                    track.lost_count += 1
                    if track.lost_count > self.max_lost:
                        self.slots[slot] = None
            else:
                if cand is not None:
                    prev = self.pending[slot]
                    confirmed = False
                    if prev is not None and (frame_idx - prev["frame_idx"] <= 2):
                        iou = box_iou(prev["bbox"], cand["bbox"])
                        c_prev = ((prev["bbox"][0] + prev["bbox"][2]) / 2.0, (prev["bbox"][1] + prev["bbox"][3]) / 2.0)
                        c_det = ((cand["bbox"][0] + cand["bbox"][2]) / 2.0, (cand["bbox"][1] + cand["bbox"][3]) / 2.0)
                        dist = np.hypot(c_prev[0] - c_det[0], c_prev[1] - c_det[1])
                        if iou >= 0.15 or dist < 120.0:
                            confirmed = True

                    if confirmed:
                        tid = self.slot_ids[slot]
                        init_hand = 0.0 if slot == "left" else 1.0
                        new_track = FootTrack(tid, t_current, cand["bbox"], initial_hand_prob=init_hand)
                        new_track.hits = self.min_hits
                        cx, cy, size = new_track.get_crop_params(crop_padding=0.25)
                        M, M_inv = get_crop_affine_transform(cx, cy, size, out_size=256, rot=0.0)
                        crop = warp_image_crop(frame, M, out_size=256, border_value=(114, 114, 114))
                        crop_inp = crop.astype(np.float32) / 255.0
                        crop_inp = np.transpose(crop_inp, (2, 0, 1))[None, ...]

                        rtm_outputs = rtmpose_sess.run(None, {rtmpose_sess.get_inputs()[0].name: crop_inp})
                        simcc_x_t = torch.from_numpy(rtm_outputs[0])
                        simcc_y_t = torch.from_numpy(rtm_outputs[1])
                        vis_t = torch.from_numpy(rtm_outputs[3]) if len(rtm_outputs) > 3 else None
                        pose_hand = float(rtm_outputs[5][0]) if len(rtm_outputs) > 5 else float(cand["class_id"])

                        preds_dict = {"simcc_x": simcc_x_t, "simcc_y": simcc_y_t, "vis": vis_t}
                        simcc_kps = decode_simcc(preds_dict, simcc_split_ratio=2.0, win=12)[0]
                        frame_kps = invert_keypoints(simcc_kps, M_inv)
                        new_track.update_keypoints(t_current, frame_kps, pose_hand)
                        self.slots[slot] = new_track
                        self.pending[slot] = None
                    else:
                        self.pending[slot] = {"bbox": cand["bbox"], "score": cand["score"], "frame_idx": frame_idx}
                else:
                    if self.pending[slot] is not None and (frame_idx - self.pending[slot]["frame_idx"] > 2):
                        self.pending[slot] = None

        if self.slots["left"] is not None and self.slots["right"] is not None:
            trk_l = self.slots["left"]
            trk_r = self.slots["right"]
            overlap_iou = box_iou(trk_l.smoothed_bbox, trk_r.smoothed_bbox)
            if overlap_iou > 0.35:
                l_conf = 1.0 - trk_l.hand_prob
                r_conf = trk_r.hand_prob
                if l_conf >= r_conf:
                    self.slots["right"] = None
                else:
                    self.slots["left"] = None

        results = []
        for slot in ["left", "right"]:
            trk = self.slots[slot]
            if trk is not None and trk.hits >= self.min_hits and trk.lost_count <= 2 and trk.last_kps is not None:
                results.append({
                    "slot": slot,
                    "track_id": trk.track_id,
                    "bbox": trk.smoothed_bbox,
                    "kps": trk.last_kps,
                    "is_left": (slot == "left"),
                    "hand_prob": trk.hand_prob,
                    "lost": trk.lost_count
                })
        return results


def draw_skeleton(frame, kpts, is_left=True):
    colors = COLOR_LEFT if is_left else COLOR_RIGHT
    for p1, p2 in SKELETON_PAIRS:
        x1, y1, v1 = kpts[p1]
        x2, y2, v2 = kpts[p2]
        if v1 > 0.20 and v2 > 0.20:
            cv2.line(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), colors["skeleton"], 2, cv2.LINE_AA)

    for i in range(13):
        x, y, v = kpts[i]
        ix, iy = int(round(x)), int(round(y))
        if v >= 0.35:
            cv2.circle(frame, (ix, iy), 4, colors["kpt_vis"], -1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 5, (0, 0, 0), 1, cv2.LINE_AA)
        elif v >= 0.15:
            cv2.circle(frame, (ix, iy), 4, colors["kpt_occ"], 1, cv2.LINE_AA)


def run_video_pipeline(
    detector_sess: ort.InferenceSession,
    detector_type: str,
    rtmpose_sess: ort.InferenceSession,
    video_path: Path,
    output_path: Path,
    conf_thresh: float = 0.30
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[!] Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Downscale massive 4K output writing to 1080p max dimension for fast encoding and small file size
    scale_out = min(1.0, 1920.0 / max(w0, h0))
    out_w = int(round(w0 * scale_out))
    out_h = int(round(h0 * scale_out))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

    tracker = DualFootTracker(min_hits=2, max_lost=4)
    frame_idx = 0
    t_start = time.time()
    latencies = []

    det_input_name = detector_sess.get_inputs()[0].name
    det_img_size = 320 if detector_type == "nanodet" else 256

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        t_current = frame_idx / fps
        t_f0 = time.perf_counter()

        # Step 1: Detector ONNX Inference
        frame_resized = cv2.resize(frame, (det_img_size, det_img_size))
        inp = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]

        detected_feet = []

        if detector_type == "nanodet":
            det_outs = detector_sess.run(None, {det_input_name: inp})
            boxes_raw = det_outs[0][0]   # [N, 4]
            scores_raw = det_outs[1][0]  # [N, 2]

            max_scores = np.max(scores_raw, axis=-1)
            cls_ids = np.argmax(scores_raw, axis=-1)
            mask = max_scores >= conf_thresh

            if np.any(mask):
                f_boxes = boxes_raw[mask]
                f_scores = max_scores[mask]
                f_cls = cls_ids[mask]

                keep_idx = nms_numpy(f_boxes, f_scores, iou_thresh=0.20)
                sx = w0 / float(det_img_size)
                sy = h0 / float(det_img_size)

                for idx in keep_idx:
                    bx = f_boxes[idx]
                    scaled_box = np.array([bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy], dtype=np.float32)
                    y_cent = (scaled_box[1] + scaled_box[3]) / 2.0
                    # Spatial floor prior
                    if y_cent < (0.45 * h0) and f_scores[idx] < 0.95:
                        continue
                    detected_feet.append({
                        "bbox": scaled_box,
                        "class_id": int(f_cls[idx]),
                        "score": float(f_scores[idx])
                    })
        else:  # YOLO Stage A
            det_outs = detector_sess.run(None, {det_input_name: inp})
            preds = det_outs[0][0].T  # [1344, 45]
            cls0 = preds[:, 4]
            cls1 = preds[:, 5]
            max_scores = np.maximum(cls0, cls1)
            cls_ids = np.where(cls0 >= cls1, 0, 1)

            mask = max_scores >= conf_thresh
            if np.any(mask):
                f_preds = preds[mask]
                f_scores = max_scores[mask]
                f_cls = cls_ids[mask]

                # Convert cx, cy, w, h -> x1, y1, x2, y2
                cx, cy, bw, bh = f_preds[:, 0], f_preds[:, 1], f_preds[:, 2], f_preds[:, 3]
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0
                f_boxes = np.stack([x1, y1, x2, y2], axis=-1)

                keep_idx = nms_numpy(f_boxes, f_scores, iou_thresh=0.20)
                sx = w0 / float(det_img_size)
                sy = h0 / float(det_img_size)

                for idx in keep_idx:
                    bx = f_boxes[idx]
                    scaled_box = np.array([bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy], dtype=np.float32)
                    y_cent = (scaled_box[1] + scaled_box[3]) / 2.0
                    if y_cent < (0.45 * h0) and f_scores[idx] < 0.95:
                        continue
                    detected_feet.append({
                        "bbox": scaled_box,
                        "class_id": int(f_cls[idx]),
                        "score": float(f_scores[idx])
                    })

        # Step 2: Track & Pose Inference
        confirmed_feet = tracker.update(
            t_current=t_current,
            frame_idx=frame_idx,
            candidates=detected_feet,
            rtmpose_sess=rtmpose_sess,
            frame=frame,
            use_postprocessing=True
        )

        # Step 3: Draw on frame
        for foot in confirmed_feet:
            s_bbox = foot["bbox"]
            smooth_kps = foot["kps"]
            is_left = foot["is_left"]
            tid = foot["track_id"]
            hand_prob = foot["hand_prob"]
            colors = COLOR_LEFT if is_left else COLOR_RIGHT

            box_x1 = int(round(s_bbox[0]))
            box_y1 = int(round(s_bbox[1]))
            box_x2 = int(round(s_bbox[2]))
            box_y2 = int(round(s_bbox[3]))
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), colors["bbox"], 3)

            lbl = f"ID:{tid} {colors['label']} (Hand: {1.0 - hand_prob if is_left else hand_prob:.2f})"
            cv2.putText(frame, lbl, (box_x1, max(25, box_y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.70, colors["bbox"], 2)
            draw_skeleton(frame, smooth_kps, is_left=is_left)

        elapsed_ms = (time.perf_counter() - t_f0) * 1000.0
        latencies.append(elapsed_ms)

        # Telemetry HUD
        rolling_fps = 1000.0 / max(1.0, np.mean(latencies[-30:]))
        tag = "NanoDet-Plus (Apache-2.0)" if detector_type == "nanodet" else "YOLOv8 Stage-A"
        cv2.rectangle(frame, (15, 15), (580, 110), (20, 20, 20), -1)
        cv2.rectangle(frame, (15, 15), (580, 110), (0, 255, 255), 2)
        cv2.putText(frame, f"SHOES VTO ONNX: {tag} + RTMPose-13KP", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(frame, f"Execution Engine: ONNXRuntime (Dual-Slot Stabilized)", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 180), 1)
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames} | Latency: {elapsed_ms:.1f}ms ({rolling_fps:.1f} FPS)", (25, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
        cv2.putText(frame, f"Active Confirmed Feet: {len(confirmed_feet)} (Max 2)", (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 200, 255), 1)

        # Scale down if needed
        if scale_out < 1.0:
            frame_out = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            frame_out = frame

        writer.write(frame_out)

    cap.release()
    writer.release()
    dur = time.time() - t_start
    print(f"  [OK] Processed {video_path.name}: {total_frames} frames in {dur:.1f}s ({total_frames / max(0.1, dur):.1f} FPS, {np.mean(latencies):.1f} ms/frame).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", choices=["nanodet", "yolo", "both"], default="both")
    parser.add_argument("--input_dir", type=str, default="deliverables/stage_a_13kp/demo/collection_defferent_videos/original")
    parser.add_argument("--output_dir", type=str, default="deliverables/stage_a_13kp/demo/collection_defferent_videos/output")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.mp4"))
    if not videos:
        print(f"[!] No mp4 videos found in {input_dir}")
        return

    print("=" * 70)
    print("  SHOES VTO PURE ONNX BATCH VIDEO INFERENCE")
    print(f"  Videos Found:      {len(videos)}")
    print(f"  Pipeline Mode:     {args.pipeline}")
    print(f"  Input Directory:   {input_dir}")
    print(f"  Output Directory:  {output_dir}")
    print("=" * 70)

    # Preferred providers: CUDA -> CPU
    available_providers = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]
    print(f"[*] ONNX Runtime Providers: {providers}")

    # Load RTMPose ONNX
    rtm_path = "deliverables/qualcomm_experiments/models/rtmpose-13kp-full-champion-256.onnx"
    print(f"[*] Loading RTMPose ONNX: {rtm_path}...")
    rtm_sess = ort.InferenceSession(rtm_path, providers=providers)

    pipelines_to_run = ["nanodet", "yolo"] if args.pipeline == "both" else [args.pipeline]

    for p_type in pipelines_to_run:
        print("\n" + "#" * 70)
        if p_type == "nanodet":
            det_path = "deliverables/qualcomm_experiments/models/nanodet-plus-foot-2class-320.onnx"
            out_subdir = output_dir / "nanodet_rtmpose_onnx"
            conf = 0.30
            print(f"  RUNNING PIPELINE 1: NanoDet-Plus (2-Class) + RTMPose-13KP [ONNX]")
        else:
            det_path = "deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.onnx"
            out_subdir = output_dir / "yolo_rtmpose_onnx"
            conf = 0.25
            print(f"  RUNNING PIPELINE 2: YOLOv8 Stage-A + RTMPose-13KP [ONNX]")
        print(f"  Detector Model:   {det_path}")
        print(f"  Output Subfolder: {out_subdir}")
        print("#" * 70)

        out_subdir.mkdir(parents=True, exist_ok=True)
        det_sess = ort.InferenceSession(det_path, providers=providers)

        t_pipe_start = time.time()
        for idx, vid in enumerate(videos, 1):
            out_file = out_subdir / vid.name
            print(f"\n[{idx}/{len(videos)}] Processing: {vid.name} -> {out_file.name}")
            run_video_pipeline(
                detector_sess=det_sess,
                detector_type=p_type,
                rtmpose_sess=rtm_sess,
                video_path=vid,
                output_path=out_file,
                conf_thresh=conf
            )

        total_pipe_time = time.time() - t_pipe_start
        print(f"\n[+] Pipeline {p_type.upper()} finished all {len(videos)} videos in {total_pipe_time:.1f}s.")

    print("\n" + "=" * 70)
    print("  ALL VIDEO INFERENCE COMPLETE!")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
