"""
generate_teacher_groundtruth.py
───────────────────────────────
Generates and caches per-frame Golden Predictions using the Teacher 16-KP model
(deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx) on test_video.mp4.

Extracts for each frame:
  - Bounding boxes [x1, y1, x2, y2]
  - Class id & name (0: left_foot, 1: right_foot)
  - 4 Coarse Keypoints:
      0: toe_tip     (Teacher index 11)
      1: heel_back    (Teacher index 1)
      2: ball_medial  (Teacher index 3)
      3: ball_lateral (Teacher index 4)

Saves results to:
  outputs/evaluation/teacher_video_groundtruth.json
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# Audited 16-KP names
KEYPOINT_NAMES_16 = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial",
    "ball_lateral", "ball_top", "instep_top", "arch_medial",
    "midfoot_lateral", "malleolus_medial", "malleolus_lateral", "toe_tip",
    "ankle_center", "throat", "achilles", "shin_mid"
]

# Mapping 4 Coarse Landmarks -> Teacher 16-KP indices
COARSE_KP_INDEX_MAP = {
    0: {"name": "toe_tip", "teacher_idx": 11, "fallback": 0},
    1: {"name": "heel", "teacher_idx": 14, "fallback": 1},
    2: {"name": "ball_medial", "teacher_idx": 3, "fallback": 7},
    3: {"name": "ball_lateral", "teacher_idx": 4, "fallback": 8}
}


def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-7)


def determine_geometric_chirality(kpts):
    p_toe = kpts[11]
    p_heel = kpts[1]
    p_lat = kpts[4]
    if p_toe["conf"] < 0.15 or p_heel["conf"] < 0.15 or p_lat["conf"] < 0.15:
        return None
    v_axis_x = p_toe["x"] - p_heel["x"]
    v_axis_y = p_toe["y"] - p_heel["y"]
    v_lat_x = p_lat["x"] - p_heel["x"]
    v_lat_y = p_lat["y"] - p_heel["y"]
    cross = v_axis_x * v_lat_y - v_axis_y * v_lat_x
    return 1 if cross >= 0 else 0


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


def decode_teacher_output(raw_output, orig_w, orig_h, scale, pad_x, pad_y, conf_thresh=0.28, nms_thresh=0.45, dedup_iou=0.60):
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

        kpts_16 = []
        for k in range(16):
            kx = ((out[6 + k * 3, i] - pad_x) / scale)
            ky = ((out[7 + k * 3, i] - pad_y) / scale)
            kv = float(out[8 + k * 3, i])
            kpts_16.append({"name": KEYPOINT_NAMES_16[k], "x": float(kx), "y": float(ky), "conf": kv})

        geom_cls = determine_geometric_chirality(kpts_16)
        if geom_cls is not None:
            class_id = geom_cls

        candidates.append({
            "class_id": class_id,
            "class_name": "left_foot" if class_id == 0 else "right_foot",
            "conf": max_score,
            "bbox": [x1, y1, x2, y2],
            "keypoints_16": kpts_16
        })

    # Per-Class NMS
    nmed = []
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
        nmed.extend(kept)

    # Cross-Class Deduplication
    nmed.sort(key=lambda d: d["conf"], reverse=True)
    deduped = []
    for det in nmed:
        dup = False
        for k in deduped:
            if compute_iou(det["bbox"], k["bbox"]) > dedup_iou:
                dup = True
                break
        if not dup:
            # Extract 4 Coarse Keypoints
            kpts_4 = []
            for coarse_i in range(4):
                info = COARSE_KP_INDEX_MAP[coarse_i]
                t_idx = info["teacher_idx"]
                t_kp = det["keypoints_16"][t_idx]
                if t_kp["conf"] < 0.20 and "fallback" in info:
                    fb_kp = det["keypoints_16"][info["fallback"]]
                    if fb_kp["conf"] > t_kp["conf"]:
                        t_kp = fb_kp
                kpts_4.append({
                    "name": str(info["name"]),
                    "x": float(t_kp["x"]),
                    "y": float(t_kp["y"]),
                    "conf": float(t_kp["conf"])
                })
            
            deduped.append({
                "class_id": int(det["class_id"]),
                "class_name": str(det["class_name"]),
                "confidence": float(det["conf"]),
                "bbox": [float(v) for v in det["bbox"]],
                "keypoints_4": kpts_4,
                "keypoints_16": [{
                    "name": str(kp["name"]),
                    "x": float(kp["x"]),
                    "y": float(kp["y"]),
                    "conf": float(kp["conf"])
                } for kp in det["keypoints_16"]]
            })

    return deduped


def generate_golden_dataset(
    video_path: str = "deliverables/stage_a_16kp/test_video.mp4",
    teacher_model_path: str = "deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx",
    output_json_path: str = "outputs/evaluation/teacher_video_groundtruth.json"
):
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    print("=" * 70)
    print("  [>] GENERATING TEACHER 16-KP GOLDEN GROUND TRUTH ON VIDEO")
    print(f"  Input Video   : {video_path}")
    print(f"  Teacher Model : {teacher_model_path}")
    print(f"  Output Cache  : {output_json_path}")
    print("=" * 70)

    session = ort.InferenceSession(teacher_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    print(f"  [Engine] Initialized on: {session.get_providers()[0]}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_records = []
    frame_idx = 0
    t0_all = time.perf_counter()

    pbar = tqdm(total=total_frames, desc="Generating Golden Frames")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        lb_img, scale, pad_x, pad_y = letterbox(frame, size=320)
        lb_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor_in = np.transpose(lb_rgb, (2, 0, 1))[np.newaxis]

        raw_out = session.run(None, {input_name: tensor_in})
        detections = decode_teacher_output(raw_out, orig_w, orig_h, scale, pad_x, pad_y)

        frame_records.append({
            "frame_idx": frame_idx,
            "width": orig_w,
            "height": orig_h,
            "num_feet": len(detections),
            "detections": detections
        })

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    t_total = time.perf_counter() - t0_all

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "video_path": video_path,
                "teacher_model": teacher_model_path,
                "total_frames": total_frames,
                "fps": video_fps,
                "resolution": [orig_w, orig_h],
                "date": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "frames": frame_records
        }, f, indent=2)

    print(f"\n[SUCCESS] Cached {len(frame_records)} golden frames to: {output_json_path}")
    print(f"Processed in {t_total:.1f}s ({total_frames/t_total:.1f} FPS)\n")


if __name__ == "__main__":
    generate_golden_dataset()
