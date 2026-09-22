"""
generate_fixtures.py
────────────────────
Generates REAL reproducible fixtures from test_video.mp4:
  1. Extracts source frames (PNG) from the video
  2. Runs the Stage A model on each frame
  3. Saves raw output tensors as .npy files
  4. Saves expected-decoded.json with real model outputs

This makes the delivery fully reproducible:
  - Anyone runs the model on the same frames → gets the same tensors.

Usage:
    python tools/generate_fixtures.py
"""

import os
import json
import hashlib
import numpy as np
import cv2
import onnxruntime as ort


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


MODEL_PATH = "deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx"
VIDEO_PATH = "data/test_video.mp4"
OUT_FRAMES = "deliverables/stage_a_16kp/fixtures/source-frames-or-clips"
OUT_TENSORS = "deliverables/stage_a_16kp/fixtures/raw-output-tensors"
OUT_DECODED = "deliverables/stage_a_16kp/fixtures/expected-decoded.json"

# Which frames to extract as fixtures
FIXTURE_FRAME_IDS = [0, 30, 60, 100, 150, 200, 250, 300]

IMG_SIZE = 320
CONF_THRESH = 0.25
KPT_THRESH = 0.30
NMS_THRESH = 0.45

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



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def letterbox(img, size=320):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
    return canvas, scale, pad_x, pad_y


def decode_output(output, orig_w, orig_h, scale, pad_x, pad_y):
    """Decode [1, 54, N] raw tensor into detections."""
    out = output[0][0]  # [54, N] — output is list, first elem is [1,54,N]
    N = out.shape[1]
    detections = []


    for i in range(N):
        cx, cy, bw, bh = out[0, i], out[1, i], out[2, i], out[3, i]
        score_left = float(sigmoid(out[4, i]))
        score_right = float(sigmoid(out[5, i]))
        max_score = max(score_left, score_right)

        if max_score < CONF_THRESH:
            continue

        class_id = 0 if score_left >= score_right else 1

        x1 = ((cx - bw/2) - pad_x) / scale
        y1 = ((cy - bh/2) - pad_y) / scale
        x2 = ((cx + bw/2) - pad_x) / scale
        y2 = ((cy + bh/2) - pad_y) / scale

        keypoints = []
        for k in range(16):
            kx = float((out[6 + k*3, i] - pad_x) / scale)
            ky = float((out[7 + k*3, i] - pad_y) / scale)
            kv = float(sigmoid(out[8 + k*3, i]))
            keypoints.append({
                "index": k,
                "name": KEYPOINT_NAMES[k],
                "x": round(max(0, min(orig_w, kx)), 2),
                "y": round(max(0, min(orig_h, ky)), 2),
                "confidence": round(kv, 4),
                "visible": kv >= KPT_THRESH
            })

        detections.append({
            "class_id": class_id,
            "class_name": "left_foot" if class_id == 0 else "right_foot",
            "confidence": round(max_score, 4),
            "bbox": {
                "x1": round(max(0, x1), 2),
                "y1": round(max(0, y1), 2),
                "x2": round(min(orig_w, x2), 2),
                "y2": round(min(orig_h, y2), 2)
            },
            "keypoints": keypoints
        })

    # Per-class NMS (same as reference-decoder.js)
    final = []
    for cls_id in [0, 1]:
        cls_dets = sorted([d for d in detections if d["class_id"] == cls_id],
                          key=lambda d: -d["confidence"])
        kept = []
        for det in cls_dets:
            suppress = False
            for k in kept:
                b1, b2 = det["bbox"], k["bbox"]
                ix1 = max(b1["x1"], b2["x1"])
                iy1 = max(b1["y1"], b2["y1"])
                ix2 = min(b1["x2"], b2["x2"])
                iy2 = min(b1["y2"], b2["y2"])
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                a1 = (b1["x2"]-b1["x1"]) * (b1["y2"]-b1["y1"])
                a2 = (b2["x2"]-b2["x1"]) * (b2["y2"]-b2["y1"])
                iou = inter / (a1 + a2 - inter + 1e-7)
                if iou > NMS_THRESH:
                    suppress = True
                    break
            if not suppress:
                kept.append(det)
        final.extend(kept)

    return sorted(final, key=lambda d: -d["confidence"])


def main():
    os.makedirs(OUT_FRAMES, exist_ok=True)
    os.makedirs(OUT_TENSORS, exist_ok=True)

    print("=" * 60)
    print("  GENERATING REAL REPRODUCIBLE FIXTURES")
    print(f"  Model:  {MODEL_PATH}")
    print(f"  Video:  {VIDEO_PATH}")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return

    # Load ONNX model
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"  Model loaded. Input: {sess.get_inputs()[0].shape}")

    # Open video or use images directory as fallback
    use_video = os.path.exists(VIDEO_PATH)
    frames = {}

    if use_video:
        cap = cv2.VideoCapture(VIDEO_PATH)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Video: {total} frames @ {fps:.1f} FPS")

        frame_ids = [fid for fid in FIXTURE_FRAME_IDS if fid < total]
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
            ret, frame = cap.read()
            if ret:
                frames[fid] = frame
        cap.release()
    else:
        # Fallback: generate synthetic frames
        print("  [WARN] No test video found. Generating 3 synthetic test frames.")
        for fid in [0, 1, 2]:
            frames[fid] = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
        fps = 30.0

    fixture_records = []
    tensor_index = []

    for fid, frame in frames.items():
        orig_h, orig_w = frame.shape[:2]
        ts_ms = round((fid / fps) * 1000, 1) if use_video else fid * 33.3

        # Save source frame
        frame_filename = f"frame_{fid:05d}.png"
        frame_path = os.path.join(OUT_FRAMES, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Preprocess
        lb_img, scale, pad_x, pad_y = letterbox(frame, IMG_SIZE)
        lb_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor_in = np.transpose(lb_rgb, (2, 0, 1))[np.newaxis]  # [1,3,320,320]

        # Run inference
        raw_output = sess.run(None, {input_name: tensor_in})  # [1, 54, 2100]

        # Save raw tensor
        tensor_filename = f"frame_{fid:05d}_output.npy"
        tensor_path = os.path.join(OUT_TENSORS, tensor_filename)
        np.save(tensor_path, raw_output[0])

        # Hash of input tensor (for reproducibility verification)
        input_hash = hashlib.sha256(tensor_in.tobytes()).hexdigest()[:16]

        # Decode
        decoded = decode_output(raw_output, orig_w, orig_h, scale, pad_x, pad_y)

        record = {
            "frame_id": fid,
            "timestamp_ms": ts_ms,
            "source_frame": frame_filename,
            "raw_tensor_file": tensor_filename,
            "input_sha256_prefix": input_hash,
            "orig_dims": {"width": orig_w, "height": orig_h},
            "preprocessing": {
                "scale": round(scale, 6),
                "pad_x": pad_x,
                "pad_y": pad_y,
                "input_shape": [1, 3, IMG_SIZE, IMG_SIZE]
            },
            "detections": decoded
        }
        fixture_records.append(record)

        feet_count = len(decoded)
        print(f"  Frame {fid:4d} | t={ts_ms:7.1f}ms | {feet_count} foot/feet detected | {tensor_filename}")

        tensor_index.append({
            "frame_id": fid,
            "file": tensor_filename,
            "shape": list(raw_output[0].shape),
            "dtype": str(raw_output[0].dtype)
        })

    # Save decoded JSON
    full_fixture = {
        "fixture_version": "2.0.0",
        "model": MODEL_PATH,
        "model_sha256": hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest(),
        "preprocessing": {
            "input_size": IMG_SIZE,
            "padding_value": 114,
            "normalization": "divide by 255",
            "color": "RGB"
        },
        "decoding": {
            "conf_thresh": CONF_THRESH,
            "kpt_thresh": KPT_THRESH,
            "nms_thresh": NMS_THRESH,
            "nms_policy": "independent per class"
        },
        "frames": fixture_records
    }

    with open(OUT_DECODED, "w", encoding="utf-8") as f:
        json.dump(full_fixture, f, indent=2, cls=NumpyEncoder)

    # Save tensor index
    tensor_index_path = os.path.join(OUT_TENSORS, "index.json")
    with open(tensor_index_path, "w", encoding="utf-8") as f:
        json.dump(tensor_index, f, indent=2, cls=NumpyEncoder)


    print("\n" + "=" * 60)
    print(f"  FIXTURES GENERATED:")
    print(f"  Source frames : {OUT_FRAMES}/  ({len(frames)} PNG files)")
    print(f"  Raw tensors   : {OUT_TENSORS}/  ({len(frames)} .npy files)")
    print(f"  Decoded JSON  : {OUT_DECODED}")
    print("=" * 60)


if __name__ == "__main__":
    main()
