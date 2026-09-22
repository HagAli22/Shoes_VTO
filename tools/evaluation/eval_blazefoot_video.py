"""
eval_blazefoot_video.py
───────────────────────
Benchmark and visual video evaluation for Google BlazeFoot ONNX models on data/test_video.mp4.

Usage:
    python tools/evaluation/eval_blazefoot_video.py
"""

import os
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort

from src.models.blazefoot.dataset import generate_blaze_anchors


def letterbox(img, new_shape=(320, 320), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def decode_blazefoot_18ch(raw_out, r, dw, dh, orig_w, orig_h, conf_thresh=0.25):
    """
    Decodes BlazeFoot Path A output: [1, 18, 1050].
      0..3: cx, cy, w, h
      4..5: left_score, right_score
      6..17: 4 keypoints (x, y, conf for toe_tip, heel_back, ball_medial, ball_lateral)
    """
    raw = raw_out[0] # [18, 1050]
    cx = raw[0] * 320.0
    cy = raw[1] * 320.0
    w  = raw[2] * 320.0
    h  = raw[3] * 320.0
    s_left  = raw[4]
    s_right = raw[5]

    x1 = np.clip((cx - w / 2.0 - dw) / r, 0, orig_w)
    y1 = np.clip((cy - h / 2.0 - dh) / r, 0, orig_h)
    x2 = np.clip((cx + w / 2.0 - dw) / r, 0, orig_w)
    y2 = np.clip((cy + h / 2.0 - dh) / r, 0, orig_h)

    detections = []
    for cls_name, scores, c_id in [("left_foot", s_left, 0), ("right_foot", s_right, 1)]:
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        if best_score >= conf_thresh:
            kpts = []
            for k in range(4):
                kx = ((raw[6 + k * 3, best_idx] * 320.0) - dw) / r
                ky = ((raw[7 + k * 3, best_idx] * 320.0) - dh) / r
                kv = float(raw[8 + k * 3, best_idx])
                kpts.append((float(kx), float(ky), kv))

            detections.append({
                "class_name": cls_name,
                "confidence": best_score,
                "box": [float(x1[best_idx]), float(y1[best_idx]), float(x2[best_idx]), float(y2[best_idx])],
                "kpts": kpts
            })

    return detections


def benchmark_blazefoot():
    video_path = "data/test_video.mp4"
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found")
        return

    model_path = "shoes-vto-ai-v2/models/stage-a-320-blazefoot-fp16.onnx"
    out_video  = "data/test_video_results/blazefoot_fp16_benchmark.mp4"
    os.makedirs("data/test_video_results", exist_ok=True)

    print("=" * 70)
    print(f"  Benchmarking Google BlazeFoot on {video_path}")
    print(f"  Model: {model_path} ({os.path.getsize(model_path)/1e6:.2f} MB)")
    print("=" * 70)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_vid, (w, h))

    latencies = []
    for f_idx in range(total_f):
        ret, frame = cap.read()
        if not ret: break

        img, r, (dw, dh) = letterbox(frame, new_shape=(320, 320))
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)

        t0 = time.perf_counter()
        raw_out = sess.run(None, {inp_name: blob})[0]
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        dets = decode_blazefoot_18ch(raw_out, r, dw, dh, w, h, conf_thresh=0.25)

        annotated = frame.copy()
        for d in dets:
            color = (0, 255, 117) if d["class_name"] == "left_foot" else (0, 160, 255)
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            cv2.putText(annotated, f"{d['class_name']}: {d['confidence']*100:.0f}%",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        writer.write(annotated)

    cap.release()
    writer.release()

    mean_lat = np.mean(latencies)
    fps = 1000.0 / mean_lat
    print(f"\n  [OK] BlazeFoot Video Benchmark Finished:")
    print(f"       Inference Latency : {mean_lat:.2f} ms")
    print(f"       Inference Speed   : {fps:.1f} FPS")
    print(f"       Annotated Video   : {out_video}")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_blazefoot()

