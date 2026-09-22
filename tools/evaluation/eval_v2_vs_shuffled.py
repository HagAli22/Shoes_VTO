"""
eval_v2_vs_shuffled.py
─────────────────────
Quick head-to-head benchmark on data/test_video.mp4:
  - run_v2_1/weights/best.pt  (baseline: 159-image training)
  - run_shuffled_v3/weights/best.pt  (new: 880-image clean shuffled training)

Reports per-model: FPS, detection rate (>=1 foot), both-feet rate, mean confidence.
Saves two annotated output videos into data/test_video_results/.
"""

import os
import sys
import time
import cv2
import numpy as np
from ultralytics import YOLO

VIDEO_PATH   = "data/test_video.mp4"
OUTPUT_DIR   = "data/test_video_results"
CONF_THRESH  = 0.25
IOU_THRESH   = 0.45
IMGSZ        = 320

MODELS = [
    {
        "id":           "run_v2_1",
        "label":        "run_v2_1 (baseline, 159 imgs)",
        "weights":      "outputs/stage_a/run_v2_1/weights/best.pt",
        "color_left":   (0, 200, 60),
        "color_right":  (0, 120, 255),
        "out_file":     "run_v2_1_benchmark.mp4",
    },
    {
        "id":           "run_shuffled_v3",
        "label":        "run_shuffled_v3 (new, 880 imgs)",
        "weights":      "outputs/stage_a/run_shuffled_v3/weights/best.pt",
        "color_left":   (0, 255, 180),
        "color_right":  (255, 120, 0),
        "out_file":     "run_shuffled_v3_benchmark.mp4",
    },
]


def draw_hud(frame, label, fps, det_rate, both_rate, f_idx, total, dets):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 115), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    cv2.putText(frame, label, (18, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:5.1f}   Frame: {f_idx+1}/{total}", (18, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Det >=1: {det_rate:5.1f}%   Both Feet: {both_rate:5.1f}%", (18, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (160, 255, 160), 2, cv2.LINE_AA)

    # Detection indicators
    l_det = next((d for d in dets if d["cls"] == 0), None)
    r_det = next((d for d in dets if d["cls"] == 1), None)
    for det, side, cx in [(l_det, "LEFT",  w-320), (r_det, "RIGHT", w-160)]:
        if det:
            txt = f"{side}: {det['conf']*100:.0f}%"
            col = (0, 230, 80) if side == "LEFT" else (0, 140, 255)
        else:
            txt = f"{side}: --"
            col = (80, 80, 80)
        cv2.putText(frame, txt, (cx, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    return frame


def run_benchmark(cfg):
    print(f"\n{'='*65}")
    print(f"  Benchmarking: {cfg['label']}")
    print(f"  Weights: {cfg['weights']}")
    print(f"{'='*65}")

    if not os.path.exists(cfg["weights"]):
        print(f"  [SKIP] weights not found: {cfg['weights']}")
        return None

    model = YOLO(cfg["weights"])
    # Warm-up
    dummy = np.zeros((IMGSZ, IMGSZ, 3), dtype=np.uint8)
    for _ in range(5):
        model.predict(dummy, imgsz=IMGSZ, conf=CONF_THRESH, iou=IOU_THRESH,
                      device=0, verbose=False)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {VIDEO_PATH}")
        return None

    fps_vid   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path  = os.path.join(OUTPUT_DIR, cfg["out_file"])
    writer    = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                fps_vid, (orig_w, orig_h))

    latencies, frames_any, frames_both = [], 0, 0
    left_confs, right_confs = [], []

    for f_idx in range(total_f):
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model.predict(frame, imgsz=IMGSZ, conf=CONF_THRESH,
                                iou=IOU_THRESH, device=0, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        latencies.append(dt)

        # Parse detections
        dets = []
        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                cls  = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                dets.append({"cls": cls, "conf": conf,
                             "box": [x1, y1, x2, y2]})

        has_left  = any(d["cls"] == 0 for d in dets)
        has_right = any(d["cls"] == 1 for d in dets)
        if has_left or has_right:
            frames_any += 1
        if has_left and has_right:
            frames_both += 1
        for d in dets:
            (left_confs if d["cls"] == 0 else right_confs).append(d["conf"])

        # Draw bounding boxes
        annotated = frame.copy()
        for d in dets:
            color = cfg["color_left"] if d["cls"] == 0 else cfg["color_right"]
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            lbl = f"{'Left' if d['cls']==0 else 'Right'} {d['conf']*100:.0f}%"
            cv2.putText(annotated, lbl, (x1 + 6, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Draw keypoints if available
        if results.keypoints is not None:
            kpts_data = results.keypoints.xy.cpu().numpy()
            for ki, kp_row in enumerate(kpts_data):
                kp_col = cfg["color_left"] if (ki < len(dets) and dets[ki]["cls"] == 0) else cfg["color_right"]
                for kx, ky in kp_row:
                    if kx > 0 or ky > 0:
                        cv2.circle(annotated, (int(kx), int(ky)), 5, kp_col, -1)
                        cv2.circle(annotated, (int(kx), int(ky)), 7, (255, 255, 255), 1)

        run_fps   = 1000.0 / max(np.mean(latencies), 1e-3)
        det_rate  = (frames_any  / max(f_idx + 1, 1)) * 100.0
        both_rate = (frames_both / max(f_idx + 1, 1)) * 100.0

        annotated = draw_hud(annotated, cfg["label"], run_fps,
                             det_rate, both_rate, f_idx, total_f, dets)
        writer.write(annotated)

        if (f_idx + 1) % 100 == 0 or (f_idx + 1) == total_f:
            print(f"  [{f_idx+1:4d}/{total_f}] FPS:{run_fps:5.1f} "
                  f"Det:{det_rate:5.1f}% Both:{both_rate:5.1f}%")

    cap.release()
    writer.release()

    mean_lat  = float(np.mean(latencies))
    mean_fps  = 1000.0 / mean_lat
    det_any   = (frames_any  / total_f) * 100.0
    det_both  = (frames_both / total_f) * 100.0
    ml_conf   = float(np.mean(left_confs))  if left_confs  else 0.0
    mr_conf   = float(np.mean(right_confs)) if right_confs else 0.0

    result = {
        "model_id":        cfg["id"],
        "label":           cfg["label"],
        "mean_latency_ms": round(mean_lat, 2),
        "mean_fps":        round(mean_fps, 1),
        "det_rate_any":    round(det_any,  1),
        "det_rate_both":   round(det_both, 1),
        "mean_left_conf":  round(ml_conf,  3),
        "mean_right_conf": round(mr_conf,  3),
        "output_video":    out_path,
    }

    print(f"\n  RESULT: Latency {mean_lat:.2f} ms | FPS {mean_fps:.1f} | "
          f"Det {det_any:.1f}% | Both {det_both:.1f}%")
    print(f"  Saved: {out_path}")
    return result


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        sys.exit(1)

    all_results = []
    for cfg in MODELS:
        r = run_benchmark(cfg)
        if r:
            all_results.append(r)

    print("\n" + "=" * 78)
    print("  HEAD-TO-HEAD BENCHMARK: run_v2_1  vs  run_shuffled_v3  on test_video.mp4")
    print("=" * 78)
    hdr = f"{'Model':<38} {'FPS':>6} {'Det>=1':>7} {'BothFeet':>9} {'L-Conf':>7} {'R-Conf':>7}"
    print(hdr)
    print("-" * 78)
    for r in all_results:
        print(f"{r['label']:<38} {r['mean_fps']:>6.1f} "
              f"{r['det_rate_any']:>6.1f}% {r['det_rate_both']:>8.1f}% "
              f"{r['mean_left_conf']*100:>6.1f}% {r['mean_right_conf']*100:>6.1f}%")
    print("=" * 78)

    import json
    out_json = os.path.join(OUTPUT_DIR, "v2_vs_shuffled_benchmark.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results → {out_json}")


if __name__ == "__main__":
    main()

