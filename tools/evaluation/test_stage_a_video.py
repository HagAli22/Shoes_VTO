"""
test_stage_a_video.py
─────────────────────
Stage A — Video Inference with Corrected Anatomical Keypoint Mapping

Features:
  1. Exact 1:1 Roboflow-to-Anatomy Index Mapping.
  2. Direct point-anchored labels with leader lines.
  3. Clean skeleton linkages connecting anatomically continuous chains.
  4. Cross-class NMS to eliminate duplicate boxes.
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.models.stage_a.postprocess import postprocess_stage_a

CANDIDATE_MODELS = [
    os.path.join(PROJECT_ROOT, "outputs", "stage_a", "run_v2_1", "weights", "best.pt"),
    os.path.join(PROJECT_ROOT, "outputs", "stage_a", "run12", "weights", "best.pt"),
    os.path.join(PROJECT_ROOT, "outputs", "stage_a", "run1", "weights", "best.pt"),
]
DEFAULT_MODEL = next((p for p in CANDIDATE_MODELS if os.path.exists(p)), CANDIDATE_MODELS[0])
DEFAULT_VIDEO = os.path.join(PROJECT_ROOT, "data", "test_video.mp4")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "stage_a", "test_results")

# ── Exact Roboflow Export Index -> True Anatomical Definition ─────────────────
ROBOFLOW_TO_ANATOMY = {
    0:  {"name": "0: toe_tip",      "zone": "sole",  "color": (50,  255, 50)},    # Bright Green
    11: {"name": "1: toe_ground",   "zone": "sole",  "color": (0,   220, 120)},   # Sea Green
    3:  {"name": "4: ball_medial",  "zone": "ball",  "color": (0,   165, 255)},   # Orange
    4:  {"name": "5: ball_lateral", "zone": "ball",  "color": (255, 180, 0)},     # Sky Blue
    5:  {"name": "6: ball_top",     "zone": "ball",  "color": (200, 255, 0)},     # Lime
    6:  {"name": "7: instep_top",   "zone": "mid",   "color": (255, 130, 30)},    # Blue-Cyan
    7:  {"name": "8: arch_medial",  "zone": "mid",   "color": (180, 50,  255)},   # Purple
    8:  {"name": "9: midfoot_lat",  "zone": "mid",   "color": (255, 50,  220)},   # Pink
    9:  {"name": "10: malle_medial","zone": "ankle", "color": (0,   100, 255)},   # Dark Orange
    10: {"name": "11: malle_lat",   "zone": "ankle", "color": (50,  200, 255)},   # Light Orange
    12: {"name": "3: heel_ground",  "zone": "sole",  "color": (0,   255, 255)},   # Yellow
    14: {"name": "14: achilles",    "zone": "ankle", "color": (0,   50,  255)},   # Red-Orange
    13: {"name": "2: heel_back",    "zone": "sole",  "color": (255, 0,   150)},   # Indigo
    15: {"name": "15: shin_mid",    "zone": "leg",   "color": (0,   255, 200)},   # Aqua
}

CLASS_COLORS = {
    0: (0, 215, 255),    # Gold for left_foot
    1: (255, 110, 40)    # Royal Blue for right_foot
}

# Anatomical skeleton linkages using Roboflow Export channel indices
SKELETON_ROBOFLOW_INDICES = [
    (0, 11),           # toe_tip -> toe_ground
    (11, 3), (11, 4),  # toe_ground -> ball_medial & ball_lateral
    (3, 4),            # Metatarsal width line: ball_medial <-> ball_lateral
    (3, 5), (4, 5),    # ball_medial/lateral -> ball_top
    (5, 6),            # ball_top -> instep_top
    (3, 7), (7, 12),   # Medial chain: ball_medial -> arch_medial -> heel_ground
    (4, 8), (8, 12),   # Lateral chain: ball_lateral -> midfoot_lateral -> heel_ground
    (6, 9), (6, 10),   # Instep -> malleoli
    (9, 14), (10, 14), # Malleoli -> achilles
    (12, 13),          # heel_ground -> heel_back
    (14, 15),          # Achilles -> shin_mid
]


def draw_verified_detections(frame: np.ndarray, processed_feet: list) -> np.ndarray:
    annotated = frame.copy()
    h, w = frame.shape[:2]

    scale = max(1.0, min(w, h) / 720.0)
    dot_radius = int(round(6.0 * scale))
    line_thick = max(1, int(round(2.0 * scale)))
    box_thick  = max(2, int(round(2.5 * scale)))
    font_scale = 0.36 * scale
    font_thick = max(1, int(round(1.2 * scale)))

    for foot in processed_feet:
        x1, y1, x2, y2 = foot["box"]
        cls = foot["cls"]
        cls_name = foot["cls_name"]
        conf = foot["conf"]
        pts = foot["keypoints"]
        kc = foot["kp_conf"]
        box_color = CLASS_COLORS.get(cls, (180, 180, 180))

        # 1. Bounding Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thick)
        label = f"{cls_name}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, font_thick + 1)
        badge_y1 = max(0, y1 - th - int(10 * scale))
        badge_y2 = y1
        cv2.rectangle(annotated, (x1, badge_y1), (x1 + tw + int(12 * scale), badge_y2), box_color, -1)
        cv2.putText(annotated, label, (x1 + int(6 * scale), badge_y2 - int(4 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (10, 10, 10), font_thick + 1, cv2.LINE_AA)

        # 2. Skeleton Lines
        for a, b in SKELETON_ROBOFLOW_INDICES:
            if a < len(pts) and b < len(pts) and a in ROBOFLOW_TO_ANATOMY and b in ROBOFLOW_TO_ANATOMY:
                if kc[a] > 0.30 and kc[b] > 0.30:
                    pa = (int(round(pts[a][0])), int(round(pts[a][1])))
                    pb = (int(round(pts[b][0])), int(round(pts[b][1])))
                    if pa[0] > 0 and pa[1] > 0 and pb[0] > 0 and pb[1] > 0:
                        ca = ROBOFLOW_TO_ANATOMY[a]["color"]
                        cb = ROBOFLOW_TO_ANATOMY[b]["color"]
                        bone_col = (int((ca[0] + cb[0]) / 2),
                                    int((ca[1] + cb[1]) / 2),
                                    int((ca[2] + cb[2]) / 2))
                        cv2.line(annotated, pa, pb, (20, 20, 20), line_thick + 2, cv2.LINE_AA)
                        cv2.line(annotated, pa, pb, bone_col, line_thick, cv2.LINE_AA)

        # 3. Keypoints & Direct Point-Anchored Labels
        for k in range(min(len(pts), 16)):
            if k not in ROBOFLOW_TO_ANATOMY:
                continue

            conf_k = float(kc[k]) if k < len(kc) else 0.0
            if conf_k < 0.30:
                continue

            px, py = int(round(pts[k, 0])), int(round(pts[k, 1]))
            if px <= 0 and py <= 0:
                continue

            meta = ROBOFLOW_TO_ANATOMY[k]
            c_kp = meta["color"]
            tag_name = meta["name"]

            # Keypoint dot
            cv2.circle(annotated, (px, py), dot_radius + 2, (15, 15, 15), -1, cv2.LINE_AA)
            cv2.circle(annotated, (px, py), dot_radius, c_kp, -1, cv2.LINE_AA)
            cv2.circle(annotated, (px, py), max(1, int(dot_radius * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)

            # Direct Label Badge with Leader Line
            (ntw, nth), _ = cv2.getTextSize(tag_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)

            # Alternate placement to avoid overlap
            if k % 2 == 0:
                tx = min(w - ntw - 6, px + int(12 * scale))
                ty = max(nth + 4, min(h - 6, py + int(4 * scale)))
                anchor_x = tx
            else:
                tx = max(4, px - ntw - int(12 * scale))
                ty = max(nth + 4, min(h - 6, py + int(4 * scale)))
                anchor_x = tx + ntw

            # Leader Line
            cv2.line(annotated, (px, py), (anchor_x, ty - int(nth / 2)), (30, 30, 30), 2, cv2.LINE_AA)
            cv2.line(annotated, (px, py), (anchor_x, ty - int(nth / 2)), c_kp, 1, cv2.LINE_AA)

            # Badge
            bg_pad = int(3 * scale)
            cv2.rectangle(annotated, (tx - bg_pad, ty - nth - bg_pad), (tx + ntw + bg_pad, ty + bg_pad), (20, 20, 20), -1)
            cv2.rectangle(annotated, (tx - bg_pad, ty - nth - bg_pad), (tx + ntw + bg_pad, ty + bg_pad), c_kp, 1)
            cv2.putText(annotated, tag_name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return annotated


def parse_args():
    p = argparse.ArgumentParser(description="Stage A video test with corrected anatomical mapping.")
    p.add_argument("--model",        default=DEFAULT_MODEL)
    p.add_argument("--video",        default=DEFAULT_VIDEO)
    p.add_argument("--box-conf",     type=float, default=0.30)
    p.add_argument("--cross-iou",    type=float, default=0.35)
    p.add_argument("--imgsz",        type=int, default=320)
    p.add_argument("--device",       default="0")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snapshots_dir = os.path.join(OUTPUT_DIR, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    from ultralytics import YOLO

    print("\n" + "=" * 65)
    print("      STAGE A — VIDEO INFERENCE (ANATOMICALLY ALIGNED)")
    print("=" * 65)
    print(f"  Model Path         : {args.model}")
    print(f"  Video Path         : {args.video}")
    print(f"  Box Conf Threshold : {args.box_conf}")
    print("=" * 65 + "\n")

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in       = cap.get(cv2.CAP_PROP_FPS)
    w_in         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(OUTPUT_DIR, "annotated_video_anatomical_correct.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps_in, (w_in, h_in))

    frame_idx = 0
    print("  Processing video frames with verified anatomical mapping...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        raw_results = model.predict(source=frame, imgsz=args.imgsz, conf=0.15, device=args.device, verbose=False)
        inf_ms = (time.perf_counter() - t0) * 1000.0

        cleaned_feet = postprocess_stage_a(
            raw_results[0],
            box_conf_thresh=args.box_conf,
            cross_class_iou_thresh=args.cross_iou,
            min_kp_conf=0.30,
            max_feet=2
        )

        n_l = sum(1 for f in cleaned_feet if f["cls"] == 0)
        n_r = sum(1 for f in cleaned_feet if f["cls"] == 1)

        annotated = draw_verified_detections(frame, cleaned_feet)

        hud_scale = max(1.0, min(w_in, h_in) / 720.0)
        cv2.putText(annotated, f"FPS: {1000.0 / max(0.1, inf_ms):.1f} ({inf_ms:.1f}ms)",
                    (int(14 * hud_scale), int(35 * hud_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * hud_scale, (50, 240, 50), max(1, int(2 * hud_scale)), cv2.LINE_AA)
        cv2.putText(annotated, f"Frame {frame_idx}/{total_frames} | Feet: L={n_l} R={n_r}",
                    (int(14 * hud_scale), int(70 * hud_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * hud_scale, (255, 255, 255), max(1, int(2 * hud_scale)), cv2.LINE_AA)

        if frame_idx in [62, 100, 150, 200, 300, 363, 400]:
            cv2.imwrite(os.path.join(snapshots_dir, f"anatomical_frame_{frame_idx:04d}.jpg"), annotated)

        writer.write(annotated)
        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"  [{frame_idx:4d}/{total_frames}] {inf_ms:5.1f}ms | Feet: L={n_l} R={n_r}", flush=True)

    cap.release()
    writer.release()

    print("\n" + "=" * 65)
    print("      VIDEO COMPLETE — ANATOMICALLY VERIFIED")
    print("=" * 65)
    print(f"  Output Video: {out_video_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
