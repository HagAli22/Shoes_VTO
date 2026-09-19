"""
test_stage_a_live.py
────────────────────
Stage A — Real-Time Live Camera Test
Stage A — Real-Time Live Camera Foot & Keypoint Detection
Stage A — Live Camera with Dual-Foot Cross-Class NMS & Keypoint Filtering
Stage A — Live Camera Inference with Anatomically Verified Keypoint Mapping

Runs the Stage A foot detector on your webcam in real time.
Shows:
  - Bounding boxes (left_foot=yellow, right_foot=blue)
  - 16 keypoints color-coded by zone
  - Real-time FPS counter
  - Per-frame inference time
  - Detection confidence

Usage
-----
  conda activate yolo
  python tools/evaluation/test_stage_a_live.py
  python tools/evaluation/test_stage_a_live.py --camera 0 --conf 0.30

Controls
--------
Controls:
  Q / ESC       — Quit
  S             — Save current frame as PNG
  S             — Save current frame snapshot PNG
  N             — Cycle keypoint name labels: Full -> Short -> None
  S             — Save snapshot image
  R             — Reset FPS counter
  H             — Toggle keypoint labels on/off
  Q / ESC — Quit
  S       — Save Snapshot
  R       — Reset FPS
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODEL = os.path.join(PROJECT_ROOT, "outputs", "stage_a", "run12", "weights", "best.pt")
from src.models.stage_a.postprocess import postprocess_stage_a

CANDIDATE_MODELS = [
    os.path.join(PROJECT_ROOT, "outputs", "stage_a", "run_v2_1", "weights", "best.pt"),
    os.path.join(PROJECT_ROOT, "outputs", "stage_a", "run12", "weights", "best.pt"),
    os.path.join(PROJECT_ROOT, "outputs", "stage_a", "run1", "weights", "best.pt"),
]
DEFAULT_MODEL = next((p for p in CANDIDATE_MODELS if os.path.exists(p)), CANDIDATE_MODELS[0])
SNAPSHOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "stage_a", "snapshots")

# ── Keypoint schema ───────────────────────────────────────────────────────────
KP_NAMES = [
    "toe_tip",    "toe_ground", "heel_back",  "heel_ground",
    "ball_med",   "ball_lat",   "ball_top",   "instep_top",
    "arch_med",   "midft_lat",  "malle_med",  "malle_lat",
    "ank_ctr",    "throat",     "achilles",   "shin_mid",
KP_NAMES_FULL = [
    "0: toe_tip", "1: toe_ground", "2: heel_back", "3: heel_ground",
    "4: ball_med", "5: ball_lat", "6: ball_top", "7: instep_top",
    "8: arch_med", "9: midft_lat", "10: mal_med", "11: mal_lat",
    "12: ank_ctr", "13: throat", "14: achilles", "15: shin_mid"
]

# BGR colors per anatomical zone
ZONE_COLOR = {
    "sole":  (80,  220, 80),    # green
    "ball":  (40,  160, 255),   # orange
    "mid":   (255, 120, 40),    # blue
    "ankle": (180, 40,  255),   # purple
    "leg":   (0,   220, 220),   # yellow-green
KP_NAMES_SHORT = [
    "toe_tip", "toe_grnd", "heel_bk", "heel_grnd",
    "ball_m", "ball_l", "b_top", "instep",
    "arch", "mid_l", "mal_m", "mal_l",
    "ank", "throat", "achilles", "shin"
]

ZONE_COLORS = {
    "sole":  (60,  230, 80),    # Green
    "ball":  (0,   165, 255),   # Orange
    "mid":   (255, 130, 30),    # Cyan-Blue
    "ankle": (220, 50,  230),   # Purple
    "leg":   (0,   230, 230),   # Yellow
ROBOFLOW_TO_ANATOMY = {
    0:  {"name": "0: toe_tip",      "color": (50,  255, 50)},
    11: {"name": "1: toe_ground",   "color": (0,   220, 120)},
    3:  {"name": "4: ball_medial",  "color": (0,   165, 255)},
    4:  {"name": "5: ball_lateral", "color": (255, 180, 0)},
    5:  {"name": "6: ball_top",     "color": (200, 255, 0)},
    6:  {"name": "7: instep_top",   "color": (255, 130, 30)},
    7:  {"name": "8: arch_medial",  "color": (180, 50,  255)},
    8:  {"name": "9: midfoot_lat",  "color": (255, 50,  220)},
    9:  {"name": "10: malle_med",   "color": (0,   100, 255)},
    10: {"name": "11: malle_lat",   "color": (50,  200, 255)},
    12: {"name": "3: heel_ground",  "color": (0,   255, 255)},
    14: {"name": "14: achilles",    "color": (0,   50,  255)},
    13: {"name": "2: heel_back",    "color": (255, 0,   150)},
    15: {"name": "15: shin_mid",    "color": (0,   255, 200)},
}
KP_ZONE  = ["sole"]*4 + ["ball"]*3 + ["mid"]*3 + ["ankle"]*5 + ["leg"]*1
KP_ZONE = ["sole"] * 4 + ["ball"] * 3 + ["mid"] * 3 + ["ankle"] * 5 + ["leg"] * 1

CLASS_COLORS = {
    0: (0, 215, 255),    # left_foot (gold)
    1: (255, 110, 40)    # right_foot (blue)
    0: (0, 215, 255),    # Gold for left_foot
    1: (255, 110, 40)    # Royal Blue for right_foot
}
CLASS_NAMES = {0: "left_foot", 1: "right_foot"}

def kp_col(i):
    return ZONE_COLOR[KP_ZONE[i]]


CLASS_COLOR = {0: (0, 220, 255), 1: (255, 100, 30)}   # left=gold, right=blue-orange
CLASS_NAME  = {0: "left_foot",   1: "right_foot"}

SKELETON = [
    (0, 1), (1, 3), (3, 2),             # toe chain → heel_back
    (3, 8), (8, 4), (4, 5),             # heel → arch → ball line
    (4, 6), (5, 6), (6, 7),             # ball → instep
    (3, 9), (9, 5),                     # heel → midfoot_lateral
    (7, 12), (12, 13),                  # instep → ankle_center → throat
    (12, 10), (12, 11),                 # ankle_center → malleoli
    (10, 14), (11, 14),                 # malleoli → achilles
    (12, 15),                           # ankle → shin
    (0, 1), (1, 3), (3, 2),
    (3, 8), (8, 4), (4, 5),
    (4, 6), (5, 6), (6, 7),
    (3, 9), (9, 5),
    (7, 12), (12, 13),
    (12, 10), (12, 11),
    (10, 14), (11, 14),
    (12, 15),
SKELETON_ROBOFLOW_INDICES = [
    (0, 11), (11, 3), (11, 4), (3, 4),
    (3, 5), (4, 5), (5, 6),
    (3, 7), (7, 12), (4, 8), (8, 12),
    (6, 9), (6, 10), (9, 14), (10, 14),
    (12, 13), (14, 15)
]


def draw_frame(frame: np.ndarray, result,
               conf_thresh: float, show_labels: bool) -> np.ndarray:
    """Draw detections on frame and return annotated copy."""
def draw_frame(frame: np.ndarray, result, conf_thresh: float, name_mode: str) -> np.ndarray:
def draw_live_cleaned(frame: np.ndarray, processed_feet: list) -> np.ndarray:
def draw_live_verified(frame: np.ndarray, processed_feet: list) -> np.ndarray:
    out = frame.copy()
    h, w = frame.shape[:2]

    scale = max(1.0, min(w, h) / 720.0)
    dot_radius = int(round(6.0 * scale))
    line_thick = max(1, int(round(2.0 * scale)))
    box_thick  = max(2, int(round(2.5 * scale)))
    font_scale = 0.38 * scale
    font_scale = 0.36 * scale
    font_thick = max(1, int(round(1.2 * scale)))

    if result.boxes is None or len(result.boxes) == 0:
        return out

    kps_all = result.keypoints

    for i, box in enumerate(result.boxes):
        conf = float(box.conf[0])
        if conf < conf_thresh:
            continue
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        col = CLASS_COLOR.get(cls, (128, 128, 128))
    for foot in processed_feet:
        x1, y1, x2, y2 = foot["box"]
        cls = foot["cls"]
        cls_name = foot["cls_name"]
        conf = foot["conf"]
        pts = foot["keypoints"]
        kp_status = foot["kp_status"]
        kc = foot["kp_conf"]
        col = CLASS_COLORS.get(cls, (180, 180, 180))

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"{CLASS_NAME.get(cls, str(cls))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, box_thick)
        label = f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}"
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, font_thick + 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - int(8 * scale))), (x1 + tw + int(10 * scale), y1), col, -1)
        cv2.putText(out, label, (x1 + int(4 * scale), y1 - int(4 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 0, 0), font_thick + 1, cv2.LINE_AA)

        # Keypoints + skeleton
        if kps_all is None or i >= len(kps_all.xy):
            continue

        pts = kps_all.xy[i].cpu().numpy().astype(int)    # [16, 2]
        vis = kps_all.conf[i].cpu().numpy() \
              if kps_all.conf is not None else np.ones(16)
        pts = kps_all.xy[i].cpu().numpy().astype(int)
        vis = kps_all.conf[i].cpu().numpy() if kps_all.conf is not None else np.ones(len(pts))

        # Skeleton lines (only between confident visible KPs)
        # Skeleton
        for a, b in SKELETON:
            if a < len(pts) and b < len(pts):
                if vis[a] > 0.3 and vis[b] > 0.3:
                    pa = tuple(pts[a])
                    pb = tuple(pts[b])
                    if pa[0] > 0 and pb[0] > 0:
                        mid_col = tuple(
                            int((kp_col(a)[c] + kp_col(b)[c]) / 2)
                            for c in range(3)
                        )
                        cv2.line(out, pa, pb, mid_col, 1, cv2.LINE_AA)
                if vis[a] > 0.25 and vis[b] > 0.25:
                    pa, pb = tuple(pts[a]), tuple(pts[b])
                    if pa[0] > 0 and pb[0] > 0 and pa[1] > 0 and pb[1] > 0:
                if kp_status[a] != "hidden" and kp_status[b] != "hidden":
                    pa, pb = (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1]))
        for a, b in SKELETON_ROBOFLOW_INDICES:
            if a < len(pts) and b < len(pts) and a in ROBOFLOW_TO_ANATOMY and b in ROBOFLOW_TO_ANATOMY:
                if kc[a] > 0.30 and kc[b] > 0.30:
                    pa = (int(round(pts[a][0])), int(round(pts[a][1])))
                    pb = (int(round(pts[b][0])), int(round(pts[b][1])))
                    if pa[0] > 0 and pa[1] > 0 and pb[0] > 0 and pb[1] > 0:
                        ca = ZONE_COLORS[KP_ZONE[a]]
                        cb = ZONE_COLORS[KP_ZONE[b]]
                        ca = ROBOFLOW_TO_ANATOMY[a]["color"]
                        cb = ROBOFLOW_TO_ANATOMY[b]["color"]
                        bone_col = (int((ca[0]+cb[0])/2), int((ca[1]+cb[1])/2), int((ca[2]+cb[2])/2))
                        cv2.line(out, pa, pb, (20, 20, 20), line_thick + 2, cv2.LINE_AA)
                        cv2.line(out, pa, pb, bone_col, line_thick, cv2.LINE_AA)

        # KP dots
        # Points & Names
        # Points & Labels
        for k in range(min(len(pts), 16)):
            px, py = pts[k, 0], pts[k, 1]
            if px == 0 and py == 0:
            status = kp_status[k]
            if status == "hidden":
            if k not in ROBOFLOW_TO_ANATOMY:
                continue
            if float(kc[k]) < 0.30:
                continue

            px, py = int(pts[k, 0]), int(pts[k, 1])
            px, py = int(round(pts[k, 0])), int(round(pts[k, 1]))
            if px <= 0 and py <= 0:
                continue
            v = float(vis[k])
            c = kp_col(k)
            r = 5 if v > 0.6 else 3
            cv2.circle(out, (px, py), r, c, -1 if v > 0.4 else 1, cv2.LINE_AA)
            if show_labels:
                cv2.putText(out, str(k), (px + 5, py - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, c, 1, cv2.LINE_AA)
            v = float(vis[k]) if k < len(vis) else 1.0

            c_zone = ZONE_COLORS[KP_ZONE[k]]
            is_occluded = (status == "occluded_anchor")
            meta = ROBOFLOW_TO_ANATOMY[k]
            c_kp = meta["color"]
            tag_name = meta["name"]

            cv2.circle(out, (px, py), dot_radius + 2, (10, 10, 10), -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), dot_radius, c_zone, -1 if v > 0.35 else 2, cv2.LINE_AA)
            if v > 0.35:
            if is_occluded:
                cv2.circle(out, (px, py), dot_radius + 2, (10, 10, 10), 2, cv2.LINE_AA)
                cv2.circle(out, (px, py), dot_radius, c_zone, 2, cv2.LINE_AA)
            else:
                cv2.circle(out, (px, py), dot_radius + 2, (10, 10, 10), -1, cv2.LINE_AA)
                cv2.circle(out, (px, py), dot_radius, c_zone, -1, cv2.LINE_AA)
                cv2.circle(out, (px, py), max(1, int(dot_radius * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), dot_radius, c_kp, -1, cv2.LINE_AA)
            cv2.circle(out, (px, py), max(1, int(dot_radius * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)

            if name_mode != "none":
                name_str = KP_NAMES_FULL[k] if name_mode == "full" else KP_NAMES_SHORT[k]
                (ntw, nth), _ = cv2.getTextSize(name_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
                tx = max(4, min(w - ntw - 6, px + int(8 * scale)))
                ty = max(nth + 4, min(h - 6, py - int(6 * scale)))
            tag_name = f"{KP_NAMES_FULL[k]} [OCC]" if is_occluded else KP_NAMES_FULL[k]
            (ntw, nth), _ = cv2.getTextSize(tag_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            tx = max(4, min(w - ntw - 6, px + int(8 * scale)))
            ty = max(nth + 4, min(h - 6, py - int(6 * scale)))
            tx = max(4, min(w - ntw - 6, px + int(10 * scale)))
            ty = max(nth + 4, min(h - 6, py - int(4 * scale)))

                pad = int(2 * scale)
                cv2.rectangle(out, (tx - pad, ty - nth - pad), (tx + ntw + pad, ty + pad), (20, 20, 20), -1)
                cv2.rectangle(out, (tx - pad, ty - nth - pad), (tx + ntw + pad, ty + pad), c_zone, 1)
                cv2.putText(out, name_str, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
            cv2.line(out, (px, py), (tx, ty - int(nth/2)), (30, 30, 30), 2, cv2.LINE_AA)
            cv2.line(out, (px, py), (tx, ty - int(nth/2)), c_kp, 1, cv2.LINE_AA)

            pad = int(2 * scale)
            cv2.rectangle(out, (tx - pad, ty - nth - pad), (tx + ntw + pad, ty + pad), (20, 20, 20), -1)
            cv2.rectangle(out, (tx - pad, ty - nth - pad), (tx + ntw + pad, ty + pad), c_zone, 1)
            text_col = (180, 220, 255) if is_occluded else (255, 255, 255)
            cv2.putText(out, tag_name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_col, font_thick, cv2.LINE_AA)
            cv2.rectangle(out, (tx - pad, ty - nth - pad), (tx + ntw + pad, ty + pad), c_kp, 1)
            cv2.putText(out, tag_name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    return out


def draw_overlay(frame: np.ndarray, fps: float, inf_ms: float,
                 n_left: int, n_right: int, show_labels: bool) -> np.ndarray:
    """Draw HUD overlay on top-left corner."""
def draw_hud(frame: np.ndarray, fps: float, inf_ms: float, n_l: int, n_r: int, name_mode: str):
    h, w = frame.shape[:2]
    scale = max(1.0, min(w, h) / 720.0)

    # Semi-transparent panel
    panel_w = int(320 * scale)
    panel_h = int(140 * scale)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    lines = [
        (f"FPS: {fps:.1f}", (0, 255, 0)),
        (f"Inf: {inf_ms:.1f}ms  budget<12ms", (255, 220, 0) if inf_ms < 12 else (0, 100, 255)),
        (f"Left : {n_left} | Right: {n_right}", (200, 200, 200)),
        (f"Labels: {'ON' if show_labels else 'OFF'}  [H=toggle]", (150, 150, 150)),
        ("Q=Quit  S=Save  R=Reset FPS", (120, 120, 120)),
        (f"FPS: {fps:.1f}  ({inf_ms:.1f}ms)", (50, 255, 50)),
        (f"Budget (Stage A): {'PASS (<12ms)' if inf_ms < 12 else 'GPU (~15ms)'}", (0, 220, 255) if inf_ms < 12 else (50, 180, 255)),
        (f"Detected: L={n_l} | R={n_r}", (220, 220, 220)),
        (f"Labels: {name_mode.upper()}  [Press N to cycle]", (255, 200, 100)),
        ("Keys: Q=Quit  S=Save  R=Reset", (160, 160, 160)),
    ]
    for row, (text, col) in enumerate(lines):
        cv2.putText(frame, text, (8, 22 + row * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return frame
    for row, (t, col) in enumerate(lines):
        cv2.putText(frame, t, (int(10 * scale), int((22 + row * 24) * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52 * scale, col, max(1, int(1.2 * scale)), cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
def parse_args():
    p = argparse.ArgumentParser(description="Stage A live camera test.")
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--camera", type=int, default=0,
                   help="Camera index (0=default webcam)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--imgsz",  type=int, default=320)
    p.add_argument("--device", default="0")
    p = argparse.ArgumentParser(description="Stage A live camera test with postprocessing.")
    p = argparse.ArgumentParser(description="Stage A live camera test with verified mapping.")
    p.add_argument("--model",        default=DEFAULT_MODEL)
    p.add_argument("--camera",       type=int, default=0)
    p.add_argument("--box-conf",     type=float, default=0.30)
    p.add_argument("--cross-iou",    type=float, default=0.35)
    p.add_argument("--kp-conf",       type=float, default=0.45)
    p.add_argument("--imgsz",        type=int, default=320)
    p.add_argument("--device",       default="0")
    return p.parse_args()


def main() -> None:
def main():
    args = parse_args()
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

    from ultralytics import YOLO

    print(f"\n{'='*55}")
    print("  Stage A — Live Camera Test")
    print(f"{'='*55}")
    print("\n" + "=" * 55)
    print("      STAGE A — LIVE CAMERA INFERENCE")
    print("      STAGE A — LIVE CAMERA INFERENCE (CLEANED)")
    print("      STAGE A — LIVE CAMERA (ANATOMICALLY VERIFIED)")
    print("=" * 55)
    print(f"  Model  : {args.model}")
    print(f"  Camera : {args.camera}")
    print(f"  Conf   : {args.conf}")
    print(f"  Device : {args.device}")
    print(f"{'='*55}")
    print("\n  Controls: Q=Quit  S=Save  R=Reset  H=Toggle labels\n")
    print(f"  Model       : {args.model}")
    print(f"  Camera      : {args.camera}")
    print(f"  Box Conf    : {args.box_conf}")
    print(f"  Cross-NMS   : {args.cross_iou}")
    print("=" * 55)
    print("\nControls: Q=Quit | S=Save Snapshot | N=Toggle Names | R=Reset\n")
    print("\nControls: Q=Quit | S=Save Snapshot | R=Reset\n")

    if not os.path.exists(args.model):
        print(f"[ERROR] Model not found: {args.model}")
        print("  Train: python src/models/stage_a/train_stage_a.py")
        sys.exit(1)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}.")
        print(f"[ERROR] Could not open webcam index {args.camera}")
        sys.exit(1)

    # Try for 1280×720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps_window  = deque(maxlen=30)
    show_labels = False
    snap_count  = 0
    fps_win = deque(maxlen=30)
    name_modes = ["full", "short", "none"]
    mode_idx = 0
    snap_count = 0

    print(f"  Camera opened: "
          f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}×"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("  Warming up model...")

    # ── Warm-up pass ──────────────────────────────────────────────────────────
    ret, frame0 = cap.read()
    ret, f0 = cap.read()
    if ret:
        model.predict(source=frame0, imgsz=args.imgsz,
                      conf=args.conf, device=args.device, verbose=False)
    print("  Ready! Starting live detection.\n")
        model.predict(f0, imgsz=args.imgsz, device=args.device, verbose=False)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("  Camera read failed — stopping.")
            break

        t0 = time.perf_counter()
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        inf_ms = (time.perf_counter() - t0) * 1000
        fps_window.append(1000.0 / inf_ms)
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)
        raw_results = model.predict(frame, imgsz=args.imgsz, conf=0.15, device=args.device, verbose=False)
        inf_ms = (time.perf_counter() - t0) * 1000.0
        fps_win.append(1000.0 / max(0.1, inf_ms))

        result  = results[0]
        n_left  = n_right = 0
        cleaned_feet = postprocess_stage_a(
            raw_results[0],
            box_conf_thresh=args.box_conf,
            cross_class_iou_thresh=args.cross_iou,
            min_kp_conf=args.kp_conf,
            min_kp_conf=0.30,
            max_feet=2
        )

        result = results[0]
        n_l = n_r = 0
        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls[0])
                if float(box.conf[0]) >= args.conf:
                    if cls == 0:
                        n_left += 1
                    elif cls == 1:
                        n_right += 1
            for b in result.boxes:
                if float(b.conf[0]) >= args.conf:
                    if int(b.cls[0]) == 0:
                        n_l += 1
                    else:
                        n_r += 1
        n_l = sum(1 for f in cleaned_feet if f["cls"] == 0)
        n_r = sum(1 for f in cleaned_feet if f["cls"] == 1)

        annotated = draw_frame(frame, result, args.conf, show_labels)
        avg_fps   = float(np.mean(fps_window))
        draw_overlay(annotated, avg_fps, inf_ms, n_left, n_right, show_labels)
        cur_mode = name_modes[mode_idx]
        annotated = draw_frame(frame, result, args.conf, cur_mode)
        annotated = draw_live_cleaned(frame, cleaned_feet)
        annotated = draw_live_verified(frame, cleaned_feet)
        avg_fps = float(np.mean(fps_win)) if len(fps_win) > 0 else 0.0
        draw_hud(annotated, avg_fps, inf_ms, n_l, n_r, cur_mode)

        cv2.imshow("Stage A — Live Foot Detection  (Q=Quit)", annotated)
        # HUD
        scale = max(1.0, min(frame.shape[1], frame.shape[0]) / 720.0)
        cv2.putText(annotated, f"FPS: {avg_fps:.1f} ({inf_ms:.1f}ms)", (int(10 * scale), int(30 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (50, 255, 50), max(1, int(1.5 * scale)), cv2.LINE_AA)
        cv2.putText(annotated, f"Clean Feet: L={n_l} R={n_r}", (int(10 * scale), int(60 * scale)),
        cv2.putText(annotated, f"Feet: L={n_l} R={n_r}", (int(10 * scale), int(60 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65 * scale, (255, 255, 255), max(1, int(1.5 * scale)), cv2.LINE_AA)

        cv2.imshow("Shoes_VTO Stage A — Real-Time Foot Perception", annotated)
        cv2.imshow("Shoes_VTO Stage A — Real-Time Clean Foot Detection", annotated)
        cv2.imshow("Shoes_VTO Stage A — Verified Foot Keypoints", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or ESC
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("s"), ord("S")):
            snap_path = os.path.join(SNAPSHOTS_DIR, f"snap_{snap_count:04d}.png")
            cv2.imwrite(snap_path, annotated)
            print(f"  Snapshot saved: {snap_path}")
            p = os.path.join(SNAPSHOTS_DIR, f"live_snap_{snap_count:04d}.png")
            p = os.path.join(SNAPSHOTS_DIR, f"live_clean_snap_{snap_count:04d}.png")
            p = os.path.join(SNAPSHOTS_DIR, f"verified_snap_{snap_count:04d}.png")
            cv2.imwrite(p, annotated)
            print(f"  Snapshot saved: {p}")
            snap_count += 1
        elif key in (ord("n"), ord("N")):
            mode_idx = (mode_idx + 1) % len(name_modes)
            print(f"  Label mode: {name_modes[mode_idx].upper()}")
        elif key in (ord("r"), ord("R")):
            fps_window.clear()
            print("  FPS counter reset.")
        elif key in (ord("h"), ord("H")):
            show_labels = not show_labels
            print(f"  KP labels: {'ON' if show_labels else 'OFF'}")
            fps_win.clear()

    cap.release()
    cv2.destroyAllWindows()

    if len(fps_window) > 0:
        print(f"\n  Session avg FPS: {float(np.mean(fps_window)):.1f}")
    print("  Done.\n")


if __name__ == "__main__":
    main()

