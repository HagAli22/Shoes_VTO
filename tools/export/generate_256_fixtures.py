#!/usr/bin/env python3
"""
generate_256_fixtures.py
────────────────────────
Generates golden regression fixtures for the Stage A 256 Mobile delivery package:
1. Scans deliverables/stage_a_256_13kp/fixtures/source-frames/ for all images.
2. Inters with stage-a-256-13kp-fp32.onnx at 256x256 letterbox.
3. Saves raw output tensors [1, 45, 1344] (.npy) in fixtures/raw-output-tensors/.
4. Generates golden expected-decoded.json using the frozen 13-KP schema + 15-KP derivation.
5. Generates crop-transform-cases.json.
6. Generates annotated visualization images in fixtures/source-frames-output/.
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.stage_a.postprocess import derive_secondary_keypoints_15kp

PACKAGE_DIR = PROJECT_ROOT / "deliverables" / "stage_a_256_13kp"
MODEL_PATH = PACKAGE_DIR / "models" / "stage-a-256-13kp-fp32.onnx"
FIXTURES_DIR = PACKAGE_DIR / "fixtures"
SRC_FRAMES_DIR = FIXTURES_DIR / "source-frames"
RAW_TENSORS_DIR = FIXTURES_DIR / "raw-output-tensors"
ANNOTATED_OUTPUT_DIR = FIXTURES_DIR / "source-frames-output"
EXPECTED_DECODED_PATH = FIXTURES_DIR / "expected-decoded.json"
TRANSFORM_CASES_PATH = FIXTURES_DIR / "crop-transform-cases.json"

RAW_TENSORS_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEYPOINT_NAMES_13 = [
    "toe_ground",        # Index 0
    "heel_back",         # Index 1
    "heel_ground",       # Index 2
    "ball_medial",       # Index 3
    "ball_lateral",      # Index 4
    "instep_top",        # Index 5
    "arch_medial",       # Index 6
    "midfoot_lateral",   # Index 7
    "malleolus_medial",  # Index 8
    "malleolus_lateral", # Index 9
    "toe_tip",           # Index 10
    "throat",            # Index 11
    "achilles"           # Index 12
]

SKELETON_PAIRS_13 = [
    (2, 1), (2, 6), (6, 3), (3, 10), (10, 0), (0, 4), (4, 7), (7, 2),
    (10, 5), (5, 11),
    (8, 11), (9, 11), (8, 12), (9, 12), (1, 12)
]

COLOR_LEFT = {
    "bbox": (255, 191, 0),       # Cyan / Deep Sky Blue (BGR)
    "skeleton": (255, 140, 0),
    "kpt_vis": (0, 255, 255),    # Yellow
    "kpt_occ": (0, 200, 200),
    "name": "Left Foot"
}
COLOR_RIGHT = {
    "bbox": (180, 105, 255),     # Pink / Magenta (BGR)
    "skeleton": (147, 20, 255),
    "kpt_vis": (0, 255, 128),    # Light Green
    "kpt_occ": (0, 180, 90),
    "name": "Right Foot"
}

def letterbox(img, target_size=256):
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_x = (target_size - new_w) / 2.0
    pad_y = (target_size - new_h) / 2.0
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return canvas, scale, pad_x, pad_y

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def nms_per_class(candidates, iou_threshold=0.45):
    by_class = {0: [], 1: []}
    for c in candidates:
        by_class[c["class_id"]].append(c)

    results = []
    for cls_id in [0, 1]:
        dets = sorted(by_class[cls_id], key=lambda x: x["confidence"], reverse=True)
        kept = []
        for cand in dets:
            overlap = False
            for k in kept:
                if compute_iou(cand["bbox"], k["bbox"]) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                kept.append(cand)
        results.extend(kept)
    return results

def deduplicate_physical_feet(candidates, iou_threshold=0.60):
    sorted_dets = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    kept = []
    for cand in sorted_dets:
        overlap = False
        for k in kept:
            if compute_iou(cand["bbox"], k["bbox"]) > iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(cand)
    return kept

def determine_geometric_chirality_13kp(keypoints):
    toe = keypoints[10]       # toe_tip
    ball_lat = keypoints[4]   # ball_lateral
    ball_med = keypoints[3]   # ball_medial

    # Fallback rear anchor: heel_ground is preferred over heel_back top-down
    if keypoints[2]["visibility"] > 0.20:
        heel = keypoints[2]
    elif keypoints[1]["visibility"] > 0.20:
        heel = keypoints[1]
    else:
        heel = {
            "x": 0.5 * (keypoints[8]["x"] + keypoints[9]["x"]),
            "y": 0.5 * (keypoints[8]["y"] + keypoints[9]["y"]),
            "visibility": 0.5 * (keypoints[8]["visibility"] + keypoints[9]["visibility"])
        }

    v_axis_x = toe["x"] - heel["x"]
    v_axis_y = toe["y"] - heel["y"]

    v_lat_x = ball_lat["x"] - heel["x"]
    v_lat_y = ball_lat["y"] - heel["y"]
    cross_lat = v_axis_x * v_lat_y - v_axis_y * v_lat_x

    v_med_x = ball_med["x"] - heel["x"]
    v_med_y = ball_med["y"] - heel["y"]
    cross_med = v_axis_x * v_med_y - v_axis_y * v_med_x

    diff = cross_lat - cross_med
    return 0 if diff < 0 else 1

def draw_annotations(frame, detections, case_id=""):
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Draw header banner
    banner_h = 36
    cv2.rectangle(canvas, (0, 0), (w, banner_h), (25, 25, 25), -1)
    title_text = f"Stage A 256 Mobile Fixture: {case_id} ({len(detections)} feet detected)"
    cv2.putText(canvas, title_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

    for det in detections:
        cls_id = det["class_id"]
        cfg = COLOR_LEFT if cls_id == 0 else COLOR_RIGHT
        bx = [int(round(v)) for v in det["bbox"]]
        kpts = det["keypoints_13"]

        # Bounding box
        cv2.rectangle(canvas, (bx[0], bx[1]), (bx[2], bx[3]), cfg["bbox"], 2)

        # Badge
        badge_text = f"{cfg['name']} {det['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(canvas, (bx[0], max(0, bx[1] - th - 8)), (bx[0] + tw + 8, bx[1]), cfg["bbox"], -1)
        cv2.putText(canvas, badge_text, (bx[0] + 4, max(th + 2, bx[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # Skeleton
        for i1, i2 in SKELETON_PAIRS_13:
            p1, p2 = kpts[i1], kpts[i2]
            if p1["visibility"] >= 0.15 and p2["visibility"] >= 0.15:
                pt1 = (int(round(p1["x"])), int(round(p1["y"])))
                pt2 = (int(round(p2["x"])), int(round(p2["y"])))
                thick = 2 if (p1["visibility"] >= 0.35 and p2["visibility"] >= 0.35) else 1
                cv2.line(canvas, pt1, pt2, cfg["skeleton"], thick, cv2.LINE_AA)

        # Keypoints
        for kp in kpts:
            vis = kp["visibility"]
            if vis < 0.15:
                continue
            pt = (int(round(kp["x"])), int(round(kp["y"])))
            if vis >= 0.35:
                cv2.circle(canvas, pt, 4, cfg["kpt_vis"], -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, 4, (0, 0, 0), 1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pt, 3, cfg["kpt_occ"], 1, cv2.LINE_AA)

            # Draw small index
            cv2.putText(canvas, str(kp["index"]), (pt[0] + 4, pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas

def main():
    print(f"[*] Loading ONNX session for 256 Model: {MODEL_PATH}")
    sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # Collect all image files in source-frames/
    valid_exts = {".png", ".jpg", ".jpeg"}
    src_files = [f for f in sorted(SRC_FRAMES_DIR.iterdir()) if f.suffix.lower() in valid_exts]
    print(f"[*] Found {len(src_files)} image files in {SRC_FRAMES_DIR}:")
    for f in src_files:
        print(f"    - {f.name}")

    expected_decoded = {}
    crop_transforms = []

    # Track valid case IDs to clean up orphaned raw tensors
    valid_stems = set()

    for img_path in src_files:
        case_id = img_path.stem
        valid_stems.add(case_id)
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[!] Warning: Could not read {img_path}")
            continue

        orig_h, orig_w = frame.shape[:2]

        # 1. Letterbox Preprocessing
        canvas, scale, pad_x, pad_y = letterbox(frame, target_size=256)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None, ...]

        # 2. Run ONNX Inference
        raw_out = sess.run([out_name], {inp_name: blob})[0]  # shape [1, 45, 1344]
        npy_path = RAW_TENSORS_DIR / f"{case_id}.npy"
        np.save(npy_path, raw_out)

        preds = raw_out[0]  # [45, 1344]
        candidates = []

        for a in range(preds.shape[1]):
            raw_s0 = float(preds[4, a])
            raw_s1 = float(preds[5, a])
            s_left = raw_s0 if (0.0 <= raw_s0 <= 1.0) else sigmoid(raw_s0)
            s_right = raw_s1 if (0.0 <= raw_s1 <= 1.0) else sigmoid(raw_s1)
            max_conf = max(s_left, s_right)
            if max_conf < 0.25:
                continue

            cls_raw = 0 if s_left >= s_right else 1
            cx, cy, w, h = preds[0:4, a]
            x1 = (cx - w / 2.0 - pad_x) / scale
            y1 = (cy - h / 2.0 - pad_y) / scale
            x2 = (cx + w / 2.0 - pad_x) / scale
            y2 = (cy + h / 2.0 - pad_y) / scale

            # Clip box to frame boundaries
            x1 = max(0.0, min(float(orig_w), x1))
            y1 = max(0.0, min(float(orig_h), y1))
            x2 = max(0.0, min(float(orig_w), x2))
            y2 = max(0.0, min(float(orig_h), y2))

            kpts13 = []
            for k in range(13):
                kx = (preds[6 + k * 3, a] - pad_x) / scale
                ky = (preds[6 + k * 3 + 1, a] - pad_y) / scale
                raw_kv = float(preds[6 + k * 3 + 2, a])
                kv = raw_kv if (0.0 <= raw_kv <= 1.0) else sigmoid(raw_kv)
                kpts13.append({
                    "index": k,
                    "name": KEYPOINT_NAMES_13[k],
                    "x": float(round(kx, 2)),
                    "y": float(round(ky, 2)),
                    "visibility": float(round(kv, 4))
                })

            candidates.append({
                "bbox": [float(round(v, 2)) for v in [x1, y1, x2, y2]],
                "class_id": cls_raw,
                "confidence": float(round(max_conf, 4)),
                "keypoints": kpts13
            })

        # 3. NMS and Deduplication
        filtered = nms_per_class(candidates, iou_threshold=0.45)
        deduped = deduplicate_physical_feet(filtered, iou_threshold=0.60)

        # 4. Deterministic Geometric Chirality & 15-KP Derivation
        final_case_dets = []
        for det in deduped:
            geom_cls = determine_geometric_chirality_13kp(det["keypoints"])
            det["class_id"] = geom_cls
            det["class_name"] = "left_foot" if geom_cls == 0 else "right_foot"

            kpts15 = derive_secondary_keypoints_15kp(det["keypoints"], bbox=det["bbox"])
            kpts15_clean = [{
                "index": kp["index"],
                "name": kp["name"],
                "x": float(round(kp["x"], 2)),
                "y": float(round(kp["y"], 2)),
                "visibility": float(round(kp["visibility"], 4))
            } for kp in kpts15]

            final_case_dets.append({
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "keypoints_13": det["keypoints"],
                "keypoints_15_derived": kpts15_clean
            })

        expected_decoded[case_id] = {
            "image_width": orig_w,
            "image_height": orig_h,
            "letterbox_scale": float(round(scale, 6)),
            "pad_x": float(round(pad_x, 2)),
            "pad_y": float(round(pad_y, 2)),
            "detections": final_case_dets
        }

        crop_transforms.append({
            "case_id": case_id,
            "orig_size": [orig_w, orig_h],
            "target_size": 256,
            "scale": float(round(scale, 6)),
            "pad": [float(round(pad_x, 2)), float(round(pad_y, 2))]
        })

        # 5. Render & Save Annotated Output Image
        annotated_img = draw_annotations(frame, final_case_dets, case_id=case_id)
        out_img_path = ANNOTATED_OUTPUT_DIR / f"{case_id}_annotated.png"
        cv2.imwrite(str(out_img_path), annotated_img)

        print(f"  [OK] {case_id} ({orig_w}x{orig_h}) -> {len(final_case_dets)} feet. Saved tensor & annotated image.")

    # Remove any orphan tensors not in valid_stems
    for npy_f in RAW_TENSORS_DIR.glob("*.npy"):
        if npy_f.stem not in valid_stems:
            print(f"  [-] Removing orphaned tensor: {npy_f.name}")
            npy_f.unlink()

    # Save expected-decoded.json
    with open(EXPECTED_DECODED_PATH, "w", encoding="utf-8") as f:
        json.dump(expected_decoded, f, indent=2)
    print(f"\n[OK] Saved golden expected-decoded.json ({len(expected_decoded)} cases)")

    # Save crop-transform-cases.json
    with open(TRANSFORM_CASES_PATH, "w", encoding="utf-8") as f:
        json.dump(crop_transforms, f, indent=2)
    print(f"[OK] Saved crop-transform-cases.json ({len(crop_transforms)} cases)")
    print(f"[OK] All annotated images saved in: {ANNOTATED_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
