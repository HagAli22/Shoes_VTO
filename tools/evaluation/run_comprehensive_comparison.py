#!/usr/bin/env python3
"""
run_comprehensive_comparison.py
───────────────────────────────
Rigorous, reproducible comparison between INT8 and FP16 models:
  Test 1: Video speed, latency distribution, end-to-end FPS, and keypoint jitter.
  Test 2: Keypoint accuracy (NME, PCK@0.05, PCK@0.10, PCK@5px, PCK@10px) against
          ground-truth on 500 randomly sampled held-out images from data/shuffled_v4_2.
"""

import os
import sys
import time
import math
import json
import random
import hashlib
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

# Add demo directory for decoder & visualizer utilities
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = PROJECT_ROOT / "deliverables" / "stage_a_256_13kp" / "demo"
sys.path.insert(0, str(DEMO_DIR))

from run_video_demo import (
    VideoInferencePipeline,
    deduplicate_physical_feet,
    determine_geometric_chirality_13kp,
    KeypointSmoother,
    draw_visuals,
    draw_hud,
)

KEYPOINT_NAMES_13 = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "instep_top", "arch_medial", "midfoot_lateral",
    "malleolus_medial", "malleolus_lateral", "toe_tip", "throat", "achilles"
]

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "deliverables" / "stage_a_256_13kp" / "models"
MODEL_FP16_PATH = MODELS_DIR / "stage-a-256-13kp-fp16.onnx"
MODEL_INT8_PATH = MODELS_DIR / "stage-a-256-13kp-int8.onnx"

VIDEO_PATH = DEMO_DIR / "collection_defferent_videos" / "original" / "6950859-hd_1920_1080_25fps.mp4"
DATA_DIR = PROJECT_ROOT / "data" / "shuffled_v4_2"
REPORTS_DIR = PROJECT_ROOT / "deliverables" / "stage_a_256_13kp" / "reports"
VIZ_DIR = REPORTS_DIR / "visualizations"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── Mapping 13 Model Keypoints -> 16 Ground Truth Keypoints ───────────────────
# In shuffled_v4_2, 16 keypoints are:
# 0: toe_ground, 1: heel_back, 2: heel_ground, 3: ball_medial, 4: ball_lateral,
# 5: ball_top, 6: instep_top, 7: arch_medial, 8: midfoot_lateral,
# 9: malleolus_medial, 10: malleolus_lateral, 11: toe_tip, 12: ankle_center,
# 13: throat, 14: achilles, 15: shin_mid
KP13_TO_GT16 = {
    0: 0,   # toe_ground
    1: 1,   # heel_back
    2: 2,   # heel_ground
    3: 3,   # ball_medial
    4: 4,   # ball_lateral
    5: 6,   # instep_top
    6: 7,   # arch_medial
    7: 8,   # midfoot_lateral
    8: 9,   # malleolus_medial
    9: 10,  # malleolus_lateral
    10: 11, # toe_tip
    11: 13, # throat
    12: 14  # achilles
}

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_box_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Video Speed, Latency Distribution & Keypoint Jitter
# ══════════════════════════════════════════════════════════════════════════════
def run_test1_video_comparison():
    print("\n" + "=" * 70)
    print("TEST 1: VIDEO SPEED, LATENCY DISTRIBUTION & KEYPOINT JITTER")
    print("=" * 70)
    print(f"Target Video: {VIDEO_PATH.name}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"Video Specs: {w}x{h} @ {fps_in:.1f} FPS | Total Frames: {total_frames}")

    results = {}

    for model_name, model_path in [("FP16", MODEL_FP16_PATH), ("INT8", MODEL_INT8_PATH)]:
        print(f"\n[*] Benchmarking {model_name} Model: {model_path.name}")
        file_size_mb = model_path.stat().st_size / 1e6
        file_sha = sha256_file(model_path)
        print(f"    Size: {file_size_mb:.2f} MB | SHA-256: {file_sha[:16]}...")

        # Initialize pipeline with explicit CPU provider for fairness
        pipeline = VideoInferencePipeline(
            model_path=str(model_path),
            input_size=256,
            conf_thresh=0.25,
            nms_thresh=0.45
        )

        smoothers = {
            0: KeypointSmoother(num_keypoints=13),
            1: KeypointSmoother(num_keypoints=13)
        }

        cap = cv2.VideoCapture(str(VIDEO_PATH))
        frame_idx = 0
        per_frame_data = []

        # Previous frame keypoints for jitter calculation: foot_cls -> keypoints array
        prev_raw_kpts = {}
        prev_filt_kpts = {}
        prev_box_diag = {}

        raw_jitters = []
        filt_jitters = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            now = frame_idx / fps_in

            # Time end-to-end frame processing
            t_e2e_start = time.perf_counter()

            # 1. Inference
            dets, lat_infer = pipeline.infer(frame)

            # 2. Deduplication
            dets = deduplicate_physical_feet(dets, iou_thresh=0.60)

            # 3. Geometric Chirality & Tracking / Smoothing
            current_raw_kpts = {}
            current_filt_kpts = {}
            current_box_diag = {}

            processed_dets = []
            frame_raw_jitter_samples = []
            frame_filt_jitter_samples = []

            for det in dets:
                kpts = det["keypoints"]
                if len(kpts) >= 11:
                    geom_cls = determine_geometric_chirality_13kp(kpts)
                    det["class_id"] = geom_cls
                    bx = det["bbox"]
                    diag = float(np.sqrt((bx[2]-bx[0])**2 + (bx[3]-bx[1])**2))
                    current_box_diag[geom_cls] = diag

                    # Store raw coordinates
                    raw_coords = np.array([[kp["x"], kp["y"]] for kp in kpts], dtype=np.float32)
                    visibilities = np.array([kp["visibility"] for kp in kpts], dtype=np.float32)
                    current_raw_kpts[geom_cls] = (raw_coords, visibilities)

                    # Compute smoothed keypoints
                    smoothed_kpts = smoothers[geom_cls].smooth(kpts, now)
                    det["keypoints"] = smoothed_kpts
                    filt_coords = np.array([[kp["x"], kp["y"]] for kp in smoothed_kpts], dtype=np.float32)
                    current_filt_kpts[geom_cls] = filt_coords

                    # Calculate frame-to-frame displacement (jitter) if foot tracked in previous frame
                    if geom_cls in prev_raw_kpts and diag > 10.0:
                        p_raw_prev, _ = prev_raw_kpts[geom_cls]
                        p_filt_prev = prev_filt_kpts[geom_cls]
                        d_prev = prev_box_diag.get(geom_cls, diag)
                        avg_diag = 0.5 * (diag + d_prev)

                        # Jitter normalized by foot box diagonal
                        vis_mask = visibilities > 0.20
                        if np.any(vis_mask):
                            disp_raw = np.linalg.norm(raw_coords[vis_mask] - p_raw_prev[vis_mask], axis=1) / avg_diag
                            disp_filt = np.linalg.norm(filt_coords[vis_mask] - p_filt_prev[vis_mask], axis=1) / avg_diag
                            frame_raw_jitter_samples.extend(disp_raw.tolist())
                            frame_filt_jitter_samples.extend(disp_filt.tolist())

                processed_dets.append(det)

            t_e2e = (time.perf_counter() - t_e2e_start) * 1000.0
            instant_fps = 1000.0 / max(t_e2e, 1.0)

            mean_raw_j = float(np.mean(frame_raw_jitter_samples)) if frame_raw_jitter_samples else np.nan
            mean_filt_j = float(np.mean(frame_filt_jitter_samples)) if frame_filt_jitter_samples else np.nan

            if not np.isnan(mean_raw_j):
                raw_jitters.append(mean_raw_j)
            if not np.isnan(mean_filt_j):
                filt_jitters.append(mean_filt_j)

            per_frame_data.append({
                "model": model_name,
                "frame_idx": frame_idx,
                "infer_latency_ms": round(lat_infer, 2),
                "e2e_latency_ms": round(t_e2e, 2),
                "e2e_fps": round(instant_fps, 1),
                "active_feet": len(processed_dets),
                "raw_jitter_pct": round(mean_raw_j * 100, 3) if not np.isnan(mean_raw_j) else "",
                "filt_jitter_pct": round(mean_filt_j * 100, 3) if not np.isnan(mean_filt_j) else ""
            })

            prev_raw_kpts = current_raw_kpts
            prev_filt_kpts = current_filt_kpts
            prev_box_diag = current_box_diag

        cap.release()

        infer_lats = [d["infer_latency_ms"] for d in per_frame_data]
        e2e_lats = [d["e2e_latency_ms"] for d in per_frame_data]
        e2e_fps_list = [d["e2e_fps"] for d in per_frame_data]

        results[model_name] = {
            "per_frame": per_frame_data,
            "file_size_mb": file_size_mb,
            "sha256": file_sha,
            "infer_median_ms": float(np.median(infer_lats)),
            "infer_p95_ms": float(np.percentile(infer_lats, 95)),
            "infer_mean_ms": float(np.mean(infer_lats)),
            "e2e_median_ms": float(np.median(e2e_lats)),
            "e2e_p95_ms": float(np.percentile(e2e_lats, 95)),
            "e2e_fps_mean": float(np.mean(e2e_fps_list)),
            "raw_jitter_pct": float(np.mean(raw_jitters) * 100) if raw_jitters else 0.0,
            "filt_jitter_pct": float(np.mean(filt_jitters) * 100) if filt_jitters else 0.0,
        }

        print(f"  [{model_name}] Infer Latency: Median = {results[model_name]['infer_median_ms']:.2f} ms | P95 = {results[model_name]['infer_p95_ms']:.2f} ms")
        print(f"  [{model_name}] End-to-End: Latency = {results[model_name]['e2e_median_ms']:.2f} ms | Mean FPS = {results[model_name]['e2e_fps_mean']:.1f}")
        print(f"  [{model_name}] Jitter: Raw = {results[model_name]['raw_jitter_pct']:.2f}% | Filtered = {results[model_name]['filt_jitter_pct']:.2f}%")

    # Export CSV
    csv_path = REPORTS_DIR / "video_speed_stability_measurements.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("model,frame_idx,infer_latency_ms,e2e_latency_ms,e2e_fps,active_feet,raw_jitter_pct,filt_jitter_pct\n")
        for m in ["FP16", "INT8"]:
            for d in results[m]["per_frame"]:
                f.write(f"{d['model']},{d['frame_idx']},{d['infer_latency_ms']},{d['e2e_latency_ms']},{d['e2e_fps']},{d['active_feet']},{d['raw_jitter_pct']},{d['filt_jitter_pct']}\n")

    print(f"\n[OK] Video speed & stability CSV saved to: {csv_path}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: Keypoint Accuracy against Ground Truth (500 Samples from shuffled_v4_2)
# ══════════════════════════════════════════════════════════════════════════════
def run_test2_ground_truth_accuracy(num_samples=500, seed=42):
    print("\n" + "=" * 70)
    print(f"TEST 2: KEYPOINT ACCURACY ON {num_samples} GROUND TRUTH SAMPLES")
    print(f"Dataset: {DATA_DIR} | Fixed Random Seed: {seed}")
    print("=" * 70)

    # 1. Discover and sample test & valid pairs
    random.seed(seed)
    candidate_pairs = []
    for split in ["test", "valid"]:
        img_dir = DATA_DIR / split / "images"
        lbl_dir = DATA_DIR / split / "labels"
        for img_p in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists() and lbl_p.stat().st_size > 0:
                candidate_pairs.append((img_p, lbl_p, split))

    print(f"Found {len(candidate_pairs)} eligible held-out image-label pairs.")
    sampled_pairs = random.sample(candidate_pairs, min(num_samples, len(candidate_pairs)))
    print(f"Selected {len(sampled_pairs)} samples for reproducible benchmarking.")

    # 2. Initialize both pipelines
    pipe_fp16 = VideoInferencePipeline(str(MODEL_FP16_PATH), input_size=256, conf_thresh=0.25)
    pipe_int8 = VideoInferencePipeline(str(MODEL_INT8_PATH), input_size=256, conf_thresh=0.25)

    pipelines = {"FP16": pipe_fp16, "INT8": pipe_int8}
    metrics = {
        "FP16": {
            "total_gt_feet": 0, "detected_feet": 0,
            "errors_visible": [], "errors_occluded": [], "errors_all": [],
            "pck05_count": 0, "pck10_count": 0, "pck5px_count": 0, "pck10px_count": 0,
            "per_kpt_errors": {i: [] for i in range(13)},
            "detailed_rows": []
        },
        "INT8": {
            "total_gt_feet": 0, "detected_feet": 0,
            "errors_visible": [], "errors_occluded": [], "errors_all": [],
            "pck05_count": 0, "pck10_count": 0, "pck5px_count": 0, "pck10px_count": 0,
            "per_kpt_errors": {i: [] for i in range(13)},
            "detailed_rows": []
        }
    }

    # Store interesting visualization candidates (clean, dynamic, occluded, failure)
    viz_candidates = []

    for idx, (img_path, lbl_path, split) in enumerate(sampled_pairs, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        h_img, w_img = img_bgr.shape[:2]

        # Parse Ground Truth
        gt_instances = []
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5 + 16 * 3:
                    continue
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                # Convert to pixel bbox [x1, y1, x2, y2]
                bx1 = (cx - bw / 2.0) * w_img
                by1 = (cy - bh / 2.0) * h_img
                bx2 = (cx + bw / 2.0) * w_img
                by2 = (cy + bh / 2.0) * h_img
                bdiag = float(np.sqrt((bx2 - bx1)**2 + (by2 - by1)**2))

                # Parse 16 keypoints
                kpts16 = []
                for kp_i in range(16):
                    kx = float(parts[5 + kp_i * 3]) * w_img
                    ky = float(parts[5 + kp_i * 3 + 1]) * h_img
                    kv = float(parts[5 + kp_i * 3 + 2])
                    kpts16.append({"x": kx, "y": ky, "v": int(round(kv))})

                gt_instances.append({
                    "class_id": cls_id,
                    "bbox": [bx1, by1, bx2, by2],
                    "diag": bdiag,
                    "keypoints": kpts16
                })

        if not gt_instances:
            continue

        # Evaluate both models on this identical frame
        preds_for_viz = {}
        for m_name in ["FP16", "INT8"]:
            pipe = pipelines[m_name]
            dets, _ = pipe.infer(img_bgr)
            dets = deduplicate_physical_feet(dets, iou_thresh=0.60)

            # Apply geometric chirality to detections
            for det in dets:
                if len(det["keypoints"]) >= 11:
                    det["class_id"] = determine_geometric_chirality_13kp(det["keypoints"])

            preds_for_viz[m_name] = dets
            metrics[m_name]["total_gt_feet"] += len(gt_instances)

            # Match detections to GT
            for gt_idx, gt in enumerate(gt_instances):
                best_det = None
                best_iou = -1.0
                for d in dets:
                    iou = compute_box_iou(d["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det = d

                # Accept detection match if IoU > 0.30 (standard detection threshold)
                if best_det is not None and best_iou >= 0.30:
                    metrics[m_name]["detected_feet"] += 1
                    pred_kpts = {kp["index"]: kp for kp in best_det["keypoints"]}

                    for k13, k16 in KP13_TO_GT16.items():
                        gt_pt = gt["keypoints"][k16]
                        v = gt_pt["v"]
                        if v == 0:
                            continue  # unlabeled / outside frame

                        if k13 in pred_kpts:
                            px = pred_kpts[k13]["x"]
                            py = pred_kpts[k13]["y"]
                            gx = gt_pt["x"]
                            gy = gt_pt["y"]

                            dist_px = float(np.sqrt((px - gx)**2 + (py - gy)**2))
                            nme = dist_px / max(gt["diag"], 10.0)

                            metrics[m_name]["errors_all"].append(nme)
                            metrics[m_name]["per_kpt_errors"][k13].append(nme)

                            if v == 2:
                                metrics[m_name]["errors_visible"].append(nme)
                            elif v == 1:
                                metrics[m_name]["errors_occluded"].append(nme)

                            if nme <= 0.05:
                                metrics[m_name]["pck05_count"] += 1
                            if nme <= 0.10:
                                metrics[m_name]["pck10_count"] += 1
                            if dist_px <= 5.0:
                                metrics[m_name]["pck5px_count"] += 1
                            if dist_px <= 10.0:
                                metrics[m_name]["pck10px_count"] += 1

                            metrics[m_name]["detailed_rows"].append({
                                "model": m_name,
                                "image": img_path.name,
                                "split": split,
                                "gt_foot_id": gt_idx,
                                "class_id": gt["class_id"],
                                "kpt_idx_13": k13,
                                "kpt_name": KEYPOINT_NAMES_13[k13],
                                "gt_vis": v,
                                "dist_px": round(dist_px, 2),
                                "nme_pct": round(nme * 100, 3),
                                "box_iou": round(best_iou, 3)
                            })

        # Save potential visualization candidates
        if len(viz_candidates) < 10:
            viz_candidates.append({
                "img_path": img_path,
                "img_bgr": img_bgr,
                "gt_instances": gt_instances,
                "preds": preds_for_viz
            })

        if idx % 100 == 0 or idx == len(sampled_pairs):
            print(f"  Evaluated [{idx}/{len(sampled_pairs)}] images...")

    # Write detailed CSV
    csv_acc_path = REPORTS_DIR / "keypoint_accuracy_measurements.csv"
    with open(csv_acc_path, "w", encoding="utf-8") as f:
        f.write("model,image,split,gt_foot_id,class_id,kpt_idx_13,kpt_name,gt_vis,dist_px,nme_pct,box_iou\n")
        for m in ["FP16", "INT8"]:
            for r in metrics[m]["detailed_rows"]:
                f.write(f"{r['model']},{r['image']},{r['split']},{r['gt_foot_id']},{r['class_id']},{r['kpt_idx_13']},{r['kpt_name']},{r['gt_vis']},{r['dist_px']},{r['nme_pct']},{r['box_iou']}\n")

    print(f"\n[OK] Keypoint accuracy measurements CSV saved to: {csv_acc_path}")

    # Compute final statistics
    acc_summary = {}
    for m in ["FP16", "INT8"]:
        tot_pts = len(metrics[m]["errors_all"])
        det_rate = (metrics[m]["detected_feet"] / max(metrics[m]["total_gt_feet"], 1)) * 100.0
        nme_all = np.mean(metrics[m]["errors_all"]) * 100.0 if tot_pts > 0 else 0.0
        nme_vis = np.mean(metrics[m]["errors_visible"]) * 100.0 if metrics[m]["errors_visible"] else 0.0
        nme_occ = np.mean(metrics[m]["errors_occluded"]) * 100.0 if metrics[m]["errors_occluded"] else 0.0

        pck05 = (metrics[m]["pck05_count"] / max(tot_pts, 1)) * 100.0
        pck10 = (metrics[m]["pck10_count"] / max(tot_pts, 1)) * 100.0
        pck5px = (metrics[m]["pck5px_count"] / max(tot_pts, 1)) * 100.0
        pck10px = (metrics[m]["pck10px_count"] / max(tot_pts, 1)) * 100.0

        per_kpt_stat = {}
        for k13 in range(13):
            k_errs = metrics[m]["per_kpt_errors"][k13]
            per_kpt_stat[KEYPOINT_NAMES_13[k13]] = {
                "nme_pct": float(np.mean(k_errs) * 100) if k_errs else 0.0,
                "pck05": float(np.sum(np.array(k_errs) <= 0.05) / max(len(k_errs), 1) * 100) if k_errs else 0.0
            }

        acc_summary[m] = {
            "total_gt_feet": metrics[m]["total_gt_feet"],
            "detection_rate_pct": det_rate,
            "total_evaluated_keypoints": tot_pts,
            "nme_all_pct": nme_all,
            "nme_visible_pct": nme_vis,
            "nme_occluded_pct": nme_occ,
            "pck05_pct": pck05,
            "pck10_pct": pck10,
            "pck5px_pct": pck5px,
            "pck10px_pct": pck10px,
            "per_kpt": per_kpt_stat
        }

        print(f"\n  [{m} Accuracy Summary]")
        print(f"    Detection Rate: {det_rate:.2f}% ({metrics[m]['detected_feet']}/{metrics[m]['total_gt_feet']})")
        print(f"    NME (All Labeled): {nme_all:.2f}% | Visible (v=2): {nme_vis:.2f}% | Occluded (v=1): {nme_occ:.2f}%")
        print(f"    PCK@0.05 (diag): {pck05:.2f}% | PCK@0.10: {pck10:.2f}%")
        print(f"    PCK@5px: {pck5px:.2f}% | PCK@10px: {pck10px:.2f}%")

    return acc_summary, viz_candidates

# ══════════════════════════════════════════════════════════════════════════════
# PART 3: Side-by-Side Visualizations
# ══════════════════════════════════════════════════════════════════════════════
def generate_side_by_side_visualizations(viz_candidates):
    print("\n" + "=" * 70)
    print("GENERATING SIDE-BY-SIDE VISUALIZATIONS (FP16 vs INT8)")
    print("=" * 70)

    # 1. Generate video side-by-side on target frames: Frame 45 and Frame 131
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    video_frames_to_save = [30, 45, 131, 200]
    frame_map = {}
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        f_idx += 1
        if f_idx in video_frames_to_save:
            frame_map[f_idx] = frame
    cap.release()

    pipe_fp16 = VideoInferencePipeline(str(MODEL_FP16_PATH), input_size=256)
    pipe_int8 = VideoInferencePipeline(str(MODEL_INT8_PATH), input_size=256)

    for f_idx, frame in frame_map.items():
        # FP16
        dets_16, lat_16 = pipe_fp16.infer(frame)
        dets_16 = deduplicate_physical_feet(dets_16)
        for d in dets_16:
            if len(d["keypoints"]) >= 11:
                d["class_id"] = determine_geometric_chirality_13kp(d["keypoints"])
        vis_16 = draw_visuals(frame.copy(), dets_16, show_labels=True)
        cv2.putText(vis_16, f"FP16 ({lat_16:.1f} ms) Frame {f_idx}", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # INT8
        dets_8, lat_8 = pipe_int8.infer(frame)
        dets_8 = deduplicate_physical_feet(dets_8)
        for d in dets_8:
            if len(d["keypoints"]) >= 11:
                d["class_id"] = determine_geometric_chirality_13kp(d["keypoints"])
        vis_8 = draw_visuals(frame.copy(), dets_8, show_labels=True)
        cv2.putText(vis_8, f"INT8 ({lat_8:.1f} ms) Frame {f_idx}", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 128), 2)

        # Stack side by side (resize if 1080p so width is manageable)
        scale = 720.0 / vis_16.shape[0]
        v16_s = cv2.resize(vis_16, (int(vis_16.shape[1] * scale), 720))
        v8_s = cv2.resize(vis_8, (int(vis_8.shape[1] * scale), 720))
        side_by_side = np.hstack([v16_s, v8_s])

        out_viz_path = VIZ_DIR / f"video_frame_{f_idx:03d}_fp16_vs_int8.png"
        cv2.imwrite(str(out_viz_path), side_by_side)
        print(f"  [OK] Saved video side-by-side comparison: {out_viz_path.name}")

    # 2. Generate side-by-side on 4 selected ground-truth dataset samples
    for i, c in enumerate(viz_candidates[:4], 1):
        img_bgr = c["img_bgr"]
        img_name = c["img_path"].stem

        # FP16
        vis_16 = draw_visuals(img_bgr.copy(), c["preds"]["FP16"], show_labels=True)
        cv2.putText(vis_16, f"FP16 Prediction: {img_name[:20]}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # INT8
        vis_8 = draw_visuals(img_bgr.copy(), c["preds"]["INT8"], show_labels=True)
        cv2.putText(vis_8, f"INT8 Prediction: {img_name[:20]}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)

        scale = 512.0 / vis_16.shape[0]
        v16_s = cv2.resize(vis_16, (int(vis_16.shape[1] * scale), 512))
        v8_s = cv2.resize(vis_8, (int(vis_8.shape[1] * scale), 512))
        combined = np.hstack([v16_s, v8_s])

        out_gt_viz = VIZ_DIR / f"dataset_sample_{i:02d}_{img_name[:15]}_comparison.png"
        cv2.imwrite(str(out_gt_viz), combined)
        print(f"  [OK] Saved dataset sample comparison: {out_gt_viz.name}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t_start = time.time()

    # 1. Run Test 1
    t1_results = run_test1_video_comparison()

    # 2. Run Test 2
    t2_results, viz_candidates = run_test2_ground_truth_accuracy(num_samples=500, seed=42)

    # 3. Generate Visualizations
    generate_side_by_side_visualizations(viz_candidates)

    # 4. Save Consolidated Summary JSON
    summary_path = REPORTS_DIR / "int8_vs_fp16_consolidated_summary.json"
    full_summary = {
        "test1_video_stability": {
            "FP16": {k: v for k, v in t1_results["FP16"].items() if k != "per_frame"},
            "INT8": {k: v for k, v in t1_results["INT8"].items() if k != "per_frame"}
        },
        "test2_ground_truth_accuracy": t2_results
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"[SUCCESS] ALL BENCHMARKS COMPLETED IN {total_time:.1f} SECONDS!")
    print(f"Consolidated Summary JSON: {summary_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
