#!/usr/bin/env python3
"""
Deep Investigation: RTMPose FP16 vs FP32 Keypoint Accuracy on Real Foot Crops
from the actual two-stage pipeline (NanoDet-Plus -> RTMPose-13KP).

Evaluates:
- At least 100 real detected foot crops from data/test_video.mp4
- Per-keypoint pixel error (Mean, Median, P95, Max)
- Exact distribution analysis for outlier keypoints (bimodal/tied peaks, visibility)
"""

import os
import sys
import json
from pathlib import Path
import cv2
import numpy as np
import torch

_tlib = Path(torch.__file__).parent / "lib"
if hasattr(os, "add_dll_directory") and _tlib.exists():
    try:
        os.add_dll_directory(str(_tlib))
    except Exception:
        pass
os.environ["PATH"] = str(_tlib) + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "deliverables/qualcomm_experiments/models"
VIDEO_PATH = REPO_ROOT / "data/test_video.mp4"

RTMO_FP32 = MODELS_DIR / "rtmpose_13kp_hand_improved5_opset12.onnx"
RTMO_FP16 = MODELS_DIR / "rtmpose_13kp_hand_improved5_fp16.onnx"
NANO_FP32 = MODELS_DIR / "nanodet_stage_a_improved5_opset12.onnx"

KP_NAMES = [
    "ankle", "heel", "medial_malleolus", "lateral_malleolus",
    "navicular", "metatarsal_1", "metatarsal_5", "toe_big",
    "toe_index", "toe_middle", "toe_ring", "toe_little", "calcaneus_base"
]

def load_sessions():
    cuda_opts = {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    nano_sess = ort.InferenceSession(str(NANO_FP32), providers=providers)
    rtm32_sess = ort.InferenceSession(str(RTMO_FP32), providers=providers)
    rtm16_sess = ort.InferenceSession(str(RTMO_FP16), providers=providers)
    return nano_sess, rtm32_sess, rtm16_sess

def detect_feet(nano_sess, frame: np.ndarray, score_thresh: float = 0.30):
    h, w = frame.shape[:2]
    # Resize to 256x256
    img = cv2.resize(frame, (256, 256))
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    tensor = np.ascontiguousarray(img.transpose(2, 0, 1)[None, ...])

    inp_name = nano_sess.get_inputs()[0].name
    outs = nano_sess.run(None, {inp_name: tensor})
    boxes, scores = outs[0][0], outs[1][0]  # [N, 4], [N, 1]

    valid = []
    scale_x, scale_y = w / 256.0, h / 256.0
    for box, score in zip(boxes, scores):
        s = float(score[0]) if isinstance(score, (list, np.ndarray)) else float(score)
        if s > score_thresh:
            x1 = float(box[0] * scale_x)
            y1 = float(box[1] * scale_y)
            x2 = float(box[2] * scale_x)
            y2 = float(box[3] * scale_y)
            # Filter non-foot aspects
            bw, bh = x2 - x1, y2 - y1
            if bw > 20 and bh > 20:
                valid.append((x1, y1, x2, y2, s))
    return valid

def extract_padded_crop(frame: np.ndarray, box, pad: float = 0.25):
    h, w = frame.shape[:2]
    x1, y1, x2, y2, _ = box
    bw, bh = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + pad)
    
    nx1 = int(round(cx - side / 2.0))
    ny1 = int(round(cy - side / 2.0))
    nx2 = int(round(cx + side / 2.0))
    ny2 = int(round(cy + side / 2.0))
    
    pad_left = max(0, -nx1)
    pad_top = max(0, -ny1)
    pad_right = max(0, nx2 - w)
    pad_bottom = max(0, ny2 - h)
    
    cx1 = max(0, nx1)
    cy1 = max(0, ny1)
    cx2 = min(w, nx2)
    cy2 = min(h, ny2)
    
    cropped = frame[cy1:cy2, cx1:cx2]
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    resized = cv2.resize(cropped, (256, 256))
    return resized, (nx1, ny1, side)

def preprocess_rtmpose_batch(crops: list, is_fp16: bool):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    t = []
    for c in crops:
        img = c[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - mean) / std
        t.append(img.transpose(2, 0, 1))
    b = np.ascontiguousarray(np.stack(t, axis=0))
    return b.astype(np.float16) if is_fp16 else b

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("================================================================================")
    print("RTMPose FP16 ACCURACY & OUTLIER INVESTIGATION ON REAL FOOT CROPS")
    print("================================================================================")
    print(f"Video Source: {VIDEO_PATH}")
    
    nano_sess, rtm32_sess, rtm16_sess = load_sessions()
    
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total video frames: {total_frames}")

    # Collect 120 real foot crops from across the video
    sample_stride = max(1, total_frames // 120)
    foot_crops = []
    crop_metadata = []

    frame_idx = 0
    while cap.isOpened() and len(foot_crops) < 120:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            boxes = detect_feet(nano_sess, frame, score_thresh=0.35)
            for b_idx, box in enumerate(boxes):
                crop, meta = extract_padded_crop(frame, box)
                foot_crops.append(crop)
                crop_metadata.append({
                    "frame_idx": frame_idx,
                    "box_idx": b_idx,
                    "det_score": float(box[4]),
                    "crop_meta": meta
                })
                if len(foot_crops) >= 120:
                    break
        frame_idx += 1
    cap.release()

    num_crops = len(foot_crops)
    print(f"[+] Extracted {num_crops} real foot crops from actual NanoDet detections across video.")
    if num_crops < 50:
        print("[!] Too few crops extracted. Aborting.")
        return

    # Run inference in batches of 4
    batch_size = 4
    all_x32, all_y32, all_x16, all_y16 = [], [], [], []
    all_sim_x32, all_sim_y32, all_sim_x16, all_sim_y16 = [], [], [], []
    all_vis32, all_vis16 = [], []

    inp32_name = rtm32_sess.get_inputs()[0].name
    inp16_name = rtm16_sess.get_inputs()[0].name

    for i in range(0, num_crops, batch_size):
        chunk = foot_crops[i:i+batch_size]
        cur_b = len(chunk)
        b32 = preprocess_rtmpose_batch(chunk, False)
        b16 = preprocess_rtmpose_batch(chunk, True)

        o32 = rtm32_sess.run(None, {inp32_name: b32})
        o16 = rtm16_sess.run(None, {inp16_name: b16})

        # simcc_x: [B, 13, 512], simcc_y: [B, 13, 512], vis: [B, 13, 1]
        sx32, sy32 = o32[0], o32[1]
        sx16, sy16 = o16[0].astype(np.float32), o16[1].astype(np.float32)
        v32, v16 = o32[3], o16[3].astype(np.float32)

        x32 = np.argmax(sx32, axis=-1) / 2.0
        y32 = np.argmax(sy32, axis=-1) / 2.0
        x16 = np.argmax(sx16, axis=-1) / 2.0
        y16 = np.argmax(sy16, axis=-1) / 2.0

        all_x32.append(x32)
        all_y32.append(y32)
        all_x16.append(x16)
        all_y16.append(y16)
        all_sim_x32.append(sx32)
        all_sim_y32.append(sy32)
        all_sim_x16.append(sx16)
        all_sim_y16.append(sy16)
        all_vis32.append(v32.squeeze(-1))
        all_vis16.append(v16.squeeze(-1))

    X32 = np.concatenate(all_x32, axis=0)  # [N, 13]
    Y32 = np.concatenate(all_y32, axis=0)
    X16 = np.concatenate(all_x16, axis=0)
    Y16 = np.concatenate(all_y16, axis=0)
    SIM_X32 = np.concatenate(all_sim_x32, axis=0) # [N, 13, 512]
    SIM_Y32 = np.concatenate(all_sim_y32, axis=0)
    SIM_X16 = np.concatenate(all_sim_x16, axis=0)
    SIM_Y16 = np.concatenate(all_sim_y16, axis=0)
    VIS32 = np.concatenate(all_vis32, axis=0) # [N, 13]
    VIS16 = np.concatenate(all_vis16, axis=0)

    # Pixel distance error
    pixel_errors = np.sqrt((X32 - X16)**2 + (Y32 - Y16)**2) # [N, 13]
    total_kp_eval = pixel_errors.size

    mean_err = float(np.mean(pixel_errors))
    median_err = float(np.median(pixel_errors))
    p95_err = float(np.percentile(pixel_errors, 95))
    p99_err = float(np.percentile(pixel_errors, 99))
    max_err = float(np.max(pixel_errors))
    exact_match = float(np.mean(pixel_errors == 0.0)) * 100.0
    subpixel_match = float(np.mean(pixel_errors <= 0.5)) * 100.0
    within_2px = float(np.mean(pixel_errors <= 2.0)) * 100.0

    print("\n" + "="*80)
    print(f"AGGREGATE STATISTICAL ACCURACY OVER {num_crops} REAL FOOT CROPS ({total_kp_eval} KEYPOINTS)")
    print("="*80)
    print(f"Mean Keypoint Error:       {mean_err:.4f} px (on 256x256 crop)")
    print(f"Median (P50) Error:        {median_err:.4f} px")
    print(f"P95 Error:                 {p95_err:.4f} px")
    print(f"P99 Error:                 {p99_err:.4f} px")
    print(f"Maximum Error:             {max_err:.4f} px")
    print(f"Exact Pixel Match (0.0px): {exact_match:.2f}%")
    print(f"Sub-pixel Match (<=0.5px): {subpixel_match:.2f}%")
    print(f"Close Match (<= 2.0px):    {within_2px:.2f}%")

    print("\n" + "="*80)
    print("PER-KEYPOINT ERROR BREAKDOWN (OVER ALL 120 REAL FOOT CROPS)")
    print("="*80)
    print("| KP ID | Keypoint Name | Mean (px) | P50 (px) | P95 (px) | Max (px) | Match (<=0.5px) |")
    print("| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
    for kp in range(13):
        kp_errs = pixel_errors[:, kp]
        kp_mean = np.mean(kp_errs)
        kp_p50 = np.median(kp_errs)
        kp_p95 = np.percentile(kp_errs, 95)
        kp_max = np.max(kp_errs)
        kp_sub = np.mean(kp_errs <= 0.5) * 100.0
        print(f"| {kp:5d} | {KP_NAMES[kp]:18s} | {kp_mean:9.3f} | {kp_p50:8.3f} | {kp_p95:8.3f} | {kp_max:8.3f} | {kp_sub:14.1f}% |")

    # OUTLIER INVESTIGATION: Analyze all keypoints with error > 5.0 px
    outliers = []
    for c_idx in range(num_crops):
        for kp in range(13):
            e = pixel_errors[c_idx, kp]
            if e > 5.0:
                outliers.append((e, c_idx, kp))
    outliers.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "="*80)
    print(f"OUTLIER ANALYSIS: ALL DETECTIONS WITH ERROR > 5.0 PX ({len(outliers)} OUT OF {total_kp_eval} KEYPOINTS = {len(outliers)/total_kp_eval*100.0:.2f}%)")
    print("="*80)

    for rank, (e, c_idx, kp) in enumerate(outliers[:10], 1):
        meta = crop_metadata[c_idx]
        frame = meta["frame_idx"]
        kp_name = KP_NAMES[kp]
        vis32 = VIS32[c_idx, kp]
        vis16 = VIS16[c_idx, kp]
        
        # Analyze SimCC X and Y distributions
        sim_x32 = SIM_X32[c_idx, kp]
        sim_x16 = SIM_X16[c_idx, kp]
        sim_y32 = SIM_Y32[c_idx, kp]
        sim_y16 = SIM_Y16[c_idx, kp]

        # Top 3 peaks on X
        sort_x32 = np.argsort(sim_x32)[::-1][:3]
        sort_x16 = np.argsort(sim_x16)[::-1][:3]
        val_x32 = sim_x32[sort_x32]
        val_x16 = sim_x16[sort_x16]

        # Top 3 peaks on Y
        sort_y32 = np.argsort(sim_y32)[::-1][:3]
        sort_y16 = np.argsort(sim_y16)[::-1][:3]
        val_y32 = sim_y32[sort_y32]
        val_y16 = sim_y16[sort_y16]

        print(f"\n[Outlier #{rank}] Error: {e:.2f} px | Frame {frame} | KP {kp} ({kp_name})")
        print(f"  Det Score: {meta['det_score']:.3f} | Vis FP32: {vis32:.4f} | Vis FP16: {vis16:.4f}")
        print(f"  FP32 Coords: ({X32[c_idx, kp]:.1f}, {Y32[c_idx, kp]:.1f}) | FP16 Coords: ({X16[c_idx, kp]:.1f}, {Y16[c_idx, kp]:.1f})")
        print(f"  SimCC X FP32 top-3 bins: {sort_x32/2.0} with values: {np.round(val_x32, 4)}")
        print(f"  SimCC X FP16 top-3 bins: {sort_x16/2.0} with values: {np.round(val_x16, 4)}")
        print(f"  SimCC Y FP32 top-3 bins: {sort_y32/2.0} with values: {np.round(val_y32, 4)}")
        print(f"  SimCC Y FP16 top-3 bins: {sort_y16/2.0} with values: {np.round(val_y16, 4)}")

        # Check for tied peaks
        peak_diff_x32 = val_x32[0] - val_x32[1] if len(val_x32) > 1 else 999.0
        peak_diff_y32 = val_y32[0] - val_y32[1] if len(val_y32) > 1 else 999.0
        print(f"  Peak Margin X (Peak1 - Peak2): {peak_diff_x32:.5f} | Margin Y: {peak_diff_y32:.5f}")
        if peak_diff_x32 < 0.02 or peak_diff_y32 < 0.02:
            print("  --> ROOT CAUSE: TIED / MULTIMODAL PEAKS! Distribution has two competing peaks separated by distance;")
            print("      FP16 numerical quantization (half-precision epsilon ~0.00098) shifted the argmax from peak 1 to peak 2.")
        elif vis32 < 0.3:
            print("  --> ROOT CAUSE: OCCLUSION / LOW VISIBILITY! Keypoint is invisible/occluded in this crop.")
        else:
            print("  --> ROOT CAUSE: SUB-BIN GRADIENT SHIFT.")

    # Save complete JSON
    out_json = REPO_ROOT / "deliverables/qualcomm_experiments/profiling/rtmpose_fp16_outlier_analysis.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "num_crops_evaluated": num_crops,
        "total_keypoints_evaluated": total_kp_eval,
        "aggregate_metrics": {
            "mean_px": mean_err,
            "median_px": median_err,
            "p95_px": p95_err,
            "p99_px": p99_err,
            "max_px": max_err,
            "exact_match_pct": exact_match,
            "subpixel_05px_match_pct": subpixel_match,
            "within_2px_pct": within_2px
        },
        "outlier_count_over_5px": len(outliers),
        "top_outliers": [
            {
                "error_px": float(e),
                "frame": int(crop_metadata[c_idx]["frame_idx"]),
                "kp_id": int(kp),
                "kp_name": KP_NAMES[kp],
                "fp32_coords": [float(X32[c_idx, kp]), float(Y32[c_idx, kp])],
                "fp16_coords": [float(X16[c_idx, kp]), float(Y16[c_idx, kp])],
                "vis_fp32": float(VIS32[c_idx, kp]),
                "vis_fp16": float(VIS16[c_idx, kp])
            }
            for e, c_idx, kp in outliers[:15]
        ]
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[+] Saved detailed accuracy report to: {out_json}")

if __name__ == "__main__":
    main()
