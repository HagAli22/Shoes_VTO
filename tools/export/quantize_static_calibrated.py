#!/usr/bin/env python3
"""
tools/export/quantize_static_calibrated.py
──────────────────────────────────────────
Performs Static Post-Training Quantization (QDQ) with Calibration on RTMPose Improved5.
Extracts representative calibration crops from data/test_video.mp4, runs calibration,
generates static QDQ INT8 models, and benchmarks against FP32 and Dynamic INT8.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType
)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> List[int]:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def compute_crop_box(cx: float, cy: float, bw: float, bh: float, crop_padding: float = 0.25) -> Tuple[float, float, float]:
    pad_w = bw * (1.0 + crop_padding * 2.0)
    pad_h = bh * (1.0 + crop_padding * 2.0)
    size = max(pad_w, pad_h, 32.0)
    return cx, cy, size


def get_crop_affine_transform(center_x: float, center_y: float, size: float, out_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    half_size = size / 2.0
    src = np.array([
        [center_x, center_y],
        [center_x, center_y - half_size],
        [center_x + half_size, center_y]
    ], dtype=np.float32)

    half_out = out_size / 2.0
    dst = np.array([
        [half_out, half_out],
        [half_out, 0.0],
        [out_size, half_out]
    ], dtype=np.float32)

    M = cv2.getAffineTransform(src, dst)
    M_inv = cv2.getAffineTransform(dst, src)
    return M, M_inv


def warp_image_crop(image: np.ndarray, M: np.ndarray, out_size: int = 256, border_value=(114, 114, 114)) -> np.ndarray:
    return cv2.warpAffine(image, M, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)


def extract_calibration_crops(video_path: str, nanodet_path: str, max_crops: int = 150) -> List[np.ndarray]:
    print(f"[*] Extracting up to {max_crops} calibration foot crops from {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    det_sess = ort.InferenceSession(nanodet_path, providers=["CPUExecutionProvider"])
    det_in_name = det_sess.get_inputs()[0].name

    crops = []
    frame_idx = 0
    stride = 3  # sample every 3 frames to get visual variety

    while cap.isOpened() and len(crops) < max_crops:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        orig_h, orig_w = frame.shape[:2]
        det_img = cv2.resize(frame, (256, 256))
        det_blob = (det_img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]

        boxes_out, scores_out = det_sess.run(None, {det_in_name: det_blob})
        raw_boxes = boxes_out[0]
        raw_scores = scores_out[0, :, 0]
        mask = raw_scores >= 0.35
        if not np.any(mask):
            continue

        filt_boxes = raw_boxes[mask]
        filt_scores = raw_scores[mask]
        keep = nms_boxes(filt_boxes, filt_scores, iou_thresh=0.45)

        scale_x = orig_w / 256.0
        scale_y = orig_h / 256.0

        for idx in keep[:2]:  # At most 2 feet
            bx = filt_boxes[idx]
            x1 = float(np.clip(bx[0] * scale_x, 0, orig_w))
            y1 = float(np.clip(bx[1] * scale_y, 0, orig_h))
            x2 = float(np.clip(bx[2] * scale_x, 0, orig_w))
            y2 = float(np.clip(bx[3] * scale_y, 0, orig_h))
            bw, bh = x2 - x1, y2 - y1
            if bw < 16 or bh < 16:
                continue

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            crop_cx, crop_cy, size = compute_crop_box(cx, cy, bw, bh, crop_padding=0.25)
            M, _ = get_crop_affine_transform(crop_cx, crop_cy, size, out_size=256)
            crop = warp_image_crop(frame, M, out_size=256)
            crop_blob = (crop.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
            crops.append(crop_blob)
            if len(crops) >= max_crops:
                break

    cap.release()
    print(f"[+] Successfully extracted {len(crops)} foot calibration crops.")
    return crops


class CropCalibrationDataReader(CalibrationDataReader):
    def __init__(self, crops: List[np.ndarray], input_name: str):
        self.crops = crops
        self.input_name = input_name
        self.index = 0

    def get_next(self) -> Dict[str, np.ndarray]:
        if self.index < len(self.crops):
            blob = self.crops[self.index]
            self.index += 1
            return {self.input_name: blob}
        return None

    def rewind(self):
        self.index = 0


def run_static_quantization(fp32_model_path: str, output_model_path: str, calib_crops: List[np.ndarray], quant_format: QuantFormat):
    print(f"\n[*] Running Static Quantization -> {output_model_path}")
    print(f"    QuantFormat: {quant_format}")

    # Determine input name from model
    m = onnx.load(fp32_model_path)
    in_name = m.graph.input[0].name

    data_reader = CropCalibrationDataReader(calib_crops, in_name)

    quantize_static(
        model_input=fp32_model_path,
        model_output=output_model_path,
        calibration_data_reader=data_reader,
        quant_format=quant_format,
        per_channel=False,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        calibrate_method=CalibrationMethod.MinMax
    )

    size_mb = os.path.getsize(output_model_path) / (1024 * 1024)
    print(f"[+] Static quantized model saved: {output_model_path} ({size_mb:.2f} MB)")


def benchmark_model(model_path: str, calib_crops: List[np.ndarray], num_warmup: int = 15, num_runs: int = 80, threads: int = 4) -> Dict[str, Any]:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    # Warmup
    for i in range(min(num_warmup, len(calib_crops))):
        sess.run(None, {in_name: calib_crops[i % len(calib_crops)]})

    # Benchmark
    latencies = []
    for i in range(num_runs):
        blob = calib_crops[i % len(calib_crops)]
        t0 = time.perf_counter()
        sess.run(None, {in_name: blob})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    m = onnx.load(model_path)
    from collections import Counter
    op_counts = Counter(n.op_type for n in m.graph.node)

    return {
        "model": Path(model_path).name,
        "size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "total_nodes": len(m.graph.node),
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.median(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "fps": 1000.0 / float(np.mean(latencies)),
        "top_ops": op_counts.most_common(6)
    }


def main():
    print("=" * 80)
    print("  STATIC CALIBRATED QUANTIZATION (QDQ & QOPERATOR) vs DYNAMIC INT8 & FP32")
    print("=" * 80)

    video_path = "data/test_video.mp4"
    nanodet_path = "tools/mobile_web_demo/public/models/nanodet-improved5-fp32.onnx"
    fp32_pose_path = "tools/mobile_web_demo/public/models/rtmpose-13kp-hand-improved5-fp32.onnx"
    dyn_int8_pose_path = "tools/mobile_web_demo/public/models/rtmpose-13kp-hand-improved5-int8.onnx"

    out_qdq_path = "tools/mobile_web_demo/public/models/rtmpose-13kp-hand-improved5-static-qdq-int8.onnx"
    out_qop_path = "tools/mobile_web_demo/public/models/rtmpose-13kp-hand-improved5-static-qop-int8.onnx"

    crops = extract_calibration_crops(video_path, nanodet_path, max_crops=120)

    # 1. Run Static QDQ Quantization
    run_static_quantization(fp32_pose_path, out_qdq_path, crops, QuantFormat.QDQ)

    # 2. Run Static QOperator Quantization
    run_static_quantization(fp32_pose_path, out_qop_path, crops, QuantFormat.QOperator)

    # 3. Benchmark all versions
    print("\n" + "=" * 80)
    print("  CPU / WASM SIMD LATENCY BENCHMARK (4 Threads)")
    print("=" * 80)

    models_to_test = [
        fp32_pose_path,
        dyn_int8_pose_path,
        out_qdq_path,
        out_qop_path
    ]

    results = []
    for p in models_to_test:
        if os.path.exists(p):
            res = benchmark_model(p, crops, num_runs=100, threads=4)
            results.append(res)

    print("\n{:<45} | {:<7} | {:<7} | {:<10} | {:<8}".format(
        "Model Name", "Size MB", "Nodes", "Mean Latency", "Throughput"
    ))
    print("-" * 85)
    for r in results:
        print("{:<45} | {:<7.2f} | {:<7} | {:<10.2f} ms | {:<8.1f} FPS".format(
            r["model"], r["size_mb"], r["total_nodes"], r["mean_latency_ms"], r["fps"]
        ))
        print("   Top Ops:", r["top_ops"])

    # Save benchmark json
    out_json = "deliverables/qualcomm_experiments/static_vs_dynamic_int8_benchmark.json"
    with open(out_json, "w") as f:
        import json
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved detailed comparison to {out_json}")


if __name__ == "__main__":
    main()
