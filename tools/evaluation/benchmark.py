"""
benchmark.py
────────────
Latency and FPS benchmarking for the full AI Perception pipeline.

Reports:
  - Per-module latency (detector, keypoint estimator, PnP, smoothing)
  - End-to-end FPS
  - Memory footprint of ONNX models
  - Comparison table across devices (recorded manually)

Usage
-----
  # Benchmark on a video file:
  python tools/evaluation/benchmark.py \\
      --detector  outputs/exports/foot_detector.onnx \\
      --keypoint  outputs/exports/rtmpose_tiny_18kp.onnx \\
      --video     test_video.mp4 \\
      --n-frames  200

  # Benchmark on webcam:
  python tools/evaluation/benchmark.py \\
      --detector  outputs/exports/foot_detector.onnx \\
      --keypoint  outputs/exports/rtmpose_tiny_18kp.onnx \\
      --camera    0 \\
      --n-frames  300
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Dict, List, Optional

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Timer context manager
# ──────────────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self) -> None:
        self._data: Dict[str, List[float]] = defaultdict(list)
        self._start: Dict[str, float] = {}

    def start(self, name: str) -> None:
        self._start[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        elapsed = (time.perf_counter() - self._start[name]) * 1000  # ms
        self._data[name].append(elapsed)
        return elapsed

    def report(self, name: str) -> Dict:
        data = np.array(self._data.get(name, [0.0]))
        return {
            "mean_ms":   float(data.mean()),
            "std_ms":    float(data.std()),
            "min_ms":    float(data.min()),
            "max_ms":    float(data.max()),
            "p95_ms":    float(np.percentile(data, 95)),
            "fps":       float(1000.0 / max(data.mean(), 1e-6)),
            "n_frames":  len(data),
        }

    def all_reports(self) -> Dict[str, Dict]:
        return {name: self.report(name) for name in self._data}


# ──────────────────────────────────────────────────────────────────────────────
# Model size helper
# ──────────────────────────────────────────────────────────────────────────────

def model_size_mb(path: str) -> float:
    import os
    return os.path.getsize(path) / (1024 * 1024)


# ──────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ──────────────────────────────────────────────────────────────────────────────

def run_benchmark(
    detector_path:  str,
    keypoint_path:  str,
    source:         str,    # "0" for webcam or path to video
    n_frames:       int     = 200,
    warmup_frames:  int     = 10,
    providers:      Optional[List[str]] = None,
) -> Dict:
    """
    Run benchmark on a video source.

    Parameters
    ----------
    detector_path  : ONNX path for foot detector.
    keypoint_path  : ONNX path for keypoint model.
    source         : Camera index (int as string) or video file path.
    n_frames       : Number of frames to benchmark.
    warmup_frames  : Frames to skip before recording (warm up GPU/CPU).
    providers      : ONNX Runtime providers.
    """
    from src.models.detector.foot_detector  import FootDetector
    from src.models.keypoint.keypoint_model import FootKeypointEstimator
    from src.models.geometry.pnp_solver     import FootPnPSolver
    from src.models.geometry.camera_model   import get_default_camera

    ep = providers or ["CPUExecutionProvider"]

    detector   = FootDetector(detector_path, providers=ep)
    keypoints  = FootKeypointEstimator(keypoint_path, providers=ep)

    # Open video source
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    ret, sample_frame = cap.read()
    h, w = sample_frame.shape[:2]
    camera = get_default_camera(w, h)

    pnp = FootPnPSolver(camera=camera)
    timer = Timer()

    frame_count = 0
    total_count = n_frames + warmup_frames

    print(f"\nBenchmarking on {total_count} frames ({warmup_frames} warmup)...")

    while frame_count < total_count:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        is_warmup = frame_count < warmup_frames

        # ── Detector ─────────────────────────────────────────────────────
        if not is_warmup:
            timer.start("detector")
        detections = detector.detect(frame)
        if not is_warmup:
            timer.stop("detector")

        # ── Keypoints ─────────────────────────────────────────────────────
        if detections:
            bbox = detections[0]["bbox"]
            if not is_warmup:
                timer.start("keypoints")
            kps = keypoints.predict(frame, tuple(bbox))
            if not is_warmup:
                timer.stop("keypoints")

            # ── PnP ───────────────────────────────────────────────────────
            left_kps, right_kps = keypoints.split_feet(kps)
            if not is_warmup:
                timer.start("pnp")
            pnp.solve(right_kps, "right")
            if not is_warmup:
                timer.stop("pnp")
        else:
            # Still count frames for consistent comparison
            if not is_warmup:
                for name in ["keypoints", "pnp"]:
                    timer._data[name].append(0.0)

        # ── End-to-end ────────────────────────────────────────────────────
        frame_count += 1

    cap.release()

    # Calculate end-to-end FPS from sum of stage means
    reports = timer.all_reports()
    e2e_mean = sum(
        reports.get(k, {}).get("mean_ms", 0)
        for k in ["detector", "keypoints", "pnp"]
    )

    return {
        "modules":         reports,
        "e2e_mean_ms":     e2e_mean,
        "e2e_fps":         1000.0 / max(e2e_mean, 1e-6),
        "detector_size_mb":  model_size_mb(detector_path),
        "keypoint_size_mb":  model_size_mb(keypoint_path),
        "frame_size":      f"{w}×{h}",
        "n_frames":        n_frames,
    }


def print_benchmark_report(result: Dict) -> None:
    print("\n" + "=" * 65)
    print("  AI PERCEPTION LAYER — BENCHMARK REPORT")
    print("=" * 65)
    print(f"  Frame resolution : {result['frame_size']}")
    print(f"  Frames evaluated : {result['n_frames']}")
    print()
    print(f"  {'Module':<20} {'Mean':>8}  {'Std':>7}  {'P95':>7}  {'FPS':>7}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    for name in ["detector", "keypoints", "pnp"]:
        r = result["modules"].get(name, {})
        print(
            f"  {name:<20} "
            f"{r.get('mean_ms',0):>7.1f}ms  "
            f"{r.get('std_ms',0):>6.1f}ms  "
            f"{r.get('p95_ms',0):>6.1f}ms  "
            f"{r.get('fps',0):>6.1f}"
        )
    print(f"  {'-'*20}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
    print(
        f"  {'E2E (sum)':<20} "
        f"{result['e2e_mean_ms']:>7.1f}ms  "
        f"{'':>7}  {'':>7}  "
        f"{result['e2e_fps']:>6.1f}"
    )
    print()
    print(f"  Model sizes:")
    print(f"    Detector  : {result['detector_size_mb']:.1f} MB")
    print(f"    Keypoints : {result['keypoint_size_mb']:.1f} MB")
    print("=" * 65 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AI Perception pipeline.")
    parser.add_argument("--detector",  required=True, help="Detector ONNX path.")
    parser.add_argument("--keypoint",  required=True, help="Keypoint ONNX path.")
    parser.add_argument("--video",     default=None,  help="Video file path.")
    parser.add_argument("--camera",    default=None,  help="Camera index (e.g. 0).")
    parser.add_argument("--n-frames",  type=int, default=200)
    parser.add_argument("--save",      default=None,  help="Save report to JSON.")
    args = parser.parse_args()

    if args.video:
        source = args.video
    elif args.camera is not None:
        source = str(args.camera)
    else:
        parser.error("Provide --video or --camera.")

    result = run_benchmark(
        detector_path=args.detector,
        keypoint_path=args.keypoint,
        source=source,
        n_frames=args.n_frames,
    )
    print_benchmark_report(result)

    if args.save:
        import json
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Benchmark saved to: {args.save}")


if __name__ == "__main__":
    main()

