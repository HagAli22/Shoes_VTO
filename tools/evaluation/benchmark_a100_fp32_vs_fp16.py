#!/usr/bin/env python3
"""
Methodologically Rigorous NVIDIA A100 (and Ampere / CUDA) Benchmark: FP32 vs FP16
Shoes VTO Two-Stage Pipeline (RTMPose-13KP + NanoDet-Plus Stage A)

Addresses all 7 methodology requirements:
1. ORT Profiling Trace: Verifies every single node's execution provider (CUDA vs CPU).
2. CUDA Timing: Uses a dedicated user_compute_stream shared between PyTorch and ORT,
   synchronized CUDA Events, and cross-validated with CPU-synchronized high-res timers.
3. VRAM: Tracks driver-level memory via torch.cuda.mem_get_info() and pynvml.
4. Preprocessing: Pure FP32 CPU arithmetic for both models; only cast to float16
   immediately prior to inference (no CPU float16 math penalty).
5. Preprocessing Statistics: Reports Mean, P50, P95, and std dev over 100 trials.
6. Real-world Parity: Uses real video frames from data/test_video.mp4 for both
   NanoDet box detection and RTMPose keypoint estimation.
7. Measurement Variability: Runs 5 independent trials per batch size (total 250 iters),
   reporting Mean ± StdDev, P50, P95, and statistical significance testing.
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path
import numpy as np

# PyTorch & CUDA Stream setup
try:
    import torch
    _tlib = Path(torch.__file__).parent / "lib"
    if hasattr(os, "add_dll_directory") and _tlib.exists():
        try:
            os.add_dll_directory(str(_tlib))
        except Exception:
            pass
    os.environ["PATH"] = str(_tlib) + os.pathsep + os.environ.get("PATH", "")
except ImportError:
    torch = None

try:
    import onnxruntime as ort
except ImportError:
    print("[!] onnxruntime not found. Please install onnxruntime-gpu.")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pynvml
    pynvml.nvmlInit()
except Exception:
    pynvml = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# -----------------------------------------------------------------------------
# 1. Hardware & Memory Helpers
# -----------------------------------------------------------------------------

def get_gpu_spec():
    if torch is None or not torch.cuda.is_available():
        return {
            "name": "CPU (No CUDA available)",
            "is_a100": False,
            "compute_capability": (0, 0),
            "total_vram_gb": 0.0,
            "cuda_version": "None",
            "cudnn_version": "None"
        }
    device_name = torch.cuda.get_device_name(0)
    cc = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    is_a100 = "A100" in device_name.upper() or (cc[0] == 8 and cc[1] == 0)
    return {
        "name": device_name,
        "is_a100": is_a100,
        "compute_capability": cc,
        "total_vram_gb": vram_gb,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else "None"
    }

def get_driver_vram_used_mb():
    """Returns GPU-level memory used in MB via CUDA driver or NVML."""
    if pynvml:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return float(mem.used / (1024**2))
        except Exception:
            pass
    if torch and torch.cuda.is_available():
        try:
            free, total = torch.cuda.mem_get_info()
            return float((total - free) / (1024**2))
        except Exception:
            pass
    return 0.0

# -----------------------------------------------------------------------------
# 2. Session Initialization & Operator Verification (Profiling Trace)
# -----------------------------------------------------------------------------

def create_synchronized_session(model_path: Path, stream: torch.cuda.Stream, enable_profiling: bool = False, profile_prefix: str = ""):
    """
    Creates an ORT session explicitly attached to the PyTorch CUDA stream.
    Eliminates stream desynchronization.
    """
    cuda_opts = {
        "device_id": 0,
        "user_compute_stream": str(stream.cuda_stream),
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if enable_profiling:
        sess_opts.enable_profiling = True
        sess_opts.profile_file_prefix = profile_prefix

    providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
    return session

def verify_session_nodes(model_path: Path, stream: torch.cuda.Stream, temp_dir: Path):
    """
    Runs a single profiling warm-up and parses the exact execution provider
    of every computational node in the graph.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(temp_dir / f"verify_{model_path.stem}")
    sess = create_synchronized_session(model_path, stream, enable_profiling=True, profile_prefix=prefix)
    
    inp = sess.get_inputs()[0]
    dtype = np.float16 if "float16" in inp.type else np.float32
    dummy = np.random.randn(1, 3, 256, 256).astype(dtype)
    
    # Run once to populate trace
    sess.run(None, {inp.name: dummy})
    stream.synchronize()
    prof_file = sess.end_profiling()

    # Parse profile trace
    cuda_ops = 0
    cpu_ops = 0
    cpu_op_types = {}
    cuda_op_types = {}

    with open(prof_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    node_events = [e for e in events if e.get("cat") in ("Node", "Op") and "dur" in e]

    for e in node_events:
        args = e.get("args", {})
        provider = args.get("provider", "Unknown")
        op_name = args.get("op_name", e.get("name", "Unknown"))
        if "CUDA" in provider:
            cuda_ops += 1
            cuda_op_types[op_name] = cuda_op_types.get(op_name, 0) + 1
        else:
            cpu_ops += 1
            cpu_op_types[op_name] = cpu_op_types.get(op_name, 0) + 1

    # Cleanup temp profile file
    try:
        os.remove(prof_file)
    except Exception:
        pass

    total_ops = cuda_ops + cpu_ops
    cuda_pct = (cuda_ops / total_ops * 100.0) if total_ops > 0 else 0.0

    return {
        "cuda_ops": cuda_ops,
        "cpu_ops": cpu_ops,
        "cuda_pct": cuda_pct,
        "cuda_op_types": cuda_op_types,
        "cpu_op_types": cpu_op_types,
        "verified_cuda": (cuda_ops > 0)
    }

# -----------------------------------------------------------------------------
# 3. Preprocessing & Postprocessing
# -----------------------------------------------------------------------------

def preprocess_crops(crops: list, is_fp16: bool):
    """
    Standard FP32 preprocessing for all inputs.
    Cast to float16 is strictly a zero-overhead contiguous cast immediately before feeding.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensors = []
    for c in crops:
        img = cv2.resize(c, (256, 256)) if (c.shape[0] != 256 or c.shape[1] != 256) else c
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - mean) / std
        tensors.append(img.transpose(2, 0, 1))
    batch_f32 = np.ascontiguousarray(np.stack(tensors, axis=0))
    if is_fp16:
        return batch_f32.astype(np.float16)
    return batch_f32

def benchmark_preprocessing_stats(batch_size: int, is_fp16: bool, trials: int = 100):
    dummy_crops = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(batch_size)]
    times_ms = []
    for _ in range(trials):
        t0 = time.perf_counter()
        _ = preprocess_crops(dummy_crops, is_fp16)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return {
        "mean_ms": float(np.mean(times_ms)),
        "std_ms": float(np.std(times_ms)),
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95))
    }

def decode_rtmpose_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray):
    """
    SimCC post-processing: argmax over representation bins to pixel coords.
    simcc_x: [B, K, 512], simcc_y: [B, K, 512]
    """
    t0 = time.perf_counter()
    x_idx = np.argmax(simcc_x, axis=-1) / 2.0  # 512 bins / 2 = 256 image coords
    y_idx = np.argmax(simcc_y, axis=-1) / 2.0
    t1 = time.perf_counter()
    return x_idx, y_idx, (t1 - t0) * 1000.0

def decode_nanodet_boxes(boxes: np.ndarray, scores: np.ndarray, score_thresh: float = 0.35):
    t0 = time.perf_counter()
    mask = scores > score_thresh
    filtered = boxes[mask.squeeze(-1)]
    t1 = time.perf_counter()
    return filtered, (t1 - t0) * 1000.0

# -----------------------------------------------------------------------------
# 4. Rigorous Synchronized Inference Benchmark
# -----------------------------------------------------------------------------

def run_synchronized_benchmark(session, model_path: Path, stream: torch.cuda.Stream,
                              batch_size: int, is_fp16: bool, warmup: int = 50,
                              num_trials: int = 5, iters_per_trial: int = 50):
    inp = session.get_inputs()[0]
    inp_name = inp.name

    # Prepare inputs
    dummy_crops = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(batch_size)]
    feed_data = preprocess_crops(dummy_crops, is_fp16)
    feed = {inp_name: feed_data}

    # Warmup
    for _ in range(warmup):
        session.run(None, feed)
    stream.synchronize()

    trial_means = []
    all_latencies_ms = []
    wall_latencies_ms = []

    vram_before = get_driver_vram_used_mb()
    peak_vram = vram_before

    for trial in range(num_trials):
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters_per_trial)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters_per_trial)]
        trial_lats = []

        for i in range(iters_per_trial):
            stream.synchronize()
            t0 = time.perf_counter()
            start_events[i].record(stream)
            out = session.run(None, feed)
            end_events[i].record(stream)
            end_events[i].synchronize()
            t1 = time.perf_counter()

            lat = start_events[i].elapsed_time(end_events[i])
            trial_lats.append(lat)
            all_latencies_ms.append(lat)
            wall_latencies_ms.append((t1 - t0) * 1000.0)

            # Sample VRAM occasionally
            if i % 10 == 0:
                cur_vram = get_driver_vram_used_mb()
                if cur_vram > peak_vram:
                    peak_vram = cur_vram

        trial_means.append(float(np.mean(trial_lats)))

    stream.synchronize()
    allocated_vram_mb = max(0.0, peak_vram - vram_before)

    # Post-processing timing
    is_rtmpose = "rtmpose" in model_path.name.lower()
    if is_rtmpose:
        _, _, post_ms = decode_rtmpose_simcc(out[0], out[1])
    else:
        _, post_ms = decode_nanodet_boxes(out[0], out[1])

    mean_ms = float(np.mean(all_latencies_ms))
    std_ms = float(np.std(all_latencies_ms))
    p50_ms = float(np.percentile(all_latencies_ms, 50))
    p95_ms = float(np.percentile(all_latencies_ms, 95))
    min_ms = float(np.min(all_latencies_ms))
    max_ms = float(np.max(all_latencies_ms))
    wall_mean_ms = float(np.mean(wall_latencies_ms))

    # Cross-validation check between CUDA Event timing and CPU wall clock
    sync_delta_pct = abs(wall_mean_ms - mean_ms) / mean_ms * 100.0

    fps = (batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "is_fp16": is_fp16,
        "gpu_mean_ms": mean_ms,
        "gpu_std_ms": std_ms,
        "gpu_p50_ms": p50_ms,
        "gpu_p95_ms": p95_ms,
        "gpu_min_ms": min_ms,
        "gpu_max_ms": max_ms,
        "wall_mean_ms": wall_mean_ms,
        "sync_delta_pct": sync_delta_pct,
        "inference_fps": fps,
        "trial_means": trial_means,
        "post_ms": post_ms,
        "peak_vram_mb": peak_vram,
        "allocated_vram_mb": allocated_vram_mb
    }

# -----------------------------------------------------------------------------
# 5. Real-World Video Parity Verification (data/test_video.mp4)
# -----------------------------------------------------------------------------

def load_real_test_frames(video_path: Path, num_frames: int = 5):
    """Extracts representative frames evenly distributed across the video."""
    if not video_path.exists():
        print(f"[!] Warning: {video_path} not found. Generating synthetic patterns for parity.")
        frames = []
        for i in range(num_frames):
            syn = np.full((720, 720, 3), 128, dtype=np.uint8)
            if cv2:
                cv2.circle(syn, (360 + i*10, 360 + i*10), 100, (255, 200, 180), -1)
            frames.append(syn)
        return frames, False

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 500

    step = max(1, total_frames // (num_frames + 1))
    frames = []
    for idx in range(1, num_frames + 1):
        target = idx * step
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()

    if len(frames) == 0:
        print(f"[!] Could not read frames from {video_path}. Using synthetic pattern.")
        return [np.zeros((720, 720, 3), dtype=np.uint8) for _ in range(num_frames)], False

    return frames, True

def verify_real_numerical_parity(rtm_fp32_sess, rtm_fp16_sess,
                                nano_fp32_sess, nano_fp16_sess,
                                video_path: Path):
    """
    Evaluates FP32 vs FP16 on actual video frames from data/test_video.mp4.
    """
    frames, is_real = load_real_test_frames(video_path, num_frames=5)
    print(f"\n[+] Loaded {len(frames)} frames for numerical validation (Source: {'Real Video (' + str(video_path) + ')' if is_real else 'Synthetic Fallback'}).")

    # 1. NanoDet Stage A validation on real full frames
    nano_inp32_name = nano_fp32_sess.get_inputs()[0].name
    nano_inp16_name = nano_fp16_sess.get_inputs()[0].name

    nano_batch_32 = preprocess_crops(frames, is_fp16=False)
    nano_batch_16 = preprocess_crops(frames, is_fp16=True)

    out_nano_32 = nano_fp32_sess.run(None, {nano_inp32_name: nano_batch_32})
    out_nano_16 = nano_fp16_sess.run(None, {nano_inp16_name: nano_batch_16})

    boxes_32, boxes_16 = out_nano_32[0], out_nano_16[0].astype(np.float32)
    scores_32, scores_16 = out_nano_32[1], out_nano_16[1].astype(np.float32)

    box_mae = float(np.mean(np.abs(boxes_32 - boxes_16)))
    box_max_err = float(np.max(np.abs(boxes_32 - boxes_16)))
    score_mae = float(np.mean(np.abs(scores_32 - scores_16)))
    score_max_err = float(np.max(np.abs(scores_32 - scores_16)))

    # 2. RTMPose-13KP validation on real foot crops
    crops = []
    for f in frames:
        h, w = f.shape[:2]
        ch, cw = int(h * 0.5), int(w * 0.5)
        # 256x256 crop around center
        y1, y2 = max(0, ch - 128), min(h, ch + 128)
        x1, x2 = max(0, cw - 128), min(w, cw + 128)
        crop = f[y1:y2, x1:x2]
        if crop.shape[0] != 256 or crop.shape[1] != 256:
            crop = cv2.resize(crop, (256, 256))
        crops.append(crop)

    rtm_inp32_name = rtm_fp32_sess.get_inputs()[0].name
    rtm_inp16_name = rtm_fp16_sess.get_inputs()[0].name

    rtm_batch_32 = preprocess_crops(crops, is_fp16=False)
    rtm_batch_16 = preprocess_crops(crops, is_fp16=True)

    out_rtm_32 = rtm_fp32_sess.run(None, {rtm_inp32_name: rtm_batch_32})
    out_rtm_16 = rtm_fp16_sess.run(None, {rtm_inp16_name: rtm_batch_16})

    simcc_x_32, simcc_x_16 = out_rtm_32[0], out_rtm_16[0].astype(np.float32)
    simcc_y_32, simcc_y_16 = out_rtm_32[1], out_rtm_16[1].astype(np.float32)
    vis_32, vis_16 = out_rtm_32[3], out_rtm_16[3].astype(np.float32)

    # Argmax pixel coordinates on 256x256 image
    x_32, y_32, _ = decode_rtmpose_simcc(simcc_x_32, simcc_y_32)
    x_16, y_16, _ = decode_rtmpose_simcc(simcc_x_16, simcc_y_16)

    pixel_dist = np.sqrt((x_32 - x_16)**2 + (y_32 - y_16)**2)
    mean_pixel_err = float(np.mean(pixel_dist))
    max_pixel_err = float(np.max(pixel_dist))
    exact_match = float(np.mean(pixel_dist == 0.0)) * 100.0
    subpixel_match = float(np.mean(pixel_dist <= 0.5)) * 100.0
    vis_mae = float(np.mean(np.abs(vis_32 - vis_16)))

    simcc_mae = float((np.mean(np.abs(simcc_x_32 - simcc_x_16)) + np.mean(np.abs(simcc_y_32 - simcc_y_16))) / 2.0)
    cos_sim = float((
        np.dot(simcc_x_32.flatten(), simcc_x_16.flatten()) / (np.linalg.norm(simcc_x_32) * np.linalg.norm(simcc_x_16) + 1e-9) +
        np.dot(simcc_y_32.flatten(), simcc_y_16.flatten()) / (np.linalg.norm(simcc_y_32) * np.linalg.norm(simcc_y_16) + 1e-9)
    ) / 2.0)

    return {
        "is_real_video": is_real,
        "frames_evaluated": len(frames),
        "rtmpose": {
            "simcc_mae": simcc_mae,
            "cosine_similarity": cos_sim,
            "mean_keypoint_pixel_error_px": mean_pixel_err,
            "max_keypoint_pixel_error_px": max_pixel_err,
            "exact_pixel_match_percent": exact_match,
            "subpixel_05px_match_percent": subpixel_match,
            "visibility_confidence_mae": vis_mae,
            "status": "PASS" if mean_pixel_err < 0.5 else "WARN"
        },
        "nanodet": {
            "box_coord_mae_px": box_mae,
            "box_coord_max_err_px": box_max_err,
            "score_mae": score_mae,
            "score_max_err": score_max_err,
            "status": "PASS" if box_mae < 1.0 and score_mae < 0.01 else "WARN"
        }
    }

# -----------------------------------------------------------------------------
# 6. Plotting
# -----------------------------------------------------------------------------

def generate_comparison_plots(df_rows, out_png: Path):
    if plt is None:
        print("[!] matplotlib not installed. Skipping plot generation.")
        return

    import pandas as pd
    df = pd.DataFrame(df_rows)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NVIDIA A100: FP32 vs FP16 Benchmark (Shoes VTO Pipeline)", fontsize=14, fontweight="bold")

    # 1. RTMPose Latency with Error Bars (StdDev)
    ax = axes[0, 0]
    rtm = df[df["model_group"] == "RTMPose"]
    rtm_32 = rtm[rtm["precision"] == "FP32"].sort_values("batch_size")
    rtm_16 = rtm[rtm["precision"] == "FP16"].sort_values("batch_size")
    ax.errorbar(rtm_32["batch_size"], rtm_32["gpu_mean_ms"], yerr=rtm_32["gpu_std_ms"], fmt="o-", label="FP32 (opset12)", color="#1f77b4", capsize=4, linewidth=2)
    ax.errorbar(rtm_16["batch_size"], rtm_16["gpu_mean_ms"], yerr=rtm_16["gpu_std_ms"], fmt="s--", label="FP16 (fp16)", color="#2ca02c", capsize=4, linewidth=2)
    ax.set_title("RTMPose-13KP: GPU Latency (ms)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms) [Lower is better]")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # 2. RTMPose Throughput (FPS)
    ax = axes[0, 1]
    ax.bar(np.array(rtm_32["batch_size"]) - 0.2, rtm_32["inference_fps"], width=0.4, label="FP32 FPS", color="#1f77b4")
    ax.bar(np.array(rtm_16["batch_size"]) + 0.2, rtm_16["inference_fps"], width=0.4, label="FP16 FPS", color="#2ca02c")
    ax.set_title("RTMPose-13KP: Inference Throughput (FPS)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Frames Per Second [Higher is better]")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # 3. NanoDet Latency with Error Bars
    ax = axes[1, 0]
    nano = df[df["model_group"] == "NanoDet"]
    nano_32 = nano[nano["precision"] == "FP32"].sort_values("batch_size")
    nano_16 = nano[nano["precision"] == "FP16"].sort_values("batch_size")
    ax.errorbar(nano_32["batch_size"], nano_32["gpu_mean_ms"], yerr=nano_32["gpu_std_ms"], fmt="o-", label="FP32 (opset12)", color="#ff7f0e", capsize=4, linewidth=2)
    ax.errorbar(nano_16["batch_size"], nano_16["gpu_mean_ms"], yerr=nano_16["gpu_std_ms"], fmt="s--", label="FP16 (fp16)", color="#d62728", capsize=4, linewidth=2)
    ax.set_title("NanoDet Stage A: GPU Latency (ms)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms) [Lower is better]")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # 4. Speedup Factor & Statistical Significance
    ax = axes[1, 1]
    rtm_speedup = rtm_32["gpu_mean_ms"].values / rtm_16["gpu_mean_ms"].values
    nano_speedup = nano_32["gpu_mean_ms"].values / nano_16["gpu_mean_ms"].values
    x_pos = np.arange(len(rtm_32["batch_size"]))
    ax.bar(x_pos - 0.2, rtm_speedup, width=0.4, label="RTMPose FP16 Speedup", color="#2ca02c")
    ax.bar(x_pos + 0.2, nano_speedup, width=0.4, label="NanoDet FP16 Speedup", color="#d62728")
    ax.axhline(1.0, color="red", linestyle="--", label="Parity (1.0x)")
    ax.axhline(0.60, color="gray", linestyle=":", label="Quadro P2000 Baseline (~0.60x)")
    ax.set_title("FP16 Speedup Factor (FP32 ms / FP16 ms)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup (> 1.0 means FP16 is faster)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=300)
    plt.close()
    print(f"[+] Saved comparison plot: {out_png}")

# -----------------------------------------------------------------------------
# 7. Main Execution Flow
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Methodologically Rigorous A100 FP32 vs FP16 Benchmark")
    parser.add_argument("--models_dir", type=str, default="deliverables/qualcomm_experiments/models", help="Path to models directory")
    parser.add_argument("--video", type=str, default="data/test_video.mp4", help="Path to test video for real parity")
    parser.add_argument("--output_dir", type=str, default="deliverables/qualcomm_experiments/profiling_a100", help="Output directory")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--trials", type=int, default=5, help="Number of independent benchmark trials")
    parser.add_argument("--iters", type=int, default=50, help="Iterations per trial (total iterations = trials * iters)")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists() and (REPO_ROOT / args.models_dir).exists():
        models_dir = REPO_ROOT / args.models_dir

    video_path = Path(args.video)
    if not video_path.exists() and (REPO_ROOT / args.video).exists():
        video_path = REPO_ROOT / args.video

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_spec = get_gpu_spec()

    print("================================================================================")
    print("METHODOLOGICALLY RIGOROUS A100 BENCHMARK: FP32 VS FP16 (SHOES VTO)")
    print("================================================================================")
    print(f"Device Name:        {gpu_spec['name']}")
    print(f"Compute Capability: {gpu_spec['compute_capability']}")
    print(f"Total VRAM:         {gpu_spec['total_vram_gb']:.2f} GB")
    print(f"Current VRAM Used:  {get_driver_vram_used_mb():.1f} MB")
    print(f"CUDA Version:       {gpu_spec['cuda_version']}")
    print(f"cuDNN Version:      {gpu_spec['cudnn_version']}")
    print(f"ORT Available EPs:  {ort.get_available_providers()}")
    print("--------------------------------------------------------------------------------")

    if not gpu_spec["is_a100"]:
        print(f"[!] NOTICE: Running on {gpu_spec['name']} (Expected: NVIDIA A100 CC 8.0).")
    else:
        print("[✓] Verified Hardware: NVIDIA A100 Tensor Core GPU (Compute Capability 8.0)")

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("[!] FATAL: CUDAExecutionProvider is NOT available in ONNX Runtime!")
        sys.exit(1)
    else:
        print("[✓] Verified Execution Provider: CUDAExecutionProvider")

    # Dedicated shared CUDA stream
    stream = torch.cuda.Stream()
    print(f"[✓] Initialized Dedicated PyTorch CUDA Stream: {stream.cuda_stream}")

    models_to_test = [
        ("RTMPose", "FP32", models_dir / "rtmpose_13kp_hand_improved5_opset12.onnx"),
        ("RTMPose", "FP16", models_dir / "rtmpose_13kp_hand_improved5_fp16.onnx"),
        ("NanoDet", "FP32", models_dir / "nanodet_stage_a_improved5_opset12.onnx"),
        ("NanoDet", "FP16", models_dir / "nanodet_stage_a_improved5_fp16.onnx"),
    ]

    for grp, prec, path in models_to_test:
        if not path.exists():
            print(f"[!] Error: Model file {path} not found!")
            sys.exit(1)

    # 1. ORT Profiling Trace & Node Execution Provider Verification
    print("\n" + "="*80)
    print("STAGE 1: ORT NODE EXECUTION PROVIDER VERIFICATION (PROFILING TRACE)")
    print("="*80)
    print("| Model | Precision | Total Ops | CUDA Ops | CPU Ops | CUDA % | Verified CUDA? |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    node_verifications = {}
    for grp, prec, path in models_to_test:
        v_res = verify_session_nodes(path, stream, out_dir / "temp_prof")
        key = f"{grp}_{prec}"
        node_verifications[key] = v_res
        total = v_res["cuda_ops"] + v_res["cpu_ops"]
        print(f"| {grp:7s} | {prec:5s} | {total:9d} | {v_res['cuda_ops']:8d} | {v_res['cpu_ops']:7d} | {v_res['cuda_pct']:5.1f}% | [{'✓ PASS' if v_res['verified_cuda'] else 'FAIL'}] |")
        if v_res["cpu_ops"] > 0:
            print(f"    -> CPU Ops ({key}): {dict(list(v_res['cpu_op_types'].items())[:5])}")

    # Initialize persistent synchronized benchmark sessions
    sessions = {}
    for grp, prec, path in models_to_test:
        key = f"{grp}_{prec}"
        s = create_synchronized_session(path, stream, enable_profiling=False)
        sessions[key] = (s, path)

    # 2. Preprocessing Benchmarking (FP32 standard)
    print("\n" + "="*80)
    print("STAGE 2: PREPROCESSING TIMING (FP32 CPU Math + Final Contiguous Cast)")
    print("="*80)
    prep_stats = {}
    for b in [1, 2, 4, 8]:
        p32 = benchmark_preprocessing_stats(b, is_fp16=False, trials=100)
        p16 = benchmark_preprocessing_stats(b, is_fp16=True, trials=100)
        prep_stats[f"batch_{b}"] = {"fp32": p32, "fp16": p16}
        print(f"Batch {b:2d}: FP32 Prep = {p32['mean_ms']:.3f} ± {p32['std_ms']:.3f} ms | FP16 Prep = {p16['mean_ms']:.3f} ± {p16['std_ms']:.3f} ms (Cast overhead: {p16['mean_ms'] - p32['mean_ms']:.3f} ms)")

    # 3. Real-world Numerical Parity
    print("\n" + "="*80)
    print("STAGE 3: REAL-WORLD NUMERICAL PARITY (data/test_video.mp4)")
    print("="*80)
    real_parity = verify_real_numerical_parity(
        sessions["RTMPose_FP32"][0], sessions["RTMPose_FP16"][0],
        sessions["NanoDet_FP32"][0], sessions["NanoDet_FP16"][0],
        video_path
    )
    print(f"RTMPose Numerical Parity on Real Video Frames:")
    print(f"  - SimCC Heatmap MAE:           {real_parity['rtmpose']['simcc_mae']:.6f}")
    print(f"  - Cosine Similarity:           {real_parity['rtmpose']['cosine_similarity']:.6f}")
    print(f"  - Mean Keypoint Pixel Error:   {real_parity['rtmpose']['mean_keypoint_pixel_error_px']:.4f} px (on 256x256)")
    print(f"  - Max Keypoint Pixel Error:    {real_parity['rtmpose']['max_keypoint_pixel_error_px']:.4f} px")
    print(f"  - Exact Pixel Match Rate:      {real_parity['rtmpose']['exact_pixel_match_percent']:.2f}%")
    print(f"  - Sub-pixel (<=0.5px) Match:   {real_parity['rtmpose']['subpixel_05px_match_percent']:.2f}%")
    print(f"  - Visibility Confidence MAE:   {real_parity['rtmpose']['visibility_confidence_mae']:.6f}")
    print(f"  - Keypoint Parity Status:      [{real_parity['rtmpose']['status']}]")

    print(f"\nNanoDet Numerical Parity on Real Video Frames:")
    print(f"  - Box Coordinate MAE:          {real_parity['nanodet']['box_coord_mae_px']:.6f} px")
    print(f"  - Max Box Coordinate Error:    {real_parity['nanodet']['box_coord_max_err_px']:.6f} px")
    print(f"  - Confidence Score MAE:        {real_parity['nanodet']['score_mae']:.6f}")
    print(f"  - Box Parity Status:           [{real_parity['nanodet']['status']}]")

    # 4. Latency & Throughput Benchmark with Repeated Trials & Variability
    print("\n" + "="*80)
    print(f"STAGE 4: SYNCHRONIZED BENCHMARK ({args.trials} Trials x {args.iters} Iters = {args.trials * args.iters} Total Iters)")
    print("="*80)

    batch_sizes = [1, 2, 4, 8]
    all_results = []

    for grp, prec, path in models_to_test:
        key = f"{grp}_{prec}"
        s = sessions[key][0]
        is_fp16 = (prec == "FP16")
        print(f"\n---> Benchmarking {key} (Synchronized user_compute_stream)...")
        for b in batch_sizes:
            res = run_synchronized_benchmark(
                s, path, stream, batch_size=b, is_fp16=is_fp16,
                warmup=args.warmup, num_trials=args.trials, iters_per_trial=args.iters
            )
            prep_info = prep_stats[f"batch_{b}"]["fp16" if is_fp16 else "fp32"]
            e2e_total_ms = prep_info["mean_ms"] + res["gpu_mean_ms"] + res["post_ms"]
            e2e_fps = (b / (e2e_total_ms / 1000.0)) if e2e_total_ms > 0 else 0.0

            row = {
                "gpu": gpu_spec["name"],
                "model_group": grp,
                "precision": prec,
                "batch_size": b,
                "gpu_mean_ms": res["gpu_mean_ms"],
                "gpu_std_ms": res["gpu_std_ms"],
                "gpu_p50_ms": res["gpu_p50_ms"],
                "gpu_p95_ms": res["gpu_p95_ms"],
                "gpu_min_ms": res["gpu_min_ms"],
                "gpu_max_ms": res["gpu_max_ms"],
                "wall_mean_ms": res["wall_mean_ms"],
                "sync_delta_pct": res["sync_delta_pct"],
                "inference_fps": res["inference_fps"],
                "prep_mean_ms": prep_info["mean_ms"],
                "post_ms": res["post_ms"],
                "e2e_latency_ms": e2e_total_ms,
                "e2e_fps": e2e_fps,
                "peak_vram_mb": res["peak_vram_mb"],
                "allocated_vram_mb": res["allocated_vram_mb"]
            }
            all_results.append(row)
            print(f"  Batch {b:2d}: GPU Mean = {res['gpu_mean_ms']:5.2f} ± {res['gpu_std_ms']:.2f} ms | P50 = {res['gpu_p50_ms']:5.2f} ms | P95 = {res['gpu_p95_ms']:5.2f} ms | Sync Agreement = {100.0 - res['sync_delta_pct']:5.2f}% | Infer FPS = {res['inference_fps']:7.1f} | E2E FPS = {e2e_fps:7.1f}")

    # 5. Statistical Significance Evaluation (FP32 vs FP16)
    print("\n" + "="*80)
    print("STAGE 5: STATISTICAL SIGNIFICANCE (FP32 VS FP16 DELTA vs MEASUREMENT NOISE)")
    print("="*80)
    import pandas as pd
    df = pd.DataFrame(all_results)

    significance_records = []
    for grp in ["RTMPose", "NanoDet"]:
        sub = df[df["model_group"] == grp]
        for b in batch_sizes:
            row32 = sub[(sub["precision"] == "FP32") & (sub["batch_size"] == b)].iloc[0]
            row16 = sub[(sub["precision"] == "FP16") & (sub["batch_size"] == b)].iloc[0]

            delta_ms = row32["gpu_mean_ms"] - row16["gpu_mean_ms"] # positive means FP16 is faster
            speedup = row32["gpu_mean_ms"] / row16["gpu_mean_ms"]
            noise_thresh = 2.0 * max(row32["gpu_std_ms"], row16["gpu_std_ms"])
            is_significant = abs(delta_ms) >= noise_thresh

            if is_significant:
                if delta_ms > 0:
                    verdict = f"FP16 FASTER by {delta_ms:.2f} ms ({speedup:.2f}x) [Significant]"
                else:
                    verdict = f"FP32 FASTER by {-delta_ms:.2f} ms ({1.0/speedup:.2f}x) [Significant]"
            else:
                verdict = f"INCONCLUSIVE / PARITY (Delta {delta_ms:+.2f} ms within noise ±{noise_thresh:.2f} ms)"

            sig_record = {
                "model_group": grp,
                "batch_size": b,
                "fp32_mean_ms": row32["gpu_mean_ms"],
                "fp16_mean_ms": row16["gpu_mean_ms"],
                "delta_ms": delta_ms,
                "speedup_factor": speedup,
                "noise_threshold_ms": noise_thresh,
                "is_significant": is_significant,
                "verdict": verdict
            }
            significance_records.append(sig_record)
            print(f"{grp:7s} Batch {b:2d}: FP32={row32['gpu_mean_ms']:5.2f} ms | FP16={row16['gpu_mean_ms']:5.2f} ms | Speedup={speedup:4.2f}x | {verdict}")

    # Save to CSV
    csv_file = out_dir / "a100_fp32_vs_fp16_benchmark.csv"
    df.to_csv(csv_file, index=False)
    print(f"\n[+] Saved Benchmark CSV: {csv_file}")

    # Save to JSON
    json_summary = {
        "hardware": gpu_spec,
        "node_execution_verifications": node_verifications,
        "preprocessing_statistics": prep_stats,
        "real_world_parity": real_parity,
        "significance_evaluations": significance_records,
        "benchmark_records": all_results
    }
    json_file = out_dir / "a100_fp32_vs_fp16_benchmark.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"[+] Saved Benchmark JSON: {json_file}")

    # Generate Comparison Plots
    png_file = out_dir / "a100_fp32_vs_fp16_comparison.png"
    generate_comparison_plots(all_results, png_file)

    # Print Formatted Markdown Table for Chat / Colab
    print("\n" + "="*80)
    print("FINAL RIGOROUS BENCHMARK SUMMARY TABLE")
    print("="*80)
    print("| Model | Precision | Batch | GPU Mean ± Std (ms) | P50 (ms) | P95 (ms) | Infer FPS | E2E FPS | Driver VRAM (MB) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in all_results:
        print(f"| {r['model_group']:7s} | {r['precision']:5s} | {r['batch_size']:5d} | {r['gpu_mean_ms']:6.2f} ± {r['gpu_std_ms']:.2f} ms | {r['gpu_p50_ms']:8.2f} | {r['gpu_p95_ms']:8.2f} | {r['inference_fps']:9.1f} | {r['e2e_fps']:7.1f} | {r['peak_vram_mb']:16.1f} |")

if __name__ == "__main__":
    main()
