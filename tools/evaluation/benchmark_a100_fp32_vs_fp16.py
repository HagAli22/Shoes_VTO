#!/usr/bin/env python3
"""
Comprehensive NVIDIA A100 (and Ampere / CUDA) Benchmark: FP32 vs FP16
Shoes VTO Two-Stage Pipeline (RTMPose-13KP + NanoDet-Plus Stage A)

Verifies:
1. GPU Hardware & Compute Capability (A100 CC 8.0 verification)
2. CUDAExecutionProvider active
3. Batch sizes 1, 2, 4, 8 for FP32 and FP16 models
4. Synchronized CUDA Events timing (Pure Inference vs Pre/Post vs Transfers)
5. Numerical accuracy / parity (Keypoint pixel MAE, cosine similarity, max error)
6. VRAM peak allocation and throughput (FPS)
7. CSV report, JSON summary, and comparison plots
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

# Ensure torch CUDA DLLs are found on Windows / Linux
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

def create_session(model_path: Path):
    cuda_opts = {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
    return session

def simulate_rtmpose_preprocess(batch_size: int, dtype: np.dtype):
    # Simulated 256x256 BGR crops
    crop = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    t0 = time.perf_counter()
    img = crop[:, :, ::-1].astype(dtype) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=dtype)
    std = np.array([0.229, 0.224, 0.225], dtype=dtype)
    img = (img - mean) / std
    chw = np.transpose(img, (2, 0, 1))
    batch = np.ascontiguousarray(np.stack([chw] * batch_size, axis=0))
    t1 = time.perf_counter()
    return batch, (t1 - t0) * 1000.0

def decode_rtmpose_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray):
    """
    SimCC post-processing: argmax over 512 representation bins to pixel coords.
    simcc_x: [B, K, 512], simcc_y: [B, K, 512]
    """
    t0 = time.perf_counter()
    # Argmax over representation bins
    x_idx = np.argmax(simcc_x, axis=-1) / 2.0  # 512 bins / 2 = 256 image coords
    y_idx = np.argmax(simcc_y, axis=-1) / 2.0
    t1 = time.perf_counter()
    return x_idx, y_idx, (t1 - t0) * 1000.0

def decode_nanodet_boxes(boxes: np.ndarray, scores: np.ndarray, score_thresh: float = 0.35):
    t0 = time.perf_counter()
    # boxes: [B, N, 4], scores: [B, N, 1]
    mask = scores > score_thresh
    filtered = boxes[mask.squeeze(-1)]
    t1 = time.perf_counter()
    return filtered, (t1 - t0) * 1000.0

def benchmark_model_batch(session, model_path: Path, batch_size: int, warmup: int = 50, iters: int = 100):
    inp = session.get_inputs()[0]
    inp_name = inp.name
    is_fp16 = "float16" in inp.type
    dtype = np.float16 if is_fp16 else np.float32

    # Measure preprocessing
    sample_batch, prep_ms = simulate_rtmpose_preprocess(batch_size, dtype)
    feed = {inp_name: sample_batch}

    # Warmup
    for _ in range(warmup):
        session.run(None, feed)
    if torch and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Synchronized benchmark
    has_cuda = torch is not None and torch.cuda.is_available()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)] if has_cuda else []
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)] if has_cuda else []
    wall_latencies = []

    if has_cuda:
        torch.cuda.reset_peak_memory_stats()

    last_outputs = None
    for i in range(iters):
        t0 = time.perf_counter()
        if has_cuda:
            start_events[i].record()
        out = session.run(None, feed)
        if has_cuda:
            end_events[i].record()
        t1 = time.perf_counter()
        wall_latencies.append((t1 - t0) * 1000.0)
        last_outputs = out

    if has_cuda:
        torch.cuda.synchronize()
        gpu_latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        gpu_latencies = wall_latencies
        peak_vram_mb = 0.0

    # Measure post-processing
    is_rtmpose = "rtmpose" in model_path.name.lower()
    if is_rtmpose:
        _, _, post_ms = decode_rtmpose_simcc(last_outputs[0], last_outputs[1])
    else:
        _, post_ms = decode_nanodet_boxes(last_outputs[0], last_outputs[1])

    mean_gpu = float(np.mean(gpu_latencies))
    p50_gpu = float(np.percentile(gpu_latencies, 50))
    p95_gpu = float(np.percentile(gpu_latencies, 95))
    fps = (batch_size / (mean_gpu / 1000.0)) if mean_gpu > 0 else 0.0
    e2e_total_ms = prep_ms + mean_gpu + post_ms
    e2e_fps = (batch_size / (e2e_total_ms / 1000.0)) if e2e_total_ms > 0 else 0.0

    return {
        "batch_size": batch_size,
        "is_fp16": is_fp16,
        "gpu_mean_ms": mean_gpu,
        "gpu_p50_ms": p50_gpu,
        "gpu_p95_ms": p95_gpu,
        "inference_fps": fps,
        "prep_ms": prep_ms,
        "post_ms": post_ms,
        "e2e_latency_ms": e2e_total_ms,
        "e2e_fps": e2e_fps,
        "peak_vram_mb": peak_vram_mb,
        "last_outputs": [np.array(o) for o in last_outputs]
    }

def verify_numerical_accuracy(fp32_session, fp16_session, model_name: str, batch_size: int = 2):
    """
    Feeds identical input tensor to FP32 and FP16 models and computes numerical deviation.
    """
    # Deterministic test input
    np.random.seed(42)
    inp32 = np.random.randn(batch_size, 3, 256, 256).astype(np.float32)
    inp16 = inp32.astype(np.float16)

    inp_name = fp32_session.get_inputs()[0].name
    out32 = fp32_session.run(None, {inp_name: inp32})
    out16 = fp16_session.run(None, {inp_name: inp16})

    res = {
        "model": model_name,
        "batch_size": batch_size,
        "outputs_tested": len(out32)
    }

    if "rtmpose" in model_name.lower():
        # Outputs: 0: simcc_x, 1: simcc_y, 2: vis_logits, 3: vis, 4: hand_logits, 5: hand
        simcc_x_32, simcc_x_16 = out32[0], out16[0].astype(np.float32)
        simcc_y_32, simcc_y_16 = out32[1], out16[1].astype(np.float32)

        # Raw representation error
        mae_x = float(np.mean(np.abs(simcc_x_32 - simcc_x_16)))
        mae_y = float(np.mean(np.abs(simcc_y_32 - simcc_y_16)))
        max_err_x = float(np.max(np.abs(simcc_x_32 - simcc_x_16)))
        max_err_y = float(np.max(np.abs(simcc_y_32 - simcc_y_16)))

        # Argmax pixel coordinate error
        x_pix_32, y_pix_32, _ = decode_rtmpose_simcc(simcc_x_32, simcc_y_32)
        x_pix_16, y_pix_16, _ = decode_rtmpose_simcc(simcc_x_16, simcc_y_16)

        pixel_diff_x = np.abs(x_pix_32 - x_pix_16)
        pixel_diff_y = np.abs(y_pix_32 - y_pix_16)
        pixel_dist = np.sqrt(pixel_diff_x**2 + pixel_diff_y**2)

        mean_pixel_error = float(np.mean(pixel_dist))
        max_pixel_error = float(np.max(pixel_dist))
        exact_match_ratio = float(np.mean(pixel_dist == 0.0))

        # Cosine similarity across heatmaps
        cos_sim_x = float(np.dot(simcc_x_32.flatten(), simcc_x_16.flatten()) / (np.linalg.norm(simcc_x_32) * np.linalg.norm(simcc_x_16) + 1e-9))
        cos_sim_y = float(np.dot(simcc_y_32.flatten(), simcc_y_16.flatten()) / (np.linalg.norm(simcc_y_32) * np.linalg.norm(simcc_y_16) + 1e-9))

        res.update({
            "simcc_mae": (mae_x + mae_y) / 2.0,
            "simcc_max_error": max(max_err_x, max_err_y),
            "simcc_cosine_similarity": (cos_sim_x + cos_sim_y) / 2.0,
            "mean_keypoint_pixel_error": mean_pixel_error,
            "max_keypoint_pixel_error": max_pixel_error,
            "exact_pixel_match_percent": exact_match_ratio * 100.0,
            "status": "PASS" if mean_pixel_error < 0.5 else "WARN"
        })
    else:
        # NanoDet: 0: boxes, 1: scores
        box32, box16 = out32[0], out16[0].astype(np.float32)
        score32, score16 = out32[1], out16[1].astype(np.float32)

        box_mae = float(np.mean(np.abs(box32 - box16)))
        box_max_err = float(np.max(np.abs(box32 - box16)))
        score_mae = float(np.mean(np.abs(score32 - score16)))
        score_max_err = float(np.max(np.abs(score32 - score16)))

        res.update({
            "box_coord_mae": box_mae,
            "box_coord_max_error": box_max_err,
            "score_mae": score_mae,
            "score_max_error": score_max_err,
            "status": "PASS" if box_mae < 1.0 and score_mae < 0.01 else "WARN"
        })

    return res

def generate_comparison_plots(df_rows, out_png: Path):
    if plt is None:
        print("[!] matplotlib not installed. Skipping plot generation.")
        return

    import pandas as pd
    df = pd.DataFrame(df_rows)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NVIDIA A100: FP32 vs FP16 Benchmark (Shoes VTO Two-Stage Pipeline)", fontsize=14, fontweight="bold")

    # 1. RTMPose Latency
    ax = axes[0, 0]
    rtm = df[df["model_group"] == "RTMPose"]
    rtm_32 = rtm[rtm["precision"] == "FP32"].sort_values("batch_size")
    rtm_16 = rtm[rtm["precision"] == "FP16"].sort_values("batch_size")
    ax.plot(rtm_32["batch_size"], rtm_32["gpu_mean_ms"], "o-", label="FP32 (opset12)", color="#1f77b4", linewidth=2)
    ax.plot(rtm_16["batch_size"], rtm_16["gpu_mean_ms"], "s--", label="FP16 (fp16)", color="#2ca02c", linewidth=2)
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

    # 3. NanoDet Latency
    ax = axes[1, 0]
    nano = df[df["model_group"] == "NanoDet"]
    nano_32 = nano[nano["precision"] == "FP32"].sort_values("batch_size")
    nano_16 = nano[nano["precision"] == "FP16"].sort_values("batch_size")
    ax.plot(nano_32["batch_size"], nano_32["gpu_mean_ms"], "o-", label="FP32 (opset12)", color="#ff7f0e", linewidth=2)
    ax.plot(nano_16["batch_size"], nano_16["gpu_mean_ms"], "s--", label="FP16 (fp16)", color="#d62728", linewidth=2)
    ax.set_title("NanoDet Stage A: GPU Latency (ms)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms) [Lower is better]")
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # 4. FP16 Speedup Factor on A100 vs Quadro P2000
    ax = axes[1, 1]
    # Speedup = FP32_ms / FP16_ms (>1 means FP16 is faster)
    rtm_speedup = rtm_32["gpu_mean_ms"].values / rtm_16["gpu_mean_ms"].values
    nano_speedup = nano_32["gpu_mean_ms"].values / nano_16["gpu_mean_ms"].values
    x_pos = np.arange(len(rtm_32["batch_size"]))
    ax.bar(x_pos - 0.2, rtm_speedup, width=0.4, label="RTMPose FP16 Speedup", color="#2ca02c")
    ax.bar(x_pos + 0.2, nano_speedup, width=0.4, label="NanoDet FP16 Speedup", color="#d62728")
    ax.axhline(1.0, color="red", linestyle="--", label="Parity (1.0x)")
    ax.axhline(0.60, color="gray", linestyle=":", label="Quadro P2000 Baseline (0.60x)")
    ax.set_title("FP16 Speedup Factor (FP32 Latency / FP16 Latency)")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Speedup Ratio (> 1.0 means FP16 faster)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    plt.tight_layout()
    plt.savefig(str(out_png), dpi=300)
    plt.close()
    print(f"[+] Saved comparison plot: {out_png}")

def main():
    parser = argparse.ArgumentParser(description="A100 FP32 vs FP16 Benchmark")
    parser.add_argument("--models_dir", type=str, default="deliverables/qualcomm_experiments/models", help="Path to models directory")
    parser.add_argument("--output_dir", type=str, default="deliverables/qualcomm_experiments/profiling_a100", help="Output directory")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    if not models_dir.exists() and (REPO_ROOT / args.models_dir).exists():
        models_dir = REPO_ROOT / args.models_dir

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_spec = get_gpu_spec()

    print("================================================================================")
    print("NVIDIA A100 BENCHMARK: FP32 VS FP16 COMPARISON (SHOES VTO)")
    print("================================================================================")
    print(f"Device Name:        {gpu_spec['name']}")
    print(f"Compute Capability: {gpu_spec['compute_capability']}")
    print(f"Total VRAM:         {gpu_spec['total_vram_gb']:.2f} GB")
    print(f"CUDA Version:       {gpu_spec['cuda_version']}")
    print(f"cuDNN Version:      {gpu_spec['cudnn_version']}")
    print(f"ORT Available EPs:  {ort.get_available_providers()}")
    print("--------------------------------------------------------------------------------")

    if not gpu_spec["is_a100"]:
        print(f"[!] NOTICE: Running on {gpu_spec['name']} (Target requested: NVIDIA A100 CC 8.0).")
        print("    Benchmark will execute with full CUDA synchronizations and metrics regardless.")
    else:
        print("[✓] Verified Hardware: NVIDIA A100 Tensor Core GPU (Compute Capability 8.0)")

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("[!] FATAL: CUDAExecutionProvider is NOT available in ONNX Runtime!")
        print("    In Colab, run:")
        print("    !pip uninstall -y onnxruntime")
        print("    !pip install onnxruntime-gpu")
        sys.exit(1)
    else:
        print("[✓] Verified Execution Provider: CUDAExecutionProvider")

    models_to_test = [
        ("RTMPose", "FP32", models_dir / "rtmpose_13kp_hand_improved5_opset12.onnx"),
        ("RTMPose", "FP16", models_dir / "rtmpose_13kp_hand_improved5_fp16.onnx"),
        ("NanoDet", "FP32", models_dir / "nanodet_stage_a_improved5_opset12.onnx"),
        ("NanoDet", "FP16", models_dir / "nanodet_stage_a_improved5_fp16.onnx"),
    ]

    # Verify model files exist
    for grp, prec, path in models_to_test:
        if not path.exists():
            print(f"[!] Error: Model file {path} not found!")
            sys.exit(1)

    batch_sizes = [1, 2, 4, 8]
    all_results = []
    sessions = {}

    # Initialize sessions
    for grp, prec, path in models_to_test:
        key = f"{grp}_{prec}"
        print(f"\n[+] Initializing Session: {key} ({path.name})...")
        s = create_session(path)
        providers_active = s.get_providers()
        print(f"    Active Providers: {providers_active}")
        sessions[key] = (s, path)

    # 1. Numerical Parity Verification
    print("\n" + "="*80)
    print("NUMERICAL ACCURACY / PARITY VERIFICATION (FP32 VS FP16)")
    print("="*80)
    acc_rtmpose = verify_numerical_accuracy(sessions["RTMPose_FP32"][0], sessions["RTMPose_FP16"][0], "RTMPose-13KP", batch_size=2)
    acc_nanodet = verify_numerical_accuracy(sessions["NanoDet_FP32"][0], sessions["NanoDet_FP16"][0], "NanoDet-StageA", batch_size=2)

    print(f"RTMPose Numerical Parity:")
    print(f"  - SimCC Heatmap MAE:         {acc_rtmpose['simcc_mae']:.6f}")
    print(f"  - Cosine Similarity:         {acc_rtmpose['simcc_cosine_similarity']:.6f}")
    print(f"  - Mean Keypoint Pixel Error: {acc_rtmpose['mean_keypoint_pixel_error']:.4f} px (on 256x256)")
    print(f"  - Max Keypoint Pixel Error:  {acc_rtmpose['max_keypoint_pixel_error']:.4f} px")
    print(f"  - Exact Pixel Match:         {acc_rtmpose['exact_pixel_match_percent']:.2f}%")
    print(f"  - Parity Status:             [{acc_rtmpose['status']}]")

    print(f"\nNanoDet Numerical Parity:")
    print(f"  - Box Coordinate MAE:        {acc_nanodet['box_coord_mae']:.6f} px")
    print(f"  - Confidence Score MAE:      {acc_nanodet['score_mae']:.6f}")
    print(f"  - Parity Status:             [{acc_nanodet['status']}]")

    # 2. Latency & Throughput Benchmark
    print("\n" + "="*80)
    print("LATENCY & THROUGHPUT BENCHMARK (Batch Sizes 1, 2, 4, 8)")
    print("="*80)

    for grp, prec, path in models_to_test:
        key = f"{grp}_{prec}"
        s = sessions[key][0]
        print(f"\n---> Benchmarking {key}...")
        for b in batch_sizes:
            res = benchmark_model_batch(s, path, batch_size=b, warmup=args.warmup, iters=args.iters)
            row = {
                "gpu": gpu_spec["name"],
                "model_group": grp,
                "precision": prec,
                "batch_size": b,
                "gpu_mean_ms": res["gpu_mean_ms"],
                "gpu_p50_ms": res["gpu_p50_ms"],
                "gpu_p95_ms": res["gpu_p95_ms"],
                "inference_fps": res["inference_fps"],
                "prep_ms": res["prep_ms"],
                "post_ms": res["post_ms"],
                "e2e_latency_ms": res["e2e_latency_ms"],
                "e2e_fps": res["e2e_fps"],
                "peak_vram_mb": res["peak_vram_mb"]
            }
            all_results.append(row)
            print(f"  Batch {b}: GPU Mean = {res['gpu_mean_ms']:6.2f} ms | P50 = {res['gpu_p50_ms']:6.2f} ms | P95 = {res['gpu_p95_ms']:6.2f} ms | FPS = {res['inference_fps']:7.1f} | E2E FPS = {res['e2e_fps']:7.1f}")

    # Save to CSV
    csv_file = out_dir / "a100_fp32_vs_fp16_benchmark.csv"
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(csv_file, index=False)
    print(f"\n[+] Saved Benchmark CSV: {csv_file}")

    # Save to JSON
    json_summary = {
        "hardware": gpu_spec,
        "numerical_parity": {
            "rtmpose": acc_rtmpose,
            "nanodet": acc_nanodet
        },
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
    print("BENCHMARK SUMMARY TABLE")
    print("="*80)
    print("| Model | Precision | Batch | GPU Mean (ms) | P50 (ms) | P95 (ms) | Infer FPS | E2E FPS | VRAM (MB) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in all_results:
        print(f"| {r['model_group']:7s} | {r['precision']:5s} | {r['batch_size']:5d} | {r['gpu_mean_ms']:13.2f} | {r['gpu_p50_ms']:8.2f} | {r['gpu_p95_ms']:8.2f} | {r['inference_fps']:9.1f} | {r['e2e_fps']:7.1f} | {r['peak_vram_mb']:9.1f} |")

if __name__ == "__main__":
    main()

