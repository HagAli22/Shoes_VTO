"""
benchmark_onnx_cpu.py
─────────────────────
Level 1: Python CPU/GPU Benchmark for Stage A ONNX Models.
Measures:
  - Cold start time (first inference including model load)
  - Warm p50 / p95 latency (median and 95th percentile across N runs)
  - Throughput (FPS)
  - Memory footprint (process RSS before/after load)

Usage:
    python tools/benchmark_onnx_cpu.py
    python tools/benchmark_onnx_cpu.py --runs 200 --imgsz 320
"""

import argparse
import os
import time
import numpy as np
import psutil
import onnxruntime as ort

# ── Models to benchmark ────────────────────────────────────────────────────────
MODELS = {
    "FP32 (16-KP run_shuffled_v3)": "deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx",
    "FP16 (16-KP run_shuffled_v3)": "deliverables/stage_a_16kp/models/stage-a-320-16kp-fp16.onnx",
    "INT8 (16-KP run_shuffled_v3)": "deliverables/stage_a_16kp/models/stage-a-320-16kp-int8.onnx",
}


def measure_ram_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / 1e6


def benchmark_model(name: str, model_path: str, runs: int, imgsz: int) -> dict:
    if not os.path.exists(model_path):
        return {"error": f"File not found: {model_path}"}

    model_size_mb = os.path.getsize(model_path) / 1e6
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # ── Cold Start: includes session creation ─────────────────────────────────
    ram_before = measure_ram_mb()
    t_load_start = time.perf_counter()
    session = ort.InferenceSession(model_path, providers=providers)
    t_load_end = time.perf_counter()
    cold_load_ms = (t_load_end - t_load_start) * 1000.0

    actual_provider = session.get_providers()[0]
    ram_after = measure_ram_mb()
    ram_delta_mb = ram_after - ram_before

    # Prepare dummy input with correct dtype
    input_name = session.get_inputs()[0].name
    input_type = session.get_inputs()[0].type
    if "float16" in input_type or "fp16" in name.lower():
        dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float16)
    else:
        dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    # ── Cold Inference: first run after load ──────────────────────────────────
    t0 = time.perf_counter()
    session.run(None, {input_name: dummy})
    t1 = time.perf_counter()
    cold_inference_ms = (t1 - t0) * 1000.0

    # ── Warmup: 10 runs to fill caches ────────────────────────────────────────
    for _ in range(10):
        session.run(None, {input_name: dummy})

    # ── Warm Benchmark: N runs ────────────────────────────────────────────────
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    timings = np.array(timings)
    output_shape = session.run(None, {input_name: dummy})[0].shape

    return {
        "name": name,
        "model_size_mb": model_size_mb,
        "provider": actual_provider,
        "output_shape": str(output_shape),
        "cold_load_ms": cold_load_ms,
        "cold_inference_ms": cold_inference_ms,
        "warm_p50_ms": float(np.percentile(timings, 50)),
        "warm_p95_ms": float(np.percentile(timings, 95)),
        "warm_mean_ms": float(np.mean(timings)),
        "warm_min_ms": float(np.min(timings)),
        "warm_max_ms": float(np.max(timings)),
        "fps_p50": 1000.0 / float(np.percentile(timings, 50)),
        "ram_delta_mb": ram_delta_mb,
        "n_runs": runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX Stage A models on CPU/GPU.")
    parser.add_argument("--runs", type=int, default=100, help="Number of warm benchmark iterations")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("  SHOES VTO STAGE A — ONNX CPU/GPU BENCHMARK (LEVEL 1)")
    print(f"  Runs: {args.runs} warm iterations per model | Input: [1,3,{args.imgsz},{args.imgsz}]")
    print(f"  OnnxRuntime Version: {ort.__version__}")
    print(f"  Available Providers: {ort.get_available_providers()}")
    print("=" * 75 + "\n")

    results = []
    for name, path in MODELS.items():
        print(f"  Benchmarking: {name} ...")
        result = benchmark_model(name, path, args.runs, args.imgsz)
        results.append(result)

        if "error" in result:
            print(f"    [SKIP] {result['error']}\n")
            continue

        print(f"    Model Size  : {result['model_size_mb']:.2f} MB")
        print(f"    Provider    : {result['provider']}")
        print(f"    Output Shape: {result['output_shape']}")
        print(f"    Cold Load   : {result['cold_load_ms']:.1f} ms")
        print(f"    Cold Infer  : {result['cold_inference_ms']:.1f} ms")
        print(f"    Warm  p50   : {result['warm_p50_ms']:.2f} ms  ->  {result['fps_p50']:.1f} FPS")
        print(f"    Warm  p95   : {result['warm_p95_ms']:.2f} ms")
        print(f"    Warm  Min   : {result['warm_min_ms']:.2f} ms")
        print(f"    Warm  Max   : {result['warm_max_ms']:.2f} ms")
        print(f"    RAM Delta   : +{result['ram_delta_mb']:.1f} MB\n")

    # ── Final Summary Table ───────────────────────────────────────────────────
    print("=" * 75)
    print("  SUMMARY TABLE — Copy this into browser-performance.md")
    print("=" * 75)
    print(f"  {'Model':<35} {'Size':>7} {'Provider':>20} {'p50 ms':>8} {'p95 ms':>8} {'FPS':>8}")
    print("  " + "-" * 73)
    for r in results:
        if "error" in r:
            continue
        print(f"  {r['name']:<35} {r['model_size_mb']:>6.2f}MB {r['provider']:>20} {r['warm_p50_ms']:>8.2f} {r['warm_p95_ms']:>8.2f} {r['fps_p50']:>7.1f}")
    print("=" * 75)

    # ── Write real results to report ──────────────────────────────────────────
    import platform, datetime

    report_path = "deliverables/stage_a_16kp/reports/browser-performance.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"""# Stage A Browser & Mobile Performance Report

> **Measurement Method:** Automated Python benchmark using `onnxruntime {ort.__version__}`.  
> **Host Machine:** {platform.processor()} | {platform.system()} {platform.version()[:30]}  
> **Date:** {datetime.date.today()}  
> **Warm Iterations:** {args.runs} per model after 10 warm-up runs.

---

## 1. Local Machine Benchmark (ONNX Runtime — CPU / GPU)

| Model | File Size | Execution Provider | Cold Load | Cold Infer | Warm p50 | Warm p95 | FPS (p50) | RAM Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
""")
        for r in results:
            if "error" in r:
                continue
            f.write(f"| **{r['name']}** | {r['model_size_mb']:.2f} MB | {r['provider']} | {r['cold_load_ms']:.0f} ms | {r['cold_inference_ms']:.0f} ms | **{r['warm_p50_ms']:.1f} ms** | {r['warm_p95_ms']:.1f} ms | **{r['fps_p50']:.0f} FPS** | +{r['ram_delta_mb']:.0f} MB |\n")

        f.write("""
---

## 2. Browser WebGPU / WASM Benchmark

> ⚠️ **Measurements Pending** — Run `tools/benchmark_browser/index.html` in Chrome/Safari on target devices and paste results here.

| Device | Browser | Provider | Precision | Warm p50 | Warm p95 | FPS | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| _To be measured_ | _Chrome_ | _WebGPU_ | FP16 | _TBD_ | _TBD_ | _TBD_ | Open `tools/benchmark_browser/index.html` |
| _To be measured_ | _Safari_ | _WebGPU_ | FP16 | _TBD_ | _TBD_ | _TBD_ | Open `tools/benchmark_browser/index.html` |
| _To be measured_ | _Chrome_ | _WASM_ | INT8 | _TBD_ | _TBD_ | _TBD_ | Open `tools/benchmark_browser/index.html` |

---

## 3. Target Latency Requirements (from AI_TEAM_ACTION_REQUIRED.md)

| Stage | Requirement | Status |
| :--- | :--- | :--- |
| Stage A acquisition / recovery | ≤ 12 ms per invocation | 🔄 Pending browser measurement |
| Stage B per foot | ≤ 7 ms | 🔄 Stage B not yet delivered |
| Two Stage B + crop prep | ≤ 16 ms steady-state | 🔄 Pending |
""")

    print(f"\n  [✅] Updated report saved: {report_path}")
    print("  [!]  Browser WebGPU/WASM numbers still need measurement via index.html")


if __name__ == "__main__":
    main()
