import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

# Ensure torch CUDA DLLs are found by ONNX Runtime
_tlib = Path(torch.__file__).parent / "lib"
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(_tlib))
    except Exception:
        pass
os.environ["PATH"] = str(_tlib) + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort

MODELS_DIR = Path("deliverables/qualcomm_experiments/models")
OUTPUT_DIR = Path("deliverables/qualcomm_experiments/profiling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RTMO_FP32 = MODELS_DIR / "rtmpose_13kp_hand_improved5_opset12.onnx"
RTMO_FP16 = MODELS_DIR / "rtmpose_13kp_hand_improved5_fp16.onnx"
NANO_FP32 = MODELS_DIR / "nanodet_stage_a_improved5_opset12.onnx"
NANO_FP16 = MODELS_DIR / "nanodet_stage_a_improved5_fp16.onnx"

def get_gpu_info():
    info = {
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "compute_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "None",
        "total_memory_mb": torch.cuda.get_device_properties(0).total_memory / (1024**2) if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "None",
        "ort_version": ort.__version__,
        "ort_providers": ort.get_available_providers()
    }
    return info

def create_session(model_path: Path, enable_cuda_graph: bool = False, enable_profiling: bool = False):
    cuda_opts = {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 3 * 1024 * 1024 * 1024,
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    if enable_cuda_graph:
        cuda_opts["enable_cuda_graph"] = True

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if enable_profiling:
        sess_opts.enable_profiling = True
        sess_opts.profile_file_prefix = str(OUTPUT_DIR / f"ort_profile_{model_path.stem}")

    providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
    return session

def analyze_profile_json(profile_file: str):
    with open(profile_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # In ORT profiling JSON, events are either in data directly or in data['traceEvents']
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    
    node_events = [e for e in events if e.get("cat") in ("Node", "Op") and "dur" in e]
    
    cuda_nodes = 0
    cpu_nodes = 0
    op_breakdown = {}
    provider_breakdown = {}
    node_durations = []

    for e in node_events:
        dur_us = e.get("dur", 0)
        args = e.get("args", {})
        provider = args.get("provider", "Unknown")
        op_name = args.get("op_name", e.get("name", "Unknown"))
        node_name = e.get("name", "Unknown")

        provider_breakdown[provider] = provider_breakdown.get(provider, 0.0) + dur_us
        if "CUDA" in provider:
            cuda_nodes += 1
        elif "CPU" in provider:
            cpu_nodes += 1

        if op_name not in op_breakdown:
            op_breakdown[op_name] = {"count": 0, "total_dur_us": 0.0, "provider": provider}
        op_breakdown[op_name]["count"] += 1
        op_breakdown[op_name]["total_dur_us"] += dur_us

        node_durations.append({
            "name": node_name,
            "op": op_name,
            "provider": provider,
            "dur_us": dur_us
        })

    # Sort node durations descending
    node_durations.sort(key=lambda x: x["dur_us"], reverse=True)
    # Sort op breakdown descending
    sorted_ops = sorted(op_breakdown.items(), key=lambda x: x[1]["total_dur_us"], reverse=True)

    return {
        "cuda_nodes": cuda_nodes,
        "cpu_nodes": cpu_nodes,
        "provider_breakdown_ms": {k: v / 1000.0 for k, v in provider_breakdown.items()},
        "top_ops": [{"op": k, "count": v["count"], "total_ms": v["total_dur_us"] / 1000.0, "provider": v["provider"]} for k, v in sorted_ops[:10]],
        "top_slowest_nodes": [{"name": n["name"], "op": n["op"], "dur_ms": n["dur_us"] / 1000.0, "provider": n["provider"]} for n in node_durations[:10]]
    }

def benchmark_inference(model_path: Path, batch_size: int = 1, warmup: int = 30, iters: int = 100, use_io_binding: bool = False, enable_cuda_graph: bool = False):
    session = create_session(model_path, enable_cuda_graph=enable_cuda_graph, enable_profiling=False)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_type = input_info.type # 'tensor(float)' or 'tensor(float16)'

    dtype = np.float32 if "float16" not in input_type else np.float16
    shape = (batch_size, 3, 256, 256)
    np_input = np.random.randn(*shape).astype(dtype)

    # I/O Binding setup if requested
    io_binding = None
    if use_io_binding:
        io_binding = session.io_binding()
        torch_dtype = torch.float32 if dtype == np.float32 else torch.float16
        t_input = torch.from_numpy(np_input).cuda()
        io_binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=0,
            element_type=dtype,
            shape=shape,
            buffer_ptr=t_input.data_ptr()
        )
        for out in session.get_outputs():
            io_binding.bind_output(out.name, "cuda")

    # Warmup
    for _ in range(warmup):
        if use_io_binding:
            session.run_with_iobinding(io_binding)
        else:
            session.run(None, {input_name: np_input})
    torch.cuda.synchronize()

    # Benchmark with CUDA Events (GPU Time)
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    cpu_wall_times = []

    for i in range(iters):
        t0 = time.perf_counter()
        start_events[i].record()
        if use_io_binding:
            session.run_with_iobinding(io_binding)
        else:
            session.run(None, {input_name: np_input})
        end_events[i].record()
        t1 = time.perf_counter()
        cpu_wall_times.append((t1 - t0) * 1000.0)

    torch.cuda.synchronize()

    gpu_latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    return {
        "batch_size": batch_size,
        "input_type": input_type,
        "gpu_p50_ms": float(np.percentile(gpu_latencies, 50)),
        "gpu_mean_ms": float(np.mean(gpu_latencies)),
        "gpu_p95_ms": float(np.percentile(gpu_latencies, 95)),
        "cpu_wall_p50_ms": float(np.percentile(cpu_wall_times, 50)),
        "cpu_wall_mean_ms": float(np.mean(cpu_wall_times)),
        "cpu_wall_p95_ms": float(np.percentile(cpu_wall_times, 95)),
        "use_io_binding": use_io_binding,
        "enable_cuda_graph": enable_cuda_graph
    }

def benchmark_data_transfers_and_stages(model_path: Path, batch_size: int = 2):
    session = create_session(model_path, enable_cuda_graph=False, enable_profiling=False)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_type = input_info.type
    dtype = np.float32 if "float16" not in input_type else np.float16
    torch_dtype = torch.float32 if dtype == np.float32 else torch.float16

    # 1. Preprocessing timing (synthetic 2 crops 256x256 BGR uint8)
    crop = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    prep_times = []
    for _ in range(50):
        t0 = time.perf_counter()
        # standard preprocess
        img = crop[:, :, ::-1].astype(dtype) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=dtype)) / np.array([0.229, 0.224, 0.225], dtype=dtype)
        tensor = np.transpose(img, (2, 0, 1))
        batch = np.stack([tensor] * batch_size, axis=0)
        t1 = time.perf_counter()
        prep_times.append((t1 - t0) * 1000.0)

    # 2. Host-to-Device transfer timing
    h2d_times = []
    d2h_times = []
    cpu_tensor = torch.from_numpy(batch).pin_memory()
    gpu_tensor = torch.empty(batch.shape, dtype=torch_dtype, device="cuda")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(50):
        start_event.record()
        gpu_tensor.copy_(cpu_tensor, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        h2d_times.append(start_event.elapsed_time(end_event))

    # D2H transfer timing (assume simulated 512x13 output)
    out_cpu = torch.empty((batch_size, 13, 512), dtype=torch_dtype).pin_memory()
    out_gpu = torch.randn((batch_size, 13, 512), dtype=torch_dtype, device="cuda")
    for _ in range(50):
        start_event.record()
        out_cpu.copy_(out_gpu, non_blocking=True)
        end_event.record()
        torch.cuda.synchronize()
        d2h_times.append(start_event.elapsed_time(end_event))

    return {
        "preprocessing_mean_ms": float(np.mean(prep_times)),
        "h2d_copy_mean_ms": float(np.mean(h2d_times)),
        "d2h_copy_mean_ms": float(np.mean(d2h_times))
    }

def run_ort_profiling(model_path: Path, batch_size: int = 2):
    session = create_session(model_path, enable_cuda_graph=False, enable_profiling=True)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    dtype = np.float32 if "float16" not in input_info.type else np.float16
    np_input = np.random.randn(batch_size, 3, 256, 256).astype(dtype)

    # Run warm up
    for _ in range(10):
        session.run(None, {input_name: np_input})
    torch.cuda.synchronize()

    # Profile run (10 iterations)
    for _ in range(20):
        session.run(None, {input_name: np_input})
    torch.cuda.synchronize()

    prof_file = session.end_profiling()
    analysis = analyze_profile_json(prof_file)
    analysis["profile_file"] = prof_file
    return analysis

def measure_memory_and_utilization(model_path: Path, batch_size: int = 2):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated() / (1024**2)

    session = create_session(model_path, enable_cuda_graph=False, enable_profiling=False)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    dtype = np.float32 if "float16" not in input_info.type else np.float16
    np_input = np.random.randn(batch_size, 3, 256, 256).astype(dtype)

    for _ in range(20):
        session.run(None, {input_name: np_input})
    torch.cuda.synchronize()

    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    mem_after = torch.cuda.memory_allocated() / (1024**2)

    # Read nvidia-smi if possible
    import subprocess
    smi_out = ""
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        ).strip()
    except Exception as e:
        smi_out = str(e)

    return {
        "torch_mem_before_mb": mem_before,
        "torch_peak_mem_mb": peak_mem,
        "torch_mem_after_mb": mem_after,
        "nvidia_smi_snapshot": smi_out
    }

def main():
    gpu_info = get_gpu_info()
    print("================================================================================")
    print("QUADRO P2000 PROFILING: FP32 VS FP16 INVESTIGATION")
    print("================================================================================")
    print(f"GPU: {gpu_info['device_name']} | Compute Capability: {gpu_info['compute_capability']}")
    print(f"VRAM: {gpu_info['total_memory_mb']:.1f} MB | CUDA: {gpu_info['cuda_version']} | cuDNN: {gpu_info['cudnn_version']}")
    print(f"ONNX Runtime: {gpu_info['ort_version']} | Providers: {gpu_info['ort_providers']}")
    print("--------------------------------------------------------------------------------")

    results = {
        "gpu_info": gpu_info,
        "models": {}
    }

    test_models = [
        ("RTMPose_FP32", RTMO_FP32),
        ("RTMPose_FP16", RTMO_FP16),
        ("NanoDet_FP32", NANO_FP32),
        ("NanoDet_FP16", NANO_FP16),
    ]

    for label, path in test_models:
        print(f"\n[+] Profiling Model: {label} ({path.name})...")
        model_res = {}

        # 1. ORT Operator Profiling
        print("  - Running ORT operator-level trace...")
        prof_res = run_ort_profiling(path, batch_size=2)
        model_res["profiling"] = prof_res
        print(f"    CUDA Operators: {prof_res['cuda_nodes']}, CPU Operators: {prof_res['cpu_nodes']}")
        print(f"    Provider durations: {prof_res['provider_breakdown_ms']}")

        # 2. Benchmarking Batch 1 & Batch 2 (Standard)
        print("  - Benchmarking standard inference (Batch 1 & 2)...")
        b1_res = benchmark_inference(path, batch_size=1, warmup=30, iters=100, use_io_binding=False)
        b2_res = benchmark_inference(path, batch_size=2, warmup=30, iters=100, use_io_binding=False)
        model_res["bench_b1_standard"] = b1_res
        model_res["bench_b2_standard"] = b2_res
        print(f"    Batch 1 GPU Mean: {b1_res['gpu_mean_ms']:.2f} ms (P50: {b1_res['gpu_p50_ms']:.2f} ms, P95: {b1_res['gpu_p95_ms']:.2f} ms)")
        print(f"    Batch 2 GPU Mean: {b2_res['gpu_mean_ms']:.2f} ms (P50: {b2_res['gpu_p50_ms']:.2f} ms, P95: {b2_res['gpu_p95_ms']:.2f} ms)")

        # 3. Benchmarking with I/O Binding
        print("  - Benchmarking with I/O Binding (Batch 2)...")
        try:
            b2_iob = benchmark_inference(path, batch_size=2, warmup=30, iters=100, use_io_binding=True)
            model_res["bench_b2_io_binding"] = b2_iob
            print(f"    Batch 2 I/O Binding GPU Mean: {b2_iob['gpu_mean_ms']:.2f} ms")
        except Exception as e:
            print(f"    [!] I/O Binding failed: {e}")
            model_res["bench_b2_io_binding"] = {"error": str(e)}

        # 4. Benchmarking with CUDA Graph (if supported)
        print("  - Testing CUDA Graphs (Batch 2)...")
        try:
            b2_cg = benchmark_inference(path, batch_size=2, warmup=30, iters=100, enable_cuda_graph=True)
            model_res["bench_b2_cuda_graph"] = b2_cg
            print(f"    Batch 2 CUDA Graph GPU Mean: {b2_cg['gpu_mean_ms']:.2f} ms")
        except Exception as e:
            print(f"    [!] CUDA Graph not supported or failed: {e}")
            model_res["bench_b2_cuda_graph"] = {"error": str(e)}

        # 5. Data Transfers & Stages
        print("  - Measuring Transfers & Preprocessing...")
        trans_res = benchmark_data_transfers_and_stages(path, batch_size=2)
        model_res["transfers_and_prep"] = trans_res
        print(f"    Prep: {trans_res['preprocessing_mean_ms']:.3f} ms | H2D: {trans_res['h2d_copy_mean_ms']:.3f} ms | D2H: {trans_res['d2h_copy_mean_ms']:.3f} ms")

        # 6. Memory
        mem_res = measure_memory_and_utilization(path, batch_size=2)
        model_res["memory"] = mem_res
        print(f"    Peak VRAM Alloc: {mem_res['torch_peak_mem_mb']:.2f} MB | nvidia-smi: {mem_res['nvidia_smi_snapshot']}")

        results["models"][label] = model_res

    # Save complete JSON
    out_json = OUTPUT_DIR / "fp32_vs_fp16_quadro_p2000_profile.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved full profiling report to: {out_json}")

if __name__ == "__main__":
    main()

