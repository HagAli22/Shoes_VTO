#!/usr/bin/env python3
"""
Deep Investigation: NanoDet 284 CPU Nodes
Identifies:
1. Which nodes are shape/index math vs numerical feature map computations
2. Exact execution time in microseconds from ORT profiling trace
3. Graph location (Backbone, FPN, Head, Anchor Grid decoding)
4. Feasibility and impact of moving them to GPU
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper
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
NANO_FP32 = MODELS_DIR / "nanodet_stage_a_improved5_opset12.onnx"
NANO_FP16 = MODELS_DIR / "nanodet_stage_a_improved5_fp16.onnx"
OUT_DIR = REPO_ROOT / "deliverables/qualcomm_experiments/profiling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_nanodet_graph_and_profiling(model_path: Path):
    model = onnx.load(str(model_path))
    graph = model.graph

    # 1. Run ORT profiling trace
    cuda_opts = {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }
    sess_opts = ort.SessionOptions()
    sess_opts.enable_profiling = True
    prefix = str(OUT_DIR / f"nanodet_prof_{model_path.stem}")
    sess_opts.profile_file_prefix = prefix

    session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    dtype = np.float16 if "float16" in session.get_inputs()[0].type else np.float32
    dummy = np.random.randn(2, 3, 256, 256).astype(dtype)

    # Warmup
    for _ in range(10):
        session.run(None, {inp_name: dummy})
    torch.cuda.synchronize()

    # Profile runs (20 iterations)
    for _ in range(20):
        session.run(None, {inp_name: dummy})
    torch.cuda.synchronize()

    prof_file = session.end_profiling()

    with open(prof_file, "r", encoding="utf-8") as f:
        trace_data = json.load(f)
    events = trace_data if isinstance(trace_data, list) else trace_data.get("traceEvents", [])
    node_events = [e for e in events if e.get("cat") in ("Node", "Op") and "dur" in e]

    # Map node events
    cuda_events = []
    cpu_events = []
    for e in node_events:
        prov = e.get("args", {}).get("provider", "")
        if "CUDA" in prov:
            cuda_events.append(e)
        elif "CPU" in prov:
            cpu_events.append(e)

    # Clean profile file
    try:
        os.remove(prof_file)
    except Exception:
        pass

    # Aggregate CPU events per node name
    cpu_node_stats = {}
    for e in cpu_events:
        name = e.get("name", "")
        dur_us = e.get("dur", 0)
        op_name = e.get("args", {}).get("op_name", "")
        if name not in cpu_node_stats:
            cpu_node_stats[name] = {"op": op_name, "total_us": 0.0, "count": 0}
        cpu_node_stats[name]["total_us"] += dur_us
        cpu_node_stats[name]["count"] += 1

    total_cuda_dur_ms = sum(e.get("dur", 0) for e in cuda_events) / 1000.0 / 20.0 # per iteration
    total_cpu_dur_ms = sum(e.get("dur", 0) for e in cpu_events) / 1000.0 / 20.0

    # Categorize nodes from ONNX graph
    node_by_name = {n.name: n for n in graph.node}
    
    cpu_op_breakdown = {}
    shape_metadata_nodes = []
    numerical_math_nodes = []

    for name, st in cpu_node_stats.items():
        base_name = name.replace("_kernel_time", "")
        node = node_by_name.get(base_name)
        op = st["op"]
        cpu_op_breakdown[op] = cpu_op_breakdown.get(op, 0.0) + (st["total_us"] / 1000.0 / 20.0)

        # Check node inputs/outputs in ONNX graph
        # If node belongs to head grid generation vs backbone/fpn
        is_shape_only = False
        category = "Unknown"
        if "backbone" in base_name:
            category = "Backbone (Shape/Slicing)"
            is_shape_only = True
        elif "fpn" in base_name:
            category = "FPN (Shape/Slicing)"
            is_shape_only = True
        elif "head" in base_name or "Concat" in base_name or "Div" in base_name or "Add" in base_name or "Gather" in base_name or "Unsqueeze" in base_name:
            if op in ("Gather", "Unsqueeze", "Shape"):
                is_shape_only = True
                category = "Anchor Grid Coordinate Indexing"
            elif op in ("Div", "Add", "Concat"):
                is_shape_only = False
                category = "Anchor Coordinate Decoding Math"

        item = {
            "name": base_name,
            "op": op,
            "dur_ms_per_iter": st["total_us"] / 1000.0 / 20.0,
            "category": category,
            "is_shape_only": is_shape_only
        }
        if is_shape_only:
            shape_metadata_nodes.append(item)
        else:
            numerical_math_nodes.append(item)

    shape_dur_ms = sum(n["dur_ms_per_iter"] for n in shape_metadata_nodes)
    math_dur_ms = sum(n["dur_ms_per_iter"] for n in numerical_math_nodes)

    return {
        "model": model_path.name,
        "total_cuda_dur_ms": total_cuda_dur_ms,
        "total_cpu_dur_ms": total_cpu_dur_ms,
        "total_iter_dur_ms": total_cuda_dur_ms + total_cpu_dur_ms,
        "cpu_percentage": (total_cpu_dur_ms / (total_cuda_dur_ms + total_cpu_dur_ms) * 100.0),
        "cpu_node_count": len(cpu_node_stats),
        "cpu_op_breakdown_ms": cpu_op_breakdown,
        "shape_nodes_count": len(shape_metadata_nodes),
        "shape_nodes_dur_ms": shape_dur_ms,
        "math_nodes_count": len(numerical_math_nodes),
        "math_nodes_dur_ms": math_dur_ms,
        "top_slowest_cpu_nodes": sorted(list(cpu_node_stats.values()), key=lambda x: x["total_us"], reverse=True)[:10]
    }

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("================================================================================")
    print("NANODET 284 CPU NODES INVESTIGATION & RUNTIME MEASUREMENT")
    print("================================================================================")
    
    res32 = analyze_nanodet_graph_and_profiling(NANO_FP32)
    res16 = analyze_nanodet_graph_and_profiling(NANO_FP16)

    for res in [res32, res16]:
        print(f"\nMODEL: {res['model']}")
        print(f"  Total GPU Kernel Time:     {res['total_cuda_dur_ms']:.3f} ms")
        print(f"  Total CPU Operator Time:   {res['total_cpu_dur_ms']:.3f} ms")
        print(f"  Total Model Execution:     {res['total_iter_dur_ms']:.3f} ms")
        print(f"  CPU Operator Share:        {res['cpu_percentage']:.2f}% of total latency")
        print(f"  Total Unique CPU Nodes:    {res['cpu_node_count']}")
        print("\n  CPU Operator Runtime Breakdown:")
        for op, ms in sorted(res['cpu_op_breakdown_ms'].items(), key=lambda x: x[1], reverse=True):
            print(f"    - {op:15s}: {ms:7.4f} ms")
        print(f"\n  Functional Node Classification:")
        print(f"    - Shape / Index Metadata: {res['shape_nodes_count']} nodes -> {res['shape_nodes_dur_ms']:.4f} ms ({res['shape_nodes_dur_ms']/res['total_cpu_dur_ms']*100.0:.1f}% of CPU time)")
        print(f"    - Numerical Box Math:     {res['math_nodes_count']} nodes -> {res['math_nodes_dur_ms']:.4f} ms ({res['math_nodes_dur_ms']/res['total_cpu_dur_ms']*100.0:.1f}% of CPU time)")

    # Save to JSON
    out_json = OUT_DIR / "nanodet_cpu_nodes_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"nanodet_fp32": res32, "nanodet_fp16": res16}, f, indent=2)
    print(f"\n[+] Saved detailed analysis to: {out_json}")

if __name__ == "__main__":
    main()

