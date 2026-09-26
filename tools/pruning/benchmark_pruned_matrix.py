"""
benchmark_pruned_matrix.py
──────────────────────────
Unified Evaluation & Benchmarking Suite across 16-KP and 4-KP Model Families.

Computes a complete side-by-side comparison table including:
  - Parameters & Architecture Family
  - Format (PyTorch vs FP32 ONNX)
  - FP32 File Size (MB) & Target Budget Compliance (< 5.0 MB)
  - Detection Metrics (Precision, Recall, mAP50)
  - Keypoint Metrics (Pose Precision, Recall, mAP50, mAP50-95)
  - Inference Latency (ms) & FPS (averaged across 50 runs)
  - Outputs structured report to experiments/reports/final_comparison.csv
"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def resolve_data_yaml(candidate_yaml: str, family: str) -> str:
    """Finds existing data yaml file or returns candidate."""
    if os.path.exists(candidate_yaml):
        return candidate_yaml
    
    fallbacks_4kp = [
        "configs/shuffled_v3_4kp.yaml",
        "data/shuffled_v3_4kp/data.yaml",
        "configs/shoes_v2_4kp.yaml",
    ]
    fallbacks_16kp = [
        "configs/shuffled_v3_16kp.yaml",
        "data/shuffled_v3/data.yaml",
        "configs/shuffled_v3_4kp.yaml",
    ]
    
    candidates = fallbacks_4kp if family == "4-KP" else fallbacks_16kp
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidate_yaml


def benchmark_single_model(
    model_path: str,
    data_yaml: str,
    device: str = "cpu",
    family: str = "4-KP",
) -> dict:
    is_onnx = model_path.endswith(".onnx")
    file_size_mb = os.path.getsize(model_path) / (1024.0 * 1024.0)
    
    resolved_yaml = resolve_data_yaml(data_yaml, family)
    has_dataset = os.path.exists(resolved_yaml)
    
    box_p, box_r, box_map50 = 0.0, 0.0, 0.0
    pose_p, pose_r, pose_map50, pose_map50_95 = 0.0, 0.0, 0.0, 0.0
    
    # Validation via Ultralytics if dataset exists
    if has_dataset:
        try:
            model = YOLO(model_path)
            # Use CPU for onnx validation to avoid provider mismatch warnings
            val_device = "cpu" if is_onnx else device
            val_res = model.val(data=resolved_yaml, imgsz=320, split="test", device=val_device, verbose=False)
            
            box_p = float(val_res.box.p.mean()) if hasattr(val_res.box, "p") and len(val_res.box.p) > 0 else float(val_res.box.mp)
            box_r = float(val_res.box.r.mean()) if hasattr(val_res.box, "r") and len(val_res.box.r) > 0 else float(val_res.box.mr)
            box_map50 = float(val_res.box.map50)
            
            pose_p = float(val_res.pose.p.mean()) if hasattr(val_res.pose, "p") and len(val_res.pose.p) > 0 else float(val_res.pose.mp)
            pose_r = float(val_res.pose.r.mean()) if hasattr(val_res.pose, "r") and len(val_res.pose.r) > 0 else float(val_res.pose.mr)
            pose_map50 = float(val_res.pose.map50)
            pose_map50_95 = float(val_res.pose.map)
        except Exception as e:
            print(f"    [Warning] Val metrics skipped for {model_path}: {e}")
    else:
        print(f"    [Notice] Dataset yaml '{data_yaml}' not found on disk. Benchmarking latency & size only.")

    # Latency profiling
    dummy = np.random.randn(1, 3, 320, 320).astype(np.float32)
    latencies = []
    
    try:
        if is_onnx:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            in_name = session.get_inputs()[0].name
            # Warmup
            for _ in range(10):
                session.run(None, {in_name: dummy})
            # Measure
            for _ in range(50):
                t0 = time.time()
                session.run(None, {in_name: dummy})
                latencies.append((time.time() - t0) * 1000.0)
        else:
            model = YOLO(model_path)
            net = model.model
            net.eval()
            t_dummy = torch.from_numpy(dummy)
            with torch.no_grad():
                for _ in range(10):
                    net(t_dummy)
                for _ in range(50):
                    t0 = time.time()
                    net(t_dummy)
                    latencies.append((time.time() - t0) * 1000.0)
    except Exception as e:
        print(f"    [Warning] Latency benchmark error: {e}")
        latencies = [0.0]
                
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    return {
        "Model_Name": os.path.basename(model_path),
        "Family": family,
        "Format": "ONNX" if is_onnx else "PyTorch",
        "File_Size_MB": round(file_size_mb, 2),
        "Budget_Compliant (<5MB)": "YES" if file_size_mb < 5.0 else "NO",
        "Box_Precision": round(box_p, 4),
        "Box_Recall": round(box_r, 4),
        "Box_mAP50": round(box_map50, 4),
        "Pose_Precision": round(pose_p, 4),
        "Pose_Recall": round(pose_r, 4),
        "Pose_mAP50": round(pose_map50, 4),
        "Pose_mAP50-95": round(pose_map50_95, 4),
        "Latency_ms": round(avg_latency, 2),
        "FPS": round(fps, 1),
        "Model_Path": model_path
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark matrix across all pruned and baseline models.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or 0)")
    parser.add_argument("--output_csv", type=str, default="experiments/reports/final_comparison.csv", help="Report CSV")
    args = parser.parse_args()

    candidate_models = [
        # 16-KP Baseline Models
        ("outputs/stage_a/run_shuffled_v3/weights/best.pt", "configs/shuffled_v3_16kp.yaml", "16-KP"),
        ("deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx", "configs/shuffled_v3_16kp.yaml", "16-KP"),
        # 4-KP Baseline Models
        ("outputs/stage1_4kp/finetune_shuffled_v3/weights/best.pt", "configs/shuffled_v3_4kp.yaml", "4-KP"),
        ("outputs/stage1_4kp/finetune_shuffled_v3/stage-a-320-yolo4kp-fp32.onnx", "configs/shuffled_v3_4kp.yaml", "4-KP"),
        ("deliverables/stage_a_4kp/stage-a-320-yolo4kp-fp32.onnx", "configs/shuffled_v3_4kp.yaml", "4-KP"),
    ]

    # Include any existing pruned exports from experiments/pruned_4kp/exports
    pruned_dirs = ["experiments/pruned_4kp/exports", "deliverables/stage_a_4kp", "experiments/pilot_16kp/exports"]
    for pdir in pruned_dirs:
        if os.path.exists(pdir):
            for f in sorted(os.listdir(pdir)):
                if f.endswith(".onnx"):
                    fam = "16-KP" if "16kp" in f else "4-KP"
                    conf = "configs/shuffled_v3_16kp.yaml" if fam == "16-KP" else "configs/shuffled_v3_4kp.yaml"
                    candidate_models.append((os.path.join(pdir, f), conf, fam))

    # Remove duplicates preserving order
    seen_paths = set()
    models_to_test = []
    for m_path, d_yaml, fam in candidate_models:
        clean_path = os.path.normpath(m_path)
        if clean_path not in seen_paths and os.path.exists(m_path):
            seen_paths.add(clean_path)
            models_to_test.append((m_path, d_yaml, fam))

    print("=" * 80)
    print("      RUNNING COMPLETE MULTI-MODEL BENCHMARK MATRIX      ")
    print("=" * 80)
    print(f"Found {len(models_to_test)} model(s) to benchmark.")

    results = []
    for m_path, d_yaml, fam in models_to_test:
        print(f"\nEvaluating: {m_path} ({fam})...")
        r = benchmark_single_model(m_path, d_yaml, device=args.device, family=fam)
        results.append(r)
        print(f"  -> Pose mAP50: {r['Pose_mAP50']} | Latency: {r['Latency_ms']} ms | Size: {r['File_Size_MB']} MB (Budget <5MB: {r['Budget_Compliant (<5MB)']})")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    
    fieldnames = [
        "Model_Name", "Family", "Format", "File_Size_MB", "Budget_Compliant (<5MB)",
        "Box_Precision", "Box_Recall", "Box_mAP50",
        "Pose_Precision", "Pose_Recall", "Pose_mAP50", "Pose_mAP50-95",
        "Latency_ms", "FPS", "Model_Path"
    ]
    
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if results:
            writer.writerows(results)

    print("\n" + "=" * 80)
    print(f"✅ Benchmark Complete! Saved {len(results)} model record(s) to: {args.output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()
