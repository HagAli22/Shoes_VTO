"""
benchmark_pruned_matrix.py
──────────────────────────
Unified Evaluation & Benchmarking Suite across 16-KP and 4-KP Model Families.

Computes a complete side-by-side comparison table including:
  - Parameters
  - FP32 ONNX File Size (MB) & Budget Compliance (< 5.0 MB)
  - Detection Metrics (Precision, Recall, mAP50)
  - Keypoint Metrics (Pose Precision, Recall, mAP50, mAP50-95)
  - Inference Latency (ms) & FPS
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

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def benchmark_single_model(
    model_path: str,
    data_yaml: str,
    device: str = "cpu",
    family: str = "4-KP",
) -> dict:
    is_onnx = model_path.endswith(".onnx")
    file_size_mb = os.path.getsize(model_path) / (1024.0 * 1024.0)
    
    # Validation via Ultralytics
    model = YOLO(model_path)
    val_res = model.val(data=data_yaml, imgsz=320, split="test", device=device, verbose=False)
    
    box_p = float(val_res.box.p.mean()) if hasattr(val_res.box, "p") else 0.0
    box_r = float(val_res.box.r.mean()) if hasattr(val_res.box, "r") else 0.0
    box_map50 = float(val_res.box.map50)
    
    pose_p = float(val_res.pose.p.mean()) if hasattr(val_res.pose, "p") else 0.0
    pose_r = float(val_res.pose.r.mean()) if hasattr(val_res.pose, "r") else 0.0
    pose_map50 = float(val_res.pose.map50)
    pose_map50_95 = float(val_res.pose.map)
    
    # Latency profiling
    dummy = np.random.randn(1, 3, 320, 320).astype(np.float32)
    latencies = []
    
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
        # PyTorch
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
                
    avg_latency = float(np.mean(latencies))
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

    models_to_test = [
        # 16-KP Models
        ("outputs/stage_a/run_shuffled_v3/weights/best.pt", "data/shuffled_v3/data.yaml", "16-KP"),
        ("deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx", "data/shuffled_v3/data.yaml", "16-KP"),
        # 4-KP Models
        ("outputs/stage1_4kp/finetune_shuffled_v3/weights/best.pt", "data/shuffled_v3_4kp/data.yaml", "4-KP"),
        ("outputs/stage1_4kp/finetune_shuffled_v3/stage-a-320-yolo4kp-fp32.onnx", "data/shuffled_v3_4kp/data.yaml", "4-KP"),
    ]

    # Include any existing pruned exports
    pruned_exports_dir = "experiments/pruned_4kp/exports"
    if os.path.exists(pruned_exports_dir):
        for f in os.listdir(pruned_exports_dir):
            if f.endswith(".onnx"):
                models_to_test.append((os.path.join(pruned_exports_dir, f), "data/shuffled_v3_4kp/data.yaml", "4-KP"))

    results = []
    print("=" * 80)
    print("      RUNNING COMPLETE MULTI-MODEL BENCHMARK MATRIX      ")
    print("=" * 80)

    for m_path, d_yaml, fam in models_to_test:
        if not os.path.exists(m_path):
            continue
        print(f"\nEvaluating: {m_path} ({fam})...")
        try:
            r = benchmark_single_model(m_path, d_yaml, device=args.device, family=fam)
            results.append(r)
            print(f"  -> Pose mAP50: {r['Pose_mAP50']} | Latency: {r['Latency_ms']} ms | Size: {r['File_Size_MB']} MB (Budget: {r['Budget_Compliant (<5MB)']})")
        except Exception as e:
            print(f"  -> Error evaluating {m_path}: {e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    if results:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print("\n" + "=" * 80)
        print(f"  Benchmark complete! Report saved to: {args.output_csv}")
        print("=" * 80)


if __name__ == "__main__":
    main()
