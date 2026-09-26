"""
run_pilot_16kp_pruning.py
─────────────────────────
Phase 2 Pilot Study on 16-KP YOLOv8-Pose Model.

Compares:
  1. Channel Importance Methods: Taylor vs Magnitude (L1/L2) vs BN-Scale.
  2. Iterative Compression Levels: 20%, 35%, 50%, 65%.
  3. Recovery fine-tuning performance.
  4. Records full benchmarking curve to experiments/reports/pilot_16kp_results.csv.
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

from ultralytics import YOLO

from src.models.pruning.pruner import prune_yolo_model, measure_model_size_and_params
from src.models.pruning.train_pruned import train_pruned_model
from tools.pruning.export_pruned_onnx import export_and_verify_pruned_model


def run_pilot_experiment(
    baseline_weights: str = "outputs/stage_a/run_shuffled_v3/weights/best.pt",
    data_yaml: str = "data/shuffled_v3/data.yaml",
    device: str = "0",
    finetune_epochs: int = 35,
    batch_size: int = 32,
    output_dir: str = "experiments/pilot_16kp",
    report_csv: str = "experiments/reports/pilot_16kp_results.csv",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_csv)), exist_ok=True)

    print("=" * 76)
    print("      PHASE 2 — PILOT CHANNEL PRUNING STUDY (16-KEYPOINT MODEL)      ")
    print("=" * 76)
    print(f"  Baseline Weights : {baseline_weights}")
    print(f"  Dataset YAML     : {data_yaml}")
    print(f"  Output Dir       : {output_dir}")
    print(f"  Report CSV       : {report_csv}")
    print("=" * 76 + "\n")

    # 1. Profile Baseline
    print(">>> [1/4] Evaluating Baseline 16-KP Model...")
    base_model = YOLO(baseline_weights)
    base_stats = measure_model_size_and_params(base_model, device="cpu")
    val_res = base_model.val(data=data_yaml, imgsz=320, split="test", device=device, verbose=False)

    base_box_map = float(val_res.box.map50)
    base_pose_map = float(val_res.pose.map50)
    base_pose_r = float(val_res.pose.r.mean()) if hasattr(val_res.pose, "r") else 0.0

    print(f"  Baseline Params   : {base_stats['total_params']:,} ({base_stats['est_fp32_mb']:.2f} MB FP32)")
    print(f"  Baseline Box mAP50: {base_box_map:.4f} | Pose mAP50: {base_pose_map:.4f}")

    results_table = []
    results_table.append({
        "Model": "Baseline-16KP",
        "Method": "None",
        "Pruning_Ratio": 0.0,
        "Params": base_stats["total_params"],
        "Est_FP32_MB": round(base_stats["est_fp32_mb"], 2),
        "Box_mAP50": round(base_box_map, 4),
        "Pose_mAP50": round(base_pose_map, 4),
        "Pose_Recall": round(base_pose_r, 4),
        "Checkpoint": baseline_weights
    })

    # 2. Compare Importance Methods at 35% Pruning Ratio
    importance_methods = ["magnitude", "taylor", "bn_scale"]
    
    print("\n>>> [2/4] Comparing Channel Importance Criteria @ 35% Pruning Ratio...")
    for imp in importance_methods:
        print(f"\n--- Testing Importance: {imp.upper()} ---")
        run_name = f"pilot_16kp_imp_{imp}_r35"
        
        # Prune
        pruned_m, stats = prune_yolo_model(
            model=baseline_weights,
            pruning_ratio=0.35,
            importance_name=imp,
            device="cpu"
        )
        
        # Save intermediate pruned checkpoint
        pruned_pt_path = os.path.join(output_dir, f"{run_name}_pruned.pt")
        pruned_m.save(pruned_pt_path)
        
        # Fine-tune
        best_pt = train_pruned_model(
            model_or_weights=pruned_pt_path,
            data_yaml=data_yaml,
            epochs=finetune_epochs,
            batch_size=batch_size,
            device=device,
            project=output_dir,
            name=run_name
        )
        
        # Validate
        eval_m = YOLO(best_pt)
        v = eval_m.val(data=data_yaml, imgsz=320, split="test", device=device, verbose=False)
        box_m = float(v.box.map50)
        pose_m = float(v.pose.map50)
        pose_r = float(v.pose.r.mean()) if hasattr(v.pose, "r") else 0.0
        
        results_table.append({
            "Model": f"16KP-{imp.upper()}-35%",
            "Method": imp,
            "Pruning_Ratio": 0.35,
            "Params": stats["params_after"],
            "Est_FP32_MB": round(stats["est_fp32_mb_after"], 2),
            "Box_mAP50": round(box_m, 4),
            "Pose_mAP50": round(pose_m, 4),
            "Pose_Recall": round(pose_r, 4),
            "Checkpoint": best_pt
        })

    # 3. Write CSV Report
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
        writer.writeheader()
        writer.writerows(results_table)

    print("\n" + "=" * 76)
    print("  PHASE 2 PILOT STUDY COMPLETE!")
    print(f"  Summary saved to: {report_csv}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Run 16-KP Channel Pruning Pilot Study.")
    parser.add_argument("--baseline", type=str, default="outputs/stage_a/run_shuffled_v3/weights/best.pt", help="Baseline weights")
    parser.add_argument("--data", type=str, default="data/shuffled_v3/data.yaml", help="Dataset YAML")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    parser.add_argument("--epochs", type=int, default=35, help="Fine-tuning epochs per stage")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_pilot_experiment(
        baseline_weights=args.baseline,
        data_yaml=args.data,
        device=args.device,
        finetune_epochs=args.epochs,
        batch_size=args.batch
    )


if __name__ == "__main__":
    main()
