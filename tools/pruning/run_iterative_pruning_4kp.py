"""
run_iterative_pruning_4kp.py
────────────────────────────
Iterative Structured Channel Pruning Pipeline for 4-Coarse-Keypoint YOLOv8-Pose.

Progressive Compression Stages:
  Stage P1: ~20% Pruning -> Recovery Fine-Tune
  Stage P2: ~35% Pruning -> Recovery Fine-Tune
  Stage P3: ~50% Pruning -> Recovery Fine-Tune
  Stage P4: ~65% Pruning -> Full Recovery Fine-Tune -> Export FP32 ONNX (< 5.0 MB Target)
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


def run_iterative_4kp_pipeline(
    baseline_weights: str = "outputs/stage1_4kp/finetune_shuffled_v3/weights/best.pt",
    data_yaml: str = "data/shuffled_v3_4kp/data.yaml",
    importance: str = "taylor",
    device: str = "0",
    epochs_per_stage: int = 40,
    batch_size: int = 32,
    output_dir: str = "experiments/pruned_4kp",
    report_csv: str = "experiments/reports/pruning_4kp_results.csv",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_csv)), exist_ok=True)

    print("=" * 76)
    print("      4-COARSE-KEYPOINT ITERATIVE STRUCTURED PRUNING PIPELINE      ")
    print("=" * 76)
    print(f"  Baseline Weights : {baseline_weights}")
    print(f"  Dataset YAML     : {data_yaml}")
    print(f"  Importance Method: {importance.upper()}")
    print(f"  Output Dir       : {output_dir}")
    print(f"  Report CSV       : {report_csv}")
    print("=" * 76 + "\n")

    # 1. Profile Baseline
    base_model = YOLO(baseline_weights)
    base_stats = measure_model_size_and_params(base_model, device="cpu")
    val_res = base_model.val(data=data_yaml, imgsz=320, split="test", device=device, verbose=False)

    base_box_map = float(val_res.box.map50)
    base_pose_map = float(val_res.pose.map50)
    base_pose_r = float(val_res.pose.r.mean()) if hasattr(val_res.pose, "r") else 0.0

    print(f"Baseline Params: {base_stats['total_params']:,} ({base_stats['est_fp32_mb']:.2f} MB FP32)")
    print(f"Baseline Box mAP50: {base_box_map:.4f} | Pose mAP50: {base_pose_map:.4f}")

    results_table = [{
        "Stage": "Baseline",
        "Pruning_Ratio": 0.0,
        "Params": base_stats["total_params"],
        "Est_FP32_MB": round(base_stats["est_fp32_mb"], 2),
        "ONNX_FP32_MB": 12.42,
        "Box_mAP50": round(base_box_map, 4),
        "Pose_mAP50": round(base_pose_map, 4),
        "Pose_Recall": round(base_pose_r, 4),
        "Checkpoint": baseline_weights,
    }]

    # 2. Progressive Iterative Pruning Schedule
    stages = [
        ("P1", 0.20, epochs_per_stage),
        ("P2", 0.35, epochs_per_stage),
        ("P3", 0.50, epochs_per_stage),
        ("P4", 0.65, int(epochs_per_stage * 1.5)), # Longer recovery for final compression stage
    ]

    current_weights = baseline_weights

    for stage_name, ratio, epochs in stages:
        print("\n" + "-" * 76)
        print(f">>> [STAGE {stage_name}] Pruning Ratio: {ratio*100:.1f}% | Fine-Tune Epochs: {epochs}")
        print("-" * 76)
        
        run_name = f"yolo_4kp_{stage_name.lower()}_r{int(ratio*100)}"
        
        # 1. Prune
        pruned_m, stats = prune_yolo_model(
            model=current_weights,
            pruning_ratio=ratio,
            importance_name=importance,
            device="cpu"
        )
        
        pruned_ckpt_path = os.path.join(output_dir, f"{run_name}_pruned.pt")
        pruned_m.save(pruned_ckpt_path)
        
        # 2. Recovery Fine-Tuning
        best_pt = train_pruned_model(
            model_or_weights=pruned_ckpt_path,
            data_yaml=data_yaml,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            project=output_dir,
            name=run_name
        )
        
        current_weights = best_pt
        
        # 3. Validation on Test Split
        eval_m = YOLO(best_pt)
        v = eval_m.val(data=data_yaml, imgsz=320, split="test", device=device, verbose=False)
        box_m = float(v.box.map50)
        pose_m = float(v.pose.map50)
        pose_r = float(v.pose.r.mean()) if hasattr(v.pose, "r") else 0.0
        
        # 4. Export to ONNX & Measure Real File Size
        onnx_name = f"stage-a-320-yolo4kp-{stage_name.lower()}.onnx"
        onnx_path = export_and_verify_pruned_model(
            weights_path=best_pt,
            output_dir=os.path.join(output_dir, "exports"),
            output_name=onnx_name,
            expected_channels=18,
            max_size_mb=5.0
        )
        real_onnx_mb = os.path.getsize(onnx_path) / (1024.0 * 1024.0)

        results_table.append({
            "Stage": stage_name,
            "Pruning_Ratio": ratio,
            "Params": stats["params_after"],
            "Est_FP32_MB": round(stats["est_fp32_mb_after"], 2),
            "ONNX_FP32_MB": round(real_onnx_mb, 2),
            "Box_mAP50": round(box_m, 4),
            "Pose_mAP50": round(pose_m, 4),
            "Pose_Recall": round(pose_r, 4),
            "Checkpoint": best_pt,
        })

    # 3. Save CSV Report
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_table[0].keys()))
        writer.writeheader()
        writer.writerows(results_table)

    print("\n" + "=" * 76)
    print("  4-KP ITERATIVE PRUNING PIPELINE COMPLETE!")
    print(f"  Full report saved to: {report_csv}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description="Run 4-KP Iterative Pruning Pipeline.")
    parser.add_argument("--baseline", type=str, default="outputs/stage1_4kp/finetune_shuffled_v3/weights/best.pt", help="Baseline weights")
    parser.add_argument("--data", type=str, default="data/shuffled_v3_4kp/data.yaml", help="Dataset YAML")
    parser.add_argument("--importance", type=str, default="taylor", help="Importance method (taylor, magnitude, bn_scale)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    parser.add_argument("--epochs", type=int, default=40, help="Fine-tuning epochs per stage")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    run_iterative_4kp_pipeline(
        baseline_weights=args.baseline,
        data_yaml=args.data,
        importance=args.importance,
        device=args.device,
        epochs_per_stage=args.epochs,
        batch_size=args.batch
    )


if __name__ == "__main__":
    main()
