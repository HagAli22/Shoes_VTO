"""
train_shuffled_v3.py
───────────────────
Main production training & fine-tuning script for the Stage A foot detector.
Supports both the new 13-keypoint schema and legacy checkpoints.

Key capabilities:
  1. Fine-tuning from adapted 13-KP checkpoints (zero keypoint knowledge loss).
  2. Automatic adaptation if a 16-KP checkpoint is provided directly.
  3. Scratch training from base yolov8n-pose.pt.
  4. Layer freezing option (--freeze) for backbone stabilization.
  5. Comprehensive post-training validation and metrics summary.

Usage examples:
  # Fine-tune 13-KP model (Recommended):
  python -m src.models.detector.train_shuffled_v3 \
      --weights checkpoints/stage_a_13kp_adapted.pt \
      --data data/shuffled_v4_13kp/data.yaml \
      --epochs 100 --batch 16 --lr0 0.001 --name run_v4_13kp

  # Auto-adapt from previous 16-KP best.pt and fine-tune:
  python -m src.models.detector.train_shuffled_v3 \
      --weights outputs/stage_a/run_shuffled_v3/weights/best.pt \
      --data data/shuffled_v4_13kp/data.yaml \
      --auto_adapt --name run_v4_13kp

  # Train from baseline COCO yolov8n-pose.pt:
  python -m src.models.detector.train_shuffled_v3 \
      --weights yolov8n-pose.pt \
      --data data/shuffled_v4_13kp/data.yaml \
      --epochs 150 --batch 16 --lr0 0.002 --name run_v4_13kp_scratch
"""

import os
import sys
import time
import argparse
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.adapt_checkpoint_keypoints import adapt_pose_checkpoint


def check_and_prepare_weights(weights_path: Path, data_yaml_path: Path, auto_adapt: bool) -> str:
    """
    Checks if weights need keypoint adaptation to match data.yaml.
    Returns path to the weights to pass to YOLO().
    """
    if not weights_path.exists():
        print(f"[*] Weights file '{weights_path}' not found locally. Ultralytics will attempt download if it is a stock model.")
        return str(weights_path)

    # Read target kpt_shape from data.yaml
    target_kps = 13
    if data_yaml_path.exists():
        try:
            with open(data_yaml_path, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f)
                if "kpt_shape" in d:
                    target_kps = d["kpt_shape"][0]
        except Exception:
            pass

    # Inspect checkpoint
    try:
        ckpt = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model = ckpt["model"]
            head = None
            for m in model.model:
                if hasattr(m, "kpt_shape") and hasattr(m, "cv4"):
                    head = m
                    break
            if head and head.kpt_shape:
                src_kps = head.kpt_shape[0]
                if src_kps != target_kps:
                    print(f"[!] Checkpoint keypoint mismatch: Source has {src_kps} KPs, dataset requires {target_kps} KPs.")
                    if auto_adapt:
                        adapted_dst = REPO_ROOT / "checkpoints" / f"{weights_path.stem}_adapted_{target_kps}kp.pt"
                        print(f"[*] --auto_adapt is enabled. Adapting weights to {adapted_dst}...")
                        adapt_pose_checkpoint(
                            src_path=weights_path,
                            dst_path=adapted_dst,
                            remove_indices=[5, 12, 15] if src_kps == 16 and target_kps == 13 else []
                        )
                        return str(adapted_dst)
                    else:
                        print("[!] Note: Pose head will be reinitialized by Ultralytics unless --auto_adapt is passed.")
    except Exception as e:
        print(f"[*] Note during checkpoint inspection: {e}")

    return str(weights_path)


def main():
    parser = argparse.ArgumentParser(description="Main Stage A Foot Detector Training & Fine-Tuning Script.")
    
    # Model & Data Paths
    parser.add_argument("--weights", default="checkpoints/stage_a_13kp_adapted.pt",
                        help="Initial checkpoint (adapted 13-KP checkpoint, 16-KP best.pt, or yolov8n-pose.pt)")
    parser.add_argument("--data", default="data/shuffled_v4_13kp/data.yaml",
                        help="Dataset YAML configuration path")
    parser.add_argument("--auto_adapt", action="store_true",
                        help="Automatically adapt a 16-KP checkpoint to 13-KP before training if mismatch detected")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=150, help="Total training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Input resolution (letterboxed)")
    parser.add_argument("--device", default="", help="Device index (e.g. 0) or 'cpu'. Default auto-detects CUDA")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (0.001 for fine-tuning, 0.002 for scratch)")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument("--freeze", type=int, default=0, help="Number of backbone layers to freeze (e.g. 10 for fine-tuning)")
    
    # Output Settings
    parser.add_argument("--project", default="outputs/stage_a", help="Destination project directory")
    parser.add_argument("--name", default="run_v4_13kp", help="Experiment name")
    parser.add_argument("--exist_ok", action="store_true", help="Overwrite existing project/name folder")
    args = parser.parse_args()

    # Auto-detect device if empty
    if not args.device:
        args.device = "0" if torch.cuda.is_available() else "cpu"

    weights_file = Path(args.weights)
    data_file = Path(args.data)

    # Check weights and adapt if needed
    final_weights = check_and_prepare_weights(weights_file, data_file, args.auto_adapt)

    print("=" * 75)
    print("  STAGE A FOOT DETECTOR — PRODUCTION TRAINING PIPELINE                      ")
    print(f"  Base weights: {final_weights}")
    print(f"  Dataset:      {args.data}")
    print(f"  Device:       {args.device} ({torch.cuda.get_device_name(0) if args.device == '0' and torch.cuda.is_available() else 'CPU'})")
    print(f"  Epochs:       {args.epochs} (patience={args.patience})")
    print(f"  Resolution:   {args.imgsz}x{args.imgsz} | Batch: {args.batch} | LR0: {args.lr0}")
    print(f"  Freeze:       {args.freeze} layers")
    print(f"  Output:       {args.project}/{args.name}")
    print("=" * 75)

    model = YOLO(final_weights)
    t0 = time.time()

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "patience": args.patience,
        "save": True,
        "verbose": True,
        "plots": True,
        "warmup_epochs": 3,
        "cos_lr": True,
        "weight_decay": 0.0005,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 10.0,
        "translate": 0.1,
        "scale": 0.5,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.5,
        "mixup": 0.0,
        "erasing": 0.3,
        "close_mosaic": 10,
        "exist_ok": args.exist_ok,
    }

    if args.freeze > 0:
        train_kwargs["freeze"] = args.freeze

    # Launch training
    results = model.train(**train_kwargs)

    elapsed_min = (time.time() - t0) / 60.0
    best_ckpt = Path(args.project) / args.name / "weights" / "best.pt"

    print("\n" + "=" * 75)
    print("  Training Completed Successfully!")
    print(f"  Time Elapsed:    {elapsed_min:.1f} minutes")
    print(f"  Best Checkpoint: {best_ckpt}")
    print("=" * 75)

    # Post-training evaluation
    if best_ckpt.exists():
        print("\n[*] Running final validation on best checkpoint...")
        try:
            val_model = YOLO(str(best_ckpt))
            metrics = val_model.val(data=args.data, imgsz=args.imgsz, device=args.device)
            print(f"  [+] Validation Results:")
            if hasattr(metrics, "box") and metrics.box:
                print(f"      Box mAP@50:    {metrics.box.map50 * 100:.2f}%")
                print(f"      Box mAP@50-95: {metrics.box.map * 100:.2f}%")
            if hasattr(metrics, "pose") and metrics.pose:
                print(f"      Pose mAP@50:   {metrics.pose.map50 * 100:.2f}%")
                print(f"      Pose mAP@50-95:{metrics.pose.map * 100:.2f}%")
        except Exception as e:
            print(f"[!] Warning during final val: {e}")

    print("\n[OK] Pipeline complete.")


if __name__ == "__main__":
    main()
