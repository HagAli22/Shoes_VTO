"""
train_yolo_4kp.py
─────────────────
Dedicated training script for YOLOv8n-Pose (4 Coarse Keypoints) model.
Runs via Python directly to avoid Colab /bin/bash CLI entrypoint PATH issues.

Usage:
  # Stage 1 Pretraining:
  python -m src.models.detector.train_yolo_4kp \
    --weights yolov8n-pose.pt \
    --data configs/shoes_v2_4kp.yaml \
    --epochs 100 \
    --batch 64 \
    --lr0 0.002 \
    --name pretrain_shoes_v2

  # Stage 2 Fine-Tuning:
  python -m src.models.detector.train_yolo_4kp \
    --weights outputs/stage1_4kp/pretrain_shoes_v2/weights/best.pt \
    --data configs/shuffled_v3_4kp.yaml \
    --epochs 150 \
    --batch 64 \
    --lr0 0.001 \
    --name finetune_shuffled_v3
"""

import os
import sys
import time
import argparse
from pathlib import Path
from ultralytics import YOLO


def train_4kp(
    weights: str,
    data: str,
    epochs: int = 100,
    batch: int = 64,
    imgsz: int = 320,
    device: str = "0",
    lr0: float = 0.002,
    lrf: float = 0.01,
    patience: int = 30,
    optimizer: str = "AdamW",
    project: str = "outputs/stage1_4kp",
    name: str = "train_4kp_run",
    cache: str = "ram",
    workers: int = 8,
):
    print("=" * 70)
    print("  YOLOv8n-Pose (4 Coarse Keypoints) High-Speed GPU Engine")
    print(f"  Base Weights: {weights}")
    print(f"  Dataset YAML: {data}")
    print(f"  Epochs      : {epochs} (Patience: {patience})")
    print(f"  Batch / Size: {batch} / {imgsz}x{imgsz} | LR: {lr0}")
    print(f"  RAM Cache   : {cache} | Workers: {workers}")
    print("=" * 70)

    model = YOLO(weights)
    t0 = time.time()

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        patience=patience,
        cache=cache,
        workers=workers,
        save=True,
        verbose=True,
        plots=True,
        warmup_epochs=3,
        cos_lr=True,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        erasing=0.2,
        close_mosaic=10,
    )

    elapsed_min = (time.time() - t0) / 60.0

    # Locate actual saved checkpoint from Ultralytics save_dir
    actual_save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    actual_best = actual_save_dir / "weights" / "best.pt"
    
    target_weights_dir = Path(project) / name / "weights"
    target_weights_dir.mkdir(parents=True, exist_ok=True)
    target_best = target_weights_dir / "best.pt"

    if actual_best.exists() and str(actual_best.resolve()) != str(target_best.resolve()):
        import shutil
        shutil.copy2(actual_best, target_best)
        shutil.copy2(actual_save_dir / "weights" / "last.pt", target_weights_dir / "last.pt") if (actual_save_dir / "weights" / "last.pt").exists() else None

    print("\n" + "=" * 70)
    print("  Training Finished Successfully!")
    print(f"  Time Elapsed   : {elapsed_min:.2f} minutes")
    print(f"  Best Checkpoint: {target_best}")
    print("=" * 70)
    return str(target_best)


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n-Pose 4-KP model.")
    parser.add_argument("--weights", type=str, default="yolov8n-pose.pt", help="Initial model weights")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    parser.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer")
    parser.add_argument("--project", type=str, default="outputs/stage1_4kp", help="Project output folder")
    parser.add_argument("--name", type=str, default="train_run", help="Run experiment name")
    parser.add_argument("--cache", type=str, default="ram", help="Data cache mode (ram, disk, False)")
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")

    args = parser.parse_args()
    train_4kp(
        weights=args.weights,
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        patience=args.patience,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
        cache=args.cache,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
