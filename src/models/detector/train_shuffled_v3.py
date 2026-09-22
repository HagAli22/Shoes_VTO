"""
train_shuffled_v3.py
───────────────────
Train / Fine-tune Stage A foot detector on the clean, uniformly shuffled
dataset (data/shuffled_v3/data.yaml) containing 1,109 balanced images.

Usage:
    python -m src.models.detector.train_shuffled_v3 --epochs 150 --lr0 0.001
"""

import argparse
import time
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train on shuffled & cleaned 1,109-image dataset.")
    parser.add_argument("--weights", default="outputs/stage_a/run_v2_1/weights/best.pt",
                        help="Initial checkpoint (use run_v2_1/best.pt for fine-tuning or yolov8n-pose.pt for scratch)")
    parser.add_argument("--data", default="data/shuffled_v3/data.yaml", help="Dataset yaml path")
    parser.add_argument("--epochs", type=int, default=150, help="Max training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Image resolution")
    parser.add_argument("--device", default="0", help="CUDA device index or cpu")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate multiplier")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument("--project", default="outputs/stage_a", help="Output project directory")
    parser.add_argument("--name", default="run_shuffled_v3", help="Experiment name")
    args = parser.parse_args()

    print("=" * 70)
    print("  Starting Training on Clean Shuffled Dataset (1,109 Images)       ")
    print(f"  Base weights: {args.weights}")
    print(f"  Dataset:      {args.data}")
    print(f"  Epochs:       {args.epochs} (patience={args.patience})")
    print(f"  Image size:   {args.imgsz}x{args.imgsz} | Batch: {args.batch} | LR: {args.lr0}")
    print("=" * 70)

    model = YOLO(args.weights)
    t0 = time.time()

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
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
        erasing=0.3,
        close_mosaic=10,
    )

    elapsed_min = (time.time() - t0) / 60.0
    best_ckpt = Path(args.project) / args.name / "weights" / "best.pt"

    print("\n" + "=" * 70)
    print("  Training Complete!")
    print(f"  Time Elapsed:    {elapsed_min:.1f} minutes")
    print(f"  Best Checkpoint: {best_ckpt}")
    print("=" * 70)

if __name__ == "__main__":
    main()

