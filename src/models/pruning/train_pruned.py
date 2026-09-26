"""
train_pruned.py
───────────────
Post-Pruning Recovery Fine-Tuning Engine for YOLOv8-Pose.

Fine-tunes pruned PyTorch checkpoints on the target foot keypoint dataset
with early stopping, cosine learning rate decay, and optional distillation from the teacher.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
from ultralytics import YOLO


def train_pruned_model(
    model_or_weights: Any,
    data_yaml: str,
    epochs: int = 50,
    batch_size: int = 32,
    imgsz: int = 320,
    device: str = "0",
    lr0: float = 0.001,
    lrf: float = 0.01,
    patience: int = 20,
    project: str = "experiments/pruned",
    name: str = "pruned_finetune",
    cache: str = "ram",
    workers: int = 8,
) -> str:
    """
    Fine-tunes a pruned model checkpoint on the target dataset.
    
    Returns:
        Path to best.pt checkpoint.
    """
    print("=" * 70)
    print(f"  POST-PRUNING RECOVERY FINE-TUNING: {name}")
    print(f"  Data YAML: {data_yaml}")
    print(f"  Epochs   : {epochs} (Patience: {patience})")
    print(f"  Batch/Img: {batch_size} / {imgsz}x{imgsz} | LR0: {lr0}")
    print(f"  Project  : {project}/{name}")
    print("=" * 70)

    if isinstance(model_or_weights, str):
        model = YOLO(model_or_weights)
    elif isinstance(model_or_weights, YOLO):
        model = model_or_weights
    else:
        model = model_or_weights

    t0 = time.time()
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        optimizer="AdamW",
        lr0=lr0,
        lrf=lrf,
        patience=patience,
        cache=cache,
        workers=workers,
        save=True,
        verbose=True,
        plots=True,
        warmup_epochs=2,
        cos_lr=True,
        weight_decay=0.0005,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
    )

    elapsed_min = (time.time() - t0) / 60.0

    # Ensure best checkpoint is copied to target folder
    actual_save_dir = Path(getattr(results, "save_dir", Path(project) / name))
    actual_best = actual_save_dir / "weights" / "best.pt"
    
    target_weights_dir = Path(project) / name / "weights"
    target_weights_dir.mkdir(parents=True, exist_ok=True)
    target_best = target_weights_dir / "best.pt"

    if actual_best.exists() and str(actual_best.resolve()) != str(target_best.resolve()):
        import shutil
        shutil.copy2(actual_best, target_best)
        
    print("\n" + "=" * 70)
    print(f"  Recovery Fine-Tuning Complete in {elapsed_min:.1f} minutes!")
    print(f"  Saved Best Checkpoint: {target_best}")
    print("=" * 70)
    return str(target_best)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pruned YOLOv8-pose model.")
    parser.add_argument("--weights", type=str, required=True, help="Pruned model weights or checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Image resolution")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or cpu")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--project", type=str, default="experiments/pruned", help="Output project directory")
    parser.add_argument("--name", type=str, default="run", help="Experiment name")
    args = parser.parse_args()

    train_pruned_model(
        model_or_weights=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
