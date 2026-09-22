"""
train_compact_detector.py
─────────────────────────
Training Script for Compact Stage A Foot Detector (YOLOv8-Pico, width 0.16).

Trains the compact model on data/stage_a/data.yaml (16 keypoints, 2 foot classes)
starting from pre-initialized teacher weights (warm-start) or scratch.

Usage:
    python -m src.models.detector.train_compact_detector \\
        --weights outputs/stage_a/compact_initialized.pt \\
        --data data/stage_a/data.yaml \\
        --epochs 200 \\
        --batch 16 \\
        --imgsz 320 \\
        --device 0 \\
        --project outputs/stage_a \\
        --name compact_run
"""

import os
import time
import argparse
from ultralytics import YOLO


def train_compact(
    weights_path: str = "outputs/stage_a/compact_initialized.pt",
    data_yaml: str = "data/stage_a/data.yaml",
    epochs: int = 200,
    batch_size: int = 16,
    imgsz: int = 320,
    device: str = "0",
    patience: int = 40,
    lr0: float = 0.001,
    optimizer: str = "AdamW",
    project: str = "outputs/stage_a",
    name: str = "compact_run",
) -> str:
    """
    Executes training of the compact detector and returns path to best.pt.
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Initial weights not found: {weights_path}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    print("==================================================================")
    print("  Training Compact Stage A (YOLOv8-Pico) Detector                 ")
    print(f"  Weights   : {weights_path}")
    print(f"  Dataset   : {data_yaml}")
    print(f"  Epochs    : {epochs} (EarlyStopping patience={patience})")
    print(f"  Image Size: {imgsz}x{imgsz} | Batch Size: {batch_size}")
    print(f"  Optimizer : {optimizer} (lr0={lr0}) | Device: {device}")
    print(f"  Output Run: {os.path.join(project, name)}")
    print("==================================================================")

    model = YOLO(weights_path)
    t0 = time.time()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        optimizer=optimizer,
        lr0=lr0,
        lrf=0.01,
        patience=patience,
        save=True,
        verbose=True,
        plots=True,

        # ── LR Schedule ────────────────────────────────────────────────
        warmup_epochs=3,
        cos_lr=True,
        weight_decay=0.0005,

        # ── Photometric augmentation ────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # ── Geometric augmentation ──────────────────────────────────────
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,          # disabled — distorts KP positions
        perspective=0.0,    # disabled — not needed
        flipud=0.0,         # disabled — feet always point down
        fliplr=0.5,

        # ── Advanced augmentation ───────────────────────────────────────
        # LESSON LEARNED: mosaic=1.0 + mixup → val collapse on small datasets.
        mosaic=0.5,         # half of batches use mosaic — good diversity without collapse
        mixup=0.0,          # disabled — interpolated KPs confuse pose head
        copy_paste=0.0,
        erasing=0.2,        # simulates partial occlusion of feet
        close_mosaic=10,    # disable mosaic for final 10 epochs → stable KP convergence
    )

    elapsed_min = (time.time() - t0) / 60.0
    best_ckpt = os.path.join(project, name, "weights", "best.pt")

    print(f"\n[OK] Training completed in {elapsed_min:.2f} minutes!")
    print(f"     Best Checkpoint: {best_ckpt} ({os.path.getsize(best_ckpt)/1e6:.2f} MB)")
    print("==================================================================\n")

    return best_ckpt


def main():
    parser = argparse.ArgumentParser(description="Train compact YOLOv8-Pico Stage A detector.")
    parser.add_argument("--weights", default="outputs/stage_a/compact_initialized.pt",
                        help="Initial checkpoint (e.g. transferred weights)")
    parser.add_argument("--data", default="data/stage_a/data.yaml",
                        help="Path to dataset data.yaml")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Input image size")
    parser.add_argument("--device", default="0", help="GPU device ID (e.g. '0' or 'cpu')")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer")
    parser.add_argument("--project", default="outputs/stage_a", help="Output project directory")
    parser.add_argument("--name", default="compact_run", help="Run sub-directory name")
    args = parser.parse_args()

    train_compact(
        weights_path=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()

