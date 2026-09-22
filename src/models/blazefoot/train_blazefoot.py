"""
train_blazefoot.py
──────────────────
High-Performance PyTorch Training Pipeline for Google BlazeFoot (A100 / GPU Optimized).

Optimizations:
  - cuDNN Benchmark enabled for ultra-fast convolution kernels.
  - In-Memory RAM Caching (--cache_ram) to eliminate disk I/O bottlenecks.
  - Multi-worker parallel DataLoader with pin_memory and non_blocking CUDA transfers.
  - Mixed Precision Training (torch.amp.autocast).
  - Cosine Annealing Learning Rate Decay with AdamW optimizer.

Usage:
    python -m src.models.blazefoot.train_blazefoot --epochs 100 --batch 64 --num_workers 4 --device 0
"""

import os
import sys
import time
import argparse
import math
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler

from .blazefoot_detector import BlazeFoot
from .dataset import get_blazefoot_loaders
from .loss import BlazeFootLoss

# Enable cuDNN benchmark & TF32 for highest GPU throughput on NVIDIA GPUs (A100 / RTX / Ampere+)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: BlazeFootLoss,
    optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    model.train()
    total_loss_accum = 0.0
    loss_cls_accum = 0.0
    loss_box_accum = 0.0
    loss_kpt_accum = 0.0
    batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        tgt_cls = batch["target_cls"].to(device, non_blocking=True)
        tgt_box = batch["target_box"].to(device, non_blocking=True)
        tgt_kpt = batch["target_kpt"].to(device, non_blocking=True)
        tgt_mask = batch["target_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            cls_preds, box_preds, kpt_preds = model(images)
            loss_dict = criterion(
                cls_preds, box_preds, kpt_preds,
                tgt_cls, tgt_box, tgt_kpt, tgt_mask
            )
            loss = loss_dict["total_loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss_accum += loss.item()
        loss_cls_accum += loss_dict["loss_cls"].item()
        loss_box_accum += loss_dict["loss_box"].item()
        loss_kpt_accum += loss_dict["loss_kpt"].item()
        batches += 1

    return {
        "loss": total_loss_accum / max(batches, 1),
        "loss_cls": loss_cls_accum / max(batches, 1),
        "loss_box": loss_box_accum / max(batches, 1),
        "loss_kpt": loss_kpt_accum / max(batches, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: BlazeFootLoss,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss_accum = 0.0
    loss_cls_accum = 0.0
    loss_box_accum = 0.0
    loss_kpt_accum = 0.0
    batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        tgt_cls = batch["target_cls"].to(device, non_blocking=True)
        tgt_box = batch["target_box"].to(device, non_blocking=True)
        tgt_kpt = batch["target_kpt"].to(device, non_blocking=True)
        tgt_mask = batch["target_mask"].to(device, non_blocking=True)

        cls_preds, box_preds, kpt_preds = model(images)
        loss_dict = criterion(
            cls_preds, box_preds, kpt_preds,
            tgt_cls, tgt_box, tgt_kpt, tgt_mask
        )

        total_loss_accum += loss_dict["total_loss"].item()
        loss_cls_accum += loss_dict["loss_cls"].item()
        loss_box_accum += loss_dict["loss_box"].item()
        loss_kpt_accum += loss_dict["loss_kpt"].item()
        batches += 1

    return {
        "val_loss": total_loss_accum / max(batches, 1),
        "val_cls": loss_cls_accum / max(batches, 1),
        "val_box": loss_box_accum / max(batches, 1),
        "val_kpt": loss_kpt_accum / max(batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Google BlazeFoot detector.")
    parser.add_argument("--data", default="data/shuffled_v3", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (e.g. 64 for A100)")
    parser.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=4 if os.name != 'nt' else 0, help="DataLoader worker processes")
    parser.add_argument("--cache_ram", action="store_true", default=True, help="Cache all images in RAM for max GPU utilization")
    parser.add_argument("--device", default="0", help="GPU device ID or 'cpu'")
    parser.add_argument("--project", default="outputs/stage_a", help="Output project directory")
    parser.add_argument("--name", default="blazefoot_4kp_colab", help="Experiment name")
    parser.add_argument("--compile", action="store_true", help="Compile model using PyTorch 2.0+ torch.compile for extra speedup")
    args = parser.parse_args()

    device_str = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    device = torch.device(device_str)

    out_dir = os.path.join(args.project, args.name)
    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == "cuda" else 0.0

    print("=" * 70)
    print("  [>] Training Google BlazeFoot (MediaPipe Style) 4-KP Detector")
    print(f"  Device        : {device} ({gpu_name}, {vram_gb:.1f} GB VRAM)")
    print(f"  Dataset       : {args.data}")
    print(f"  Epochs        : {args.epochs} | Batch Size: {args.batch} | LR0: {args.lr0}")
    print(f"  Workers       : {args.num_workers} | RAM Cache: {args.cache_ram} | TF32: {torch.cuda.is_available()}")
    print(f"  Output Dir    : {out_dir}")
    print("=" * 70)

    # 1. Initialize Model (4 Keypoints strictly for Path A)
    model = BlazeFoot(num_classes=2, num_keypoints=4).to(device)
    if args.compile and hasattr(torch, "compile"):
        print("  [⚡] Compiling model with torch.compile()...")
        model = torch.compile(model)

    params = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())
    print(f"[1] BlazeFoot (4-KP Native) initialized with {params:,} parameters ({params*4/1e6:.2f} MB in FP32)")


    # 2. Load Data with RAM Caching & Multi-Workers
    train_loader, val_loader = get_blazefoot_loaders(
        args.data,
        batch_size=args.batch,
        num_workers=args.num_workers,
        cache_ram=args.cache_ram
    )
    print(f"[2] Data loaded: {len(train_loader.dataset)} train images | {len(val_loader.dataset)} val images")

    # 3. Setup Optimizer, Scheduler & Scaler
    criterion = BlazeFootLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr0, weight_decay=0.0005)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    start_time = time.time()

    print("\nStarting Training Loop...")
    print(f"{'Epoch':>7} | {'Train Loss':>10} | {'Val Loss':>10} | {'Box Loss':>9} | {'Kpt Loss':>9} | {'Time':>7}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_metrics   = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        v_loss = val_metrics["val_loss"]

        is_best = v_loss < best_val_loss
        if is_best:
            best_val_loss = v_loss
            best_path = os.path.join(weights_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": best_val_loss,
                "config": {"num_classes": 2, "num_keypoints": 4}
            }, best_path)

        last_path = os.path.join(weights_dir, "last.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": v_loss
        }, last_path)

        star = " * (best)" if is_best else ""
        print(f"{epoch:6d}/{args.epochs} | {train_metrics['loss']:10.4f} | {val_metrics['val_loss']:10.4f} | {val_metrics['val_box']:9.4f} | {val_metrics['val_kpt']:9.4f} | {elapsed:5.1f}s{star}", flush=True)

    total_time = (time.time() - start_time) / 60.0
    print("=" * 70)
    print(f"  Training Complete in {total_time:.2f} minutes!")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Saved Weights : {os.path.join(weights_dir, 'best.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
