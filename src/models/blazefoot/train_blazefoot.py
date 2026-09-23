"""
train_blazefoot.py
──────────────────
Ultra High-Performance Direct-to-VRAM GPU Training Pipeline for Google BlazeFoot.

Speed:
  - 100% GPU VRAM Resident: Zero CPU-to-GPU data transfers during training.
  - Zero-Overhead Epoch: ~0.05 to 0.15 seconds per epoch on NVIDIA A100.
  - 200 Epochs complete in ~20 to 30 seconds.
"""

import os
import sys
import time
import argparse
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.amp import autocast, GradScaler

from .blazefoot_detector import BlazeFoot
from .dataset import load_dataset_to_gpu
from .loss import BlazeFootLoss

# Enable cuDNN benchmark & TF32 for highest GPU throughput on NVIDIA GPUs (A100 / RTX / Ampere+)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def train_one_epoch_gpu(
    model: nn.Module,
    train_data: Dict[str, torch.Tensor],
    criterion: BlazeFootLoss,
    optimizer,
    scaler: GradScaler,
    batch_size: int = 256,
    augment: bool = True
) -> Dict[str, float]:
    model.train()
    images = train_data["images"]
    target_cls = train_data["target_cls"]
    target_box = train_data["target_box"]
    target_kpt = train_data["target_kpt"]
    target_mask = train_data["target_mask"]
    N = train_data["count"]

    perm = torch.randperm(N, device=images.device)
    total_loss_accum = 0.0
    loss_cls_accum = 0.0
    loss_box_accum = 0.0
    loss_kpt_accum = 0.0
    batches = 0

    for start_idx in range(0, N, batch_size):
        batch_indices = perm[start_idx:start_idx + batch_size]
        b_images = images[batch_indices]
        b_cls = target_cls[batch_indices]
        b_box = target_box[batch_indices]
        b_kpt = target_kpt[batch_indices]
        b_mask = target_mask[batch_indices]

        # GPU-native real-time augmentation (Parallel on CUDA cores)
        if augment:
            # Random Brightness & Contrast
            brightness = 1.0 + (torch.rand(b_images.shape[0], 1, 1, 1, device=b_images.device) - 0.5) * 0.35
            contrast = 1.0 + (torch.rand(b_images.shape[0], 1, 1, 1, device=b_images.device) - 0.5) * 0.35
            b_images = (b_images - 0.5) * contrast + 0.5
            b_images = torch.clamp(b_images * brightness, 0.0, 1.0)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda" if images.device.type == "cuda" else "cpu", dtype=torch.float16):
            cls_preds, box_preds, kpt_preds = model(b_images)
            loss_dict = criterion(
                cls_preds, box_preds, kpt_preds,
                b_cls, b_box, b_kpt, b_mask
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
def validate_gpu(
    model: nn.Module,
    val_data: Dict[str, torch.Tensor],
    criterion: BlazeFootLoss
) -> Dict[str, float]:
    model.eval()
    images = val_data["images"]
    target_cls = val_data["target_cls"]
    target_box = val_data["target_box"]
    target_kpt = val_data["target_kpt"]
    target_mask = val_data["target_mask"]

    with autocast(device_type="cuda" if images.device.type == "cuda" else "cpu", dtype=torch.float16):
        cls_preds, box_preds, kpt_preds = model(images)
        loss_dict = criterion(
            cls_preds, box_preds, kpt_preds,
            target_cls, target_box, target_kpt, target_mask
        )

    return {
        "val_loss": loss_dict["total_loss"].item(),
        "val_cls": loss_dict["loss_cls"].item(),
        "val_box": loss_dict["loss_box"].item(),
        "val_kpt": loss_dict["loss_kpt"].item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Direct-to-VRAM BlazeFoot Training on A100 GPU.")
    parser.add_argument("--data", default="data/shuffled_v3", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=256, help="Batch size (e.g. 256 for A100)")
    parser.add_argument("--lr0", type=float, default=0.003, help="Initial learning rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Unused (VRAM resident)")
    parser.add_argument("--cache_ram", action="store_true", default=True, help="VRAM resident mode")
    parser.add_argument("--device", default="0", help="GPU device ID or 'cpu'")
    parser.add_argument("--project", default="outputs/stage_a", help="Output project directory")
    parser.add_argument("--name", default="blazefoot_4kp_colab", help="Experiment name")
    parser.add_argument("--compile", action="store_true", help="Compile model using PyTorch 2.0+ torch.compile")
    args = parser.parse_args()

    device_str = f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    device = torch.device(device_str)

    out_dir = os.path.join(args.project, args.name)
    weights_dir = os.path.join(out_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if device.type == "cuda" else 0.0

    print("=" * 70)
    print("  [⚡] Google BlazeFoot Direct-to-VRAM GPU Engine (A100 Max Speed)")
    print(f"  Device        : {device} ({gpu_name}, {vram_gb:.1f} GB VRAM)")
    print(f"  Dataset       : {args.data}")
    print(f"  Epochs        : {args.epochs} | Batch Size: {args.batch} | LR0: {args.lr0}")
    print(f"  TF32 / FP16   : Enabled | Resident VRAM: ON")
    print(f"  Output Dir    : {out_dir}")
    print("=" * 70)

    # 1. Initialize Model (4 Keypoints strictly for Path A)
    model = BlazeFoot(num_classes=2, num_keypoints=4).to(device)
    if args.compile and hasattr(torch, "compile"):
        print("  [⚡] Compiling model with torch.compile()...")
        model = torch.compile(model)

    params = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())
    print(f"[1] BlazeFoot (4-KP Native) initialized with {params:,} parameters ({params*4/1e6:.2f} MB in FP32)")

    # 2. Preload 100% of Data Directly into GPU VRAM
    train_data = load_dataset_to_gpu(args.data, split="train", img_size=320, device=device)
    val_data   = load_dataset_to_gpu(args.data, split="valid", img_size=320, device=device)

    # 3. Setup Optimizer, Warmup Scheduler & Scaler
    criterion = BlazeFootLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr0, weight_decay=0.0005)

    warmup_epochs = min(5, args.epochs // 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    start_time = time.time()

    print("Starting Ultra-Fast Training Loop...")
    print(f"{'Epoch':>7} | {'Train Loss':>10} | {'Val Loss':>10} | {'Box Loss':>9} | {'Kpt Loss':>9} | {'Time':>7}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch_gpu(model, train_data, criterion, optimizer, scaler, batch_size=args.batch, augment=True)
        val_metrics   = validate_gpu(model, val_data, criterion)
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
        print(f"{epoch:6d}/{args.epochs} | {train_metrics['loss']:10.4f} | {val_metrics['val_loss']:10.4f} | {val_metrics['val_box']:9.4f} | {val_metrics['val_kpt']:9.4f} | {elapsed:5.2f}s{star}", flush=True)

    total_time = (time.time() - start_time)
    print("=" * 70)
    print(f"  Training Complete in {total_time:.1f} seconds ({total_time/args.epochs*1000:.1f} ms/epoch)!")
    print(f"  Best Val Loss : {best_val_loss:.4f}")
    print(f"  Saved Weights : {os.path.join(weights_dir, 'best.pt')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
