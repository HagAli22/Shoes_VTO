"""
rtmpose/train.py
────────────────
Self-contained training runner for RTMPose-13KP Foot Pose.
Trains CSPNeXt-Tiny + SimCC head on cropped foot patches.
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.common.dataset import FootPoseDataset13KP
from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP
from src.models.qualcomm_experiments.rtmpose.losses import SimCCLoss
from src.models.qualcomm_experiments.rtmpose.decoder import decode_simcc


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def collate_fn_rtmpose(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    classes = torch.stack([b["class_id"] for b in batch], dim=0)
    keypoints = torch.stack([b["keypoints"] for b in batch], dim=0)
    paths = [b["img_path"] for b in batch]
    return {
        "images": images,
        "classes": classes,
        "keypoints": keypoints,
        "paths": paths
    }


def train_rtmpose(
    data_dir: str,
    output_dir: str = "checkpoints/rtmpose_13kp_official_coco",
    epochs: int = 60,
    batch_size: int = 32,
    lr: float = 5e-4,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    img_size: int = 256,
    backbone_name: str = "cspnext",
    pretrained: bool = True,
    warmup_epochs: int = 4,
    workers: int = 4,
    resume: str = None
):
    set_seed(42)
    device = torch.device(device_name)
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("  EXPERIMENT 2: Qualcomm RTMPose-13KP Foot Pose (SimCC + GAU) Training")
    print(f"  Dataset:         {data_path}")
    print(f"  Backbone:        {backbone_name} (Pretrained: {pretrained})")
    print(f"  Device:          {device}")
    print(f"  Epochs:          {epochs} (Warmup: {warmup_epochs}) | Batch: {batch_size}")
    print(f"  Backbone LR:     {lr * 0.2:.6f} | Head LR: {lr:.6f}")
    print(f"  Resolution:      {img_size}x{img_size} (SimCC Bins: 512x512, Dec Win: 12)")
    print(f"  Data Workers:    {workers} (Pin Memory: True)")
    print("=" * 75)

    train_ds = FootPoseDataset13KP(
        images_dir=str(data_path / "train" / "images"),
        img_size=img_size,
        mode="crop",
        augment=True,
        allow_flip=False  # NO flip as instructed!
    )
    val_ds = FootPoseDataset13KP(
        images_dir=str(data_path / "valid" / "images"),
        img_size=img_size,
        mode="crop",
        augment=False,
        allow_flip=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        collate_fn=collate_fn_rtmpose
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        collate_fn=collate_fn_rtmpose
    )

    model = RTMPose13KP(
        num_keypoints=13,
        input_size=(img_size, img_size),
        simcc_split_ratio=2.0,
        backbone_name=backbone_name,
        pretrained=pretrained
    ).to(device)

    criterion = SimCCLoss(simcc_split_ratio=2.0, sigma=6.0).to(device)

    # Differential learning rate: preserve mature pretrained backbone features
    bb_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": lr * 0.2},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-4)

    # Cosine annealing with warmup
    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    start_epoch = 1
    best_val_err = float("inf")
    best_val_pck05 = 0.0

    if resume and Path(resume).exists():
        print(f"[*] Resuming checkpoint from: {resume}")
        try:
            ckpt = torch.load(resume, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_err = ckpt.get("mean_px_err", float("inf"))
        best_val_pck05 = ckpt.get("val_pck05", 0.0)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        start_t = time.time()

        for step, batch in enumerate(train_loader):
            imgs = batch["images"].to(device)
            kps = batch["keypoints"].to(device)
            targets = criterion.build_targets(
                kps, w_bins=int(img_size * 2), h_bins=int(img_size * 2), device=device
            )

            optimizer.zero_grad()
            preds = model(imgs)
            losses = criterion(preds, targets)
            loss = losses["total_loss"]
            loss.backward()

            # Gradient clipping prevents gradient explosion during early head adaptation
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= max(1, len(train_loader))

        # Validation with localization metrics
        model.eval()
        val_loss = 0.0
        val_errors = []
        val_pck_05_hits = 0
        val_pck_10_hits = 0
        total_valid_kps = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["images"].to(device)
                kps = batch["keypoints"].to(device)
                targets = criterion.build_targets(
                    kps, w_bins=int(img_size * 2), h_bins=int(img_size * 2), device=device
                )
                preds = model(imgs)
                losses = criterion(preds, targets)
                val_loss += losses["total_loss"].item()

                decoded = decode_simcc(preds, simcc_split_ratio=2.0, win=12)
                gt = kps.cpu().numpy()

                for b in range(len(decoded)):
                    valid_mask = gt[b, :, 2] > 0
                    if not np.any(valid_mask):
                        continue
                    dx = decoded[b, valid_mask, 0] - gt[b, valid_mask, 0]
                    dy = decoded[b, valid_mask, 1] - gt[b, valid_mask, 1]
                    dist = np.sqrt(dx ** 2 + dy ** 2)

                    min_x, max_x = gt[b, valid_mask, 0].min(), gt[b, valid_mask, 0].max()
                    min_y, max_y = gt[b, valid_mask, 1].min(), gt[b, valid_mask, 1].max()
                    diag = max(float(np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)), 20.0)

                    val_pck_05_hits += int(np.sum(dist <= (0.05 * diag)))
                    val_pck_10_hits += int(np.sum(dist <= (0.10 * diag)))
                    total_valid_kps += len(dist)
                    val_errors.extend(dist.tolist())

        val_loss /= max(1, len(val_loader))
        val_pck05 = (val_pck_05_hits / max(1, total_valid_kps)) * 100.0
        val_pck10 = (val_pck_10_hits / max(1, total_valid_kps)) * 100.0
        mean_px_err = float(np.mean(val_errors)) if val_errors else 0.0
        dt = time.time() - start_t

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Val PCK@0.05: {val_pck05:.2f}% | PCK@0.10: {val_pck10:.2f}% | "
            f"Mean Err: {mean_px_err:.2f}px"
        )

        # Checkpoints with complete resume states
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "val_loss": val_loss,
            "val_pck10": val_pck10,
            "val_pck05": val_pck05,
            "mean_px_err": mean_px_err,
            "backbone_name": backbone_name
        }
        torch.save(ckpt, out_path / "last.pt")

        # Select best checkpoint by lowest Mean Pixel Error (avoids PCK@0.10 saturation)
        if mean_px_err < best_val_err:
            best_val_err = mean_px_err
            best_val_pck05 = val_pck05
            torch.save(ckpt, out_path / "best.pt")
            print(f"  [*] New best checkpoint saved! (Mean Err: {mean_px_err:.2f}px, PCK@0.05: {val_pck05:.2f}%)")

    print(f"\n[OK] Training completed. Best checkpoint: {out_path / 'best.pt'} (Mean Err: {best_val_err:.2f}px, PCK@0.05: {best_val_pck05:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/shuffled_v4_13kp")
    parser.add_argument("--output", type=str, default="checkpoints/rtmpose_13kp_official_coco")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone", type=str, default="cspnext", choices=["cspnext", "mobilenetv3", "mobilenetv2"])
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_rtmpose(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device_name=args.device,
        backbone_name=args.backbone,
        pretrained=args.pretrained,
        warmup_epochs=args.warmup,
        workers=args.workers,
        resume=args.resume
    )
