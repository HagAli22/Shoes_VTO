"""
rtmpose/train_handedness.py
────────────────────────────
Fast Calibration Training for In-Model Left/Right Handedness Head (Phase 2).
Freezes the entire backbone, neck, GAU, and SimCC keypoint localization layers in eval() mode
to keep keypoint landmark predictions bit-identical to the champion 95.73% PCK model.
Optimizes only the new `head.hand` linear layer for 10 fast epochs to reach >= 97% classification accuracy.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP
from src.models.qualcomm_experiments.common.dataset import FootPoseDataset13KP


def collate_fn_handedness(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    classes = torch.stack([b["class_id"] for b in batch], dim=0)
    return {
        "images": images,
        "classes": classes
    }


def train_handedness(
    data_dir: str = "data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3",
    init_weights: str = "checkpoints/rtmpose_13kp_official_coco_full_data/best.pt",
    output_path: str = "checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt",
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    workers: int = 4,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    print("=" * 75)
    print("  PHASE 2: FAST FROZEN HANDEDNESS (LEFT/RIGHT) HEAD CALIBRATION")
    print(f"  Init Checkpoint: {init_weights}")
    print(f"  Output Model:    {output_path}")
    print(f"  Dataset:         {data_dir}")
    print(f"  Device:          {device} (Batch: {batch_size}, LR: {lr})")
    print("=" * 75)

    # 1. Build Model with Handedness Head
    model = RTMPose13KP(
        num_keypoints=13,
        input_size=(256, 256),
        simcc_split_ratio=2.0,
        backbone_name="cspnext",
        pretrained=False,
        include_handedness=True
    ).to(device)

    # 2. Warm-start weights
    ckpt = torch.load(init_weights, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[*] Loaded base weights from {init_weights}")
    print(f"    Missing keys (expected for new hand head): {missing}")

    # 3. Freeze all parameters except head.hand
    trainable_params = []
    frozen_count = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        if name.startswith("head.hand"):
            param.requires_grad = True
            trainable_params.append(param)
            trainable_count += param.numel()
        else:
            param.requires_grad = False
            frozen_count += param.numel()

    print(f"[+] Frozen parameters: {frozen_count:,} (eval mode, zero drift on keypoints)")
    print(f"[+] Trainable parameters: {trainable_count:,} (head.hand only)")

    # Keep BatchNorm running statistics completely frozen
    model.eval()

    # 4. Data Loaders
    data_path = Path(data_dir)
    train_dataset = FootPoseDataset13KP(
        images_dir=str(data_path / "train" / "images"),
        img_size=256,
        mode="crop",
        augment=True,
        allow_flip=False,  # DO NOT flip during handedness training! Flip would invert left/right
        crop_padding=0.25,
        include_negatives=False  # Negatives have no handedness
    )
    val_dataset = FootPoseDataset13KP(
        images_dir=str(data_path / "valid" / "images"),
        img_size=256,
        mode="crop",
        augment=False,
        allow_flip=False,
        crop_padding=0.25,
        include_negatives=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn_handedness
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn_handedness
    )

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    best_val_acc = 0.0

    print(f"[*] Training started: {len(train_dataset)} train crops, {len(val_dataset)} val crops\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        # Keep model in eval mode for BN stats, but enable autograd for head.hand
        model.eval()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            imgs = batch["images"].to(device, non_blocking=True)
            cls_targets = batch["classes"].to(device, dtype=torch.float32, non_blocking=True)

            # Mask valid classes (0: Left Foot, 1: Right Foot)
            valid_mask = cls_targets >= 0
            if not valid_mask.any():
                continue

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    outputs = model(imgs)
                    logits = outputs["hand_logits"]  # [B]
                    loss = F.binary_cross_entropy_with_logits(logits[valid_mask], cls_targets[valid_mask])
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * valid_mask.sum().item()
            preds = (torch.sigmoid(logits[valid_mask]) >= 0.5).float()
            train_correct += (preds == cls_targets[valid_mask]).sum().item()
            train_total += valid_mask.sum().item()

        train_loss /= max(1, train_total)
        train_acc = (train_correct / max(1, train_total)) * 100.0

        # Validation
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                for batch in val_loader:
                    imgs = batch["images"].to(device, non_blocking=True)
                    cls_targets = batch["classes"].to(device, dtype=torch.float32, non_blocking=True)
                    valid_mask = cls_targets >= 0
                    if not valid_mask.any():
                        continue
                    outputs = model(imgs)
                    logits = outputs["hand_logits"]
                    loss = F.binary_cross_entropy_with_logits(logits[valid_mask], cls_targets[valid_mask])
                    val_loss += loss.item() * valid_mask.sum().item()
                    preds = (torch.sigmoid(logits[valid_mask]) >= 0.5).float()
                    val_correct += (preds == cls_targets[valid_mask]).sum().item()
                    val_total += valid_mask.sum().item()

        val_loss /= max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100.0
        elapsed = time.time() - t0

        is_best = val_acc > best_val_acc
        best_str = " [*] BEST!" if is_best else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%{best_str}", flush=True)

        if is_best or epoch == epochs:
            best_val_acc = max(best_val_acc, val_acc)
            save_dict = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_hand_acc": val_acc,
                "backbone_name": "cspnext",
                "include_handedness": True
            }
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(save_dict, output_path)

    print("\n" + "=" * 75)
    print(f"  CALIBRATION COMPLETE: Best Val Hand Acc: {best_val_acc:.2f}%")
    print(f"  Checkpoint saved: {output_path}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3")
    parser.add_argument("--init_weights", type=str, default="checkpoints/rtmpose_13kp_official_coco_full_data/best.pt")
    parser.add_argument("--output", type=str, default="checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    train_handedness(
        data_dir=args.data,
        init_weights=args.init_weights,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        workers=args.workers,
        device_name=args.device
    )
