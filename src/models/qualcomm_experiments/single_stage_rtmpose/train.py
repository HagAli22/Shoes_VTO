"""
single_stage_rtmpose/train.py
─────────────────────────────
Complete Training Script for Single-Stage RTMPose (Apache-2.0).
Simultaneously optimizes 2-class chirality, bounding box regression, and 13 foot keypoints.
Warm-starts backbone from official COCO pretrained weights.
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.single_stage_rtmpose.model import (
    SingleStageRTMPose,
    postprocess_single_stage_pose
)
from src.models.qualcomm_experiments.single_stage_rtmpose.loss import SingleStagePoseLoss
from src.models.qualcomm_experiments.single_stage_rtmpose.dataset import (
    SingleStageFootDataset,
    collate_fn_single_stage
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class ModelEMA:
    """Exponential Moving Average with warmup ramp matching official YOLO/RTMPose recipe."""
    def __init__(self, model: nn.Module, decay: float = 0.9998):
        self.module = model
        self.decay = decay
        self.iteration = 0
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        self.iteration += 1
        d = self.decay * (1.0 - math.exp(-self.iteration / 2000.0))
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if k in self.shadow:
                    if v.dtype.is_floating_point:
                        self.shadow[k] = d * self.shadow[k] + (1.0 - d) * v
                    else:
                        self.shadow[k] = v.clone()

    def apply_shadow(self):
        self.backup = {k: v.clone() for k, v in self.module.state_dict().items()}
        self.module.load_state_dict(self.shadow)

    def restore(self):
        self.module.load_state_dict(self.backup)


def evaluate_val_metrics(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    conf_thresh: float = 0.30
) -> Dict[str, float]:
    """
    Evaluates validation metrics across bounding boxes, chirality, and 13 keypoints:
      - Box F1@0.50 (with exact Left/Right foot class match)
      - Pose PCK@0.2 (Percentage of Correct Keypoints within 20% of foot box diagonal)
      - Negative precision (% of empty carpet/floor images with zero detections)
    """
    model.eval()
    tp, fp, fn = 0, 0, 0
    clean_negatives = 0
    total_negatives = 0

    pck_hits = 0
    total_gt_kpts = 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["images"].to(device)
            gt_boxes_list = batch["boxes"]
            gt_labels_list = batch["labels"]
            gt_kpts_list = batch["keypoints"]

            # Export/eval mode outputs [B, 45, N]
            out_tensor = model(imgs, export_mode=True)
            B = imgs.shape[0]

            for b in range(B):
                preds = postprocess_single_stage_pose(out_tensor[b], conf_thresh=conf_thresh, iou_thresh=0.45)
                gt_boxes = gt_boxes_list[b].to(device)
                gt_labels = gt_labels_list[b].to(device)
                gt_kpts = gt_kpts_list[b].to(device)

                num_gt = gt_boxes.shape[0]
                num_pred = len(preds)

                if num_gt == 0:
                    total_negatives += 1
                    if num_pred == 0:
                        clean_negatives += 1
                    else:
                        fp += num_pred
                    continue

                if num_pred == 0:
                    fn += num_gt
                    total_gt_kpts += int((gt_kpts[..., 2] > 0).sum().item())
                    continue

                pred_boxes_t = torch.tensor([p["bbox"] for p in preds], device=device, dtype=torch.float32)
                inter_x1 = torch.max(pred_boxes_t[:, 0:1], gt_boxes[:, 0:1].T)
                inter_y1 = torch.max(pred_boxes_t[:, 1:2], gt_boxes[:, 1:2].T)
                inter_x2 = torch.min(pred_boxes_t[:, 2:3], gt_boxes[:, 2:3].T)
                inter_y2 = torch.min(pred_boxes_t[:, 3:4], gt_boxes[:, 3:4].T)
                inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
                inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
                inter_area = inter_w * inter_h

                pred_area = (pred_boxes_t[:, 2] - pred_boxes_t[:, 0]) * (pred_boxes_t[:, 3] - pred_boxes_t[:, 1])
                gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                union_area = pred_area.unsqueeze(1) + gt_area.unsqueeze(0) - inter_area
                iou_mat = inter_area / union_area.clamp(min=1e-7)

                matched_gt = set()
                for p_idx in range(num_pred):
                    best_iou, best_idx = iou_mat[p_idx].max(dim=-1)
                    gt_id = best_idx.item()
                    cls_match = (preds[p_idx]["class_id"] == int(gt_labels[gt_id].item()))

                    if best_iou.item() >= 0.50 and gt_id not in matched_gt and cls_match:
                        tp += 1
                        matched_gt.add(gt_id)

                        # PCK Calculation: threshold = 0.20 * box diagonal
                        bw = gt_boxes[gt_id, 2] - gt_boxes[gt_id, 0]
                        bh = gt_boxes[gt_id, 3] - gt_boxes[gt_id, 1]
                        diag = torch.sqrt(bw ** 2 + bh ** 2).item()
                        pck_thresh = max(10.0, 0.20 * diag)

                        pred_kps = preds[p_idx]["keypoints"]  # [13, 3]
                        gt_k = gt_kpts[gt_id].cpu().numpy()   # [13, 3]
                        for k_i in range(13):
                            if gt_k[k_i, 2] > 0:
                                total_gt_kpts += 1
                                dist = np.hypot(pred_kps[k_i, 0] - gt_k[k_i, 0], pred_kps[k_i, 1] - gt_k[k_i, 1])
                                if dist <= pck_thresh:
                                    pck_hits += 1
                    else:
                        fp += 1

                fn += (num_gt - len(matched_gt))

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1_50 = 2 * (prec * rec) / max(1e-6, prec + rec) * 100.0
    pck_02 = (pck_hits / max(1, total_gt_kpts)) * 100.0
    neg_prec = (clean_negatives / max(1, total_negatives)) * 100.0

    return {
        "f1_50": f1_50,
        "prec": prec * 100.0,
        "rec": rec * 100.0,
        "pck_02": pck_02,
        "neg_prec": neg_prec
    }


def train_single_stage_rtmpose(
    data_dir: str,
    output_dir: str = "checkpoints/single_stage_rtmpose_320",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_classes: int = 2,
    img_size: int = 320,
    workers: int = 4,
    amp: bool = True,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    ckpt_path: str = "checkpoints/rtmpose_tiny_coco.pth",
    simcc_ckpt: str = "checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt"
):
    device = torch.device(device_name)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    set_seed(42)

    data_path = Path(data_dir)
    train_images = data_path / "train" / "images"
    val_images = data_path / "valid" / "images"
    if not val_images.exists():
        val_images = data_path / "val" / "images"

    train_dataset = SingleStageFootDataset(
        images_dir=str(train_images),
        img_size=img_size,
        num_classes=num_classes,
        augment=True,
        allow_flip=False,  # Strict: no left/right chirality confusion
        include_negatives=True
    )
    val_dataset = SingleStageFootDataset(
        images_dir=str(val_images),
        img_size=img_size,
        num_classes=num_classes,
        augment=False,
        allow_flip=False,
        include_negatives=True
    )

    use_workers = workers if os.name != "nt" else min(workers, 2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=use_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn_single_stage,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=use_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn_single_stage
    )

    print("=" * 70)
    print("  EXPERIMENT: Single-Stage RTMPose Foot Perception (Apache-2.0)")
    print(f"  Dataset:         {data_dir} ({len(train_dataset)} train, {len(val_dataset)} val)")
    print(f"  Backbone:        CSPNeXt Multi-Scale (Pretrained COCO)")
    print(f"  Neck:            CSPNeXt-PAFPN (96 channels across strides 8, 16, 32)")
    print(f"  Head:            RoI-Align + SimCC [2 classes, 4 box, 13 SimCC KPs with Continuous Expectation]")
    print(f"  Device:          {device} (AMP: {amp})")
    print(f"  Epochs:          {epochs} | Batch: {batch_size} | LR: {lr:.6f}")
    print(f"  Resolution:      {img_size}x{img_size}")
    print("=" * 70)

    model = SingleStageRTMPose(
        num_classes=num_classes,
        num_keypoints=13,
        input_size=(img_size, img_size),
        neck_channels=96,
        strides=[8, 16, 32],
        pretrained=True,
        backbone_ckpt=ckpt_path,
        simcc_ckpt=simcc_ckpt
    ).to(device)

    criterion = SingleStagePoseLoss(
        num_classes=num_classes,
        num_keypoints=13,
        strides=[8, 16, 32],
        center_radius=2.0
    ).to(device)

    # Differential learning rate
    bb_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    neck_params = [p for n, p in model.named_parameters() if "neck" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": lr * 0.1},
        {"params": neck_params, "lr": lr * 0.5},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-4)

    warmup_epochs = 3
    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and (device.type == "cuda"))
    ema = ModelEMA(model, decay=0.9998)

    best_score = 0.0

    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        train_loss = 0.0
        train_cls = 0.0
        train_box = 0.0
        train_kpt = 0.0

        for batch in train_loader:
            imgs = batch["images"].to(device)
            boxes = batch["boxes"]
            labels = batch["labels"]
            kpts = batch["keypoints"]

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp and (device.type == "cuda")):
                preds = model(imgs, gt_boxes=boxes, gt_labels=labels, gt_kpts=kpts, export_mode=False)
                losses = criterion(preds, boxes, labels, kpts, img_size=(img_size, img_size))
                total_loss = losses["total_loss"]

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            train_loss += total_loss.item()
            train_cls += losses["loss_cls"].item()
            train_box += losses["loss_box"].item()
            train_kpt += losses["loss_simcc"].item()

        scheduler.step()
        n_steps = max(1, len(train_loader))
        train_loss /= n_steps
        train_cls /= n_steps
        train_box /= n_steps
        train_kpt /= n_steps
        dt = time.time() - t_start

        if epoch % 2 == 0 or epoch == epochs:
            ema.apply_shadow()
            metrics = evaluate_val_metrics(model, val_loader, device)
            ema.restore()

            f1 = metrics["f1_50"]
            pck = metrics["pck_02"]
            neg_p = metrics["neg_prec"]

            # Composite validation score (balancing Box F1 and Pose PCK)
            score = 0.5 * f1 + 0.5 * pck

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | "
                f"Loss: {train_loss:.4f} (Cls: {train_cls:.3f}, Box: {train_box:.3f}, Kpt: {train_kpt:.3f}) | "
                f"Box F1: {f1:.1f}% | Pose PCK: {pck:.1f}% | NegPrec: {neg_p:.1f}%",
                flush=True
            )

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "f1_50": f1,
                "pck_02": pck,
                "neg_prec": neg_p
            }
            torch.save(ckpt, out_path / "last.pt")

            if score > best_score:
                best_score = score
                torch.save(ckpt, out_path / "best.pt")
                print(f"  [*] New best model saved! (Box F1: {f1:.1f}%, Pose PCK: {pck:.1f}%)", flush=True)
        else:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | "
                f"Loss: {train_loss:.4f} (Cls: {train_cls:.3f}, Box: {train_box:.3f}, Kpt: {train_kpt:.3f})",
                flush=True
            )

    print("\n[OK] Single-Stage RTMPose training completed.")
    print(f"Best Checkpoint: {out_path / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3")
    parser.add_argument("--output", type=str, default="checkpoints/single_stage_rtmpose_320")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--simcc_ckpt", type=str, default="checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt")
    args = parser.parse_args()

    train_single_stage_rtmpose(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        num_classes=args.num_classes,
        img_size=args.img_size,
        workers=args.workers,
        amp=args.amp,
        device_name=args.device,
        simcc_ckpt=args.simcc_ckpt
    )

