"""
rtmpose/train_distill.py
────────────────────────
Teacher-Student Knowledge Distillation Trainer for Compressed RTMPose-13KP.
Distills knowledge from rtmpose_13kp_improved5 (14.2 MB) into lightweight student models:
  - Option A: RTMPose-Slim (Stride 16, ~3.04 MB FP32 / 0.76 MB INT8)
  - Option A + Lite: RTMPose-Slim-Stage4Lite (Stride 32, ~4.25 MB FP32 / 1.06 MB INT8)
  - Option B: RTMPose-Pico (Width 0.5x, ~3.32 MB FP32 / 0.83 MB INT8)

Loss components:
  1. Ground-truth SimCC KL loss + Visibility BCE loss
  2. Soft-logit SimCC KL Distillation (temperature tau=2.0)
  3. Ground-truth Chirality (Left/Right foot) BCE loss
  4. Teacher Chirality soft distillation
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.common.dataset import FootPoseDataset13KP
from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP, load_improved5_weights
from src.models.qualcomm_experiments.rtmpose.losses import SimCCLoss
from src.models.qualcomm_experiments.rtmpose.decoder import decode_simcc
from src.models.qualcomm_experiments.rtmpose.train import collate_fn_rtmpose


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class DistillationLoss(nn.Module):
    """
    Composite Teacher-Student Distillation Loss for SimCC Coordinate Distributions
    and In-Model Foot Chirality (Handedness).
    """
    def __init__(
        self,
        simcc_split_ratio: float = 2.0,
        sigma: float = 6.0,
        temperature: float = 2.0,
        alpha_kd: float = 1.0,
        beta_hand_gt: float = 0.5,
        gamma_hand_kd: float = 0.5
    ):
        super().__init__()
        self.gt_criterion = SimCCLoss(simcc_split_ratio=simcc_split_ratio, sigma=sigma)
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.beta_hand_gt = beta_hand_gt
        self.gamma_hand_kd = gamma_hand_kd
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        student_preds: Dict[str, torch.Tensor],
        teacher_preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # 1. Ground truth SimCC loss
        gt_losses = self.gt_criterion(student_preds, targets)
        loss_gt = gt_losses["total_loss"]

        # 2. Soft-logit SimCC KL distillation
        # Student & Teacher logits: [B, K, Bins]
        s_logits_x = student_preds["simcc_x"]
        s_logits_y = student_preds["simcc_y"]
        t_logits_x = teacher_preds["simcc_x"].detach()
        t_logits_y = teacher_preds["simcc_y"].detach()

        T = self.temperature
        p_t_x = F.softmax(t_logits_x / T, dim=-1)
        p_t_y = F.softmax(t_logits_y / T, dim=-1)
        log_s_x = F.log_softmax(s_logits_x / T, dim=-1)
        log_s_y = F.log_softmax(s_logits_y / T, dim=-1)

        kl_x = F.kl_div(log_s_x, p_t_x, reduction="batchmean") * (T * T)
        kl_y = F.kl_div(log_s_y, p_t_y, reduction="batchmean") * (T * T)
        loss_kd = 0.5 * (kl_x + kl_y)

        total_loss = loss_gt + self.alpha_kd * loss_kd

        # 3. Handedness (Chirality) Ground-Truth & Teacher Distillation
        loss_hand_gt = torch.tensor(0.0, device=loss_gt.device)
        loss_hand_kd = torch.tensor(0.0, device=loss_gt.device)

        if "hand_logits" in student_preds:
            s_hand_logits = student_preds["hand_logits"]
            classes = targets.get("classes", None)

            if classes is not None:
                valid_mask = classes >= 0
                if valid_mask.any():
                    gt_float = classes[valid_mask].float()
                    loss_hand_gt = self.bce(s_hand_logits[valid_mask], gt_float)
                    total_loss = total_loss + self.beta_hand_gt * loss_hand_gt

            if "hand_logits" in teacher_preds:
                t_hand_prob = teacher_preds["hand"].detach()
                loss_hand_kd = F.binary_cross_entropy_with_logits(s_hand_logits, t_hand_prob)
                total_loss = total_loss + self.gamma_hand_kd * loss_hand_kd

        return {
            "total_loss": total_loss,
            "loss_gt": loss_gt,
            "loss_kd": loss_kd,
            "loss_hand_gt": loss_hand_gt,
            "loss_hand_kd": loss_hand_kd
        }


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    simcc_split_ratio: float = 2.0
) -> Dict[str, float]:
    model.eval()
    ref_diag = math.sqrt(256.0**2 + 256.0**2)
    all_pck05, all_pck10, errs = [], [], []
    hand_correct, hand_total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["images"].to(device)
            kps_gt = batch["keypoints"].to(device)
            classes = batch["classes"].to(device)

            preds = model(imgs)
            pred_kps = decode_simcc(preds, simcc_split_ratio=simcc_split_ratio)
            gt_np = kps_gt.cpu().numpy()

            for b in range(len(imgs)):
                p05, p10, tot = 0, 0, 0
                for k in range(13):
                    if gt_np[b, k, 2] > 0:
                        dist = float(np.linalg.norm(pred_kps[b, k, :2] - gt_np[b, k, :2]))
                        errs.append(dist)
                        if dist <= 0.05 * ref_diag:
                            p05 += 1
                        if dist <= 0.10 * ref_diag:
                            p10 += 1
                        tot += 1

                if tot > 0:
                    all_pck05.append(p05 / tot)
                    all_pck10.append(p10 / tot)

                if "hand" in preds:
                    pred_cls = 1 if preds["hand"][b].item() >= 0.5 else 0
                    if classes[b].item() >= 0:
                        hand_correct += int(pred_cls == classes[b].item())
                        hand_total += 1

    mean_pck05 = float(np.mean(all_pck05) * 100.0) if all_pck05 else 0.0
    mean_pck10 = float(np.mean(all_pck10) * 100.0) if all_pck10 else 0.0
    mean_err = float(np.mean(errs)) if errs else 999.0
    hand_acc = float(hand_correct / max(1, hand_total) * 100.0) if hand_total > 0 else 0.0

    return {
        "pck05": mean_pck05,
        "pck10": mean_pck10,
        "mean_err": mean_err,
        "hand_acc": hand_acc
    }


def train_distillation(
    data_dir: str,
    teacher_weights: str = "checkpoints/rtmpose_13kp_improved5/best_with_hand.pt",
    student_arch: str = "cspnext_slim",
    output_dir: str = "checkpoints/rtmpose_13kp_slim_distill",
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 3e-4,
    temperature: float = 2.0,
    alpha_kd: float = 1.0,
    warmup_epochs: int = 3,
    workers: int = 4,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    img_size: int = 256,
    amp: bool = True
):
    set_seed(42)
    device = torch.device(device_name)
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  QUALCOMM RTMPOSE-13KP: TEACHER-STUDENT KNOWLEDGE DISTILLATION")
    print(f"  Dataset:         {data_path}")
    print(f"  Teacher Model:   {teacher_weights} (RTMPose-Tiny / CSPNeXt-384, 14.2 MB)")
    print(f"  Student Arch:    {student_arch}")
    print(f"  Device:          {device} (AMP: {amp})")
    print(f"  Epochs:          {epochs} (Warmup: {warmup_epochs}) | Batch: {batch_size}")
    print(f"  Learning Rate:   Backbone: {lr * 0.2:.6f} | Head: {lr:.6f}")
    print(f"  Distillation:    Temperature tau={temperature} | Alpha_KD={alpha_kd}")
    print("=" * 80)

    # 1. Load Frozen Teacher Model
    print(f"[*] Initializing Teacher model from {teacher_weights}...")
    teacher = RTMPose13KP(
        num_keypoints=13,
        input_size=(img_size, img_size),
        simcc_split_ratio=2.0,
        backbone_name="cspnext",
        pretrained=False,
        include_handedness=True
    ).to(device)

    try:
        t_ckpt = torch.load(teacher_weights, map_location=device, weights_only=False)
    except TypeError:
        t_ckpt = torch.load(teacher_weights, map_location=device)
    t_state = t_ckpt.get("model_state", t_ckpt)
    teacher.load_state_dict(t_state, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("[+] Teacher loaded and frozen successfully.")

    # 2. Initialize Student Model & Warm-start from Teacher
    print(f"[*] Initializing Student model ({student_arch})...")
    student = RTMPose13KP(
        num_keypoints=13,
        input_size=(img_size, img_size),
        simcc_split_ratio=2.0,
        backbone_name=student_arch,
        pretrained=False,
        include_handedness=True
    ).to(device)

    total_student_params = sum(p.numel() for p in student.parameters())
    print(f"[+] Student parameter count: {total_student_params:,} ({total_student_params * 4 / 1024**2:.2f} MB FP32)")

    # Warm-start compatible weights
    transferred = load_improved5_weights(student, teacher_weights)
    print(f"[+] Student warm-started with {transferred:,} parameters ({transferred / total_student_params * 100:.1f}%) from teacher!")

    # 3. Datasets and Loaders
    train_images = data_path / "train" / "images"
    valid_images = data_path / "valid" / "images"

    train_ds = FootPoseDataset13KP(
        images_dir=str(train_images),
        img_size=img_size,
        mode="crop",
        augment=True,
        allow_flip=False,  # Enforce chirality preservation
        include_negatives=True
    )
    val_ds = FootPoseDataset13KP(
        images_dir=str(valid_images),
        img_size=img_size,
        mode="crop",
        augment=False,
        allow_flip=False,
        include_negatives=True
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
    print(f"[+] Dataset loaded: {len(train_ds)} train crops, {len(val_ds)} validation crops.")

    # 4. Criterion, Optimizer & Cosine LR Scheduler
    distill_criterion = DistillationLoss(
        simcc_split_ratio=2.0,
        sigma=6.0,
        temperature=temperature,
        alpha_kd=alpha_kd
    ).to(device)

    bb_params = [p for n, p in student.named_parameters() if "backbone" in n and p.requires_grad]
    head_params = [p for n, p in student.named_parameters() if "backbone" not in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": lr * 0.2},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-4)

    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and (device.type == "cuda"))

    # Initial baseline validation
    print("[*] Running initial baseline validation before training...")
    init_metrics = evaluate(student, val_loader, device)
    print(f"   [Init] PCK@0.05: {init_metrics['pck05']:.2f}% | PCK@0.10: {init_metrics['pck10']:.2f}% | Mean Err: {init_metrics['mean_err']:.2f} px | Hand: {init_metrics['hand_acc']:.2f}%")

    best_mean_err = init_metrics["mean_err"]
    best_pck05 = init_metrics["pck05"]

    log_path = out_path / "trainlog.md"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Distillation Training Log: Compressed RTMPose-13KP\n\n")
        f.write(f"- Student Arch: `{student_arch}` ({total_student_params * 4 / 1024**2:.2f} MB FP32)\n")
        f.write(f"- Teacher: `{teacher_weights}`\n")
        f.write(f"- Dataset: `{data_path}`\n\n")
        f.write("| Epoch | Train Loss | GT Loss | KD Loss | Val PCK@0.05 | Val PCK@0.10 | Mean Err (px) | Hand Acc (%) | Time |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

    for epoch in range(1, epochs + 1):
        student.train()
        epoch_loss = 0.0
        epoch_gt_loss = 0.0
        epoch_kd_loss = 0.0
        start_t = time.time()

        for step, batch in enumerate(train_loader):
            imgs = batch["images"].to(device)
            kps = batch["keypoints"].to(device)
            classes = batch["classes"].to(device)

            targets = distill_criterion.gt_criterion.build_targets(
                kps, w_bins=int(img_size * 2), h_bins=int(img_size * 2), device=device
            )
            targets["classes"] = classes

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp and (device.type == "cuda")):
                with torch.no_grad():
                    teacher_preds = teacher(imgs)
                student_preds = student(imgs)
                losses = distill_criterion(student_preds, teacher_preds, targets)
                loss = losses["total_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_gt_loss += losses["loss_gt"].item()
            epoch_kd_loss += losses["loss_kd"].item()

        scheduler.step()
        elapsed = time.time() - start_t
        num_batches = max(1, len(train_loader))

        avg_loss = epoch_loss / num_batches
        avg_gt = epoch_gt_loss / num_batches
        avg_kd = epoch_kd_loss / num_batches

        # Validation
        val_metrics = evaluate(student, val_loader, device)

        is_best = val_metrics["mean_err"] < best_mean_err or (
            abs(val_metrics["mean_err"] - best_mean_err) < 0.2 and val_metrics["pck05"] > best_pck05
        )

        status_flag = " [*] BEST!" if is_best else ""
        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
            f"Train: {avg_loss:.4f} (GT: {avg_gt:.3f}, KD: {avg_kd:.3f}) | "
            f"Val PCK@0.05: {val_metrics['pck05']:.2f}% | PCK@0.10: {val_metrics['pck10']:.2f}% | "
            f"Mean Err: {val_metrics['mean_err']:.2f}px | Hand Acc: {val_metrics['hand_acc']:.1f}%{status_flag}"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"| {epoch:02d} | {avg_loss:.4f} | {avg_gt:.4f} | {avg_kd:.4f} | "
                f"{val_metrics['pck05']:.2f}% | {val_metrics['pck10']:.2f}% | "
                f"{val_metrics['mean_err']:.2f}px | {val_metrics['hand_acc']:.1f}% | {elapsed:.1f}s |\n"
            )

        # Checkpoint saving
        last_ckpt = {
            "epoch": epoch,
            "student_arch": student_arch,
            "backbone_name": student_arch,
            "model_state": student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "mean_px_err": val_metrics["mean_err"],
            "val_pck05": val_metrics["pck05"],
            "val_pck10": val_metrics["pck10"],
            "hand_acc": val_metrics["hand_acc"]
        }
        torch.save(last_ckpt, str(out_path / "last.pt"))

        if is_best:
            best_mean_err = val_metrics["mean_err"]
            best_pck05 = val_metrics["pck05"]
            torch.save(last_ckpt, str(out_path / "best.pt"))
            torch.save(last_ckpt, str(out_path / "best_with_hand.pt"))

    print(f"\n[OK] Distillation training finished! Best checkpoint saved to: {out_path / 'best_with_hand.pt'}")
    return str(out_path / "best_with_hand.pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Distillation Trainer for Compressed RTMPose-13KP")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset root containing train/ and valid/")
    parser.add_argument("--teacher", type=str, default="checkpoints/rtmpose_13kp_improved5/best_with_hand.pt", help="Path to teacher weights")
    parser.add_argument("--student_arch", type=str, default="cspnext_slim", choices=["cspnext_slim", "cspnext_slim_s4lite", "cspnext_pico"], help="Student architecture")
    parser.add_argument("--output", type=str, default="checkpoints/rtmpose_13kp_slim_distill", help="Output directory")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for head (backbone is 0.2x)")
    parser.add_argument("--temperature", type=float, default=2.0, help="Distillation temperature tau")
    parser.add_argument("--alpha_kd", type=float, default=1.0, help="Weight for SimCC logit distillation")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--amp", action="store_true", default=True, help="Use automatic mixed precision")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_distillation(
        data_dir=args.data,
        teacher_weights=args.teacher,
        student_arch=args.student_arch,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        temperature=args.temperature,
        alpha_kd=args.alpha_kd,
        warmup_epochs=args.warmup,
        workers=args.workers,
        amp=args.amp
    )

