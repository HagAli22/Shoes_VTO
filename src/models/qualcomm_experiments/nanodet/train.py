"""
nanodet/train.py
────────────────
Complete training script for NanoDet-Plus Foot Detector (Apache-2.0).
Warm-starts from official NanoDet-Plus-m COCO pretrained weights with full
regression transfer, and adapts classification heads for the unified 'foot' class.
"""

import argparse
import io
import math
import os
import random
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.nanodet.model import NanoDetPlus, nms_postprocess, DEFAULT_CKPT_PATH
from src.models.qualcomm_experiments.nanodet.loss import NanoDetPlusLoss
from src.models.qualcomm_experiments.nanodet.dataset import NanoDetFootDataset, collate_fn_nanodet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


class ModelEMA:
    """Exponential Moving Average with warmup decay matching official NanoDet / YOLO recipe."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998):
        self.module = model
        self.decay = decay
        self.iteration = 0
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model: torch.nn.Module):
        self.iteration += 1
        # Warmup decay: ramps up smoothly from 0 to decay over first 2000 iterations
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


def evaluate_val_metrics(model, val_loader, device, num_classes=1, conf_thresh=0.20):
    model.eval()
    tp_50, fp_50, fn_50 = 0, 0, 0
    tp_75, fp_75, fn_75 = 0, 0, 0
    ious_collected = []
    clean_negatives = 0
    total_negatives = 0
    max_score_seen = 0.0

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["images"].to(device)
            gt_boxes_list = batch["boxes"]
            gt_labels_list = batch["labels"]

            out = model(imgs, export_mode=False)
            boxes = out["boxes"]
            scores = out["scores"]
            max_score_seen = max(max_score_seen, float(scores.max().item()))

            B = imgs.shape[0]
            for b in range(B):
                preds = nms_postprocess(boxes[b], scores[b], conf_thresh=conf_thresh, iou_thresh=0.45)
                gt_boxes = gt_boxes_list[b].to(device)
                gt_labels = gt_labels_list[b].to(device)

                num_gt = gt_boxes.shape[0]
                num_pred = len(preds)

                if num_gt == 0:
                    total_negatives += 1
                    if num_pred == 0:
                        clean_negatives += 1
                    else:
                        fp_50 += num_pred
                        fp_75 += num_pred
                    continue

                if num_pred == 0:
                    fn_50 += num_gt
                    fn_75 += num_gt
                    continue

                pred_t = torch.tensor([p["bbox"] for p in preds], device=device, dtype=torch.float32)

                inter_x1 = torch.max(pred_t[:, 0:1], gt_boxes[:, 0:1].T)
                inter_y1 = torch.max(pred_t[:, 1:2], gt_boxes[:, 1:2].T)
                inter_x2 = torch.min(pred_t[:, 2:3], gt_boxes[:, 2:3].T)
                inter_y2 = torch.min(pred_t[:, 3:4], gt_boxes[:, 3:4].T)
                inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
                inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
                inter_area = inter_w * inter_h

                pred_area = (pred_t[:, 2] - pred_t[:, 0]) * (pred_t[:, 3] - pred_t[:, 1])
                gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
                union_area = pred_area.unsqueeze(1) + gt_area.unsqueeze(0) - inter_area
                iou_mat = inter_area / union_area.clamp(min=1e-7)

                # IoU 0.50
                matched_50 = set()
                for p_idx in range(num_pred):
                    best_iou, best_gt = iou_mat[p_idx].max(dim=-1)
                    gt_id = best_gt.item()
                    cls_match = (num_classes == 1 or preds[p_idx]["class_id"] == int(gt_labels[gt_id].item()))
                    if best_iou.item() >= 0.50 and gt_id not in matched_50 and cls_match:
                        tp_50 += 1
                        matched_50.add(gt_id)
                        ious_collected.append(best_iou.item())
                    else:
                        fp_50 += 1
                fn_50 += (num_gt - len(matched_50))

                # IoU 0.75
                matched_75 = set()
                for p_idx in range(num_pred):
                    best_iou, best_gt = iou_mat[p_idx].max(dim=-1)
                    gt_id = best_gt.item()
                    cls_match = (num_classes == 1 or preds[p_idx]["class_id"] == int(gt_labels[gt_id].item()))
                    if best_iou.item() >= 0.75 and gt_id not in matched_75 and cls_match:
                        tp_75 += 1
                        matched_75.add(gt_id)
                    else:
                        fp_75 += 1
                fn_75 += (num_gt - len(matched_75))

    p50 = tp_50 / max(1, tp_50 + fp_50)
    r50 = tp_50 / max(1, tp_50 + fn_50)
    f1_50 = 2 * (p50 * r50) / max(1e-6, p50 + r50)

    p75 = tp_75 / max(1, tp_75 + fp_75)
    r75 = tp_75 / max(1, tp_75 + fn_75)
    f1_75 = 2 * (p75 * r75) / max(1e-6, p75 + r75)

    mean_iou = float(np.mean(ious_collected)) if ious_collected else 0.0
    neg_prec = (clean_negatives / max(1, total_negatives)) * 100.0

    return {
        "f1_50": f1_50 * 100.0,
        "f1_75": f1_75 * 100.0,
        "mean_iou": mean_iou * 100.0,
        "neg_prec": neg_prec,
        "max_conf": max_score_seen * 100.0
    }


def train_nanodet(
    data_dir: str,
    output_dir: str = "checkpoints/nanodet_foot",
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_classes: int = 1,
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
    img_size: int = 256,
    workers: int = 4,
    amp: bool = True,
    pretrained: bool = True,
    ckpt_path: str = DEFAULT_CKPT_PATH,
    resume: str = None
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

    train_dataset = NanoDetFootDataset(
        images_dir=str(train_images),
        img_size=img_size,
        num_classes=num_classes,
        augment=True,
        allow_flip=(num_classes == 1),
        include_negatives=True
    )
    val_dataset = NanoDetFootDataset(
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
        collate_fn=collate_fn_nanodet,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=use_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn_nanodet
    )

    print("=" * 70)
    print("  EXPERIMENT: NanoDet-Plus Foot Detector (Apache-2.0) Training")
    print(f"  Dataset:         {data_dir} ({len(train_dataset)} train, {len(val_dataset)} val)")
    print(f"  Backbone:        ShuffleNetV2 1.0x (Pretrained COCO: {pretrained})")
    print(f"  Device:          {device} (AMP: {amp})")
    print(f"  Epochs:          {epochs} | Batch: {batch_size} | LR: {lr:.6f} | Classes: {num_classes}")
    print(f"  Resolution:      {img_size}x{img_size}")
    print("=" * 70)

    model = NanoDetPlus(
        num_classes=num_classes,
        input_size=(img_size, img_size),
        pretrained=pretrained and (resume is None),
        ckpt_path=ckpt_path
    ).to(device)

    criterion = NanoDetPlusLoss(num_classes=num_classes, reg_max=7).to(device)

    # Differential learning rate to preserve pretrained features
    bb_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    neck_params = [p for n, p in model.named_parameters() if "fpn" in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": lr * 0.1},
        {"params": neck_params, "lr": lr * 0.5},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-4)

    warmup_epochs = 2 if resume else 3
    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and (device.type == "cuda"))
    ema = ModelEMA(model, decay=0.9998)

    best_f1 = 0.0
    start_epoch = 1

    if resume:
        resume_p = Path(resume)
        if not resume_p.exists():
            print(f"[!] Warning: Checkpoint path does not exist: {resume}")
            print(f"[*] Starting training from official COCO pretrained weights.")
        else:
            file_size = resume_p.stat().st_size
            print(f"[*] Inspecting checkpoint: {resume} ({file_size / (1024*1024):.2f} MB, {file_size} bytes)")

            if file_size < 1000:
                try:
                    with open(resume_p, "r", encoding="utf-8", errors="ignore") as f_diag:
                        snippet = f_diag.read(300)
                    print(f"[!] WARNING: Checkpoint file is suspiciously small ({file_size} bytes)!\nFile Preview:\n{snippet}")
                except Exception:
                    pass
                raise RuntimeError(
                    f"Checkpoint file '{resume}' is invalid or truncated ({file_size} bytes). "
                    "If using Kaggle/Colab, please check that the file was downloaded completely."
                )

            ckpt = None
            if zipfile.is_zipfile(resume_p):
                try:
                    with zipfile.ZipFile(resume_p, "r") as zf:
                        names = zf.namelist()
                        embedded = [n for n in names if n.endswith((".pt", ".pth"))]
                        if embedded:
                            print(f"[*] Detected outer ZIP archive containing nested checkpoint: {embedded[0]}")
                            with zf.open(embedded[0]) as zf_file:
                                data = io.BytesIO(zf_file.read())
                                ckpt = torch.load(data, map_location=device, weights_only=False)
                except Exception as z_err:
                    print(f"[!] Zip inspection notice: {z_err}")

            if ckpt is None:
                try:
                    ckpt = torch.load(resume_p, map_location=device, weights_only=False)
                except Exception as load_err:
                    with open(resume_p, "rb") as f_raw:
                        hdr = f_raw.read(64)
                    print(f"[!] torch.load failed on '{resume}': {load_err}")
                    print(f"    Raw header bytes: {hdr}")
                    raise RuntimeError(
                        f"Failed to load checkpoint '{resume}' ({file_size} bytes): {load_err}\n"
                        f"The file appears to be corrupted or not a valid PyTorch checkpoint archive."
                    ) from load_err

            state_to_load = ckpt.get("ema_state", ckpt.get("model_state", ckpt))
            model.load_state_dict(state_to_load)
            if "ema_state" in ckpt:
                ema.shadow = {k: v.clone().detach() for k, v in ckpt["ema_state"].items()}

            ckpt_epoch = ckpt.get("epoch", 0)
            if ckpt_epoch < epochs and "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                start_epoch = ckpt_epoch + 1
                best_f1 = ckpt.get("f1_75", ckpt.get("f1", 0.0))
                print(f"[+] Resumed training from epoch {start_epoch} (Best F1: {best_f1:.2f}%)")
            else:
                print(f"[+] Fine-tuning initialized from checkpoint! Running fresh schedule for {epochs} epochs at lr={lr:.6f}.")
                start_epoch = 1
                best_f1 = ckpt.get("f1_75", ckpt.get("f1", 0.0))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        train_cls = 0.0
        train_giou = 0.0
        train_dfl = 0.0
        t_start = time.time()

        for step, batch in enumerate(train_loader):
            imgs = batch["images"].to(device)
            boxes = batch["boxes"]
            labels = batch["labels"]

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp and (device.type == "cuda")):
                preds = model(imgs)
                losses = criterion(preds, boxes, labels, img_size=(img_size, img_size))
                loss = losses["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            train_loss += loss.item()
            train_cls += losses["loss_cls"].item()
            train_giou += losses["loss_giou"].item()
            train_dfl += losses["loss_dfl"].item()

        scheduler.step()
        n_steps = max(1, len(train_loader))
        train_loss /= n_steps
        train_cls /= n_steps
        train_giou /= n_steps
        train_dfl /= n_steps
        dt = time.time() - t_start

        if epoch % 2 == 0 or epoch == epochs:
            ema.apply_shadow()
            metrics = evaluate_val_metrics(model, val_loader, device, num_classes=num_classes)
            ema.restore()

            f1_50 = metrics["f1_50"]
            f1_75 = metrics["f1_75"]
            m_iou = metrics["mean_iou"]
            neg_p = metrics["neg_prec"]
            m_conf = metrics["max_conf"]

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | "
                f"Loss: {train_loss:.4f} (Cls: {train_cls:.3f}, GIoU: {train_giou:.3f}, DFL: {train_dfl:.3f}) | "
                f"F1@0.50: {f1_50:.1f}% | F1@0.75: {f1_75:.1f}% | mIoU: {m_iou:.1f}% | MaxConf: {m_conf:.1f}% | NegPrec: {neg_p:.1f}%",
                flush=True
            )

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "ema_state": ema.shadow,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "f1_50": f1_50,
                "f1_75": f1_75,
                "mean_iou": m_iou,
                "neg_prec": neg_p
            }
            torch.save(ckpt, out_path / "last.pt")

            if f1_75 > best_f1:
                best_f1 = f1_75
                torch.save(ckpt, out_path / "best.pt")
                print(f"  [*] New best detector checkpoint saved! (F1@0.75: {f1_75:.2f}%, mIoU: {m_iou:.2f}%)", flush=True)
        else:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({dt:.1f}s) | "
                f"Loss: {train_loss:.4f} (Cls: {train_cls:.3f}, GIoU: {train_giou:.3f}, DFL: {train_dfl:.3f})",
                flush=True
            )

    print("\n[OK] NanoDet-Plus training finished.")
    print(f"Best Checkpoint: {out_path / 'best.pt'} (Best F1@0.75: {best_f1:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3")
    parser.add_argument("--output", type=str, default="checkpoints/nanodet_foot")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--img_size", type=int, default=320)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    train_nanodet(
        data_dir=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        num_classes=args.num_classes,
        device_name=args.device,
        img_size=args.img_size,
        workers=args.workers,
        amp=args.amp,
        ckpt_path=args.ckpt,
        resume=args.resume
    )
