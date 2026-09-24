"""
train_blazefoot.py
──────────────────
Ultra High-Performance Direct-to-VRAM GPU Training Pipeline for Google BlazeFoot.
Equipped with YOLOv8-Style Real-Time Validation & Anchor-Relative Coordinates:
  - Bounding Box: Precision(B), Recall(B), mAP50(B), mAP50-95(B)
  - 4-Keypoint Pose: Precision(P), Recall(P), mAP50(P), mAP50-95(P) (via OKS)
  - Class breakdown table (all, left_foot, right_foot)
"""

import os
import sys
import time
import argparse
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.ops as ops
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.amp import autocast, GradScaler

from .blazefoot_detector import BlazeFoot
from .dataset import load_dataset_to_gpu, generate_blaze_anchors
from .loss import BlazeFootLoss
from .metrics import evaluate_detections, print_yolo_metrics_table

# Enable cuDNN benchmark & TF32 for highest GPU throughput on NVIDIA GPUs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def decode_predictions(
    cls_preds: torch.Tensor,
    box_preds: torch.Tensor,
    kpt_preds: torch.Tensor,
    anchors: torch.Tensor,
    img_size: int = 320,
    conf_thresh: float = 0.35,
    nms_iou_thresh: float = 0.50
) -> List[Dict]:
    """
    Decodes anchor-relative predictions to pixel coordinates and applies batched NMS per image on GPU.
    """
    B = cls_preds.shape[0]
    cls_scores = torch.sigmoid(cls_preds) # [B, 1050, 2]

    # Decode boxes with anchors
    ax = anchors[:, 0]
    ay = anchors[:, 1]
    aw = anchors[:, 2]
    ah = anchors[:, 3]

    cx = (ax + box_preds[..., 0] * aw) * img_size
    cy = (ay + box_preds[..., 1] * ah) * img_size
    bw = (aw * torch.exp(torch.clamp(box_preds[..., 2], -4.0, 4.0))) * img_size
    bh = (ah * torch.exp(torch.clamp(box_preds[..., 3], -4.0, 4.0))) * img_size

    x1 = (cx - bw / 2.0).clamp(0, img_size)
    y1 = (cy - bh / 2.0).clamp(0, img_size)
    x2 = (cx + bw / 2.0).clamp(0, img_size)
    y2 = (cy + bh / 2.0).clamp(0, img_size)
    boxes_px = torch.stack([x1, y1, x2, y2], dim=-1) # [B, 1050, 4]

    # Decode keypoints with anchors
    kpts_list = []
    for k in range(4):
        kx = ((ax + kpt_preds[..., k * 3] * aw) * img_size).clamp(0, img_size)
        ky = ((ay + kpt_preds[..., k * 3 + 1] * ah) * img_size).clamp(0, img_size)
        kv = torch.sigmoid(kpt_preds[..., k * 3 + 2])
        kpts_list.append(torch.stack([kx, ky, kv], dim=-1))
    kpts_px = torch.stack(kpts_list, dim=2) # [B, 1050, 4, 3]

    batch_preds = []
    for b in range(B):
        im_cls = cls_scores[b]
        im_boxes = boxes_px[b]
        im_kpts = kpts_px[b]

        max_scores, class_ids = torch.max(im_cls, dim=-1)
        pos_mask = max_scores >= conf_thresh

        if not pos_mask.any():
            batch_preds.append({
                "boxes": torch.zeros((0, 4), device=cls_preds.device),
                "scores": torch.zeros((0,), device=cls_preds.device),
                "classes": torch.zeros((0,), dtype=torch.int64, device=cls_preds.device),
                "keypoints": torch.zeros((0, 4, 3), device=cls_preds.device)
            })
            continue

        filt_boxes = im_boxes[pos_mask]
        filt_scores = max_scores[pos_mask]
        filt_classes = class_ids[pos_mask]
        filt_kpts = im_kpts[pos_mask]

        keep_indices = ops.batched_nms(filt_boxes, filt_scores, filt_classes, nms_iou_thresh)

        batch_preds.append({
            "boxes": filt_boxes[keep_indices],
            "scores": filt_scores[keep_indices],
            "classes": filt_classes[keep_indices],
            "keypoints": filt_kpts[keep_indices]
        })

    return batch_preds


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
    criterion: BlazeFootLoss,
    anchors: torch.Tensor,
    compute_map: bool = True
) -> Dict[str, any]:
    model.eval()
    images = val_data["images"]
    target_cls = val_data["target_cls"]
    target_box = val_data["target_box"]
    target_kpt = val_data["target_kpt"]
    target_mask = val_data["target_mask"]
    raw_targets = val_data.get("raw_targets", [])

    with autocast(device_type="cuda" if images.device.type == "cuda" else "cpu", dtype=torch.float16):
        cls_preds, box_preds, kpt_preds = model(images)
        loss_dict = criterion(
            cls_preds, box_preds, kpt_preds,
            target_cls, target_box, target_kpt, target_mask
        )

    res = {
        "val_loss": loss_dict["total_loss"].item(),
        "val_cls": loss_dict["loss_cls"].item(),
        "val_box": loss_dict["loss_box"].item(),
        "val_kpt": loss_dict["loss_kpt"].item(),
        "p_box": 0.0, "r_box": 0.0, "map50_box": 0.0, "map_box": 0.0,
        "p_kpt": 0.0, "r_kpt": 0.0, "map50_kpt": 0.0, "map_kpt": 0.0,
        "metrics": {}
    }

    if compute_map and raw_targets:
        decoded_preds = decode_predictions(cls_preds, box_preds, kpt_preds, anchors=anchors, img_size=320, conf_thresh=0.35, nms_iou_thresh=0.50)
        eval_metrics = evaluate_detections(decoded_preds, raw_targets, num_classes=2)
        res["metrics"] = eval_metrics
        res["p_box"] = eval_metrics["p_box"]
        res["r_box"] = eval_metrics["r_box"]
        res["map50_box"] = eval_metrics["map50_box"]
        res["map_box"] = eval_metrics["map_box"]
        res["p_kpt"] = eval_metrics["p_kpt"]
        res["r_kpt"] = eval_metrics["r_kpt"]
        res["map50_kpt"] = eval_metrics["map50_kpt"]
        res["map_kpt"] = eval_metrics["map_kpt"]

    return res


def main():
    parser = argparse.ArgumentParser(description="Direct-to-VRAM BlazeFoot Training with YOLO-Style Real-Time Metrics.")
    parser.add_argument("--data", default="data/shuffled_v3", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (e.g. 64 for optimal gradient steps)")
    parser.add_argument("--lr0", type=float, default=0.0015, help="Initial learning rate")
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

    print("=" * 105)
    print("  [⚡] Google BlazeFoot Direct-to-VRAM GPU Engine with Anchor-Relative Decoding")
    print(f"  Device        : {device} ({gpu_name}, {vram_gb:.1f} GB VRAM)")
    print(f"  Dataset       : {args.data}")
    print(f"  Epochs        : {args.epochs} | Batch Size: {args.batch} | LR0: {args.lr0}")
    print(f"  TF32 / FP16   : Enabled | Resident VRAM: ON")
    print(f"  Output Dir    : {out_dir}")
    print("=" * 105)

    # 1. Initialize Model (4 Keypoints strictly for Path A)
    model = BlazeFoot(num_classes=2, num_keypoints=4).to(device)
    if args.compile and hasattr(torch, "compile"):
        print("  [⚡] Compiling model with torch.compile()...")
        model = torch.compile(model)

    params = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters())
    print(f"[1] BlazeFoot (4-KP Native) initialized with {params:,} parameters ({params*4/1e6:.2f} MB in FP32)\n")

    # 2. Preload 100% of Data Directly into GPU VRAM
    train_data = load_dataset_to_gpu(args.data, split="train", img_size=320, device=device)
    val_data   = load_dataset_to_gpu(args.data, split="valid", img_size=320, device=device)
    anchors    = val_data["anchors"]

    # 3. Setup Optimizer, Warmup Scheduler & Scaler
    criterion = BlazeFootLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr0, weight_decay=0.0005)

    warmup_epochs = min(5, args.epochs // 10)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.2, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_loss = float("inf")
    best_map50 = 0.0
    start_time = time.time()
    last_eval_metrics = None

    print("Starting Ultra-Fast Training Loop (Live YOLO Box & Pose Metrics)...")
    header = f"  {'Epoch':>7} | {'TrainLoss':>9} | {'ValLoss':>8} | {'Box(P':>6} {'R':>5} {'mAP50':>6} {'50-95)':>7} | {'Pose(P':>6} {'R':>5} {'mAP50':>6} {'50-95)':>7} | {'Time':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch_gpu(model, train_data, criterion, optimizer, scaler, batch_size=args.batch, augment=True)
        
        # Calculate full YOLO metrics (mAP50, mAP50-95) every epoch
        val_metrics = validate_gpu(model, val_data, criterion, anchors=anchors, compute_map=True)
        scheduler.step()

        elapsed = time.time() - t0
        v_loss = val_metrics["val_loss"]
        map50_b = val_metrics["map50_box"]
        last_eval_metrics = val_metrics.get("metrics")

        # Save best model by combined Box+Pose mAP or validation loss
        is_best = (map50_b > best_map50) or (isinstance(best_val_loss, float) and v_loss < best_val_loss)
        if is_best:
            best_val_loss = min(best_val_loss, v_loss)
            best_map50 = max(best_map50, map50_b)
            best_path = os.path.join(weights_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": v_loss,
                "map50_box": map50_b,
                "config": {"num_classes": 2, "num_keypoints": 4}
            }, best_path)

        last_path = os.path.join(weights_dir, "last.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": v_loss
        }, last_path)

        star = " *" if is_best else "  "
        print(f"  {epoch:5d}/{args.epochs} | {train_metrics['loss']:9.4f} | {v_loss:8.4f} | {val_metrics['p_box']:6.3f} {val_metrics['r_box']:5.3f} {val_metrics['map50_box']:6.3f} {val_metrics['map_box']:7.3f} | {val_metrics['p_kpt']:6.3f} {val_metrics['r_kpt']:5.3f} {val_metrics['map50_kpt']:6.3f} {val_metrics['map_kpt']:7.3f} | {elapsed:5.2f}s{star}", flush=True)

    total_time = (time.time() - start_time)
    print("=" * 105)
    print(f"  Training Complete in {total_time:.1f} seconds ({total_time/args.epochs*1000:.1f} ms/epoch)!")
    print(f"  Best Val Loss : {best_val_loss:.4f} | Best Box mAP50: {best_map50:.4f}")
    print(f"  Saved Weights : {os.path.join(weights_dir, 'best.pt')}")
    print("=" * 105)

    # Print Final YOLO Summary Table
    if last_eval_metrics:
        print_yolo_metrics_table(last_eval_metrics, title="Final Validation Results (COCO / YOLOv8 Standard)")


if __name__ == "__main__":
    main()
