"""
train_keypoint.py
─────────────────
Training script for RTMPose-Tiny foot keypoint model (18 KP).

Training setup (matching the Springer 2026 paper):
  - Backbone  : RTMPose-Tiny (via MMPose)
  - Loss      : L2 (MSE) on heatmaps, weighted by visibility mask
  - Optimizer : AdamW, lr=1e-4, weight_decay=1e-4
  - Scheduler : Exponential decay (gamma=0.95) with warmup
  - Input     : 256×192, heatmap 64×48

Usage
-----
  python src/models/keypoint/train_keypoint.py \\
      --config configs/training_config.yaml \\
      --work-dir outputs/ \\
      [--resume outputs/checkpoints/last.pth]

The script supports two training modes:
  1. MMPose mode  (preferred): uses mmpose TrainLoop if mmpose is installed.
  2. PyTorch mode (fallback): pure PyTorch training loop using FootKeypointDataset.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import yaml
from tqdm import tqdm

# Local imports
from src.dataset.foot_dataset   import FootKeypointDataset
from src.dataset.augmentations  import get_train_transforms, get_val_transforms


# ──────────────────────────────────────────────────────────────────────────────
# Heatmap MSE Loss with visibility masking
# ──────────────────────────────────────────────────────────────────────────────

class MaskedHeatmapLoss(nn.Module):
    """
    MSE loss on predicted heatmaps, zeroed out for invisible keypoints (vis=0).

    Parameters
    ----------
    use_visibility_weight : If True, zero-weight invisible keypoints.
    """

    def __init__(self, use_visibility_weight: bool = True) -> None:
        super().__init__()
        self.use_vis = use_visibility_weight
        self.mse     = nn.MSELoss(reduction="none")

    def forward(
        self,
        pred:     torch.Tensor,  # [B, 18, Hm, Wm]
        target:   torch.Tensor,  # [B, 18, Hm, Wm]
        vis_mask: torch.Tensor,  # [B, 18]
    ) -> torch.Tensor:
        """
        Returns scalar loss.
        """
        loss = self.mse(pred, target)             # [B, 18, Hm, Wm]

        if self.use_vis:
            # Expand vis_mask to heatmap shape: [B, 18, 1, 1] → broadcast
            mask = vis_mask.view(*vis_mask.shape, 1, 1).float()
            loss = loss * mask

        return loss.mean()


# ──────────────────────────────────────────────────────────────────────────────
# PCK metric (used for validation monitoring)
# ──────────────────────────────────────────────────────────────────────────────

def compute_pck(
    pred_kps:  torch.Tensor,  # [B, 18, 2]  in heatmap coords
    gt_kps:    torch.Tensor,  # [B, 18, 2]
    vis_mask:  torch.Tensor,  # [B, 18]
    threshold: float = 5.0,   # pixels in heatmap space
) -> float:
    """Compute PCK@threshold — fraction of visible keypoints within threshold pixels."""
    diff    = torch.norm(pred_kps - gt_kps, dim=-1)  # [B, 18]
    correct = (diff < threshold).float() * vis_mask
    total   = vis_mask.sum().item()
    if total == 0:
        return 0.0
    return float(correct.sum().item() / total)


def heatmaps_to_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Differentiable soft-argmax to get (x, y) from heatmaps.

    Parameters
    ----------
    heatmaps : torch.Tensor [B, K, H, W]

    Returns
    -------
    coords : torch.Tensor [B, K, 2]  (x, y) in heatmap pixel space
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device

    # Flatten and softmax
    flat = heatmaps.view(B, K, -1)
    prob = torch.softmax(flat * 10.0, dim=-1)  # temperature scaling

    xs = torch.arange(W, dtype=torch.float32, device=device)
    ys = torch.arange(H, dtype=torch.float32, device=device)

    prob_2d = prob.view(B, K, H, W)
    x_coord = (prob_2d.sum(dim=2) * xs).sum(dim=2)    # [B, K]
    y_coord = (prob_2d.sum(dim=3) * ys).sum(dim=2)    # [B, K]

    return torch.stack([x_coord, y_coord], dim=-1)    # [B, K, 2]


# ──────────────────────────────────────────────────────────────────────────────
# Model builder (RTMPose-Tiny via MMPose or lightweight fallback)
# ──────────────────────────────────────────────────────────────────────────────

def build_rtmpose_tiny(num_keypoints: int = 18) -> nn.Module:
    """
    Build RTMPose-Tiny for custom keypoint count.

    Tries MMPose first; falls back to a lightweight MobileNetV2-based model.
    """
    try:
        from mmpose.apis import init_model
        from mmengine.config import Config

        cfg_path = "configs/rtmpose_tiny_18kp.py"
        if os.path.exists(cfg_path):
            # Use MMPose config
            model = init_model(cfg_path, checkpoint=None, device="cpu")
            return model
    except ImportError:
        pass

    # Fallback: lightweight MobileNetV2 + custom head
    return _build_fallback_model(num_keypoints)


def _build_fallback_model(num_keypoints: int) -> nn.Module:
    """
    Lightweight fallback: MobileNetV2 backbone + heatmap head.
    Only used if MMPose is not installed.
    """
    import torchvision.models as models

    class FootKPModel(nn.Module):
        def __init__(self, num_kp: int) -> None:
            super().__init__()
            mv2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            self.backbone = mv2.features          # output: [B, 1280, H/32, W/32]
            self.head = nn.Sequential(
                nn.Conv2d(1280, 256, 1),
                nn.ReLU(inplace=True),
                # Upsample to heatmap size
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(256, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(128, num_kp, 1),
            )
            # Final output: [B, num_kp, H/8, W/8] → interpolate to 64×48

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,3,256,192]
            feat = self.backbone(x)                           # [B,1280,8,6]
            hm   = self.head(feat)                            # [B,K,32,24]
            hm   = torch.nn.functional.interpolate(
                hm, size=(64, 48), mode="bilinear", align_corners=False
            )
            return hm                                         # [B, K, 64, 48]

    return FootKPModel(num_keypoints)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model:      nn.Module,
    loader:     DataLoader,
    optimizer:  optim.Optimizer,
    criterion:  nn.Module,
    device:     torch.device,
    scheduler:  Optional[object] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch in tqdm(loader, desc="Train", leave=False):
        images   = batch["image"].to(device)        # [B, 3, 256, 192]
        heatmaps = batch["heatmaps"].to(device)     # [B, 18, 64, 48]
        vis_mask = batch["vis_mask"].to(device)     # [B, 18]

        optimizer.zero_grad()
        pred = model(images)                         # [B, 18, 64, 48]
        loss = criterion(pred, heatmaps, vis_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    return {"loss": total_loss / n_batches}


@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss  = 0.0
    total_pck   = 0.0
    n_batches   = len(loader)

    for batch in tqdm(loader, desc="Val  ", leave=False):
        images   = batch["image"].to(device)
        heatmaps = batch["heatmaps"].to(device)
        vis_mask = batch["vis_mask"].to(device)
        gt_kps   = batch["keypoints"].to(device)  # [B, 18, 2]

        pred = model(images)
        loss = criterion(pred, heatmaps, vis_mask)
        total_loss += loss.item()

        # PCK
        pred_kps = heatmaps_to_coords(pred)        # [B, 18, 2]
        pck = compute_pck(pred_kps, gt_kps, vis_mask, threshold=5.0)
        total_pck += pck

    return {
        "val_loss": total_loss / n_batches,
        "pck_5px":  total_pck  / n_batches,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train RTMPose-Tiny foot keypoint model.")
    parser.add_argument("--config",   default="configs/training_config.yaml")
    parser.add_argument("--work-dir", default="outputs/")
    parser.add_argument("--resume",   default=None, help="Path to checkpoint to resume from.")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = FootKeypointDataset(
        ann_file=cfg["dataset"]["train_json"],
        image_dir=cfg["dataset"]["image_dir"],
        transforms=get_train_transforms(
            flip_prob=0.5,
            brightness_limit=cfg["augmentation"]["train"]["brightness_limit"],
            contrast_limit=cfg["augmentation"]["train"]["contrast_limit"],
        ),
        sigma=cfg["model"]["sigma"],
    )
    val_ds = FootKeypointDataset(
        ann_file=cfg["dataset"]["val_json"],
        image_dir=cfg["dataset"]["image_dir"],
        transforms=get_val_transforms(),
        sigma=cfg["model"]["sigma"],
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_rtmpose_tiny(num_keypoints=cfg["model"]["num_keypoints"])
    model = model.to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = MaskedHeatmapLoss(
        use_visibility_weight=cfg["loss"]["use_visibility_weight"]
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
        betas=tuple(cfg["optimizer"]["betas"]),
    )

    # ── Warmup + Exponential LR ───────────────────────────────────────────────
    warmup_epochs = cfg["scheduler"]["warmup_epochs"]
    warmup_ratio  = cfg["scheduler"]["warmup_ratio"]

    def warmup_fn(epoch: int) -> float:
        if epoch < warmup_epochs:
            return warmup_ratio + (1.0 - warmup_ratio) * epoch / warmup_epochs
        return 1.0

    warmup_sched = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)
    exp_sched    = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=cfg["scheduler"]["gamma"]
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_pck    = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_pck    = ckpt.get("best_pck", 0.0)
        print(f"Resumed from epoch {start_epoch}, best PCK {best_pck:.4f}")

    # ── Checkpointing dirs ────────────────────────────────────────────────────
    ckpt_dir = Path(args.work_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    total_epochs = cfg["training"]["epochs"]
    val_every    = cfg["logging"]["val_every_n_epochs"]

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        # Warmup for first N epochs, then exponential
        if epoch < warmup_epochs:
            warmup_sched.step()
        else:
            exp_sched.step()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)

        log_str = (
            f"Epoch [{epoch+1:03d}/{total_epochs}]  "
            f"loss={train_metrics['loss']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"time={time.time()-t0:.1f}s"
        )

        if (epoch + 1) % val_every == 0 or epoch == total_epochs - 1:
            val_metrics = validate(model, val_loader, criterion, device)
            log_str += (
                f"  val_loss={val_metrics['val_loss']:.4f}"
                f"  PCK@5px={val_metrics['pck_5px']:.4f}"
            )

            # Save best
            if val_metrics["pck_5px"] > best_pck:
                best_pck = val_metrics["pck_5px"]
                torch.save({
                    "epoch":     epoch + 1,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_pck":  best_pck,
                    "config":    cfg,
                }, ckpt_dir / "best.pth")
                log_str += "  ✅ best model saved"

        print(log_str)

        # Save last
        if (epoch + 1) % cfg["checkpoint"]["save_every_n_epochs"] == 0:
            torch.save({
                "epoch":     epoch + 1,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_pck":  best_pck,
                "config":    cfg,
            }, ckpt_dir / "last.pth")

    print(f"\nTraining complete. Best PCK@5px = {best_pck:.4f}")
    print(f"Best model saved to: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()

