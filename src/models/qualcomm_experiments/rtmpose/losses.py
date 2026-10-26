"""
rtmpose/losses.py
─────────────────
Loss formulations for RTMPose-13KP SimCC:
  - 1D Gaussian Label Smoothing Target Generator
  - KL-Divergence loss for 1D continuous coordinate distributions
  - Binary Cross-Entropy loss for keypoint visibility
"""

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_simcc_gaussian_1d(
    coord: float, num_bins: int, sigma: float = 6.0, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generates a 1D Gaussian distribution centered at coord over num_bins."""
    bins = torch.arange(num_bins, dtype=torch.float32, device=device)
    dist = torch.exp(-0.5 * ((bins - coord) / sigma) ** 2)
    total = dist.sum()
    if total > 0:
        dist = dist / total
    return dist


class SimCCLoss(nn.Module):
    """
    Kullback-Leibler Divergence Loss for SimCC Coordinate Classification.
    Vectorized target generation eliminates GPU-CPU synchronization bottlenecks.
    """
    def __init__(self, simcc_split_ratio: float = 2.0, sigma: float = 6.0):
        super().__init__()
        self.split_ratio = simcc_split_ratio
        self.sigma = sigma

    def build_targets(
        self,
        keypoints: torch.Tensor,  # [B, 13, 3] in image coordinates [0..img_size]
        w_bins: int = 512,
        h_bins: int = 512,
        device: torch.device = torch.device("cpu")
    ) -> Dict[str, torch.Tensor]:
        # Fully vectorized target generator: zero .item() syncs!
        kps = keypoints.to(device)
        kx = kps[..., 0] * self.split_ratio  # [B, K]
        ky = kps[..., 1] * self.split_ratio  # [B, K]
        kv = kps[..., 2]                     # [B, K]

        # Valid mask: kv > 0 and within bin range
        mask = ((kv > 0) & (kx >= 0) & (kx < w_bins) & (ky >= 0) & (ky < h_bins)).float()

        # Vectorized 1D Gaussian targets
        bins_x = torch.arange(w_bins, dtype=torch.float32, device=device).view(1, 1, -1)
        bins_y = torch.arange(h_bins, dtype=torch.float32, device=device).view(1, 1, -1)

        dist_sq_x = (bins_x - kx.unsqueeze(-1)) ** 2
        dist_sq_y = (bins_y - ky.unsqueeze(-1)) ** 2

        target_x = torch.exp(-0.5 * dist_sq_x / (self.sigma ** 2))
        target_y = torch.exp(-0.5 * dist_sq_y / (self.sigma ** 2))

        # Normalize distributions
        target_x = target_x / target_x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target_y = target_y / target_y.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Zero out invalid keypoints
        target_x = target_x * mask.unsqueeze(-1)
        target_y = target_y * mask.unsqueeze(-1)

        return {
            "target_x": target_x,
            "target_y": target_y,
            "mask": mask
        }

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        pred_x = preds["simcc_x"]  # [B, K, w_bins]
        pred_y = preds["simcc_y"]  # [B, K, h_bins]
        vis_logits = preds.get("vis_logits", None)  # [B, K, 1]

        target_x = targets["target_x"]
        target_y = targets["target_y"]
        mask = targets["mask"]  # [B, K]

        # Log-softmax over 1D bins
        log_prob_x = F.log_softmax(pred_x, dim=-1)
        log_prob_y = F.log_softmax(pred_y, dim=-1)

        # KL divergence per keypoint: D_KL(target || pred) = target * (log(target) - log_prob)
        # Mask out invisible keypoints
        kl_x = F.kl_div(log_prob_x, target_x, reduction="none").sum(dim=-1) * mask
        kl_y = F.kl_div(log_prob_y, target_y, reduction="none").sum(dim=-1) * mask

        valid_kps = mask.sum().clamp(min=1.0)
        loss_x = kl_x.sum() / valid_kps
        loss_y = kl_y.sum() / valid_kps

        # Visibility loss using BCEWithLogitsLoss (AMP safe)
        if vis_logits is not None:
            loss_vis = F.binary_cross_entropy_with_logits(vis_logits.squeeze(-1), mask)
        else:
            loss_vis = F.binary_cross_entropy(preds["vis"].squeeze(-1), mask)

        total_loss = loss_x + loss_y + 0.5 * loss_vis

        return {
            "total_loss": total_loss,
            "loss_x": loss_x,
            "loss_y": loss_y,
            "loss_vis": loss_vis
        }

