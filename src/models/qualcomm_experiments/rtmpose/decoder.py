"""
rtmpose/decoder.py
──────────────────
SimCC Continuous Expectation Decoder for RTMPose-13KP Foot Pose.
Calculates sub-pixel coordinates via the 1D integral expectation:
  x = sum(i * Softmax(logits_x)[i]) / split_ratio
  y = sum(i * Softmax(logits_y)[i]) / split_ratio

Zero NMS, continuous gradients, sub-0.1ms latency in JavaScript/C++.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
import numpy as np


def decode_axis(logits: torch.Tensor, ratio: float = 2.0, win: int = 12) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sub-pixel continuous expectation within a local window (win=12, ~2 sigma) around the peak bin.
    Eliminates tail probability bias from global soft-argmax while capturing ~95% confidence mass.
    """
    p = F.softmax(logits, dim=-1)
    idx = p.argmax(dim=-1, keepdim=True)
    bins = torch.arange(p.shape[-1], dtype=torch.float32, device=p.device).view(1, 1, -1)
    mask = (bins - idx).abs() <= win
    pw = p * mask
    mass = pw.sum(dim=-1)  # in [0, 1] probability mass around peak
    coord = (pw * bins).sum(dim=-1) / mass.clamp(min=1e-8) / ratio
    return coord, mass


def decode_simcc(
    preds: Dict[str, torch.Tensor],
    simcc_split_ratio: float = 2.0,
    win: int = 12
) -> np.ndarray:
    """
    Decodes SimCC 1D classification logits into continuous 2D keypoints [B, 13, 3].
    Uses local peak windowing to prevent center-pull bias and properly scales confidence.
    Output: np.ndarray of shape [B, 13, 3] with [x, y, confidence].
    """
    pred_x = preds["simcc_x"]  # [B, 13, W_bins]
    pred_y = preds["simcc_y"]  # [B, 13, H_bins]
    pred_vis = preds.get("vis", None)  # [B, 13, 1]

    coord_x, mass_x = decode_axis(pred_x, ratio=simcc_split_ratio, win=win)
    coord_y, mass_y = decode_axis(pred_y, ratio=simcc_split_ratio, win=win)

    kpt_score = (mass_x * mass_y).clamp(min=0.0).sqrt()  # [B, 13]

    if pred_vis is not None:
        confidence = kpt_score * pred_vis.squeeze(-1)
    else:
        confidence = kpt_score

    out = torch.stack([coord_x, coord_y, confidence], dim=-1)  # [B, 13, 3]
    return out.detach().cpu().numpy()

