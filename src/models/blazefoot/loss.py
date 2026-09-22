"""
loss.py
───────
Composite Loss functions for Google BlazeFoot Training.

Components:
  1. Wing Loss (Adaptive Wing Loss for Keypoint coordinates)
  2. Complete IoU (CIoU) / Smooth L1 Loss for Bounding Boxes
  3. Binary Cross Entropy / Focal Loss for Classification & Visibility
"""

import math
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class WingLoss(nn.Module):
    """
    Wing Loss for robust keypoint regression:
      Provides non-linear logarithmic gradient for small errors (sub-pixel refinement),
      and linear L1 gradient for large errors (outlier resistance).
    """
    def __init__(self, omega: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.c = omega - omega * math.log(1.0 + omega / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        pred, target: [B, N, 32] (x, y coordinates only)
        mask: [B, N, 16] (visibility > 0)
        """
        diff = torch.abs(pred - target) # [B, N, 32]
        small_mask = diff < self.omega

        loss_small = self.omega * torch.log(1.0 + diff / self.epsilon)
        loss_large = diff - self.c
        loss = torch.where(small_mask, loss_small, loss_large)

        # Duplicate mask for x and y: [B, N, 16] -> [B, N, 32]
        mask_xy = mask.repeat_interleave(2, dim=-1) # [B, N, 32]
        valid_loss = loss * mask_xy

        denom = mask_xy.sum().clamp(min=1.0)
        return valid_loss.sum() / denom


def bbox_ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Complete IoU (CIoU) Loss between predicted and target boxes in [cx, cy, w, h] format.
    pred_boxes, target_boxes: [M, 4] where M is number of positive anchors.
    """
    if pred_boxes.numel() == 0:
        return torch.tensor(0.0, device=pred_boxes.device)

    # Convert cx, cy, w, h to x1, y1, x2, y2
    b1_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    b1_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    b1_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    b1_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    b2_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    b2_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    b2_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    b2_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection area
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area + eps
    iou = inter_area / union_area

    # Enclosing box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c_diagonal = (c_x2 - c_x1)**2 + (c_y2 - c_y1)**2 + eps

    # Center distance
    center_dist = (pred_boxes[:, 0] - target_boxes[:, 0])**2 + (pred_boxes[:, 1] - target_boxes[:, 1])**2

    # Aspect ratio term v and alpha
    w1, h1 = pred_boxes[:, 2], pred_boxes[:, 3]
    w2, h2 = target_boxes[:, 2], target_boxes[:, 3]
    v = (4.0 / (math.pi**2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (center_dist / c_diagonal) - alpha * v
    loss = 1.0 - ciou
    return loss.mean()


class BlazeFootLoss(nn.Module):
    """
    Unified Loss for BlazeFoot multi-task learning.
    """
    def __init__(self, lambda_cls: float = 1.0, lambda_box: float = 2.5, lambda_kpt: float = 4.0):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_kpt = lambda_kpt

        self.cls_bce = nn.BCEWithLogitsLoss(reduction='none')
        self.wing_loss = WingLoss(omega=10.0, epsilon=2.0)
        self.vis_bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
        self,
        cls_preds: torch.Tensor, # [B, N, 2]
        box_preds: torch.Tensor, # [B, N, 4]
        kpt_preds: torch.Tensor, # [B, N, 48]
        target_cls: torch.Tensor,# [B, N, 2]
        target_box: torch.Tensor,# [B, N, 4]
        target_kpt: torch.Tensor,# [B, N, 48]
        target_mask: torch.Tensor# [B, N] bool
    ) -> Dict[str, torch.Tensor]:

        pos_mask = target_mask # [B, N]
        num_pos = pos_mask.sum().clamp(min=1.0)

        # ── 1. Classification Loss (BCE) ──────────────────────────────────
        cls_loss_all = self.cls_bce(cls_preds, target_cls) # [B, N, 2]
        loss_cls = cls_loss_all.sum() / num_pos

        # ── 2. Box Regression Loss (CIoU on positive anchors) ─────────────
        pred_boxes_sig = torch.sigmoid(box_preds) # [B, N, 4]
        if pos_mask.any():
            pos_pred_boxes = pred_boxes_sig[pos_mask]
            pos_tgt_boxes  = target_box[pos_mask]
            loss_box = bbox_ciou_loss(pos_pred_boxes, pos_tgt_boxes)
        else:
            loss_box = torch.tensor(0.0, device=cls_preds.device)

        # ── 3. Keypoint Regression Loss (Wing Loss + Visibility BCE) ──────
        if pos_mask.any():
            pos_kpt_pred = torch.sigmoid(kpt_preds[pos_mask]) # [M, 48]
            pos_kpt_tgt  = target_kpt[pos_mask]               # [M, 48]

            # Separate (x, y) coordinates from visibility (v)
            # 16 landmarks -> 32 coords + 16 visibilities
            pred_xy = pos_kpt_pred.view(-1, 16, 3)[:, :, :2].reshape(-1, 32)
            tgt_xy  = pos_kpt_tgt.view(-1, 16, 3)[:, :, :2].reshape(-1, 32)
            tgt_vis = (pos_kpt_tgt.view(-1, 16, 3)[:, :, 2] > 0).float() # [M, 16]

            loss_kpt_coords = self.wing_loss(pred_xy.unsqueeze(0), tgt_xy.unsqueeze(0), tgt_vis.unsqueeze(0))

            pred_vis_logits = kpt_preds[pos_mask].view(-1, 16, 3)[:, :, 2]
            loss_vis = self.vis_bce(pred_vis_logits, tgt_vis).mean()

            loss_kpt = loss_kpt_coords + 0.5 * loss_vis
        else:
            loss_kpt = torch.tensor(0.0, device=cls_preds.device)

        total_loss = (
            self.lambda_cls * loss_cls +
            self.lambda_box * loss_box +
            self.lambda_kpt * loss_kpt
        )

        return {
            "total_loss": total_loss,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_kpt": loss_kpt
        }
