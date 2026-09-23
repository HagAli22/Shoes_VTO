"""
loss.py
───────
Composite Loss functions for Google BlazeFoot 4-Keypoint Training.

Components:
  1. Adaptive Wing Loss on 4 Coarse Keypoint coordinates (8 coords: 4 * 2)
  2. Complete IoU (CIoU) Loss for Bounding Boxes (4 coords: cx, cy, w, h)
  3. Binary Cross Entropy (BCE) for Foot Classification (left_foot, right_foot)
"""

import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class WingLoss(nn.Module):
    """
    Wing Loss for robust 4-keypoint regression:
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
        pred, target: [B, N, 8] (4 keypoints x 2 coordinates = 8)
        mask: [B, N, 4] (visibility of the 4 keypoints)
        """
        diff = torch.abs(pred - target)
        small_mask = diff < self.omega

        loss_small = self.omega * torch.log(1.0 + diff / self.epsilon)
        loss_large = diff - self.c
        loss = torch.where(small_mask, loss_small, loss_large)

        # Duplicate mask for x and y: [B, N, 4] -> [B, N, 8]
        mask_xy = mask.repeat_interleave(2, dim=-1)
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
    Unified Loss for BlazeFoot 4-Keypoint multi-task learning.
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
        cls_preds: torch.Tensor,   # [B, N, 2]
        box_preds: torch.Tensor,   # [B, N, 4]
        kpt_preds: torch.Tensor,   # [B, N, 12] (4 keypoints * 3)
        target_cls: torch.Tensor,  # [B, N, 2]
        target_box: torch.Tensor,  # [B, N, 4]
        target_kpt: torch.Tensor,  # [B, N, 12]
        target_mask: torch.Tensor  # [B, N] bool
    ) -> Dict[str, torch.Tensor]:

        pos_mask = target_mask
        num_pos = pos_mask.sum().clamp(min=1.0)

        # ── 1. Classification Loss (BCE) ──────────────────────────────────
        cls_loss_all = self.cls_bce(cls_preds, target_cls)
        loss_cls = cls_loss_all.sum() / num_pos

        # ── 2. Box Loss (CIoU) ───────────────────────────────────────────
        pred_boxes_sig = torch.sigmoid(box_preds)
        if pos_mask.any():
            pos_pred_boxes = pred_boxes_sig[pos_mask]
            pos_tgt_boxes  = target_box[pos_mask]
            loss_box = bbox_ciou_loss(pos_pred_boxes, pos_tgt_boxes)
        else:
            loss_box = torch.tensor(0.0, device=cls_preds.device)

        # ── 3. 4-Keypoint Loss (Wing Loss + Visibility BCE) ──────────────
        if pos_mask.any():
            pos_kpt_pred = torch.sigmoid(kpt_preds[pos_mask]) # [M, 12]
            pos_kpt_tgt  = target_kpt[pos_mask]               # [M, 12]

            # 4 Keypoints -> 8 (x,y) coords + 4 visibilities
            pred_xy = pos_kpt_pred.view(-1, 4, 3)[:, :, :2].reshape(-1, 8)
            tgt_xy  = pos_kpt_tgt.view(-1, 4, 3)[:, :, :2].reshape(-1, 8)
            tgt_vis = (pos_kpt_tgt.view(-1, 4, 3)[:, :, 2] > 0).float() # [M, 4]

            loss_kpt_coords = self.wing_loss(pred_xy.unsqueeze(0), tgt_xy.unsqueeze(0), tgt_vis.unsqueeze(0))

            pred_vis_logits = kpt_preds[pos_mask].view(-1, 4, 3)[:, :, 2]
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
