"""
loss.py
───────
Composite Loss functions for Google BlazeFoot 4-Keypoint Training with Anchor-Relative Decoding.

Components:
  1. Sigmoid Focal Loss for Foot Detection / Classification (Left/Right vs Background)
     - Down-weights easy background anchors (gamma=1.5, alpha=0.75) to prevent positive score suppression.
  2. Complete IoU (CIoU) Loss on Anchor-Decoded Bounding Boxes (cx, cy, w, h)
  3. Adaptive Wing Loss on Anchor-Decoded 4 Coarse Keypoint coordinates in pixel space (omega=10.0, eps=2.0)
  4. BCE Loss for Keypoint Visibility
"""

import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import generate_blaze_anchors


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 1.5,
    reduction: str = "none"
) -> torch.Tensor:
    """
    Original Focal Loss formulation for binary/multilabel classification:
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


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
        pred, target: [B, N, 8] (4 keypoints x 2 coordinates = 8) in pixel scale
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
    w1, h1 = pred_boxes[:, 2].clamp(min=1e-4), pred_boxes[:, 3].clamp(min=1e-4)
    w2, h2 = target_boxes[:, 2].clamp(min=1e-4), target_boxes[:, 3].clamp(min=1e-4)
    v = (4.0 / (math.pi**2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (center_dist / c_diagonal) - alpha * v
    loss = 1.0 - ciou
    return loss.mean()


class BlazeFootLoss(nn.Module):
    """
    Unified Loss for BlazeFoot 4-Keypoint multi-task learning with Anchor-Relative Decoding.
    """
    def __init__(
        self,
        lambda_cls: float = 2.0,
        lambda_box: float = 3.0,
        lambda_kpt: float = 0.20,
        focal_alpha: float = 0.75,
        focal_gamma: float = 1.5,
        img_size: int = 320
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_box = lambda_box
        self.lambda_kpt = lambda_kpt
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.img_size = float(img_size)

        self.register_buffer("anchors", generate_blaze_anchors(img_size))
        self.wing_loss = WingLoss(omega=10.0, epsilon=2.0)
        self.vis_bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(
        self,
        cls_preds: torch.Tensor,   # [B, N, 2] logits
        box_preds: torch.Tensor,   # [B, N, 4] raw delta logits
        kpt_preds: torch.Tensor,   # [B, N, 12] raw delta logits
        target_cls: torch.Tensor,  # [B, N, 2]
        target_box: torch.Tensor,  # [B, N, 4] (cx, cy, w, h) in [0, 1]
        target_kpt: torch.Tensor,  # [B, N, 12] (kx, ky, kv) in [0, 1]
        target_mask: torch.Tensor  # [B, N] bool
    ) -> Dict[str, torch.Tensor]:

        pos_mask = target_mask
        num_pos = pos_mask.sum().clamp(min=1.0)

        # ── 1. Classification Loss (Sigmoid Focal Loss) ──────────────────
        focal_loss_all = sigmoid_focal_loss(
            cls_preds,
            target_cls,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            reduction="none"
        )
        loss_cls = focal_loss_all.sum() / num_pos

        # ── 2. Box Loss (CIoU on Anchor-Decoded Boxes) ───────────────────
        if pos_mask.any():
            B, N = cls_preds.shape[:2]
            # Repeat anchors for batch dimension
            batch_anchors = self.anchors.unsqueeze(0).expand(B, N, 4)
            pos_anchors = batch_anchors[pos_mask]

            ax = pos_anchors[:, 0]
            ay = pos_anchors[:, 1]
            aw = pos_anchors[:, 2]
            ah = pos_anchors[:, 3]

            pos_bx_raw = box_preds[pos_mask]
            dec_cx = ax + pos_bx_raw[:, 0] * aw
            dec_cy = ay + pos_bx_raw[:, 1] * ah
            dec_w  = aw * torch.exp(torch.clamp(pos_bx_raw[:, 2], -4.0, 4.0))
            dec_h  = ah * torch.exp(torch.clamp(pos_bx_raw[:, 3], -4.0, 4.0))
            pos_dec_boxes = torch.stack([dec_cx, dec_cy, dec_w, dec_h], dim=-1)

            pos_tgt_boxes = target_box[pos_mask]
            loss_box = bbox_ciou_loss(pos_dec_boxes, pos_tgt_boxes)
        else:
            loss_box = torch.tensor(0.0, device=cls_preds.device)

        # ── 3. Keypoint Loss (Box-Relative Landmark Formulation) ─────────
        if pos_mask.any():
            pos_kpt_raw = kpt_preds[pos_mask] # [M, 12]
            pos_tgt_kpt = target_kpt[pos_mask] # [M, 12]

            # Decode keypoints relative to the predicted box bounds with tanh bounding
            dec_cx = pos_dec_boxes[:, 0]
            dec_cy = pos_dec_boxes[:, 1]
            half_dec_w = pos_dec_boxes[:, 2] * 0.5
            half_dec_h = pos_dec_boxes[:, 3] * 0.5

            dec_kpts_px = []
            for k in range(4):
                dec_kx = (dec_cx + torch.tanh(pos_kpt_raw[:, k * 3]) * half_dec_w) * self.img_size
                dec_ky = (dec_cy + torch.tanh(pos_kpt_raw[:, k * 3 + 1]) * half_dec_h) * self.img_size
                dec_kpts_px.extend([dec_kx, dec_ky])

            pred_xy_px = torch.stack(dec_kpts_px, dim=-1) # [M, 8]
            tgt_xy_px  = pos_tgt_kpt.view(-1, 4, 3)[:, :, :2].reshape(-1, 8) * self.img_size
            tgt_vis    = (pos_tgt_kpt.view(-1, 4, 3)[:, :, 2] > 0).float() # [M, 4]

            # Wing Loss + Box-Scale Normalized L1 (YOLO-Pose style)
            loss_wing = self.wing_loss(pred_xy_px.unsqueeze(0), tgt_xy_px.unsqueeze(0), tgt_vis.unsqueeze(0))
            
            s_box = torch.sqrt(dec_w * dec_h) * self.img_size # [M]
            mask_xy = tgt_vis.repeat_interleave(2, dim=-1)     # [M, 8]
            l1_diff = torch.abs(pred_xy_px - tgt_xy_px) * mask_xy # [M, 8]
            norm_l1 = (l1_diff.sum(dim=-1) / (s_box.clamp(min=10.0) * 8.0)).mean()

            pred_vis_logits = pos_kpt_raw.view(-1, 4, 3)[:, :, 2]
            loss_vis = self.vis_bce(pred_vis_logits, tgt_vis)

            loss_kpt = loss_wing * 0.10 + norm_l1 * 2.0 + loss_vis
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
