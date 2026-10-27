"""
single_stage_rtmpose/loss.py
────────────────────────────
Unified Multi-Task Loss for Single-Stage RTMPose with RoI-Align + SimCC (Apache-2.0):
  1. Quality Focal Loss for 2-class chirality (Left Foot / Right Foot) and negative background suppression.
  2. Complete IoU (CIoU) loss for bounding box localization.
  3. Kullback-Leibler (KL) Divergence loss over continuous 1D Gaussian SimCC bins.
  4. Keypoint visibility Binary Cross-Entropy loss.
  5. In-model chirality / handedness classification loss.
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes Complete-IoU (CIoU) between box1 and box2: [N, 4] in (x1, y1, x2, y2)."""
    x1_1, y1_1, x2_1, y2_1 = box1.unbind(-1)
    x1_2, y1_2, x2_2, y2_2 = box2.unbind(-1)

    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    w1 = (x2_1 - x1_1).clamp(min=eps)
    h1 = (y2_1 - y1_1).clamp(min=eps)
    w2 = (x2_2 - x1_2).clamp(min=eps)
    h2 = (y2_2 - y1_2).clamp(min=eps)

    union_area = w1 * h1 + w2 * h2 - inter_area + eps
    iou = inter_area / union_area

    enclose_x1 = torch.min(x1_1, x1_2)
    enclose_y1 = torch.min(y1_1, y1_2)
    enclose_x2 = torch.max(x2_1, x2_2)
    enclose_y2 = torch.max(y2_1, y2_2)
    cw = (enclose_x2 - enclose_x1).clamp(min=eps)
    ch = (enclose_y2 - enclose_y1).clamp(min=eps)
    c2 = cw ** 2 + ch ** 2 + eps

    b1_cx, b1_cy = (x1_1 + x2_1) / 2.0, (y1_1 + y2_1) / 2.0
    b2_cx, b2_cy = (x1_2 + x2_2) / 2.0, (y1_2 + y2_2) / 2.0
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    v = (4.0 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou


def quality_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25
) -> torch.Tensor:
    """Focal loss for multi-class classification and negative background rejection."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    a_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (a_t * (1.0 - p_t) ** gamma * ce).sum()


class SingleStagePoseLoss(nn.Module):
    """
    Unified multi-task loss for SingleStageRTMPose with RoI-Align + SimCC:
      L_total = loss_cls * 1.0 + loss_box * 2.0 + loss_simcc * 1.0 + loss_vis * 0.5 + loss_hand * 0.5
    """
    def __init__(
        self,
        num_classes: int = 2,
        num_keypoints: int = 13,
        strides: List[int] = [8, 16, 32],
        center_radius: float = 2.0,
        simcc_sigma: float = 6.0,
        loss_cls_weight: float = 1.0,
        loss_box_weight: float = 2.0,
        loss_simcc_weight: float = 1.0,
        loss_vis_weight: float = 0.5,
        loss_hand_weight: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.strides = strides
        self.center_radius = center_radius
        self.simcc_sigma = simcc_sigma

        self.loss_cls_weight = loss_cls_weight
        self.loss_box_weight = loss_box_weight
        self.loss_simcc_weight = loss_simcc_weight
        self.loss_vis_weight = loss_vis_weight
        self.loss_hand_weight = loss_hand_weight

    def forward(
        self,
        preds: Dict[str, Any],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_keypoints: List[torch.Tensor],
        img_size: Tuple[int, int] = (320, 320)
    ) -> Dict[str, torch.Tensor]:
        cls_scores = preds["cls_scores"]
        bbox_preds = preds["bbox_preds"]
        simcc_outputs = preds.get("simcc_outputs", None)
        rois = preds.get("rois", None)
        roi_kpts = preds.get("roi_kpts", None)
        roi_labels = preds.get("roi_labels", None)

        device = cls_scores[0].device
        B = cls_scores[0].shape[0]

        # ─────────────────────────────────────────────────────────────
        # 1. Detection Anchor Flattening & Matching
        # ─────────────────────────────────────────────────────────────
        flat_cls = []
        flat_box = []
        flat_anchors = []
        flat_strides = []

        for stride, cp, bp in zip(self.strides, cls_scores, bbox_preds):
            _, _, h, w = cp.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij"
            )
            anchor_x = (grid_x + 0.5) * stride
            anchor_y = (grid_y + 0.5) * stride
            centers = torch.stack([anchor_x, anchor_y], dim=-1).view(-1, 2)

            flat_cls.append(cp.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            flat_box.append(bp.permute(0, 2, 3, 1).reshape(B, -1, 4))
            flat_anchors.append(centers)
            flat_strides.append(torch.full((h * w,), stride, device=device, dtype=torch.float32))

        flat_cls = torch.cat(flat_cls, dim=1)
        flat_box = torch.cat(flat_box, dim=1)
        flat_anchors = torch.cat(flat_anchors, dim=0)
        flat_strides = torch.cat(flat_strides, dim=0)

        cls_targets = torch.zeros_like(flat_cls)
        pos_box_preds = []
        pos_gt_boxes = []

        for b in range(B):
            b_boxes = gt_boxes[b].to(device)
            b_labels = gt_labels[b].to(device)

            num_gt = b_boxes.shape[0]
            if num_gt == 0:
                continue

            gt_cx = (b_boxes[:, 0] + b_boxes[:, 2]) / 2.0
            gt_cy = (b_boxes[:, 1] + b_boxes[:, 3]) / 2.0

            anchor_cx = flat_anchors[:, 0].unsqueeze(1)
            anchor_cy = flat_anchors[:, 1].unsqueeze(1)
            stride_v = flat_strides.unsqueeze(1)

            dx = (anchor_cx - gt_cx.unsqueeze(0)).abs()
            dy = (anchor_cy - gt_cy.unsqueeze(0)).abs()
            max_dist = self.center_radius * stride_v

            in_center = (dx <= max_dist) & (dy <= max_dist)
            in_box = (
                (anchor_cx >= b_boxes[:, 0].unsqueeze(0)) &
                (anchor_cx <= b_boxes[:, 2].unsqueeze(0)) &
                (anchor_cy >= b_boxes[:, 1].unsqueeze(0)) &
                (anchor_cy <= b_boxes[:, 3].unsqueeze(0))
            )

            candidate_mask = in_center & in_box
            areas = (b_boxes[:, 2] - b_boxes[:, 0]) * (b_boxes[:, 3] - b_boxes[:, 1])
            candidate_areas = torch.where(candidate_mask, areas.unsqueeze(0), torch.tensor(1e9, device=device))
            min_area, best_gt_idx = candidate_areas.min(dim=-1)
            is_pos = min_area < 1e8

            pos_indices = torch.nonzero(is_pos).squeeze(-1)
            if len(pos_indices) == 0:
                continue

            matched_gt = best_gt_idx[pos_indices]
            pos_labels = b_labels[matched_gt]
            cls_targets[b, pos_indices, pos_labels] = 1.0

            p_anc = flat_anchors[pos_indices]
            p_str = flat_strides[pos_indices]
            p_box_raw = flat_box[b, pos_indices] * p_str.unsqueeze(-1)

            pred_x1 = p_anc[:, 0] - p_box_raw[:, 0]
            pred_y1 = p_anc[:, 1] - p_box_raw[:, 1]
            pred_x2 = p_anc[:, 0] + p_box_raw[:, 2]
            pred_y2 = p_anc[:, 1] + p_box_raw[:, 3]
            pred_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

            pos_box_preds.append(pred_xyxy)
            pos_gt_boxes.append(b_boxes[matched_gt])

        # Classification Loss (Quality Focal Loss across all anchors including negative images)
        num_pos = max(1, sum(len(p) for p in pos_box_preds))
        loss_cls = quality_focal_loss(flat_cls, cls_targets) / float(num_pos)

        # Dedicated background floor/carpet suppression loss for pure negative images
        neg_img_mask = torch.tensor([gt_boxes[b].shape[0] == 0 for b in range(B)], device=device)
        if neg_img_mask.any():
            neg_cls_preds = flat_cls[neg_img_mask]
            neg_targets = torch.zeros_like(neg_cls_preds)
            loss_bg = quality_focal_loss(neg_cls_preds, neg_targets) / max(1, neg_cls_preds.numel() // self.num_classes)
            loss_cls = loss_cls + loss_bg * 0.5

        # Bounding Box Loss (CIoU)
        if len(pos_box_preds) > 0:
            all_pred_boxes = torch.cat(pos_box_preds, dim=0)
            all_gt_boxes = torch.cat(pos_gt_boxes, dim=0)
            ciou_vals = bbox_ciou(all_pred_boxes, all_gt_boxes)
            loss_box = (1.0 - ciou_vals).mean()
        else:
            loss_box = torch.tensor(0.0, device=device)

        # ─────────────────────────────────────────────────────────────
        # 2. RoI-Align SimCC 13-Keypoint Loss
        # ─────────────────────────────────────────────────────────────
        loss_simcc = torch.tensor(0.0, device=device)
        loss_vis = torch.tensor(0.0, device=device)
        loss_hand = torch.tensor(0.0, device=device)

        if simcc_outputs is not None and rois is not None and rois.shape[0] > 0:
            pred_x = simcc_outputs["simcc_x"]  # [M, 13, 512]
            pred_y = simcc_outputs["simcc_y"]  # [M, 13, 512]
            vis_logits = simcc_outputs["vis_logits"]  # [M, 13, 1]
            hand_logits = simcc_outputs.get("hand_logits", None)

            M = rois.shape[0]
            w_bins = pred_x.shape[-1]
            h_bins = pred_y.shape[-1]

            bx1 = rois[:, 1].unsqueeze(-1)  # [M, 1]
            by1 = rois[:, 2].unsqueeze(-1)
            bx2 = rois[:, 3].unsqueeze(-1)
            by2 = rois[:, 4].unsqueeze(-1)
            bw = (bx2 - bx1).clamp(min=1.0)
            bh = (by2 - by1).clamp(min=1.0)

            # Keypoints relative to RoI box in [0, 1]
            k_x = roi_kpts[..., 0]  # [M, 13]
            k_y = roi_kpts[..., 1]  # [M, 13]
            k_v = roi_kpts[..., 2]  # [M, 13]

            rel_u = (k_x - bx1) / bw
            rel_v = (k_y - by1) / bh

            # Valid keypoint mask: labeled (kv > 0) and inside the box bounds
            valid_mask = (k_v > 0) & (rel_u >= 0.0) & (rel_u <= 1.0) & (rel_v >= 0.0) & (rel_v <= 1.0)

            if valid_mask.any():
                target_bin_x = (rel_u * (w_bins - 1.0)).clamp(0.0, w_bins - 1.0)
                target_bin_y = (rel_v * (h_bins - 1.0)).clamp(0.0, h_bins - 1.0)

                bins_x = torch.arange(w_bins, dtype=torch.float32, device=device).view(1, 1, -1)
                bins_y = torch.arange(h_bins, dtype=torch.float32, device=device).view(1, 1, -1)

                dist_sq_x = (bins_x - target_bin_x.unsqueeze(-1)) ** 2
                dist_sq_y = (bins_y - target_bin_y.unsqueeze(-1)) ** 2

                target_dist_x = torch.exp(-0.5 * dist_sq_x / (self.simcc_sigma ** 2))
                target_dist_y = torch.exp(-0.5 * dist_sq_y / (self.simcc_sigma ** 2))

                target_dist_x = target_dist_x / target_dist_x.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                target_dist_y = target_dist_y / target_dist_y.sum(dim=-1, keepdim=True).clamp(min=1e-8)

                log_pred_x = F.log_softmax(pred_x, dim=-1)
                log_pred_y = F.log_softmax(pred_y, dim=-1)

                # KL divergence: D_KL(P || Q) = sum(P * (log(P) - log(Q)))
                kl_x = (target_dist_x * (target_dist_x.clamp(min=1e-8).log() - log_pred_x)).sum(dim=-1)
                kl_y = (target_dist_y * (target_dist_y.clamp(min=1e-8).log() - log_pred_y)).sum(dim=-1)

                loss_simcc = (kl_x[valid_mask].mean() + kl_y[valid_mask].mean()) * 0.5

            # Visibility BCE loss
            gt_vis = (k_v > 0).float()
            loss_vis = F.binary_cross_entropy_with_logits(vis_logits.squeeze(-1), gt_vis)

            # Handedness classification loss
            if hand_logits is not None and roi_labels is not None:
                loss_hand = F.binary_cross_entropy_with_logits(hand_logits, roi_labels.float())

        total_loss = (
            self.loss_cls_weight * loss_cls +
            self.loss_box_weight * loss_box +
            self.loss_simcc_weight * loss_simcc +
            self.loss_vis_weight * loss_vis +
            self.loss_hand_weight * loss_hand
        )

        return {
            "total_loss": total_loss,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_simcc": loss_simcc,
            "loss_vis": loss_vis,
            "loss_hand": loss_hand
        }
