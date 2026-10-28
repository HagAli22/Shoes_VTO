"""
nanodet/loss.py
───────────────
Hardware-Optimized Loss formulation for Stage A Foot Detector (Apache-2.0):
  - Generalized / Quality Focal Loss for classification with Hard Negative Mining
  - Complete IoU (CIoU) Loss accounting for overlap, center distance & aspect ratio
  - Distribution Focal Loss (DFL) with expanded reg_max=16 to prevent box truncation
  - Task-Aligned Assigner (TAL) & Adaptive Center Sampling across strides [8, 16, 32]
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.qualcomm_experiments.nanodet.head import Integral


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Standard pairwise or elementwise IoU."""
    x1_1, y1_1, x2_1, y2_1 = box1.unbind(-1)
    x1_2, y1_2, x2_2, y2_2 = box2.unbind(-1)

    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area1 = (x2_1 - x1_1).clamp(min=0.0) * (y2_1 - y1_1).clamp(min=0.0)
    area2 = (x2_2 - x1_2).clamp(min=0.0) * (y2_2 - y1_2).clamp(min=0.0)
    union_area = area1 + area2 - inter_area + eps
    return inter_area / union_area


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Complete IoU (CIoU) Loss.
    Penalizes overlap, normalized center distance, and aspect ratio discrepancy.
    Essential for elongated foot shapes (2.5:1 ratio).
    """
    x1_1, y1_1, x2_1, y2_1 = box1.unbind(-1)
    x1_2, y1_2, x2_2, y2_2 = box2.unbind(-1)

    w1 = (x2_1 - x1_1).clamp(min=eps)
    h1 = (y2_1 - y1_1).clamp(min=eps)
    w2 = (x2_2 - x1_2).clamp(min=eps)
    h2 = (y2_2 - y1_2).clamp(min=eps)

    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    union_area = w1 * h1 + w2 * h2 - inter_area + eps
    iou = inter_area / union_area

    # Smallest enclosing box
    enclose_x1 = torch.min(x1_1, x1_2)
    enclose_y1 = torch.min(y1_1, y1_2)
    enclose_x2 = torch.max(x2_1, x2_2)
    enclose_y2 = torch.max(y2_1, y2_2)
    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0.0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0.0)
    c2 = enclose_w ** 2 + enclose_h ** 2 + eps

    # Center distance squared
    cx1 = (x1_1 + x2_1) / 2.0
    cy1 = (y1_1 + y2_1) / 2.0
    cx2 = (x1_2 + x2_2) / 2.0
    cy2 = (y1_2 + y2_2) / 2.0
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Aspect ratio consistency term
    arctan_gt = torch.atan(w2 / h2)
    arctan_pred = torch.atan(w1 / h1)
    v = (4.0 / (math.pi ** 2)) * (arctan_gt - arctan_pred) ** 2
    with torch.no_grad():
        alpha = v / (1.0 - iou + v).clamp(min=eps)

    ciou = iou - (rho2 / c2 + alpha * v)
    return ciou


def bbox_giou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x1_1, y1_1, x2_1, y2_1 = box1.unbind(-1)
    x1_2, y1_2, x2_2, y2_2 = box2.unbind(-1)

    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)
    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h

    area1 = (x2_1 - x1_1).clamp(min=0.0) * (y2_1 - y1_1).clamp(min=0.0)
    area2 = (x2_2 - x1_2).clamp(min=0.0) * (y2_2 - y1_2).clamp(min=0.0)
    union_area = area1 + area2 - inter_area + eps
    iou = inter_area / union_area

    enclose_x1 = torch.min(x1_1, x1_2)
    enclose_y1 = torch.min(y1_1, y1_2)
    enclose_x2 = torch.max(x2_1, x2_2)
    enclose_y2 = torch.max(y2_1, y2_2)
    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0.0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0.0)
    enclose_area = enclose_w * enclose_h + eps

    giou = iou - (enclose_area - union_area) / enclose_area
    return giou


def sigmoid_focal(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25
) -> torch.Tensor:
    """
    Sigmoid Focal Loss / Quality Focal Loss formulation.
    Balances positive/negative samples and hard/easy examples across multi-scale anchors.
    """
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    a_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (a_t * (1.0 - p_t) ** gamma * ce).sum()


def distribution_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reg_max: int = 16
) -> torch.Tensor:
    target = target.clamp(min=0.0, max=float(reg_max))
    left = target.long()
    right = (left + 1).clamp(max=reg_max)

    weight_left = right.float() - target
    weight_right = target - left.float()

    log_prob = F.log_softmax(pred, dim=-1)
    loss_left = F.nll_loss(log_prob, left, reduction="none") * weight_left
    loss_right = F.nll_loss(log_prob, right, reduction="none") * weight_right

    return loss_left + loss_right


class NanoDetPlusLoss(nn.Module):
    """
    Composite Loss for Stage A NanoDet-Plus Foot Detector:
      L_total = loss_cls + loss_bbox * bbox_weight + loss_dfl * dfl_weight
    Features:
      - CIoU loss option (aspect-ratio consistency)
      - Task-Aligned Assigner (TAL) option
      - Dedicated Hard Negative Background Penalization
    """
    def __init__(
        self,
        num_classes: int = 1,
        reg_max: int = 16,
        strides: List[int] = [8, 16, 32],
        center_radius: float = 2.0,
        loss_cls_weight: float = 1.0,
        loss_bbox_weight: float = 2.0,
        loss_dfl_weight: float = 0.25,
        loss_bbox_type: str = "ciou",
        assigner: str = "tal",
        neg_penalty_weight: float = 1.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.strides = strides
        self.center_radius = center_radius
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.loss_dfl_weight = loss_dfl_weight
        self.loss_bbox_type = loss_bbox_type
        self.assigner = assigner
        self.neg_penalty_weight = neg_penalty_weight
        self.integral = Integral(reg_max=reg_max)

    def forward(
        self,
        preds: Dict[str, Any],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        img_size: Tuple[int, int] = (256, 256)
    ) -> Dict[str, torch.Tensor]:
        cls_preds = preds["cls_preds"]
        reg_preds = preds["reg_preds"]
        device = cls_preds[0].device
        B = cls_preds[0].shape[0]

        flat_cls = []
        flat_reg = []
        flat_anchors = []
        flat_strides = []

        for stride, cp, rp in zip(self.strides, cls_preds, reg_preds):
            _, _, h, w = cp.shape
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij"
            )
            cx = (grid_x + 0.5) * stride
            cy = (grid_y + 0.5) * stride
            centers = torch.stack([cx, cy], dim=-1).view(-1, 2)

            flat_cls.append(cp.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            flat_reg.append(rp.permute(0, 2, 3, 1).reshape(B, -1, 4 * (self.reg_max + 1)))
            flat_anchors.append(centers)
            flat_strides.append(torch.full((h * w,), stride, device=device, dtype=torch.float32))

        flat_cls = torch.cat(flat_cls, dim=1)
        flat_reg = torch.cat(flat_reg, dim=1)
        flat_anchors = torch.cat(flat_anchors, dim=0)
        flat_strides = torch.cat(flat_strides, dim=0)

        pos_reg_preds = []
        pos_gt_boxes = []
        pos_anchors = []
        pos_strides = []

        cls_targets = torch.zeros_like(flat_cls)

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

            # In-box candidate condition
            in_box = (
                (anchor_cx >= b_boxes[:, 0].unsqueeze(0)) &
                (anchor_cx <= b_boxes[:, 2].unsqueeze(0)) &
                (anchor_cy >= b_boxes[:, 1].unsqueeze(0)) &
                (anchor_cy <= b_boxes[:, 3].unsqueeze(0))
            )

            # Center distance condition with adaptive margin
            dx = (anchor_cx - gt_cx.unsqueeze(0)).abs()
            dy = (anchor_cy - gt_cy.unsqueeze(0)).abs()
            max_dist = self.center_radius * stride_v
            in_center = (dx <= max_dist) & (dy <= max_dist)

            # Reach check: reg_max * stride covers maximum dimension
            box_max_dim = torch.max(b_boxes[:, 2] - b_boxes[:, 0], b_boxes[:, 3] - b_boxes[:, 1]).unsqueeze(0)
            max_reach = (self.reg_max + 1) * stride_v * 1.5
            scale_valid = (box_max_dim <= max_reach)

            candidate_mask = (in_center | in_box) & scale_valid

            if self.assigner == "tal":
                # Task-Aligned Assigner: t = score^alpha * IoU^beta
                with torch.no_grad():
                    # Decode predicted boxes for candidate anchors
                    pred_dist_b = self.integral(flat_reg[b]) * flat_strides.unsqueeze(-1)
                    px1 = flat_anchors[:, 0] - pred_dist_b[:, 0]
                    py1 = flat_anchors[:, 1] - pred_dist_b[:, 1]
                    px2 = flat_anchors[:, 0] + pred_dist_b[:, 2]
                    py2 = flat_anchors[:, 1] + pred_dist_b[:, 3]
                    p_boxes = torch.stack([px1, py1, px2, py2], dim=-1)

                    scores_b = flat_cls[b].sigmoid()  # [N_anchors, num_classes]

                    align_matrix = torch.zeros((len(flat_anchors), num_gt), device=device)
                    iou_matrix = torch.zeros((len(flat_anchors), num_gt), device=device)

                    for g in range(num_gt):
                        cands = candidate_mask[:, g]
                        if not cands.any():
                            # Fallback to nearest center
                            cands = in_box[:, g] | in_center[:, g]
                            if not cands.any():
                                continue

                        cand_idx = torch.nonzero(cands).squeeze(-1)
                        gt_b_expand = b_boxes[g:g+1].expand(len(cand_idx), 4)
                        ious = bbox_iou(p_boxes[cand_idx], gt_b_expand)
                        iou_matrix[cand_idx, g] = ious

                        cls_id = int(b_labels[g].item()) if self.num_classes > 1 else 0
                        cls_id = min(cls_id, self.num_classes - 1)
                        s = scores_b[cand_idx, cls_id]
                        t = (s.clamp(min=1e-6) ** 0.5) * (ious.clamp(min=1e-6) ** 6.0)
                        align_matrix[cand_idx, g] = t

                    # Top-K candidates per GT
                    topk = min(9, len(flat_anchors))
                    pos_mask = torch.zeros((len(flat_anchors),), dtype=torch.bool, device=device)
                    best_gt_idx = torch.zeros((len(flat_anchors),), dtype=torch.long, device=device)

                    for g in range(num_gt):
                        vals, indices = align_matrix[:, g].topk(topk)
                        valid_top = vals > 0
                        chosen = indices[valid_top]
                        if len(chosen) > 0:
                            pos_mask[chosen] = True
                            best_gt_idx[chosen] = g

                pos_indices = torch.nonzero(pos_mask).squeeze(-1)
            else:
                # Fast center sampling
                areas = (b_boxes[:, 2] - b_boxes[:, 0]) * (b_boxes[:, 3] - b_boxes[:, 1])
                candidate_areas = torch.where(candidate_mask, areas.unsqueeze(0), torch.tensor(1e9, device=device))
                min_area, best_gt = candidate_areas.min(dim=-1)
                is_pos = min_area < 1e8
                pos_indices = torch.nonzero(is_pos).squeeze(-1)
                best_gt_idx = best_gt

            if len(pos_indices) == 0:
                continue

            matched_gt = best_gt_idx[pos_indices]
            pos_labels = b_labels[matched_gt]
            if self.num_classes == 1:
                pos_labels = torch.zeros_like(pos_labels)

            cls_targets[b, pos_indices, pos_labels] = 1.0

            pos_reg_preds.append(flat_reg[b, pos_indices])
            pos_gt_boxes.append(b_boxes[matched_gt])
            pos_anchors.append(flat_anchors[pos_indices])
            pos_strides.append(flat_strides[pos_indices])

        num_pos = sum(p.shape[0] for p in pos_reg_preds) if pos_reg_preds else 0

        # Dedicated background negative penalization
        if num_pos == 0:
            # Negative image / negative batch: penalize all false positive activations
            neg_loss = sigmoid_focal(flat_cls, cls_targets, gamma=2.0, alpha=0.25)
            loss_cls = (neg_loss / max(1, flat_cls.shape[0] * flat_cls.shape[1])) * 50.0 * self.neg_penalty_weight
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "total_loss": loss_cls,
                "loss_cls": loss_cls,
                "loss_bbox": zero_loss,
                "loss_giou": zero_loss,
                "loss_dfl": zero_loss
            }


        # Sigmoid Focal Loss normalized by positive anchors
        loss_cls = sigmoid_focal(flat_cls, cls_targets, gamma=2.0, alpha=0.25) / max(num_pos, 1)

        all_pos_reg = torch.cat(pos_reg_preds, dim=0)
        all_pos_gt = torch.cat(pos_gt_boxes, dim=0)
        all_pos_anchors = torch.cat(pos_anchors, dim=0)
        all_pos_strides = torch.cat(pos_strides, dim=0)

        pred_dist = self.integral(all_pos_reg) * all_pos_strides.unsqueeze(-1)
        pred_x1 = all_pos_anchors[:, 0] - pred_dist[:, 0]
        pred_y1 = all_pos_anchors[:, 1] - pred_dist[:, 1]
        pred_x2 = all_pos_anchors[:, 0] + pred_dist[:, 2]
        pred_y2 = all_pos_anchors[:, 1] + pred_dist[:, 3]
        pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

        if self.loss_bbox_type == "ciou":
            ciou = bbox_ciou(pred_boxes, all_pos_gt)
            loss_bbox = (1.0 - ciou).mean()
        else:
            giou = bbox_giou(pred_boxes, all_pos_gt)
            loss_bbox = (1.0 - giou).mean()

        gt_l = (all_pos_anchors[:, 0] - all_pos_gt[:, 0]) / all_pos_strides
        gt_t = (all_pos_anchors[:, 1] - all_pos_gt[:, 1]) / all_pos_strides
        gt_r = (all_pos_gt[:, 2] - all_pos_anchors[:, 0]) / all_pos_strides
        gt_b = (all_pos_gt[:, 3] - all_pos_anchors[:, 1]) / all_pos_strides
        target_dist = torch.stack([gt_l, gt_t, gt_r, gt_b], dim=-1)

        reg_reshaped = all_pos_reg.view(num_pos * 4, self.reg_max + 1)
        target_reshaped = target_dist.view(num_pos * 4)
        loss_dfl = distribution_focal_loss(reg_reshaped, target_reshaped, reg_max=self.reg_max).mean()

        total_loss = (
            self.loss_cls_weight * loss_cls +
            self.loss_bbox_weight * loss_bbox +
            self.loss_dfl_weight * loss_dfl
        )

        return {
            "total_loss": total_loss,
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_bbox,
            "loss_dfl": loss_dfl
        }

