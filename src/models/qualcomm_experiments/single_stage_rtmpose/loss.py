"""
single_stage_rtmpose/loss.py
────────────────────────────
Multi-Task Unified Loss for Single-Stage RTMPose (Apache-2.0):
  1. Quality Focal Loss for 2-class chirality (Left Foot / Right Foot)
  2. Complete / Generalized IoU loss for bounding box localization
  3. Keypoint coordinate regression loss (normalized by stride and bounding box scale)
  4. Keypoint visibility BCE loss
"""

from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# Anatomical sigmas for 13 foot keypoints (matching OKS standards)
# Smaller sigmas for rigid points (sole/heel/toes), larger for fleshy instep
FOOT_OKS_SIGMAS = torch.tensor([
    0.026,  # 0: toe_ground
    0.025,  # 1: heel_back
    0.025,  # 2: heel_ground
    0.035,  # 3: ball_medial
    0.035,  # 4: ball_lateral
    0.045,  # 5: instep_top
    0.040,  # 6: arch_medial
    0.040,  # 7: midfoot_lateral
    0.030,  # 8: malleolus_medial
    0.030,  # 9: malleolus_lateral
    0.025,  # 10: toe_tip
    0.035,  # 11: throat
    0.030   # 12: achilles
], dtype=torch.float32)


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

    # Enclosing box
    enclose_x1 = torch.min(x1_1, x1_2)
    enclose_y1 = torch.min(y1_1, y1_2)
    enclose_x2 = torch.max(x2_1, x2_2)
    enclose_y2 = torch.max(y2_1, y2_2)
    cw = (enclose_x2 - enclose_x1).clamp(min=eps)
    ch = (enclose_y2 - enclose_y1).clamp(min=eps)
    c2 = cw ** 2 + ch ** 2 + eps

    # Center distance
    b1_cx, b1_cy = (x1_1 + x2_1) / 2.0, (y1_1 + y2_1) / 2.0
    b2_cx, b2_cy = (x1_2 + x2_2) / 2.0, (y1_2 + y2_2) / 2.0
    rho2 = (b1_cx - b2_cx) ** 2 + (b1_cy - b2_cy) ** 2

    # Aspect ratio consistency
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
    """Focal loss for multi-class classification and background rejection."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    a_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (a_t * (1.0 - p_t) ** gamma * ce).sum()


class SingleStagePoseLoss(nn.Module):
    """
    Unified multi-task loss for single-stage bounding box and 13-keypoint estimation:
      L_total = loss_cls * 1.0 + loss_box * 2.0 + loss_kpt * 3.0 + loss_vis * 1.0
    """
    def __init__(
        self,
        num_classes: int = 2,
        num_keypoints: int = 13,
        strides: List[int] = [8, 16, 32],
        center_radius: float = 2.0,
        loss_cls_weight: float = 1.0,
        loss_box_weight: float = 2.0,
        loss_kpt_weight: float = 3.0,
        loss_vis_weight: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.strides = strides
        self.center_radius = center_radius
        self.loss_cls_weight = loss_cls_weight
        self.loss_box_weight = loss_box_weight
        self.loss_kpt_weight = loss_kpt_weight
        self.loss_vis_weight = loss_vis_weight
        self.register_buffer("oks_sigmas", FOOT_OKS_SIGMAS.clone())

    def forward(
        self,
        preds: Dict[str, List[torch.Tensor]],
        gt_boxes: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        gt_keypoints: List[torch.Tensor],
        img_size: Tuple[int, int] = (320, 320)
    ) -> Dict[str, torch.Tensor]:
        cls_scores = preds["cls_scores"]
        bbox_preds = preds["bbox_preds"]
        kpt_preds = preds["kpt_preds"]

        device = cls_scores[0].device
        B = cls_scores[0].shape[0]

        flat_cls = []
        flat_box = []
        flat_kpt = []
        flat_anchors = []
        flat_strides = []

        for stride, cp, bp, kp in zip(self.strides, cls_scores, bbox_preds, kpt_preds):
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
            flat_kpt.append(kp.permute(0, 2, 3, 1).reshape(B, -1, self.num_keypoints, 3))
            flat_anchors.append(centers)
            flat_strides.append(torch.full((h * w,), stride, device=device, dtype=torch.float32))

        flat_cls = torch.cat(flat_cls, dim=1)
        flat_box = torch.cat(flat_box, dim=1)
        flat_kpt = torch.cat(flat_kpt, dim=1)
        flat_anchors = torch.cat(flat_anchors, dim=0)
        flat_strides = torch.cat(flat_strides, dim=0)

        cls_targets = torch.zeros_like(flat_cls)
        pos_box_preds = []
        pos_gt_boxes = []
        pos_kpt_preds = []
        pos_gt_kpts = []
        pos_strides = []

        for b in range(B):
            b_boxes = gt_boxes[b].to(device)
            b_labels = gt_labels[b].to(device)
            b_kpts = gt_keypoints[b].to(device)

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

            # Decode predicted positive bounding boxes
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

            # Keypoint predictions
            p_kpt_raw = flat_kpt[b, pos_indices]  # [M, 13, 3]
            pos_kpt_preds.append((p_kpt_raw, p_anc, p_str))
            pos_gt_kpts.append(b_kpts[matched_gt])
            pos_strides.append(p_str)

        num_pos = sum(p.shape[0] for p in pos_box_preds) if pos_box_preds else 0
        loss_cls = quality_focal_loss(flat_cls, cls_targets) / max(num_pos, 1)

        if num_pos == 0:
            zero_val = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "total_loss": loss_cls,
                "loss_cls": loss_cls,
                "loss_box": zero_val,
                "loss_kpt": zero_val,
                "loss_vis": zero_val
            }

        # 1. Bounding Box CIoU Loss
        all_pred_boxes = torch.cat(pos_box_preds, dim=0)
        all_gt_boxes = torch.cat(pos_gt_boxes, dim=0)
        ciou = bbox_ciou(all_pred_boxes, all_gt_boxes)
        loss_box = (1.0 - ciou).mean()

        # 2. Keypoint Coordinate Regression & Visibility Loss
        total_kpt_loss = torch.tensor(0.0, device=device)
        total_vis_loss = torch.tensor(0.0, device=device)
        total_vis_points = 0

        for (p_raw, p_anc, p_str), gt_k in zip(pos_kpt_preds, pos_gt_kpts):
            # p_raw: [M, 13, 3] (dx, dy, conf_logit)
            # gt_k:  [M, 13, 3] (x, y, vis)
            pred_x = p_anc[:, 0].unsqueeze(1) + p_raw[..., 0] * p_str.unsqueeze(1)
            pred_y = p_anc[:, 1].unsqueeze(1) + p_raw[..., 1] * p_str.unsqueeze(1)
            pred_conf_logit = p_raw[..., 2]

            gt_x = gt_k[..., 0]
            gt_y = gt_k[..., 1]
            gt_vis = (gt_k[..., 2] > 0).float()

            # Smooth L1 on coordinates normalized by stride
            diff_x = (pred_x - gt_x) / p_str.unsqueeze(1)
            diff_y = (pred_y - gt_y) / p_str.unsqueeze(1)
            dist_sq = diff_x ** 2 + diff_y ** 2
            huber = torch.where(dist_sq < 1.0, 0.5 * dist_sq, torch.sqrt(dist_sq + 1e-6) - 0.5)

            kpt_loss_batch = (huber * gt_vis).sum()
            total_kpt_loss = total_kpt_loss + kpt_loss_batch

            vis_loss_batch = F.binary_cross_entropy_with_logits(
                pred_conf_logit,
                gt_vis,
                reduction="sum"
            )
            total_vis_loss = total_vis_loss + vis_loss_batch
            total_vis_points += int(gt_vis.sum().item())

        loss_kpt = total_kpt_loss / max(1, total_vis_points)
        loss_vis = total_vis_loss / max(1, num_pos * self.num_keypoints)

        total_loss = (
            self.loss_cls_weight * loss_cls +
            self.loss_box_weight * loss_box +
            self.loss_kpt_weight * loss_kpt +
            self.loss_vis_weight * loss_vis
        )

        return {
            "total_loss": total_loss,
            "loss_cls": loss_cls,
            "loss_box": loss_box,
            "loss_kpt": loss_kpt,
            "loss_vis": loss_vis
        }

