"""
single_stage_rtmpose/model.py
─────────────────────────────
Complete Single-Stage RTMPose Model with RoI-Align + SimCC (Apache-2.0).
Combines CSPNeXt Backbone + CSPNeXt-PAFPN Neck + Anchor-Free Detection Head
+ In-Model RoI-Align Feature Extraction + 13-Keypoint SimCC Head with GAU.

Provides sub-pixel 13-keypoint accuracy and Left/Right chirality in a single
end-to-end forward pass without external crop preprocessing.
"""

from typing import Dict, List, Tuple, Any, Optional
import torchvision.ops as ops
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.qualcomm_experiments.single_stage_rtmpose.backbone import CSPNeXtBackbone, DEFAULT_COCO_CKPT
from src.models.qualcomm_experiments.single_stage_rtmpose.neck import CSPNeXtPAFPN
from src.models.qualcomm_experiments.single_stage_rtmpose.head import SingleStageDetectionHead
from src.models.qualcomm_experiments.single_stage_rtmpose.simcc_roi_head import RoISimCCHead

DEFAULT_SIMCC_CKPT = "checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt"


def expand_boxes(boxes: torch.Tensor, pad_ratio: float = 0.20, img_size: Tuple[int, int] = (320, 320)) -> torch.Tensor:
    """Expands bounding boxes by pad_ratio for full anatomical foot context without clipping keypoints."""
    if boxes.shape[0] == 0:
        return boxes
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    w_pad = bw * (1.0 + pad_ratio)
    h_pad = bh * (1.0 + pad_ratio)
    x1 = (cx - w_pad / 2.0).clamp(min=0.0)
    y1 = (cy - h_pad / 2.0).clamp(min=0.0)
    x2 = (cx + w_pad / 2.0).clamp(max=float(img_size[0]))
    y2 = (cy + h_pad / 2.0).clamp(max=float(img_size[1]))
    return torch.stack([x1, y1, x2, y2], dim=-1)


class SingleStageRTMPose(nn.Module):
    """
    Single-Stage RTMPose-13KP with RoI-Align + SimCC (Apache-2.0).
    Performs full-image detection and high-resolution sub-pixel keypoint estimation
    in a single unified network.
    """
    def __init__(
        self,
        num_classes: int = 2,
        num_keypoints: int = 13,
        input_size: Tuple[int, int] = (320, 320),
        neck_channels: int = 96,
        roi_channels: int = 384,
        strides: List[int] = [8, 16, 32],
        pretrained: bool = True,
        backbone_ckpt: str = DEFAULT_COCO_CKPT,
        simcc_ckpt: str = DEFAULT_SIMCC_CKPT,
        ckpt_path: Optional[str] = None
    ):
        super().__init__()
        if ckpt_path is not None:
            backbone_ckpt = ckpt_path
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.strides = strides
        self.kpt_channels = num_keypoints * 3  # 39 channels

        # In-model normalization buffers (RGB [0, 1] standardized)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 1. Multi-scale CSPNeXt backbone
        self.backbone = CSPNeXtBackbone(pretrained=pretrained, ckpt_path=backbone_ckpt)

        # 2. Multi-scale CSPNeXt-PAFPN neck
        self.neck = CSPNeXtPAFPN(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels
        )

        # 3. Decoupled Detection Head (Cls + Box across strides 8, 16, 32)
        self.det_head = SingleStageDetectionHead(
            in_channels=neck_channels,
            feat_channels=neck_channels,
            num_classes=num_classes,
            strides=strides
        )

        # 4. RoI-Align + SimCC 13-Keypoint Head (extracts from P2 at stride 8)
        self.roi_head = RoISimCCHead(
            in_channels=neck_channels,
            roi_channels=roi_channels,
            num_keypoints=num_keypoints,
            roi_output_size=(8, 8),
            spatial_scale=1.0 / 8.0,
            simcc_split_ratio=2.0,
            input_size=(256, 256),
            pretrained_weights_path=simcc_ckpt if pretrained else None
        )

    def forward(
        self,
        x: torch.Tensor,
        gt_boxes: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        gt_kpts: Optional[List[torch.Tensor]] = None,
        export_mode: bool = False,
        conf_thresh: float = 0.30,
        iou_thresh: float = 0.45,
        max_det: int = 10
    ) -> Any:
        device = x.device
        B = x.shape[0]
        img_w = x.shape[3]
        img_h = x.shape[2]

        # Standardized RGB
        x_norm = (x - self.mean) / self.std

        # 1. Backbone + Neck
        c2, c3, c4 = self.backbone(x_norm)
        p2, p3, p4 = self.neck((c2, c3, c4))

        # 2. Detection Head
        det_outputs = self.det_head((p2, p3, p4))

        # TRAINING MODE: RoI-Align on Ground-Truth foot boxes with RoI Jitter + 20% Context Expansion
        if self.training and not export_mode:
            rois_list = []
            roi_kpts_list = []
            roi_labels_list = []

            if gt_boxes is not None and gt_kpts is not None and gt_labels is not None:
                for b_idx in range(B):
                    b_boxes = gt_boxes[b_idx].to(device)
                    if b_boxes.shape[0] > 0:
                        # 1. RoI Jitter Augmentation: 50% chance of random translation/scale noise
                        # Trains SimCC to be resilient to detector localization shifts
                        if torch.rand(1).item() < 0.50:
                            bw = (b_boxes[:, 2] - b_boxes[:, 0]).clamp(min=1.0)
                            bh = (b_boxes[:, 3] - b_boxes[:, 1]).clamp(min=1.0)
                            dx = (torch.rand_like(bw) * 0.24 - 0.12) * bw
                            dy = (torch.rand_like(bh) * 0.24 - 0.12) * bh
                            dw = (torch.rand_like(bw) * 0.25 - 0.10) * bw
                            dh = (torch.rand_like(bh) * 0.25 - 0.10) * bh
                            j_x1 = (b_boxes[:, 0] + dx - dw * 0.5).clamp(min=0.0)
                            j_y1 = (b_boxes[:, 1] + dy - dh * 0.5).clamp(min=0.0)
                            j_x2 = (b_boxes[:, 2] + dx + dw * 0.5).clamp(max=float(img_w))
                            j_y2 = (b_boxes[:, 3] + dy + dh * 0.5).clamp(max=float(img_h))
                            sampled_boxes = torch.stack([j_x1, j_y1, j_x2, j_y2], dim=-1)
                        else:
                            sampled_boxes = b_boxes

                        # 2. 20% Anatomical Context Expansion (eliminates boundary clipping)
                        pad_boxes = expand_boxes(sampled_boxes, pad_ratio=0.20, img_size=(img_w, img_h))

                        b_idx_col = torch.full((b_boxes.shape[0], 1), b_idx, dtype=torch.float32, device=device)
                        rois_list.append(torch.cat([b_idx_col, pad_boxes], dim=1))
                        roi_kpts_list.append(gt_kpts[b_idx].to(device))
                        roi_labels_list.append(gt_labels[b_idx].to(device))

            if len(rois_list) > 0:
                rois = torch.cat(rois_list, dim=0)
                roi_kpts = torch.cat(roi_kpts_list, dim=0)
                roi_labels = torch.cat(roi_labels_list, dim=0)
            else:
                rois = torch.zeros((0, 5), dtype=torch.float32, device=device)
                roi_kpts = torch.zeros((0, self.num_keypoints, 3), dtype=torch.float32, device=device)
                roi_labels = torch.zeros((0,), dtype=torch.long, device=device)

            simcc_outputs = self.roi_head.forward_rois(p2, rois)

            return {
                "cls_scores": det_outputs["cls_scores"],
                "bbox_preds": det_outputs["bbox_preds"],
                "simcc_outputs": simcc_outputs,
                "rois": rois,
                "roi_kpts": roi_kpts,
                "roi_labels": roi_labels
            }

        # EVALUATION / INFERENCE / EXPORT MODE:
        # 1. Decode all candidate boxes & classification probabilities
        boxes_xyxy, cls_probs = self.det_head.decode_boxes_and_scores(
            det_outputs, img_size=(img_w, img_h)
        )

        # Construct [B, 45, max_det] unified tensor matching project contract
        batch_out_tensors = []

        for b in range(B):
            b_boxes = boxes_xyxy[b]      # [N, 4]
            b_probs = cls_probs[b]       # [N, 2]
            max_s, max_cls = b_probs.max(dim=-1)

            # Filter candidates above confidence threshold
            cand_mask = max_s >= conf_thresh
            if cand_mask.any():
                cand_boxes = b_boxes[cand_mask]
                cand_scores = max_s[cand_mask]
                cand_cls = max_cls[cand_mask]
                cand_probs = b_probs[cand_mask]

                # Class-Agnostic NMS eliminates duplicate boxes on the same foot
                keep = ops.nms(cand_boxes, cand_scores, iou_thresh)
                
                # Centroid distance suppression
                final_keep = []
                centers = (cand_boxes[:, :2] + cand_boxes[:, 2:]) / 2.0
                for idx in keep:
                    c_i = centers[idx]
                    too_close = False
                    for f_idx in final_keep:
                        dist = torch.norm(c_i - centers[f_idx]).item()
                        if dist < 25.0:
                            too_close = True
                            break
                    if not too_close:
                        final_keep.append(idx)
                    if len(final_keep) >= max_det:
                        break

                kept_boxes = cand_boxes[final_keep]
                kept_scores = cand_scores[final_keep]
                kept_cls = cand_cls[final_keep]
                kept_probs = cand_probs[final_keep]

                # RoI-Align on surviving detected boxes with 20% Context Expansion
                K = len(final_keep)
                pad_kept_boxes = expand_boxes(kept_boxes, pad_ratio=0.20, img_size=(img_w, img_h))
                b_idx_col = torch.zeros((K, 1), dtype=torch.float32, device=device)
                rois_b = torch.cat([b_idx_col, pad_kept_boxes], dim=1)

                simcc_out = self.roi_head.forward_rois(p2[b:b+1], rois_b)
                kpts_b = self.roi_head.decode_continuous_expectation(
                    simcc_out["simcc_x"],
                    simcc_out["simcc_y"],
                    simcc_out["vis_logits"],
                    rois_b
                )  # [K, 13, 3]

                # Convert boxes to cxcywh for [45, K] contract
                cx = (kept_boxes[:, 0] + kept_boxes[:, 2]) / 2.0
                cy = (kept_boxes[:, 1] + kept_boxes[:, 3]) / 2.0
                bw = (kept_boxes[:, 2] - kept_boxes[:, 0]).clamp(min=1.0)
                bh = (kept_boxes[:, 3] - kept_boxes[:, 1]).clamp(min=1.0)
                b_cxcywh = torch.stack([cx, cy, bw, bh], dim=-1)  # [K, 4]

                kpts_flat = kpts_b.view(K, self.kpt_channels)  # [K, 39]

                # [K, 45]: [box (4), cls_prob (2), kpts (39)]
                k_det_tensor = torch.cat([b_cxcywh, kept_probs, kpts_flat], dim=-1)

                # Pad to max_det if needed for uniform tensor shape
                if K < max_det:
                    pad = torch.zeros((max_det - K, 45), dtype=torch.float32, device=device)
                    b_det_tensor = torch.cat([k_det_tensor, pad], dim=0)
                else:
                    b_det_tensor = k_det_tensor[:max_det]
            else:
                b_det_tensor = torch.zeros((max_det, 45), dtype=torch.float32, device=device)

            # Permute to [45, max_det]
            batch_out_tensors.append(b_det_tensor.permute(1, 0))

        # Unified tensor: [B, 45, max_det]
        unified_tensor = torch.stack(batch_out_tensors, dim=0)

        if export_mode:
            return unified_tensor

        return unified_tensor


def postprocess_single_stage_pose(
    pred_tensor: torch.Tensor,
    conf_thresh: float = 0.30,
    iou_thresh: float = 0.45,
    max_det: int = 10
) -> List[Dict[str, Any]]:
    """
    Decodes predictions from [45, N] tensor:
      Channels 0..3:  (cx, cy, w, h)
      Channels 4..5:  (left_prob, right_prob)
      Channels 6..44: 13 * [x, y, conf]
    Returns list of detections with:
      bbox: [x1, y1, x2, y2]
      score: float
      class_id: int (0: left_foot, 1: right_foot)
      keypoints: [13, 3] (x, y, conf)
    """
    # pred_tensor: [45, N]
    device = pred_tensor.device
    N = pred_tensor.shape[1]

    # Transpose to [N, 45]
    preds = pred_tensor.permute(1, 0)

    boxes_cxcywh = preds[:, 0:4]
    cls_scores = preds[:, 4:6]
    kpts_flat = preds[:, 6:45]

    max_scores, class_ids = cls_scores.max(dim=-1)
    mask = max_scores >= conf_thresh

    if not mask.any():
        return []

    boxes_cxcywh = boxes_cxcywh[mask]
    scores = max_scores[mask]
    classes = class_ids[mask]
    kpts = kpts_flat[mask].view(-1, 13, 3)

    # Convert cxcywh to xyxy
    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2.0
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2.0
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2.0
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2.0
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    # Class-Agnostic NMS
    keep = ops.nms(boxes_xyxy, scores, iou_thresh)
    if len(keep) > max_det:
        keep = keep[:max_det]

    results = []
    for idx in keep:
        results.append({
            "bbox": boxes_xyxy[idx].cpu().numpy().tolist(),
            "score": float(scores[idx].item()),
            "class_id": int(classes[idx].item()),
            "keypoints": kpts[idx].cpu().numpy()
        })

    return results
