"""
single_stage_rtmpose/model.py
─────────────────────────────
Complete Single-Stage RTMPose Model (Apache-2.0).
Combines CSPNeXt Backbone + CSPNeXt-PAFPN Neck + Decoupled Pose Head.
Outputs [B, 45, N] tensor in one forward pass matching production contract:
  - Channels 0..3:  Bounding Box (cx, cy, w, h)
  - Channels 4..5:  Class Probabilities (left_foot, right_foot)
  - Channels 6..44: 13 Keypoints (x, y, conf)
"""

from typing import Dict, List, Tuple, Any, Optional
import torchvision.ops as ops
import torch
import torch.nn as nn

from src.models.qualcomm_experiments.single_stage_rtmpose.backbone import CSPNeXtBackbone, DEFAULT_COCO_CKPT
from src.models.qualcomm_experiments.single_stage_rtmpose.neck import CSPNeXtPAFPN
from src.models.qualcomm_experiments.single_stage_rtmpose.head import SingleStagePoseHead


class SingleStageRTMPose(nn.Module):
    """
    Single-Stage RTMPose-13KP Foot Perception Network (Apache-2.0).
    Predicts Left/Right foot bounding boxes and 13 anatomical keypoints in a single pass.
    """
    def __init__(
        self,
        num_classes: int = 2,
        num_keypoints: int = 13,
        input_size: Tuple[int, int] = (320, 320),
        neck_channels: int = 96,
        strides: List[int] = [8, 16, 32],
        pretrained: bool = True,
        ckpt_path: str = DEFAULT_COCO_CKPT
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.strides = strides

        # In-model normalization buffers (RGB [0, 1] standardized)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # 1. Multi-scale CSPNeXt backbone
        self.backbone = CSPNeXtBackbone(pretrained=pretrained, ckpt_path=ckpt_path)

        # 2. Multi-scale CSPNeXt-PAFPN neck
        self.neck = CSPNeXtPAFPN(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels
        )

        # 3. Decoupled Pose & Detection head
        self.head = SingleStagePoseHead(
            in_channels=neck_channels,
            feat_channels=neck_channels,
            num_classes=num_classes,
            num_keypoints=num_keypoints,
            strides=strides
        )

    def forward(
        self,
        x: torch.Tensor,
        export_mode: bool = False
    ) -> Any:
        # Preprocessing: standardized RGB
        x_norm = (x - self.mean) / self.std

        # Feature extraction
        c2, c3, c4 = self.backbone(x_norm)
        p2, p3, p4 = self.neck((c2, c3, c4))
        head_outputs = self.head((p2, p3, p4))

        if not self.training or export_mode:
            # Decode to [B, 45, N]
            img_size = (x.shape[3], x.shape[2])  # (W, H)
            return self.head.decode_predictions(head_outputs, img_size=img_size)

        return head_outputs


def postprocess_single_stage_pose(
    pred_tensor: torch.Tensor,
    conf_thresh: float = 0.30,
    iou_thresh: float = 0.45,
    max_det: int = 10
) -> List[Dict[str, Any]]:
    """
    Decodes and performs non-maximum suppression on [45, N] tensor:
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

    # Batched class-aware NMS
    keep = ops.batched_nms(boxes_xyxy, scores, classes, iou_thresh)
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

