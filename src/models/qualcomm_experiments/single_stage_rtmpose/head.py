"""
single_stage_rtmpose/head.py
────────────────────────────
Decoupled Single-Stage Detection Head for Single-Stage RTMPose (Apache-2.0).
Predicts:
  1. 2-class chirality (Left Foot, Right Foot) with Quality Focal Loss prior.
  2. Multi-scale bounding box distance regression (l, t, r, b) across strides 8, 16, 32.
Keypoints are handled by RoISimCCHead on dedicated foot RoIs for sub-pixel SimCC accuracy.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.qualcomm_experiments.single_stage_rtmpose.backbone import ConvModule


class SingleStageDetectionHead(nn.Module):
    """
    Decoupled anchor-free multi-scale detection head.
    Predicts:
      1. Classification (Left Foot, Right Foot)
      2. Bounding Box (l, t, r, b) in stride units
    """
    def __init__(
        self,
        in_channels: int = 96,
        feat_channels: int = 96,
        num_classes: int = 2,
        stacked_convs: int = 2,
        strides: List[int] = [8, 16, 32]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides

        # Decoupled conv towers per stride level
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        for _ in strides:
            # Classification tower
            cls_tower = []
            for i in range(stacked_convs):
                c_in = in_channels if i == 0 else feat_channels
                cls_tower.append(ConvModule(c_in, feat_channels, kernel_size=3, padding=1))
            self.cls_convs.append(nn.Sequential(*cls_tower))
            self.cls_preds.append(nn.Conv2d(feat_channels, num_classes, kernel_size=1))

            # Bounding box regression tower
            reg_tower = []
            for i in range(stacked_convs):
                c_in = in_channels if i == 0 else feat_channels
                reg_tower.append(ConvModule(c_in, feat_channels, kernel_size=3, padding=1))
            self.reg_convs.append(nn.Sequential(*reg_tower))
            self.reg_preds.append(nn.Conv2d(feat_channels, 4, kernel_size=1))

        self.init_weights()

    def init_weights(self):
        # Normal initialization for convolutional weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # Classification prior initialization: -4.595 (sigmoid(-4.595) = 0.01)
        # Prevents early background gradient explosion
        for pred in self.cls_preds:
            nn.init.constant_(pred.bias, -math.log((1.0 - 0.01) / 0.01))

    def forward(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Dict[str, List[torch.Tensor]]:
        cls_scores = []
        bbox_preds = []

        for lvl, feat in enumerate(feats):
            # Cls
            cls_feat = self.cls_convs[lvl](feat)
            cls_scores.append(self.cls_preds[lvl](cls_feat))

            # Box: ReLU enforces non-negative distances (l, t, r, b)
            reg_feat = self.reg_convs[lvl](feat)
            bbox_preds.append(F.relu(self.reg_preds[lvl](reg_feat)))

        return {
            "cls_scores": cls_scores,
            "bbox_preds": bbox_preds
        }

    def decode_boxes_and_scores(
        self,
        head_outputs: Dict[str, List[torch.Tensor]],
        img_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes raw multi-scale feature predictions into candidate boxes and class probabilities.
        Returns:
          boxes_xyxy: [B, N, 4] in (x1, y1, x2, y2)
          cls_probs:  [B, N, num_classes]
        """
        cls_scores = head_outputs["cls_scores"]
        bbox_preds = head_outputs["bbox_preds"]

        B = cls_scores[0].shape[0]
        device = cls_scores[0].device

        all_boxes = []
        all_probs = []

        for stride, cp, bp in zip(self.strides, cls_scores, bbox_preds):
            _, _, h, w = cp.shape

            # Anchor grid centers (x, y)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij"
            )
            anchor_x = (grid_x + 0.5) * stride
            anchor_y = (grid_y + 0.5) * stride
            anchors = torch.stack([anchor_x, anchor_y], dim=-1).view(1, -1, 2)  # [1, H*W, 2]

            # 1. Classification probabilities: [B, H*W, num_classes]
            cls_prob = torch.sigmoid(cp.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))

            # 2. Bounding Box: (l, t, r, b) in pixels -> (x1, y1, x2, y2)
            dist = bp.permute(0, 2, 3, 1).reshape(B, -1, 4) * stride
            l = dist[..., 0]
            t = dist[..., 1]
            r = dist[..., 2]
            b = dist[..., 3]

            x1 = (anchors[..., 0] - l).clamp(min=0.0, max=img_size[0])
            y1 = (anchors[..., 1] - t).clamp(min=0.0, max=img_size[1])
            x2 = (anchors[..., 0] + r).clamp(min=0.0, max=img_size[0])
            y2 = (anchors[..., 1] + b).clamp(min=0.0, max=img_size[1])

            boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # [B, H*W, 4]

            all_boxes.append(boxes_xyxy)
            all_probs.append(cls_prob)

        return torch.cat(all_boxes, dim=1), torch.cat(all_probs, dim=1)


# Alias for backwards compatibility
SingleStagePoseHead = SingleStageDetectionHead
