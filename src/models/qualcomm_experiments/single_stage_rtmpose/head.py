"""
single_stage_rtmpose/head.py
────────────────────────────
Decoupled Single-Stage Detection & 13-Keypoint Pose Head (Apache-2.0).
Simultaneously outputs 2-class chirality (Left/Right), bounding box regression,
and 13 anatomical foot landmarks in a single unified representation:
Channel layout: [cls (2), box (4), keypoints (13 * 3 = 39)] = 45 channels.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.qualcomm_experiments.single_stage_rtmpose.backbone import ConvModule


class SingleStagePoseHead(nn.Module):
    """
    Decoupled anchor-free multi-scale head.
    Predicts:
      1. Classification (Left Foot, Right Foot)
      2. Bounding Box (l, t, r, b) in stride units
      3. 13 Foot Keypoints (dx, dy, conf) in stride units
    """
    def __init__(
        self,
        in_channels: int = 96,
        feat_channels: int = 96,
        num_classes: int = 2,
        num_keypoints: int = 13,
        stacked_convs: int = 2,
        strides: List[int] = [8, 16, 32]
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.strides = strides
        self.kpt_channels = num_keypoints * 3  # [x, y, conf] * 13 = 39

        # Decoupled conv towers per stride level
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.kpt_convs = nn.ModuleList()

        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.kpt_preds = nn.ModuleList()

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

            # Keypoint regression tower
            kpt_tower = []
            for i in range(stacked_convs):
                c_in = in_channels if i == 0 else feat_channels
                kpt_tower.append(ConvModule(c_in, feat_channels, kernel_size=3, padding=1))
            self.kpt_convs.append(nn.Sequential(*kpt_tower))
            self.kpt_preds.append(nn.Conv2d(feat_channels, self.kpt_channels, kernel_size=1))

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
        kpt_preds = []

        for lvl, feat in enumerate(feats):
            # Cls
            cls_feat = self.cls_convs[lvl](feat)
            cls_scores.append(self.cls_preds[lvl](cls_feat))

            # Box
            reg_feat = self.reg_convs[lvl](feat)
            # ReLU enforces non-negative distances (l, t, r, b)
            bbox_preds.append(F.relu(self.reg_preds[lvl](reg_feat)))

            # Keypoints
            kpt_feat = self.kpt_convs[lvl](feat)
            kpt_preds.append(self.kpt_preds[lvl](kpt_feat))

        return {
            "cls_scores": cls_scores,
            "bbox_preds": bbox_preds,
            "kpt_preds": kpt_preds
        }

    def decode_predictions(
        self,
        head_outputs: Dict[str, List[torch.Tensor]],
        img_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Decodes raw multi-scale feature predictions into unified [B, 45, N] tensor
        matching project champion output contract:
          Channels 0..3:  Bounding Box (cx, cy, w, h)
          Channels 4..5:  Class Probabilities (left_foot, right_foot)
          Channels 6..44: 13 Keypoints (x0, y0, conf0, ..., x12, y12, conf12)
        """
        cls_scores = head_outputs["cls_scores"]
        bbox_preds = head_outputs["bbox_preds"]
        kpt_preds = head_outputs["kpt_preds"]

        B = cls_scores[0].shape[0]
        device = cls_scores[0].device

        all_outputs = []

        for stride, cp, bp, kp in zip(self.strides, cls_scores, bbox_preds, kpt_preds):
            _, _, h, w = cp.shape

            # Anchor grid centers (x, y)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij"
            )
            anchor_x = (grid_x + 0.5) * stride
            anchor_y = (grid_y + 0.5) * stride
            # Shape: [1, H*W, 2]
            anchors = torch.stack([anchor_x, anchor_y], dim=-1).view(1, -1, 2)

            # 1. Classification: [B, H*W, 2]
            cls_prob = torch.sigmoid(cp.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))

            # 2. Bounding Box: (l, t, r, b) in pixels -> (cx, cy, w, h)
            dist = bp.permute(0, 2, 3, 1).reshape(B, -1, 4) * stride
            l = dist[..., 0]
            t = dist[..., 1]
            r = dist[..., 2]
            b = dist[..., 3]

            x1 = (anchors[..., 0] - l).clamp(min=0.0, max=img_size[0])
            y1 = (anchors[..., 1] - t).clamp(min=0.0, max=img_size[1])
            x2 = (anchors[..., 0] + r).clamp(min=0.0, max=img_size[0])
            y2 = (anchors[..., 1] + b).clamp(min=0.0, max=img_size[1])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = (x2 - x1).clamp(min=1.0)
            bh = (y2 - y1).clamp(min=1.0)
            box_cxcywh = torch.stack([cx, cy, bw, bh], dim=-1)  # [B, H*W, 4]

            # 3. Keypoints: [B, H*W, 13, 3] -> [B, H*W, 39]
            kp_raw = kp.permute(0, 2, 3, 1).reshape(B, -1, self.num_keypoints, 3)
            # Offset x, y by anchor center
            kpt_x = (anchors[..., 0].unsqueeze(-1) + kp_raw[..., 0] * stride).clamp(min=0.0, max=img_size[0])
            kpt_y = (anchors[..., 1].unsqueeze(-1) + kp_raw[..., 1] * stride).clamp(min=0.0, max=img_size[1])
            kpt_conf = torch.sigmoid(kp_raw[..., 2])
            kpt_decoded = torch.stack([kpt_x, kpt_y, kpt_conf], dim=-1).reshape(B, -1, self.kpt_channels)

            # Concatenate to [B, H*W, 45]: [box (4), cls (2), kpts (39)]
            lvl_pred = torch.cat([box_cxcywh, cls_prob, kpt_decoded], dim=-1)
            all_outputs.append(lvl_pred)

        # Concatenate across all anchor locations: [B, N, 45] -> Permute to [B, 45, N]
        unified_tensor = torch.cat(all_outputs, dim=1).permute(0, 2, 1)
        return unified_tensor

