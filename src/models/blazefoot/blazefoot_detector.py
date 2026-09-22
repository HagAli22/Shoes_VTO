"""
blazefoot_detector.py
─────────────────────
Google BlazeFoot (MediaPipe / BlazePose style) Foot Detection & Keypoint Network.

Key specifications:
  - Input: [B, 3, 320, 320]
  - Architecture: Lightweight 5x5 Depthwise Separable Convolutions + Multi-Scale Anchor Heads
  - Feature Pyramid: P3 (20x20), P4 (10x10), P5 (5x5)
  - Outputs:
      - Bounding Boxes: [B, N, 4] (cx, cy, w, h)
      - Classification: [B, N, 2] (left_foot, right_foot)
      - 16 Keypoints:   [B, N, 48] (x, y, visibility for 16 landmarks)
  - Total Parameters: ~500K parameters (FP32 size: ~2.1 MB, FP16 size: ~1.05 MB)
"""

import math
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blazeblock import BlazeBlock, DoubleBlazeBlock


class BlazeFoot(nn.Module):
    """
    BlazeFoot full detector architecture.
    """
    def __init__(self, num_classes: int = 2, num_keypoints: int = 16, num_anchors_per_scale: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.num_anchors = num_anchors_per_scale

        # ── 1. Stem: Conv 5x5, s=2 (320x320 -> 160x160) ───────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(24),
            nn.PReLU(24)
        )

        # ── 2. Stage 1 (160x160) ──────────────────────────────────────────
        self.stage1 = nn.Sequential(
            BlazeBlock(24, 24, stride=1),
            BlazeBlock(24, 24, stride=1)
        )

        # ── 3. Stage 2 (160x160 -> 80x80) ─────────────────────────────────
        self.stage2 = nn.Sequential(
            BlazeBlock(24, 48, stride=2),
            BlazeBlock(48, 48, stride=1),
            BlazeBlock(48, 48, stride=1)
        )

        # ── 4. Stage 3 (80x80 -> 40x40) ───────────────────────────────────
        self.stage3 = nn.Sequential(
            DoubleBlazeBlock(48, 96, stride=2),
            DoubleBlazeBlock(96, 96, stride=1),
            DoubleBlazeBlock(96, 96, stride=1)
        )

        # ── 5. Stage 4 / P3 Scale (40x40 -> 20x20) ────────────────────────
        self.stage4 = nn.Sequential(
            DoubleBlazeBlock(96, 144, stride=2),
            DoubleBlazeBlock(144, 144, stride=1),
            DoubleBlazeBlock(144, 144, stride=1)
        )

        # ── 6. Stage 5 / P4 Scale (20x20 -> 10x10) ────────────────────────
        self.stage5 = nn.Sequential(
            DoubleBlazeBlock(144, 192, stride=2),
            DoubleBlazeBlock(192, 192, stride=1),
            DoubleBlazeBlock(192, 192, stride=1)
        )

        # ── 7. Stage 6 / P5 Scale (10x10 -> 5x5) ──────────────────────────
        self.stage6 = nn.Sequential(
            DoubleBlazeBlock(192, 256, stride=2),
            DoubleBlazeBlock(256, 256, stride=1)
        )

        # ── 8. Multi-Scale Detection Heads ────────────────────────────────
        # Channels: P3=144, P4=192, P5=256
        self.head_p3 = self._build_head(144)
        self.head_p4 = self._build_head(192)
        self.head_p5 = self._build_head(256)

    def _build_head(self, in_c: int) -> nn.ModuleDict:
        """Constructs classification, bounding box, and keypoint prediction subnets."""
        return nn.ModuleDict({
            "cls": nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=5, padding=2, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.PReLU(in_c),
                nn.Conv2d(in_c, self.num_anchors * self.num_classes, kernel_size=1)
            ),
            "box": nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=5, padding=2, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.PReLU(in_c),
                nn.Conv2d(in_c, self.num_anchors * 4, kernel_size=1)
            ),
            "kpt": nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=5, padding=2, groups=in_c, bias=False),
                nn.BatchNorm2d(in_c),
                nn.PReLU(in_c),
                nn.Conv2d(in_c, self.num_anchors * (self.num_keypoints * 3), kernel_size=1)
            )
        })

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns:
            cls_preds: [B, Total_Anchors, num_classes] (logits)
            box_preds: [B, Total_Anchors, 4] (cx, cy, w, h offsets)
            kpt_preds: [B, Total_Anchors, num_keypoints * 3] (kx, ky, kv)
        """
        b = x.shape[0]

        # Backbone forward
        x = self.stem(x)     # 160x160
        x = self.stage1(x)   # 160x160
        x = self.stage2(x)   # 80x80
        x = self.stage3(x)   # 40x40

        p3 = self.stage4(x)  # 20x20
        p4 = self.stage5(p3) # 10x10
        p5 = self.stage6(p4) # 5x5

        features = [p3, p4, p5]
        heads = [self.head_p3, self.head_p4, self.head_p5]

        all_cls, all_box, all_kpt = [], [], []

        for feat, head in zip(features, heads):
            h, w = feat.shape[2], feat.shape[3]

            # Cls: [B, num_anchors * num_classes, H, W] -> [B, H*W*num_anchors, num_classes]
            c = head["cls"](feat).view(b, self.num_anchors, self.num_classes, h, w)
            c = c.permute(0, 3, 4, 1, 2).contiguous().view(b, -1, self.num_classes)
            all_cls.append(c)

            # Box: [B, num_anchors * 4, H, W] -> [B, H*W*num_anchors, 4]
            bx = head["box"](feat).view(b, self.num_anchors, 4, h, w)
            bx = bx.permute(0, 3, 4, 1, 2).contiguous().view(b, -1, 4)
            all_box.append(bx)

            # Kpt: [B, num_anchors * (KP*3), H, W] -> [B, H*W*num_anchors, KP*3]
            kp = head["kpt"](feat).view(b, self.num_anchors, self.num_keypoints * 3, h, w)
            kp = kp.permute(0, 3, 4, 1, 2).contiguous().view(b, -1, self.num_keypoints * 3)
            all_kpt.append(kp)

        cls_out = torch.cat(all_cls, dim=1)
        box_out = torch.cat(all_box, dim=1)
        kpt_out = torch.cat(all_kpt, dim=1)

        return cls_out, box_out, kpt_out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BlazeFootPathAExport(nn.Module):
    """
    ONNX Export wrapper formatting BlazeFoot output strictly to Path A [1, 18, Total_Anchors]:
      - Rows 0..3: cx, cy, w, h
      - Rows 4..5: class scores (Sigmoid probabilities)
      - Rows 6..17: 4 coarse keypoints (toe_tip: 0, heel_back: 2, ball_medial: 4, ball_lateral: 5)
    """
    def __init__(self, model: BlazeFoot):
        super().__init__()
        self.model = model
        # Indices of the 4 coarse keypoints (0, 2, 4, 5) * 3 channels
        self.kpt_select_indices = [
            0*3, 0*3+1, 0*3+2,   # kp0: toe_tip
            2*3, 2*3+1, 2*3+2,   # kp2: heel_back
            4*3, 4*3+1, 4*3+2,   # kp4: ball_medial
            5*3, 5*3+1, 5*3+2    # kp5: ball_lateral
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_logits, box_preds, kpt_preds = self.model(x)

        # Apply sigmoid to class scores and keypoint visibilities
        cls_probs = torch.sigmoid(cls_logits)
        
        # Bounding box is sigmoid normalized
        boxes = torch.sigmoid(box_preds)

        # Extract 4 coarse keypoints
        kpts = torch.sigmoid(kpt_preds)
        kpts_4 = kpts[:, :, self.kpt_select_indices]

        # Concatenate into [B, Total_Anchors, 18] -> Transpose to [B, 18, Total_Anchors]
        out = torch.cat([boxes, cls_probs, kpts_4], dim=-1)
        out = out.permute(0, 2, 1).contiguous()
        return out

