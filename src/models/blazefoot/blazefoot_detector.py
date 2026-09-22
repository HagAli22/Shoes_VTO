"""
blazefoot_detector.py
─────────────────────
Google BlazeFoot (MediaPipe / BlazePose style) 4-Keypoint Native Perception Network.

Optimized strictly for Path A Contract:
  - Input: [B, 3, 320, 320]
  - Outputs:
      - Box Reg:   [B, N, 4]  (cx, cy, w, h)
      - Class:     [B, N, 2]  (left_foot, right_foot)
      - 4 Coarse:  [B, N, 12] (4 landmarks: toe_tip, heel_back, ball_medial, ball_lateral)
  - Native Output: [B, 18, 1050] (Zero slicing overhead!)
  - Total Parameters: ~574K parameters (FP32: ~2.30 MB, FP16: ~1.15 MB, INT8: ~0.58 MB)
"""

from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blazeblock import BlazeBlock, DoubleBlazeBlock


class BlazeFoot(nn.Module):
    """
    BlazeFoot 4-Keypoint Native Architecture.
    """
    def __init__(self, num_classes: int = 2, num_keypoints: int = 4, num_anchors_per_scale: int = 2):
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

        # ── 8. Multi-Scale Detection Heads (4 Keypoints Only) ─────────────
        self.head_p3 = self._build_head(144)
        self.head_p4 = self._build_head(192)
        self.head_p5 = self._build_head(256)

    def _build_head(self, in_c: int) -> nn.ModuleDict:
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
                nn.Conv2d(in_c, self.num_anchors * (self.num_keypoints * 3), kernel_size=1) # 4 * 3 = 12
            )
        })

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = x.shape[0]

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        p3 = self.stage4(x)  # 20x20
        p4 = self.stage5(p3) # 10x10
        p5 = self.stage6(p4) # 5x5

        features = [p3, p4, p5]
        heads = [self.head_p3, self.head_p4, self.head_p5]

        all_cls, all_box, all_kpt = [], [], []

        for feat, head in zip(features, heads):
            h, w = feat.shape[2], feat.shape[3]

            c = head["cls"](feat).view(b, self.num_anchors, self.num_classes, h, w)
            c = c.permute(0, 3, 4, 1, 2).contiguous().view(b, -1, self.num_classes)
            all_cls.append(c)

            bx = head["box"](feat).view(b, self.num_anchors, 4, h, w)
            bx = bx.permute(0, 3, 4, 1, 2).contiguous().view(b, -1, 4)
            all_box.append(bx)

            kp = head["kpt"](feat).view(b, self.num_anchors, self.num_keypoints * 3, h, w)
            kp = kp.permute(0, 3, 4, 1, 2).contiguous().view(b, -1, self.num_keypoints * 3)
            all_kpt.append(kp)

        cls_out = torch.cat(all_cls, dim=1) # [B, 1050, 2]
        box_out = torch.cat(all_box, dim=1) # [B, 1050, 4]
        kpt_out = torch.cat(all_kpt, dim=1) # [B, 1050, 12]

        return cls_out, box_out, kpt_out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BlazeFootPathAExport(nn.Module):
    """
    Direct Export to Path A [1, 18, 1050]:
      - Rows 0..3: cx, cy, w, h
      - Rows 4..5: left_foot, right_foot class probabilities
      - Rows 6..17: 4 coarse keypoints (toe_tip, heel_back, ball_medial, ball_lateral)
    """
    def __init__(self, model: BlazeFoot):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_logits, box_preds, kpt_preds = self.model(x)

        cls_probs = torch.sigmoid(cls_logits)
        boxes     = torch.sigmoid(box_preds)
        kpts      = torch.sigmoid(kpt_preds)

        # Concat [B, 1050, 18] -> Transpose to [B, 18, 1050]
        out = torch.cat([boxes, cls_probs, kpts], dim=-1)
        out = out.permute(0, 2, 1).contiguous()
        return out
