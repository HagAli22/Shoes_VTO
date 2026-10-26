"""
rtmpose/simcc_head.py
─────────────────────
SimCC (Simple Coordinate Classification) Head for RTMPose-13KP Foot Pose.
Ref: "RTMPose: Real-Time Multi-Person Pose Estimation based on MMPose" (Jiang et al.)

Transforms 2D keypoint regression into two independent 1D coordinate classification
problems with sub-pixel resolution factor k=2.0 (e.g. 256 -> 512 discrete bins).
Completely eliminates 2D Gaussian heatmaps and NMS, enabling INT8 quantization
with zero loss in sub-pixel keypoint accuracy.
"""

import math
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.qualcomm_experiments.rtmpose.backbone import CSPNeXtTiny, load_official_coco_weights


class ScaleNorm(nn.Module):
    """Scale normalization layer for stable keypoint feature representation."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.s = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.norm(dim=-1, keepdim=True) * self.s).clamp(min=self.eps) * self.g


class GAU(nn.Module):
    """
    Gated Attention Unit (GAU) for keypoint relational self-attention.
    Mixes spatial features across keypoints in linear time with spatial gating.
    """
    def __init__(self, dim: int = 256, s: int = 128, expansion: int = 2):
        super().__init__()
        self.e, self.s = dim * expansion, s
        self.ln = ScaleNorm(dim)
        self.uv = nn.Linear(dim, 2 * self.e + s, bias=False)
        self.gamma = nn.Parameter(torch.randn(2, s) * 0.02)
        self.beta = nn.Parameter(torch.zeros(2, s))
        self.o = nn.Linear(self.e, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, dim]
        h = F.silu(self.uv(self.ln(x)))
        u, v, base = torch.split(h, [self.e, self.e, self.s], dim=-1)
        q, k = (base.unsqueeze(2) * self.gamma + self.beta).unbind(2)
        kernel = F.relu(q @ k.transpose(-1, -2) / math.sqrt(self.s)) ** 2
        return x + self.o(u * (kernel @ v))


class RTMCCHead13KP(nn.Module):
    """
    Official RTMCCHead implementation for RTMPose-13KP:
      1. 7x7 conv produces 1 spatial feature map per keypoint (K channels)
      2. Spatial map flattened (8x8 = 64) and projected to hidden dim (256-d) per keypoint
      3. GAU (Gated Attention Unit) mixes relational foot keypoint tokens
      4. Per-keypoint Linear heads produce discrete 1D bin logits (w_bins, h_bins)
    """
    def __init__(
        self,
        in_channels: int = 384,
        num_keypoints: int = 13,
        feat_hw: int = 64,  # 8x8 = 64 at stride 32 for 256x256
        hidden_dim: int = 256,
        input_size: Tuple[int, int] = (256, 256),
        simcc_split_ratio: float = 2.0
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.w_bins = int(input_size[0] * simcc_split_ratio)
        self.h_bins = int(input_size[1] * simcc_split_ratio)

        # 7x7 Conv preserves per-keypoint spatial layout
        self.final = nn.Conv2d(in_channels, num_keypoints, kernel_size=7, stride=1, padding=3)
        self.mlp = nn.Sequential(ScaleNorm(feat_hw), nn.Linear(feat_hw, hidden_dim, bias=False))
        self.gau = GAU(hidden_dim)

        self.cls_x = nn.Linear(hidden_dim, self.w_bins)
        self.cls_y = nn.Linear(hidden_dim, self.h_bins)
        self.vis = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, in_channels, H, W]
        # self.final(x): [B, K, H, W] -> flatten(2): [B, K, H*W]
        f = self.gau(self.mlp(self.final(x).flatten(2)))  # [B, K, hidden_dim]

        logits_x = self.cls_x(f)  # [B, K, w_bins]
        logits_y = self.cls_y(f)  # [B, K, h_bins]
        vis_logits = self.vis(f)   # [B, K, 1]

        return {
            "simcc_x": logits_x,
            "simcc_y": logits_y,
            "vis_logits": vis_logits,
            "vis": torch.sigmoid(vis_logits)
        }


# Alias for backwards compatibility
SimCCHead13KP = RTMCCHead13KP


class RTMPose13KP(nn.Module):
    """
    Complete RTMPose-13KP Foot Pose Estimation Model.
    Pairs official CSPNeXt-Tiny backbone with RTMCCHead (7x7 conv + GAU + SimCC).
    """
    def __init__(
        self,
        num_keypoints: int = 13,
        input_size: Tuple[int, int] = (256, 256),
        simcc_split_ratio: float = 2.0,
        backbone_name: str = "cspnext",
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == "mobilenetv3":
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            m = mobilenet_v3_small(weights=weights)
            self.backbone = m.features  # 576 channels at stride 32 (8x8)
            in_channels = 576
        elif self.backbone_name == "mobilenetv2":
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            m = mobilenet_v2(weights=weights)
            self.backbone = m.features  # 1280 channels at stride 32 (8x8)
            in_channels = 1280
        elif self.backbone_name == "cspnext":
            self.backbone = CSPNeXtTiny()
            if pretrained:
                load_official_coco_weights(self.backbone)
            in_channels = 384
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        feat_h = input_size[1] // 32
        feat_w = input_size[0] // 32
        feat_hw = feat_h * feat_w

        self.head = RTMCCHead13KP(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            feat_hw=feat_hw,
            hidden_dim=256,
            input_size=input_size,
            simcc_split_ratio=simcc_split_ratio
        )

        # Register official ImageNet/COCO normalization buffers
        # Allows inputs as raw BGR in [0, 1] while feeding the exact expected RGB distribution to the backbone
        self.register_buffer("mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Preprocessing: BGR in [0, 1] -> RGB in [0, 255] -> Standardized
        x = x[:, [2, 1, 0]] * 255.0
        x = (x - self.mean) / self.std

        feat = self.backbone(x)
        return self.head(feat)

