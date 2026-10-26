"""
rtmpose/backbone.py
───────────────────
Official CSPNeXt-Tiny Backbone for RTMPose-13KP.
Architecture matches OpenMMLab / Qualcomm AI Hub RTMPose-Tiny specification:
  - Multi-stage hierarchical CSPNeXt with 5x5 depthwise separable convolutions
  - SPPF (Spatial Pyramid Pooling Fast) at Stage 4
  - Feature dimension: 384 channels at stride 32 (8x8 for 256x256 input)
  - Full 206-tensor pretrained weight transfer from official COCO checkpoint
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """Standard Convolution + BatchNorm + SiLU block with official CSPNeXt parameters."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """5x5 Depthwise Separable Convolution block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1, padding: int = 2):
        super().__init__()
        self.depthwise_conv = ConvModule(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise_conv = ConvModule(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise_conv(self.depthwise_conv(x))


class CSPNeXtBlock(nn.Module):
    """CSPNeXt residual bottleneck block with 5x5 depthwise separable conv."""
    def __init__(self, channels: int, add_identity: bool = True):
        super().__init__()
        self.conv1 = ConvModule(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = DepthwiseSeparableConv(channels, channels, kernel_size=5, stride=1, padding=2)
        self.add_identity = add_identity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.add_identity else y


class ChannelAttention(nn.Module):
    """Channel-attention module for CSPNeXt feature recalibration with Hardsigmoid."""
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class CSPLayer(nn.Module):
    """
    Cross Stage Partial (CSP) layer.
    Attention is applied to concatenated [main, short] BEFORE final_conv.
    """
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1, add_identity: bool = True, channel_attention: bool = True):
        super().__init__()
        mid_channels = out_channels // 2
        self.main_conv = ConvModule(in_channels, mid_channels, kernel_size=1)
        self.short_conv = ConvModule(in_channels, mid_channels, kernel_size=1)
        self.final_conv = ConvModule(mid_channels * 2, out_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[CSPNeXtBlock(mid_channels, add_identity=add_identity) for _ in range(num_blocks)])
        self.attention = ChannelAttention(mid_channels * 2) if channel_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.blocks(self.main_conv(x))
        short = self.short_conv(x)
        out = torch.cat([main, short], dim=1)
        if self.attention is not None:
            out = self.attention(out)
        return self.final_conv(out)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling Fast (SPPF) block at Stage 4."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = ConvModule(in_channels, mid_channels, kernel_size=1)
        self.conv2 = ConvModule(mid_channels * 4, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))


class CSPNeXtTiny(nn.Module):
    """
    Official CSPNeXt-Tiny backbone specification matching OpenMMLab / Qualcomm AI Hub.
    Output: 384 channels at stride 32 (8x8 for 256x256 input).
    """
    def __init__(self):
        super().__init__()
        # Stem: stride 2
        self.stem = nn.Sequential(
            ConvModule(3, 12, kernel_size=3, stride=2, padding=1),
            ConvModule(12, 12, kernel_size=3, stride=1, padding=1),
            ConvModule(12, 24, kernel_size=3, stride=1, padding=1)
        )
        # Stage 1: stride 4 (64x64, 48 channels)
        self.stage1 = nn.Sequential(
            ConvModule(24, 48, kernel_size=3, stride=2, padding=1),
            CSPLayer(48, 48, num_blocks=1, add_identity=True)
        )
        # Stage 2: stride 8 (32x32, 96 channels)
        self.stage2 = nn.Sequential(
            ConvModule(48, 96, kernel_size=3, stride=2, padding=1),
            CSPLayer(96, 96, num_blocks=1, add_identity=True)
        )
        # Stage 3: stride 16 (16x16, 192 channels)
        self.stage3 = nn.Sequential(
            ConvModule(96, 192, kernel_size=3, stride=2, padding=1),
            CSPLayer(192, 192, num_blocks=1, add_identity=True)
        )
        # Stage 4: stride 32 (8x8, 384 channels, add_identity=False in final stage)
        self.stage4 = nn.Sequential(
            ConvModule(192, 384, kernel_size=3, stride=2, padding=1),
            SPPF(384, 384, kernel_size=5),
            CSPLayer(384, 384, num_blocks=1, add_identity=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


def load_official_coco_weights(
    backbone_model: nn.Module,
    ckpt_path: str = "checkpoints/rtmpose_tiny_coco.pth"
) -> bool:
    """
    Transfers 100% of the 206 backbone tensors from official RTMPose-tiny checkpoint.
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        print(f"[*] Downloading official RTMPose-tiny COCO weights to {ckpt_file}...")
        url = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-tiny_simcc-coco_pt-aic-coco_420e-256x192-e613ba3f_20230127.pth"
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(ckpt_file))

    try:
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_file), map_location="cpu")
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    bb_state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
    missing, unexpected = backbone_model.load_state_dict(bb_state, strict=True)
    print(f"[+] Successfully transferred all {len(bb_state)} tensors from official RTMPose-tiny COCO checkpoint!")
    return True
