"""
blazeblock.py
─────────────
Core building blocks for Google BlazeFoot architecture (MediaPipe / BlazePose style).

Key design features:
  - 5x5 Depthwise Separable Convolutions (large receptive field with minimal FLOPs).
  - Residual skip connections with channel padding and max pooling for downsampling.
  - DoubleBlazeBlock for deeper feature extraction in high-level layers.
  - PReLU activation functions for non-linear feature representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlazeBlock(nn.Module):
    """
    Single BlazeBlock:
      Branch 1 (Conv path):
        1. 5x5 Depthwise Conv (stride s)
        2. BatchNorm + PReLU
        3. 1x1 Pointwise Conv (out_channels)
        4. BatchNorm
      Branch 2 (Skip path):
        - If stride == 2: 2x2 MaxPool + channel padding
        - If in_channels != out_channels: 1x1 Conv or channel padding
      Output = PReLU(Conv path + Skip path)
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Depthwise 5x5 + Pointwise 1x1
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, stride=stride,
            padding=2, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.PReLU(in_channels)

        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.PReLU(out_channels)

        # Skip connection handling
        if stride == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self.maxpool = None

        if in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Main branch
        h = self.depthwise(x)
        h = self.bn1(h)
        h = self.act1(h)
        h = self.pointwise(h)
        h = self.bn2(h)

        # Skip branch
        skip = x
        if self.maxpool is not None:
            skip = self.maxpool(skip)
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        # Residual sum + activation
        return self.act2(h + skip)


class DoubleBlazeBlock(nn.Module):
    """
    Double BlazeBlock (used in deeper stages of BlazePose / MediaPipe):
      Two consecutive 5x5 depthwise + 1x1 pointwise convs inside a single residual block.
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = out_channels // 2 if out_channels >= in_channels else out_channels

        # First depthwise + pointwise
        self.dw1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, stride=stride,
            padding=2, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.PReLU(in_channels)

        self.pw1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = nn.PReLU(mid_channels)

        # Second depthwise + pointwise
        self.dw2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=5, stride=1,
            padding=2, groups=mid_channels, bias=False
        )
        self.bn3 = nn.BatchNorm2d(mid_channels)
        self.act3 = nn.PReLU(mid_channels)

        self.pw2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=1, stride=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(out_channels)
        self.act4 = nn.PReLU(out_channels)

        # Skip connection handling
        if stride == 2:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self.maxpool = None

        if in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First layer
        h = self.dw1(x)
        h = self.bn1(h)
        h = self.act1(h)
        h = self.pw1(h)
        h = self.bn2(h)
        h = self.act2(h)

        # Second layer
        h = self.dw2(h)
        h = self.bn3(h)
        h = self.act3(h)
        h = self.pw2(h)
        h = self.bn4(h)

        # Skip branch
        skip = x
        if self.maxpool is not None:
            skip = self.maxpool(skip)
        if self.skip_conv is not None:
            skip = self.skip_conv(skip)

        return self.act4(h + skip)

