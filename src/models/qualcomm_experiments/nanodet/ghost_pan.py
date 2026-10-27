"""
nanodet/ghost_pan.py
────────────────────
Official Ghost-PAN (Ghost Feature Pyramid Network) for NanoDet-Plus (Apache-2.0).
Matches official RangiLyu/nanodet checkpoint architecture with 100% weight compatibility:
  - 3 Input stages from ShuffleNetV2 (116, 232, 464 ch)
  - 4 Output pyramid levels (strides 8, 16, 32, 64; 96 ch each)
  - GhostBottleneck feature aggregation with depthwise 5x5 shortcuts
"""

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
    """1x1 Conv + BatchNorm2d + LeakyReLU(0.1)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1, p: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise 5x5 Conv + BN + Pointwise 1x1 Conv + BN + LeakyReLU(0.1)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 5, s: int = 1, p: int = 2):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.dwnorm = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.pwnorm = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dwnorm(self.depthwise(x)))
        x = self.act(self.pwnorm(self.pointwise(x)))
        return x


class GhostModule(nn.Module):
    """Ghost Module splitting channels into primary 1x1 conv and cheap 3x3 depthwise conv."""
    def __init__(self, in_ch: int, out_ch: int, ratio: int = 2, activation: bool = True):
        super().__init__()
        init_ch = math.ceil(out_ch / ratio)
        cheap_ch = init_ch * (ratio - 1)
        
        primary = [
            nn.Conv2d(in_ch, init_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(init_ch)
        ]
        if activation:
            primary.append(nn.LeakyReLU(0.1, inplace=True))
        self.primary_conv = nn.Sequential(*primary)

        cheap = [
            nn.Conv2d(init_ch, cheap_ch, 3, 1, 1, groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch)
        ]
        if activation:
            cheap.append(nn.LeakyReLU(0.1, inplace=True))
        self.cheap_operation = nn.Sequential(*cheap)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck block matching official checkpoint."""
    def __init__(self, in_ch: int, out_ch: int, dw_kernel_size: int = 5):
        super().__init__()
        # 1st Ghost Module has LeakyReLU activation
        self.ghost1 = GhostModule(in_ch, out_ch, activation=True)
        # 2nd Ghost Module has NO activation (linear projection matching official NanoDet)
        self.ghost2 = GhostModule(out_ch, out_ch, activation=False)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, dw_kernel_size, 1, dw_kernel_size // 2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.ghost2(self.ghost1(x))
        return res + self.shortcut(x)


class GhostBlocks(nn.Module):
    """Container holding GhostBottleneck blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.blocks = nn.ModuleList([GhostBottleneck(in_ch, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


class GhostPAN(nn.Module):
    """
    Official Ghost-PAN Feature Pyramid Network.
    Inputs: [C2 (116ch), C3 (232ch), C4 (464ch)]
    Outputs: [N2 (stride 8), N3 (stride 16), N4 (stride 32), P5 (stride 64)] each 96 channels.
    """
    def __init__(
        self,
        in_channels: List[int] = [116, 232, 464],
        out_channels: int = 96
    ):
        super().__init__()
        self.reduce_layers = nn.ModuleList([
            ConvModule(ch, out_channels, k=1, s=1, p=0) for ch in in_channels
        ])
        self.top_down_blocks = nn.ModuleList([
            GhostBlocks(out_channels * 2, out_channels),
            GhostBlocks(out_channels * 2, out_channels)
        ])
        self.downsamples = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, k=5, s=2, p=2),
            DepthwiseSeparableConv(out_channels, out_channels, k=5, s=2, p=2)
        ])
        self.bottom_up_blocks = nn.ModuleList([
            GhostBlocks(out_channels * 2, out_channels),
            GhostBlocks(out_channels * 2, out_channels)
        ])
        self.extra_lvl_in_conv = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, k=5, s=2, p=2)
        ])
        self.extra_lvl_out_conv = nn.ModuleList([
            DepthwiseSeparableConv(out_channels, out_channels, k=5, s=2, p=2)
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        inputs = [r(f) for r, f in zip(self.reduce_layers, feats)]
        c2, c3, c4 = inputs

        # Top-down pathway (OpenMMLab / NanoDet GhostPAN exact ordering)
        # 1. p4 upsampled + c3 -> top_down_blocks[0]
        p4 = c4
        p4_up = F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)
        p3 = self.top_down_blocks[0](torch.cat([p4_up, c3], dim=1))

        # 2. p3 upsampled + c2 -> top_down_blocks[1]
        p3_up = F.interpolate(p3, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = self.top_down_blocks[1](torch.cat([p3_up, c2], dim=1))

        # Bottom-up pathway (downsampled features concatenated first)
        # 1. p2 downsampled + p3 -> bottom_up_blocks[0]
        n2 = p2
        n2_down = self.downsamples[0](n2)
        n3 = self.bottom_up_blocks[0](torch.cat([n2_down, p3], dim=1))

        # 2. n3 downsampled + p4 -> bottom_up_blocks[1]
        n3_down = self.downsamples[1](n3)
        n4 = self.bottom_up_blocks[1](torch.cat([n3_down, p4], dim=1))

        # Extra level (stride 64): parallel addition of downsampled backbone input and neck output
        p5 = self.extra_lvl_in_conv[0](c4) + self.extra_lvl_out_conv[0](n4)

        return [n2, n3, n4, p5]
