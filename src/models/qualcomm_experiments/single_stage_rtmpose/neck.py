"""
single_stage_rtmpose/neck.py
────────────────────────────
CSPNeXt-PAFPN Neck for Single-Stage RTMPose (Apache-2.0).
Provides multi-scale bidirectional feature fusion across strides 8, 16, and 32.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.qualcomm_experiments.single_stage_rtmpose.backbone import ConvModule, CSPLayer


class CSPNeXtPAFPN(nn.Module):
    """
    Path Aggregation Feature Pyramid Network built with CSPNeXt blocks.
    Input channels:  [96, 192, 384] (from Stage 2, 3, 4)
    Output channels: [out_channels, out_channels, out_channels] (default: 96)
    """
    def __init__(
        self,
        in_channels: List[int] = [96, 192, 384],
        out_channels: int = 96,
        num_csp_blocks: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Top-down feature lateral reductions
        self.reduce_c4 = ConvModule(in_channels[2], out_channels, kernel_size=1)
        self.reduce_c3 = ConvModule(in_channels[1], out_channels, kernel_size=1)
        self.reduce_c2 = ConvModule(in_channels[0], out_channels, kernel_size=1)

        # Top-down fusion CSPLayers
        self.top_down_csp1 = CSPLayer(
            out_channels * 2,
            out_channels,
            num_blocks=num_csp_blocks,
            add_identity=False
        )
        self.top_down_csp0 = CSPLayer(
            out_channels * 2,
            out_channels,
            num_blocks=num_csp_blocks,
            add_identity=False
        )

        # Bottom-up downsamplers
        self.downsample0 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.downsample1 = ConvModule(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # Bottom-up fusion CSPLayers
        self.bottom_up_csp0 = CSPLayer(
            out_channels * 2,
            out_channels,
            num_blocks=num_csp_blocks,
            add_identity=False
        )
        self.bottom_up_csp1 = CSPLayer(
            out_channels * 2,
            out_channels,
            num_blocks=num_csp_blocks,
            add_identity=False
        )

    def forward(
        self,
        feats: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2, c3, c4 = feats

        # 1. Lateral reductions
        m4 = self.reduce_c4(c4)  # stride 32, [B, out_channels, H/32, W/32]
        m3 = self.reduce_c3(c3)  # stride 16, [B, out_channels, H/16, W/16]
        m2 = self.reduce_c2(c2)  # stride 8,  [B, out_channels, H/8,  W/8]

        # 2. Top-down pathway
        # Upsample m4 to stride 16 and fuse with m3
        up_m4 = F.interpolate(m4, size=m3.shape[2:], mode="nearest")
        p3 = self.top_down_csp1(torch.cat([m3, up_m4], dim=1))

        # Upsample p3 to stride 8 and fuse with m2
        up_p3 = F.interpolate(p3, size=m2.shape[2:], mode="nearest")
        p2 = self.top_down_csp0(torch.cat([m2, up_p3], dim=1))

        # 3. Bottom-up pathway
        # Downsample p2 to stride 16 and fuse with p3
        down_p2 = self.downsample0(p2)
        n3 = self.bottom_up_csp0(torch.cat([p3, down_p2], dim=1))

        # Downsample n3 to stride 32 and fuse with m4
        down_n3 = self.downsample1(n3)
        n4 = self.bottom_up_csp1(torch.cat([m4, down_n3], dim=1))

        # Output multi-scale pyramid at strides 8, 16, 32
        return p2, n3, n4

