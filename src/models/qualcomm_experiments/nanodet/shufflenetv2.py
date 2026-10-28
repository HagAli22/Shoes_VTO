"""
nanodet/shufflenetv2.py
───────────────────────
ShuffleNetV2 1.0x Backbone for NanoDet-Plus (Apache-2.0 / BSD-3-Clause).
Extracts multi-scale features at strides 8 (stage2), 16 (stage3), and 32 (stage4).
Supports pure ReLU (native torchvision ImageNet weights, zero INT8 NPU penalty)
as well as legacy LeakyReLU(0.1) for backward compatibility with official COCO checkpoints.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights


def _relu_to_leaky(mod: nn.Module):
    """Recursively replaces all nn.ReLU with nn.LeakyReLU(0.1, inplace=True)."""
    for name, child in mod.named_children():
        if isinstance(child, nn.ReLU):
            setattr(mod, name, nn.LeakyReLU(0.1, inplace=True))
        else:
            _relu_to_leaky(child)


class ShuffleNetV2Backbone(nn.Module):
    """
    ShuffleNetV2 1.0x multi-scale feature extractor.
    Feature Channels:
      - Stage 2: 116 channels (stride 8)
      - Stage 3: 232 channels (stride 16)
      - Stage 4: 464 channels (stride 32)
    Uses native ReLU by default (100% matched to torchvision ImageNet pretrained weights
    and optimal for Qualcomm Hexagon NPU / WASM SIMD INT8 quantization).
    """
    def __init__(self, pretrained: bool = True, activation: str = "relu"):
        super().__init__()
        weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        m = shufflenet_v2_x1_0(weights=weights)
        if activation == "leaky":
            _relu_to_leaky(m)

        self.conv1 = m.conv1
        self.maxpool = m.maxpool
        self.stage2 = m.stage2  # stride 8,  116 channels
        self.stage3 = m.stage3  # stride 16, 232 channels
        self.stage4 = m.stage4  # stride 32, 464 channels

        self.out_channels = [116, 232, 464]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.maxpool(x)

        c2 = self.stage2(x)   # [B, 116, H/8,  W/8]
        c3 = self.stage3(c2)  # [B, 232, H/16, W/16]
        c4 = self.stage4(c3)  # [B, 464, H/32, W/32]

        return [c2, c3, c4]


