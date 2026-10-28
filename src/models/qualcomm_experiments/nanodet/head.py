"""
nanodet/head.py
───────────────
Official NanoDet-Plus Detection Head (Apache-2.0).
Shared depthwise-separable conv towers across 4 pyramid scales (strides 8, 16, 32, 64)
with unified GFL classification + DFL bounding box distribution heads.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.qualcomm_experiments.nanodet.ghost_pan import DepthwiseSeparableConv


class Integral(nn.Module):
    """
    Computes continuous expectation over DFL discrete probability bins:
      dist = sum_{i=0}^{reg_max} i * softmax(p_i)
    Matches buffer name `distribution_project.project` from official checkpoint.
    """
    def __init__(self, reg_max: int = 7):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer(
            "project",
            torch.linspace(0, reg_max, reg_max + 1, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 4:
            B, C, H, W = orig_shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B * H * W * 4, self.reg_max + 1)
            prob = F.softmax(x, dim=-1)
            dist = torch.matmul(prob, self.project)
            return dist.view(B, H, W, 4).permute(0, 3, 1, 2).contiguous()
        elif x.dim() == 3:
            B, N, C = orig_shape
            x = x.view(B * N * 4, self.reg_max + 1)
            prob = F.softmax(x, dim=-1)
            dist = torch.matmul(prob, self.project)
            return dist.view(B, N, 4)
        else:
            N = orig_shape[0]
            x = x.view(N * 4, self.reg_max + 1)
            prob = F.softmax(x, dim=-1)
            dist = torch.matmul(prob, self.project)
            return dist.view(N, 4)


class NanoDetPlusHead(nn.Module):
    """
    Official NanoDet-Plus Head & Hardware-Optimized Stage A Head.
    Features:
      - Multi-scale support (e.g. strides [8, 16, 32] or [8, 16, 32, 64])
      - Stacked DepthwiseSeparableConv towers (k=3 or k=5)
      - Pure ReLU or LeakyReLU activations
      - Unified GFL classification + DFL bounding box distribution heads
      - Configurable reg_max (16 for high-precision reach, 7 for legacy checkpoints)
    """
    def __init__(
        self,
        num_classes: int = 1,
        in_channels: int = 96,
        stacked_convs: int = 2,
        reg_max: int = 16,
        strides: List[int] = [8, 16, 32],
        act_type: str = "relu",
        kernel_size: int = 3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.reg_max = reg_max
        self.strides = strides
        self.act_type = act_type
        self.kernel_size = kernel_size

        self.distribution_project = Integral(reg_max=reg_max)

        self.cls_convs = nn.ModuleList()
        self.gfl_cls = nn.ModuleList()
        pad = kernel_size // 2

        for _ in strides:
            convs = nn.ModuleList()
            for _ in range(stacked_convs):
                convs.append(
                    DepthwiseSeparableConv(
                        in_channels,
                        in_channels,
                        k=kernel_size,
                        s=1,
                        p=pad,
                        act_type=act_type
                    )
                )
            self.cls_convs.append(convs)
            self.gfl_cls.append(
                nn.Conv2d(in_channels, num_classes + 4 * (reg_max + 1), kernel_size=1, stride=1, padding=0)
            )

        self._init_weights()


    def _init_weights(self):
        prior_prob = 0.01
        bias_init = -float(torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        for head in self.gfl_cls:
            nn.init.normal_(head.weight, std=0.01)
            # Classification channels initialized with prior bias
            nn.init.constant_(head.bias[:self.num_classes], bias_init)
            # Regression channels initialized with zero
            nn.init.constant_(head.bias[self.num_classes:], 0.0)

    def forward(self, feats: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        cls_scores = []
        reg_dists = []

        for feat, convs, head in zip(feats, self.cls_convs, self.gfl_cls):
            for c in convs:
                feat = c(feat)
            out = head(feat)
            cls_scores.append(out[:, :self.num_classes])
            reg_dists.append(out[:, self.num_classes:])

        return cls_scores, reg_dists
