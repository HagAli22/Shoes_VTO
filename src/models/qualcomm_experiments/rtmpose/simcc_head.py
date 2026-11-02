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
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.qualcomm_experiments.rtmpose.backbone import (
    CSPNeXtTiny,
    CSPNeXtSlim,
    CSPNeXtSlimStage4Lite,
    CSPNeXtPico,
    load_official_coco_weights,
    load_improved5_backbone_weights
)


class SeparableConv7x7(nn.Module):
    """
    Bottleneck 1x1 projection + 7x7 Depthwise Conv for high spatial receptive field
    with minimal parameters and Qualcomm NPU-friendly execution.
    Reduces head convolution parameters by >90% compared to dense 7x7 conv.
    """
    def __init__(self, in_channels: int, out_channels: int = 13, mid_channels: int = 48):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True)
        self.dw = nn.Conv2d(mid_channels, mid_channels, kernel_size=7, stride=1, padding=3, groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels, eps=1e-3, momentum=0.03)
        self.pw = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.proj(x)))
        x = self.act(self.bn2(self.dw(x)))
        return self.pw(x)


class TokenPoolingHandHead(nn.Module):
    """
    Lightweight Global Token Pooling + MLP for foot chirality (Left/Right) classification.
    Pools across the 13 keypoint tokens to produce a permutation-invariant feature,
    reducing parameters from 426K down to ~8K-16K (96% reduction) while preventing overfitting.
    """
    def __init__(self, hidden_dim: int = 128, mlp_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(mlp_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K, hidden_dim]
        pooled = x.mean(dim=1)  # [B, hidden_dim]
        return self.mlp(pooled)


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
    Modular RTMCCHead implementation for RTMPose-13KP:
      1. Head conv (Separable 7x7 or standard 7x7) produces spatial maps per keypoint
      2. Spatial map flattened (H*W) and projected to hidden dim (128-d / 256-d) per keypoint
      3. GAU (Gated Attention Unit) mixes relational foot keypoint tokens
      4. Per-keypoint Linear heads produce discrete 1D bin logits (w_bins, h_bins)
      5. Handedness head (Token Pooling or Flatten) classifies Left vs Right Foot
    """
    def __init__(
        self,
        in_channels: int = 384,
        num_keypoints: int = 13,
        feat_hw: int = 64,  # 8x8 = 64 at stride 32; 16x16 = 256 at stride 16
        hidden_dim: int = 128,
        input_size: Tuple[int, int] = (256, 256),
        simcc_split_ratio: float = 2.0,
        include_handedness: bool = True,
        head_conv_type: str = "separable",
        hand_head_type: str = "pooling"
    ):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.w_bins = int(input_size[0] * simcc_split_ratio)
        self.h_bins = int(input_size[1] * simcc_split_ratio)
        self.include_handedness = include_handedness

        # Head convolution: separable 7x7 (NPU optimized, ~12K params) vs standard 7x7 (244K params)
        if head_conv_type == "separable":
            self.final = SeparableConv7x7(in_channels, num_keypoints, mid_channels=min(48, in_channels))
        else:
            self.final = nn.Conv2d(in_channels, num_keypoints, kernel_size=7, stride=1, padding=3)

        self.mlp = nn.Sequential(ScaleNorm(feat_hw), nn.Linear(feat_hw, hidden_dim, bias=False))
        self.gau = GAU(hidden_dim)

        self.cls_x = nn.Linear(hidden_dim, self.w_bins)
        self.cls_y = nn.Linear(hidden_dim, self.h_bins)
        self.vis = nn.Linear(hidden_dim, 1)

        # In-model handedness classification (0: Left Foot, 1: Right Foot)
        if include_handedness:
            if hand_head_type == "pooling":
                # Token average pooling + MLP: permutation-invariant, ~8K-16K params (96% smaller)
                self.hand = TokenPoolingHandHead(hidden_dim=hidden_dim, mlp_dim=min(64, hidden_dim))
            else:
                # Legacy flattened linear head: 426K params
                self.hand = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(num_keypoints * hidden_dim, 128),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1)
                )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: [B, in_channels, H, W]
        # self.final(x): [B, K, H, W] -> flatten(2): [B, K, H*W]
        f = self.gau(self.mlp(self.final(x).flatten(2)))  # [B, K, hidden_dim]

        logits_x = self.cls_x(f)  # [B, K, w_bins]
        logits_y = self.cls_y(f)  # [B, K, h_bins]
        vis_logits = self.vis(f)   # [B, K, 1]

        out = {
            "simcc_x": logits_x,
            "simcc_y": logits_y,
            "vis_logits": vis_logits,
            "vis": torch.sigmoid(vis_logits)
        }

        if self.include_handedness:
            hand_logits = self.hand(f).squeeze(-1)  # [B]
            out["hand_logits"] = hand_logits
            out["hand"] = torch.sigmoid(hand_logits)

        return out


# Alias for backwards compatibility
SimCCHead13KP = RTMCCHead13KP


class RTMPose13KP(nn.Module):
    """
    Complete RTMPose-13KP Foot Pose Estimation Model.
    Supports official CSPNeXt-Tiny, CSPNeXt-Slim (stride 16), CSPNeXt-SlimStage4Lite (stride 32),
    CSPNeXt-Pico (width 0.5x), and MobileNetV3.
    """
    def __init__(
        self,
        num_keypoints: int = 13,
        input_size: Tuple[int, int] = (256, 256),
        simcc_split_ratio: float = 2.0,
        backbone_name: str = "cspnext",
        pretrained: bool = True,
        include_handedness: bool = True,
        hidden_dim: Optional[int] = None,
        head_conv_type: Optional[str] = None,
        hand_head_type: Optional[str] = None
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.include_handedness = include_handedness

        # Determine architecture defaults based on backbone choice
        if self.backbone_name == "mobilenetv3":
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            m = mobilenet_v3_small(weights=weights)
            self.backbone = m.features  # 576 channels at stride 32 (8x8)
            in_channels = 576
            stride = 32
            hidden_dim = 128 if hidden_dim is None else hidden_dim
            head_conv_type = "separable" if head_conv_type is None else head_conv_type
            hand_head_type = "pooling" if hand_head_type is None else hand_head_type
        elif self.backbone_name == "mobilenetv2":
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            m = mobilenet_v2(weights=weights)
            self.backbone = m.features  # 1280 channels at stride 32 (8x8)
            in_channels = 1280
            stride = 32
            hidden_dim = 128 if hidden_dim is None else hidden_dim
            head_conv_type = "separable" if head_conv_type is None else head_conv_type
            hand_head_type = "pooling" if hand_head_type is None else hand_head_type
        elif self.backbone_name == "cspnext":
            # Exact legacy configuration
            self.backbone = CSPNeXtTiny()
            if pretrained:
                load_official_coco_weights(self.backbone)
            in_channels = 384
            stride = 32
            hidden_dim = 256 if hidden_dim is None else hidden_dim
            head_conv_type = "standard" if head_conv_type is None else head_conv_type
            hand_head_type = "flatten" if hand_head_type is None else hand_head_type
        elif self.backbone_name in ("cspnext_slim", "slim"):
            # Option A: High-resolution stride-16 backbone (192 channels, no stage 4)
            self.backbone = CSPNeXtSlim()
            in_channels = 192
            stride = 16
            hidden_dim = 128 if hidden_dim is None else hidden_dim
            head_conv_type = "separable" if head_conv_type is None else head_conv_type
            hand_head_type = "pooling" if hand_head_type is None else hand_head_type
        elif self.backbone_name in ("cspnext_slim_s4lite", "cspnext_slim_stage4lite", "s4lite"):
            # Option A + Stage-4-Lite: Stride-32 backbone (192 channels, lightweight stage 4)
            self.backbone = CSPNeXtSlimStage4Lite()
            in_channels = 192
            stride = 32
            hidden_dim = 128 if hidden_dim is None else hidden_dim
            head_conv_type = "separable" if head_conv_type is None else head_conv_type
            hand_head_type = "pooling" if hand_head_type is None else hand_head_type
        elif self.backbone_name in ("cspnext_pico", "pico"):
            # Option B: 0.5x width backbone (192 channels at stage 4)
            self.backbone = CSPNeXtPico()
            in_channels = 192
            stride = 32
            hidden_dim = 128 if hidden_dim is None else hidden_dim
            head_conv_type = "separable" if head_conv_type is None else head_conv_type
            hand_head_type = "pooling" if hand_head_type is None else hand_head_type
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        feat_h = input_size[1] // stride
        feat_w = input_size[0] // stride
        feat_hw = feat_h * feat_w

        self.head = RTMCCHead13KP(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            feat_hw=feat_hw,
            hidden_dim=hidden_dim,
            input_size=input_size,
            simcc_split_ratio=simcc_split_ratio,
            include_handedness=include_handedness,
            head_conv_type=head_conv_type,
            hand_head_type=hand_head_type
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


def load_improved5_weights(
    student_model: nn.Module,
    ckpt_path: str = "checkpoints/rtmpose_13kp_improved5/best_with_hand.pt"
) -> int:
    """
    Transfers compatible weights from rtmpose_13kp_improved5 checkpoint into the student model.
    1. Loads matching backbone layers (Stem, Stages 1-3).
    2. Slices/loads compatible head layers (SimCC coordinate classification, visibility, scale norms).
    Returns total transferred parameters.
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        alt = ckpt_file.parent / "best.pt"
        if alt.exists():
            ckpt_file = alt
        else:
            print(f"[!] Warning: Checkpoint {ckpt_path} not found.")
            return 0

    try:
        ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(ckpt_file), map_location="cpu")

    state = ckpt.get("model_state", ckpt)
    student_state = student_model.state_dict()

    matched = {}
    transferred_params = 0

    for k, v in state.items():
        if k in student_state:
            # Exact tensor shape match (e.g. stem, stages 1-3, scale norms, biases)
            if student_state[k].shape == v.shape:
                matched[k] = v
                transferred_params += v.numel()
            # Partial slice match for linear head projections (e.g. hidden_dim 256 -> 128)
            elif (len(v.shape) == 2 and len(student_state[k].shape) == 2 and
                  student_state[k].shape[0] == v.shape[0] and student_state[k].shape[1] < v.shape[1]):
                # Slice input dimension (e.g. [512, 128] from [512, 256])
                matched[k] = v[:, :student_state[k].shape[1]].clone()
                transferred_params += matched[k].numel()
            elif (len(v.shape) == 1 and len(student_state[k].shape) == 1 and
                  student_state[k].shape[0] == v.shape[0]):
                matched[k] = v.clone()
                transferred_params += matched[k].numel()

    missing, unexpected = student_model.load_state_dict(matched, strict=False)
    total_student = sum(p.numel() for p in student_model.parameters())
    pct = transferred_params / max(1, total_student) * 100
    print(f"[+] Successfully transferred {len(matched)} tensors ({transferred_params:,} parameters, {pct:.1f}% of student) from {ckpt_file.name}!")
    return transferred_params


