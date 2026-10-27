"""
single_stage_rtmpose/simcc_roi_head.py
──────────────────────────────────────
RoI-Align + SimCC Head for Single-Stage RTMPose (Apache-2.0).
Extracts high-resolution foot feature patches directly from PAFPN P2 feature maps
and applies SimCC (Simple Coordinate Classification) with GAU relational attention
and continuous expectation decoding.

Resolves the 43x resolution gap of single-stage regression by giving each foot
a dedicated 8x8 feature patch and 512-bin continuous 1D Gaussian coordinate representation.
Supports warm-start loading from official RTMPose-13KP SimCC weights (95.73% PCK).
"""

import math
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

from src.models.qualcomm_experiments.single_stage_rtmpose.backbone import ConvModule
from src.models.qualcomm_experiments.rtmpose.simcc_head import SimCCHead13KP


class RoISimCCHead(nn.Module):
    """
    RoI-Align + SimCC 13-Keypoint Head.
      1. RoI-Align extracts [N_rois, in_channels, output_size, output_size] patch from P2 (stride 8).
      2. 1x1/3x3 Conv projects channels (e.g. 96 -> 384) to match official RTMPose SimCC dimensions.
      3. SimCCHead13KP (Conv 7x7 + GAU + 1D Linear Heads) predicts:
         - simcc_x: [N_rois, 13, 512]
         - simcc_y: [N_rois, 13, 512]
         - vis_logits: [N_rois, 13, 1]
         - hand_logits: [N_rois] (0: left foot, 1: right foot)
      4. Continuous Expectation decodes sub-pixel coordinates in [0, 1] relative to the box.
      5. Bounding box coordinates map relative [0, 1] keypoints back to image pixel space.
    """
    def __init__(
        self,
        in_channels: int = 96,
        roi_channels: int = 384,
        num_keypoints: int = 13,
        roi_output_size: Tuple[int, int] = (8, 8),
        spatial_scale: float = 1.0 / 8.0,
        simcc_split_ratio: float = 2.0,
        input_size: Tuple[int, int] = (256, 256),
        pretrained_weights_path: Optional[str] = None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.roi_channels = roi_channels
        self.num_keypoints = num_keypoints
        self.roi_output_size = roi_output_size
        self.spatial_scale = spatial_scale
        self.simcc_split_ratio = simcc_split_ratio

        self.w_bins = int(input_size[0] * simcc_split_ratio)  # 512
        self.h_bins = int(input_size[1] * simcc_split_ratio)  # 512

        # 1. Feature projection from PAFPN P2 channels (96) to SimCC channels (384)
        self.roi_proj = ConvModule(
            in_channels,
            roi_channels,
            kernel_size=3,
            padding=1
        )

        # 2. SimCC Head (13KP with GAU attention + handedness)
        feat_hw = roi_output_size[0] * roi_output_size[1]  # 8x8 = 64
        self.simcc_head = SimCCHead13KP(
            in_channels=roi_channels,
            num_keypoints=num_keypoints,
            feat_hw=feat_hw,
            hidden_dim=256,
            input_size=input_size,
            simcc_split_ratio=simcc_split_ratio,
            include_handedness=True
        )

        # Pre-computed bin index coordinate grids for continuous expectation
        # Shape: [1, 1, 512]
        self.register_buffer(
            "x_bin_coords",
            torch.arange(self.w_bins, dtype=torch.float32).view(1, 1, -1)
        )
        self.register_buffer(
            "y_bin_coords",
            torch.arange(self.h_bins, dtype=torch.float32).view(1, 1, -1)
        )

        if pretrained_weights_path is not None:
            self.load_pretrained_weights(pretrained_weights_path)

    def load_pretrained_weights(self, ckpt_path: str):
        """Warm-starts the SimCC Head and GAU from official RTMPose-13KP checkpoint."""
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
            head_weights = {}
            for k, v in state.items():
                if k.startswith("head."):
                    head_weights[k.replace("head.", "")] = v
                elif k.startswith("simcc_head."):
                    head_weights[k.replace("simcc_head.", "")] = v
                elif not k.startswith("backbone.") and not k.startswith("neck."):
                    head_weights[k] = v

            missing, unexpected = self.simcc_head.load_state_dict(head_weights, strict=False)
            loaded_count = len(head_weights) - len(unexpected)
            print(f"[+] Loaded {loaded_count} pretrained SimCC head tensors from {ckpt_path}")
            if missing:
                print(f"    Missing SimCC keys: {missing}")
        except Exception as e:
            print(f"[!] Warning: Could not load pretrained SimCC weights from {ckpt_path}: {e}")

    def extract_roi_features(
        self,
        p2_feat: torch.Tensor,
        rois: torch.Tensor
    ) -> torch.Tensor:
        """
        RoI-Align feature extraction.
        rois: [N_rois, 5] (batch_idx, x1, y1, x2, y2)
        returns: [N_rois, roi_channels, 8, 8]
        """
        # torchvision.ops.roi_align natively supported in PyTorch and ONNX (op: RoiAlign)
        roi_patches = ops.roi_align(
            p2_feat,
            rois,
            output_size=self.roi_output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=2,
            aligned=True
        )
        return self.roi_proj(roi_patches)

    def forward_rois(
        self,
        p2_feat: torch.Tensor,
        rois: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for a set of RoIs.
        rois: [N_rois, 5] (batch_idx, x1, y1, x2, y2)
        """
        if rois.shape[0] == 0:
            device = p2_feat.device
            return {
                "simcc_x": torch.zeros((0, self.num_keypoints, self.w_bins), device=device),
                "simcc_y": torch.zeros((0, self.num_keypoints, self.h_bins), device=device),
                "vis_logits": torch.zeros((0, self.num_keypoints, 1), device=device),
                "vis": torch.zeros((0, self.num_keypoints, 1), device=device),
                "hand_logits": torch.zeros((0,), device=device),
                "hand": torch.zeros((0,), device=device)
            }

        roi_feats = self.extract_roi_features(p2_feat, rois)
        return self.simcc_head(roi_feats)

    def decode_continuous_expectation(
        self,
        simcc_x: torch.Tensor,
        simcc_y: torch.Tensor,
        vis_logits: torch.Tensor,
        rois: torch.Tensor
    ) -> torch.Tensor:
        """
        Continuous Expectation Decoding for sub-pixel accuracy.
        simcc_x: [N_rois, 13, 512]
        simcc_y: [N_rois, 13, 512]
        vis_logits: [N_rois, 13, 1]
        rois: [N_rois, 5] (batch_idx, x1, y1, x2, y2)

        Returns decoded keypoints in image pixel coordinates:
        [N_rois, 13, 3] where each row is (x, y, conf)
        """
        N = simcc_x.shape[0]
        if N == 0:
            return torch.zeros((0, self.num_keypoints, 3), device=simcc_x.device)

        # 1. Softmax probability distributions over 1D bins
        prob_x = F.softmax(simcc_x, dim=-1)  # [N, 13, 512]
        prob_y = F.softmax(simcc_y, dim=-1)  # [N, 13, 512]

        # 2. Continuous sub-pixel expectation in [0, 1] relative to the box:
        # expectation = sum(p_i * i) / (num_bins - 1)
        rel_u = (prob_x * self.x_bin_coords).sum(dim=-1) / (self.w_bins - 1.0)  # [N, 13]
        rel_v = (prob_y * self.y_bin_coords).sum(dim=-1) / (self.h_bins - 1.0)  # [N, 13]

        # 3. Map relative [0, 1] back to absolute image coordinates using box bounds
        bx1 = rois[:, 1].unsqueeze(-1)  # [N, 1]
        by1 = rois[:, 2].unsqueeze(-1)
        bx2 = rois[:, 3].unsqueeze(-1)
        by2 = rois[:, 4].unsqueeze(-1)
        bw = (bx2 - bx1).clamp(min=1.0)
        bh = (by2 - by1).clamp(min=1.0)

        abs_x = bx1 + rel_u * bw
        abs_y = by1 + rel_v * bh
        conf = torch.sigmoid(vis_logits.squeeze(-1))  # [N, 13]

        # Shape: [N, 13, 3]
        return torch.stack([abs_x, abs_y, conf], dim=-1)
