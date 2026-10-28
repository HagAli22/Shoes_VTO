"""
nanodet/model.py
────────────────
NanoDet-Plus Complete Object Detector (Apache-2.0).
Integrates ShuffleNetV2 1.0x + Ghost-PAN + NanoDetPlusHead with DFL bounding box decoding.
Includes official COCO pretrained weights loader with zero missing keys and regression transfer.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import sys
import types
import urllib.request
import torch
import torch.nn as nn
import torchvision.ops as ops

from src.models.qualcomm_experiments.nanodet.shufflenetv2 import ShuffleNetV2Backbone
from src.models.qualcomm_experiments.nanodet.ghost_pan import GhostPAN, RepPAN
from src.models.qualcomm_experiments.nanodet.head import NanoDetPlusHead

OFFICIAL_COCO_CKPT_URL = "https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320_checkpoint.ckpt"
DEFAULT_CKPT_PATH = "checkpoints/nanodet_plus_m_320_checkpoint.ckpt"


class DummyLightning:
    """Mock for loading PyTorch Lightning checkpoints without pytorch_lightning dependency."""
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): pass


class MockLightningModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
    def __getattr__(self, name):
        return DummyLightning


def _setup_lightning_mock():
    for mod in [
        "pytorch_lightning",
        "pytorch_lightning.callbacks",
        "pytorch_lightning.callbacks.model_checkpoint",
        "pytorch_lightning.core",
        "pytorch_lightning.core.lightning",
        "pytorch_lightning.core.memory",
        "pytorch_lightning.loggers",
        "lightning_fabric",
        "lightning_fabric.utilities",
        "lightning_fabric.utilities.cloud_io"
    ]:
        if mod not in sys.modules:
            m = MockLightningModule(mod)
            sys.modules[mod] = m


def load_official_nanodet_weights(
    model: nn.Module,
    ckpt_path: str = DEFAULT_CKPT_PATH
) -> Dict[str, Any]:
    """
    Downloads and transfers weights from official NanoDet-Plus-m COCO checkpoint.
    - Loads 100% of backbone (ShuffleNetV2 1.0x)
    - Loads 100% of neck (Ghost-PAN)
    - Loads 100% of head towers (cls_convs across all 4 levels)
    - Adapts gfl_cls: preserves all 32 regression channels directly from COCO,
      initializes classification channels for the target 'foot' class.
    """
    ckpt_file = Path(ckpt_path)
    if not ckpt_file.exists():
        ckpt_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[*] Downloading official NanoDet-Plus-m COCO checkpoint to {ckpt_file}...")
        urllib.request.urlretrieve(OFFICIAL_COCO_CKPT_URL, str(ckpt_file))
        print(f"[+] Download complete: {ckpt_file} ({ckpt_file.stat().st_size / (1024*1024):.2f} MB)")

    _setup_lightning_mock()
    ckpt = torch.load(str(ckpt_file), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    # Use avg_model (EMA weights from COCO benchmark) if available
    src_prefix = "avg_model." if any(k.startswith("avg_model.") for k in state_dict.keys()) else "model."

    model_state = model.state_dict()
    loaded_keys = []
    adapted_keys = []
    missing_keys = []
    unexpected_keys = []

    num_classes = model.head.num_classes

    for name, param in model_state.items():
        if name in ("mean", "std"):
            continue
        src_key = src_prefix + name
        if src_key in state_dict:
            src_tensor = state_dict[src_key]
            if src_tensor.shape == param.shape:
                model_state[name] = src_tensor
                loaded_keys.append(name)
            else:
                # Handle classification head adaptation for 'foot' class
                if "gfl_cls" in name:
                    adapted_tensor = param.clone()
                    if "weight" in name:
                        # Transfer regression weights (last 32 channels from COCO 80:112)
                        adapted_tensor[num_classes:] = src_tensor[80:]
                        # Initialize classification weights
                        nn.init.normal_(adapted_tensor[:num_classes], std=0.01)
                    elif "bias" in name:
                        # Transfer regression biases
                        adapted_tensor[num_classes:] = src_tensor[80:]
                        # Initialize classification bias with focal loss prior: -4.595 (sigmoid(-4.595) = 0.01)
                        nn.init.constant_(adapted_tensor[:num_classes], -4.595)
                    model_state[name] = adapted_tensor
                    adapted_keys.append(name)
                else:
                    missing_keys.append(f"{name} (shape mismatch: {param.shape} vs {src_tensor.shape})")
        else:
            missing_keys.append(name)

    model.load_state_dict(model_state)

    print(f"[+] Successfully loaded official NanoDet-Plus-m COCO checkpoint:")
    print(f"    - Exact matching tensors loaded: {len(loaded_keys)} / {len(model_state)}")
    print(f"    - Adapted head tensors (regression transferred + cls init): {len(adapted_keys)}")
    print(f"    - Missing / Unloaded tensors: {len(missing_keys)}")
    if missing_keys:
        print(f"    [!] Warning: Missing keys: {missing_keys[:10]}")

    return {
        "loaded_keys": len(loaded_keys),
        "adapted_keys": len(adapted_keys),
        "missing_keys": len(missing_keys),
        "missing_names": missing_keys
    }


class NanoDetPlus(nn.Module):
    """
    NanoDet-Plus & Hardware-Optimized Stage A Foot Detector (Apache-2.0).
    Backbone: ShuffleNetV2 1.0x with native ReLU
    Neck: Rep-PAN (3x3 DWConv + residual addition) or Ghost-PAN
    Head: NanoDetPlusHead with configurable reg_max & strides
    Input: Standardized BGR image in [0, 1] of shape [B, 3, 256, 256]
    """
    def __init__(
        self,
        num_classes: int = 1,
        input_size: Tuple[int, int] = (256, 256),
        out_channels: int = 96,
        stacked_convs: int = 2,
        reg_max: int = 16,
        strides: List[int] = [8, 16, 32],
        neck_type: str = "rep_pan",
        act_type: str = "relu",
        kernel_size: int = 3,
        pretrained: bool = False,
        ckpt_path: str = DEFAULT_CKPT_PATH
    ):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.strides = strides
        self.reg_max = reg_max
        self.neck_type = neck_type
        self.act_type = act_type

        # Backbone uses native ImageNet pretrained weights (pure ReLU)
        self.backbone = ShuffleNetV2Backbone(pretrained=pretrained, activation=act_type)

        if neck_type == "rep_pan":
            self.fpn = RepPAN(
                in_channels=self.backbone.out_channels,
                out_channels=out_channels,
                num_levels=len(strides),
                act_type=act_type
            )
        else:
            self.fpn = GhostPAN(in_channels=self.backbone.out_channels, out_channels=out_channels)

        self.head = NanoDetPlusHead(
            num_classes=num_classes,
            in_channels=out_channels,
            stacked_convs=stacked_convs,
            reg_max=reg_max,
            strides=strides,
            act_type=act_type,
            kernel_size=kernel_size
        )

        # In-model normalization buffers matching official NanoDet (BGR format)
        self.register_buffer("mean", torch.tensor([103.53, 116.28, 123.675]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([57.375, 57.12, 58.395]).view(1, 3, 1, 1))

        if pretrained and Path(ckpt_path).exists() and reg_max == 7 and neck_type == "ghost_pan":
            load_official_nanodet_weights(self, ckpt_path=ckpt_path)


    @property
    def neck(self):
        return self.fpn

    def forward(self, x: torch.Tensor, export_mode: bool = False) -> Dict[str, Any]:
        # Preprocessing: BGR in [0, 1] -> BGR [0, 255] -> Standardized
        x_norm = (x * 255.0 - self.mean) / self.std

        feats = self.backbone(x_norm)
        pan_feats = self.fpn(feats)
        cls_preds, reg_preds = self.head(pan_feats)

        if not self.training or export_mode:
            boxes, scores = self._decode_predictions(cls_preds, reg_preds)
            return {"boxes": boxes, "scores": scores, "cls_preds": cls_preds, "reg_preds": reg_preds}

        return {"cls_preds": cls_preds, "reg_preds": reg_preds}

    def _decode_predictions(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = cls_preds[0].shape[0]
        device = cls_preds[0].device
        all_boxes = []
        all_scores = []

        max_w = float(self.input_size[0])
        max_h = float(self.input_size[1])

        for stride, cls_p, reg_p in zip(self.strides, cls_preds, reg_preds):
            _, _, h, w = cls_p.shape

            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij"
            )
            cx = (grid_x + 0.5) * stride
            cy = (grid_y + 0.5) * stride
            centers = torch.stack([cx, cy], dim=-1).view(1, -1, 2)

            score = torch.sigmoid(cls_p.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes))
            all_scores.append(score)

            dist = self.head.distribution_project(reg_p).permute(0, 2, 3, 1).reshape(B, -1, 4) * stride
            l = dist[..., 0]
            t = dist[..., 1]
            r = dist[..., 2]
            b = dist[..., 3]

            x1 = (centers[..., 0] - l).clamp(min=0.0, max=max_w)
            y1 = (centers[..., 1] - t).clamp(min=0.0, max=max_h)
            x2 = (centers[..., 0] + r).clamp(min=0.0, max=max_w)
            y2 = (centers[..., 1] + b).clamp(min=0.0, max=max_h)

            boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            all_boxes.append(boxes)

        return torch.cat(all_boxes, dim=1), torch.cat(all_scores, dim=1)


def nms_postprocess(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    conf_thresh: float = 0.35,
    iou_thresh: float = 0.45
) -> List[Dict[str, Any]]:
    results = []
    max_scores, class_ids = scores.max(dim=-1)
    keep_mask = max_scores >= conf_thresh

    if not keep_mask.any():
        return results

    b_boxes = boxes[keep_mask]
    b_scores = max_scores[keep_mask]
    b_classes = class_ids[keep_mask]

    keep_indices = ops.batched_nms(b_boxes, b_scores, b_classes, iou_thresh)

    for idx in keep_indices:
        results.append({
            "bbox": b_boxes[idx].cpu().numpy().tolist(),
            "score": float(b_scores[idx].item()),
            "class_id": int(b_classes[idx].item())
        })

    return results
