"""
nanodet/export_onnx.py
──────────────────────
Exports NanoDet-Plus to ONNX for mobile / NPU deployment (Apache-2.0).
Bakes in ImageNet/COCO standardization and decodes DFL into bounding boxes [B, N, 4].
"""

import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.nanodet.model import NanoDetPlus, DEFAULT_CKPT_PATH


def export_nanodet_onnx(
    weights_path: str = "checkpoints/nanodet_stage_a_improved/best.pt",
    output_onnx: str = "deliverables/qualcomm_experiments/models/nanodet-stage-a-improved-256.onnx",
    img_size: int = 256,
    num_classes: int = 1,
    reg_max: int = 16,
    neck_type: str = "rep_pan",
    act_type: str = "relu"
):
    out_file = Path(output_onnx)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    strides = [8, 16, 32] if neck_type == "rep_pan" else [8, 16, 32, 64]

    print(f"[*] Initializing NanoDet Stage A model for export...")
    print(f"    - Resolution: {img_size}x{img_size} | Neck: {neck_type} | Act: {act_type} | reg_max: {reg_max}")

    model = NanoDetPlus(
        num_classes=num_classes,
        input_size=(img_size, img_size),
        reg_max=reg_max,
        strides=strides,
        neck_type=neck_type,
        act_type=act_type,
        pretrained=False
    )

    if Path(weights_path).exists():
        print(f"[*] Loading weights from {weights_path}...")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("ema_state", ckpt.get("model_state", ckpt.get("model", ckpt)))
        model.load_state_dict(state_dict)
    else:
        print(f"[*] Pretrained weights file '{weights_path}' not found; exporting initialized architecture...")

    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    class NanoDetExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            out = self.m(x, export_mode=True)
            return out["boxes"], out["scores"]

    export_model = NanoDetExportWrapper(model)

    print(f"[*] Exporting to ONNX: {out_file}...")
    torch.onnx.export(
        export_model,
        dummy_input,
        str(out_file),
        input_names=["images"],
        output_names=["boxes", "scores"],
        dynamic_axes={
            "images": {0: "batch"},
            "boxes": {0: "batch"},
            "scores": {0: "batch"}
        },
        opset_version=12,
        dynamo=False
    )

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully exported NanoDet Stage A to ONNX: {out_file} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/nanodet_stage_a_improved/best.pt")
    parser.add_argument("--output", type=str, default="deliverables/qualcomm_experiments/models/nanodet-stage-a-improved-256.onnx")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=1)
    parser.add_argument("--reg_max", type=int, default=16)
    parser.add_argument("--neck", type=str, default="rep_pan")
    parser.add_argument("--act", type=str, default="relu")
    args = parser.parse_args()

    export_nanodet_onnx(
        weights_path=args.weights,
        output_onnx=args.output,
        img_size=args.img_size,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        neck_type=args.neck,
        act_type=args.act
    )

