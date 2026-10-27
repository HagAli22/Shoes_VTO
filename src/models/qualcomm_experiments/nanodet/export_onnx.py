"""
nanodet/export_onnx.py
──────────────────────
Exports NanoDet-Plus to ONNX for mobile / NPU deployment (Apache-2.0).
Bakes in ImageNet/COCO standardization and decodes DFL into bounding boxes [B, N, 4].
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.nanodet.model import NanoDetPlus, DEFAULT_CKPT_PATH


def export_nanodet_onnx(
    weights_path: str = "checkpoints/nanodet_foot/best.pt",
    output_onnx: str = "deliverables/qualcomm_experiments/models/nanodet-plus-foot-256.onnx",
    img_size: int = 256,
    num_classes: int = 1
):
    out_file = Path(output_onnx)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Initializing NanoDet-Plus model for export...")
    if Path(weights_path).exists():
        print(f"[*] Loading weights from {weights_path}...")
        model = NanoDetPlus(
            num_classes=num_classes,
            input_size=(img_size, img_size),
            pretrained=False
        )
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print(f"[*] Loading official COCO pretrained weights for architecture export...")
        model = NanoDetPlus(
            num_classes=num_classes,
            input_size=(img_size, img_size),
            pretrained=True
        )

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
    print(f"[+] Successfully exported NanoDet-Plus to ONNX: {out_file} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    export_nanodet_onnx()
