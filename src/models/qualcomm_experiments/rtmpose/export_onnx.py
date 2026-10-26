"""
rtmpose/export_onnx.py
───────────────────────
Exports Qualcomm RTMPose-13KP to ONNX for mobile / NPU deployment.
Baked with input standardization and dynamic batching.
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP


def export_rtmpose_onnx(
    weights_path: str = "checkpoints/rtmpose_13kp_official_coco/best.pt",
    output_onnx: str = "deliverables/qualcomm_experiments/models/rtmpose-13kp-coco-256.onnx",
    img_size: int = 256
):
    out_file = Path(output_onnx)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading model from {weights_path}...")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    backbone_name = ckpt.get("backbone_name", "cspnext")

    model = RTMPose13KP(
        num_keypoints=13,
        input_size=(img_size, img_size),
        simcc_split_ratio=2.0,
        backbone_name=backbone_name,
        pretrained=False
    )
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    print(f"[*] Exporting to ONNX: {out_file}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        input_names=["images"],
        output_names=["simcc_x", "simcc_y", "vis_logits", "vis"],
        dynamic_axes={
            "images": {0: "batch"},
            "simcc_x": {0: "batch"},
            "simcc_y": {0: "batch"},
            "vis_logits": {0: "batch"},
            "vis": {0: "batch"}
        },
        opset_version=12,
        dynamo=False
    )

    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully exported RTMPose-13KP to ONNX: {out_file} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    export_rtmpose_onnx()
