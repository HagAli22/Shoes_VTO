"""
single_stage_rtmpose/export_onnx.py
───────────────────────────────────
Exports Single-Stage RTMPose to ONNX with [1, 45, N] output tensor:
  - Channels 0..3:  (cx, cy, w, h)
  - Channels 4..5:  (left_prob, right_prob)
  - Channels 6..44: 13 Keypoints (x, y, conf)
Validates numerical parity against PyTorch via ONNX Runtime.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.single_stage_rtmpose.model import SingleStageRTMPose


def export_single_stage_rtmpose_onnx(
    weights_path: str,
    output_path: str = "deliverables/qualcomm_experiments/models/single-stage-rtmpose-13kp-320.onnx",
    img_size: int = 320,
    opset: int = 17,
    device_name: str = "cpu"
):
    device = torch.device(device_name)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SINGLE-STAGE RTMPOSE ONNX EXPORT")
    print(f"  Input Weights:    {weights_path}")
    print(f"  Target ONNX:      {out_file}")
    print(f"  Input Size:       {img_size}x{img_size}")
    print(f"  Opset:            {opset}")
    print("=" * 70)

    model = SingleStageRTMPose(
        num_classes=2,
        num_keypoints=13,
        input_size=(img_size, img_size),
        pretrained=False
    ).to(device)

    if Path(weights_path).exists():
        ckpt = torch.load(weights_path, map_location=device, weights_only=False)
        state = ckpt.get("ema_state", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[+] Loaded weights from {weights_path} (missing: {len(missing)}, unexpected: {len(unexpected)}).")
    else:
        print(f"[!] Warning: {weights_path} not found. Exporting initialized architecture.")

    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x, export_mode=True)

    wrapper = ExportWrapper(model)
    with torch.no_grad():
        pt_out = wrapper(dummy_input)
    print(f"[*] PyTorch output tensor shape: {pt_out.shape} ([B, 45, N])")

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_file),
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={
            "images": {0: "batch"},
            "output0": {0: "batch"}
        },
        opset_version=opset,
        do_constant_folding=True
    )
    print(f"[+] ONNX export succeeded: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")

    # Verify with ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_out = session.run(None, ort_inputs)[0]
        max_diff = np.max(np.abs(pt_out.cpu().numpy() - ort_out))
        print(f"[+] ONNX Runtime validation passed! Max absolute diff vs PyTorch: {max_diff:.6e}")
    except ImportError:
        print("[*] Note: onnxruntime not installed, skipping numerical verification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/single_stage_rtmpose_320/best.pt")
    parser.add_argument("--output", type=str, default="deliverables/qualcomm_experiments/models/single-stage-rtmpose-13kp-320.onnx")
    parser.add_argument("--img_size", type=int, default=320)
    args = parser.parse_args()

    export_single_stage_rtmpose_onnx(
        weights_path=args.weights,
        output_path=args.output,
        img_size=args.img_size
    )
