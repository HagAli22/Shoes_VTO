"""
export_blazefoot.py
───────────────────
Export, Quantization, and Size Verification for Google BlazeFoot Detector.

Produces:
  1. FP32 ONNX (Opset 12)  -> Target <= 2.5 MB
  2. True FP16 (WebGPU)     -> Target <= 1.3 MB
  3. Dynamic INT8 (WASM)    -> Target <= 0.8 MB

Outputs match Path A contract [1, 18, 1050].

Usage:
    python -m src.export.export_blazefoot --weights outputs/stage_a/blazefoot_smoke_test/weights/best.pt
"""

import os
import sys
import argparse
import hashlib
import numpy as np

import torch
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

from src.models.blazefoot.blazefoot_detector import BlazeFoot, BlazeFootPathAExport


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def export_blazefoot(
    weights_path: str,
    output_dir: str = "shoes-vto-ai-v2/models",
    prefix: str = "stage-a-320-blazefoot",
    img_size: int = 320,
    opset: int = 12
):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  [>] Exporting Google BlazeFoot Detector to ONNX & Quantizing")
    print(f"  Weights Path : {weights_path}")
    print(f"  Output Dir   : {output_dir}")
    print("=" * 70)

    # 1. Load PyTorch Checkpoint
    checkpoint = torch.load(weights_path, map_location="cpu")
    model = BlazeFoot(num_classes=2, num_keypoints=16)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    sd = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # Deduce num_keypoints from head_p3.kpt.3.weight shape: (num_anchors * kp * 3) -> 2 * kp * 3 = 6 * kp
    kpt_out_channels = sd["head_p3.kpt.3.weight"].shape[0]
    num_kp = kpt_out_channels // 6 # 24 // 6 = 4 keypoints

    model = BlazeFoot(num_classes=2, num_keypoints=num_kp)
    model.load_state_dict(sd)
    model.eval()

    export_model = BlazeFootPathAExport(model)
    export_model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    # 2. Export FP32 ONNX
    fp32_path = os.path.join(output_dir, f"{prefix}-fp32.onnx")
    torch.onnx.export(
        export_model,
        dummy_input,
        fp32_path,
        opset_version=opset,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=None
    )
    m_fp32 = onnx.load(fp32_path)
    onnx.checker.check_model(m_fp32)
    fp32_size = os.path.getsize(fp32_path) / 1e6
    print(f"[1] FP32 Exported: {fp32_path} ({fp32_size:.2f} MB)")

    # 3. Convert to FP16
    fp16_path = os.path.join(output_dir, f"{prefix}-fp16.onnx")
    m_fp16 = float16.convert_float_to_float16(m_fp32, keep_io_types=True)
    del m_fp16.graph.value_info[:]
    m_fp16 = onnx.shape_inference.infer_shapes(m_fp16)
    onnx.save(m_fp16, fp16_path)
    fp16_size = os.path.getsize(fp16_path) / 1e6
    print(f"[2] FP16 Exported: {fp16_path} ({fp16_size:.2f} MB)")

    # 4. Quantize to Dynamic INT8
    int8_path = os.path.join(output_dir, f"{prefix}-int8.onnx")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8
    )
    int8_size = os.path.getsize(int8_path) / 1e6
    print(f"[3] INT8 Exported: {int8_path} ({int8_size:.2f} MB)")

    # 5. Verification
    print("\n" + "=" * 70)
    print("  BLAZEFOOT EXPORT & SIZE BUDGET VERIFICATION")
    print("=" * 70)
    models = [
        ("FP32 (Desktop/WebGL)", fp32_path, fp32_size, 2.5),
        ("FP16 (Mobile WebGPU)", fp16_path, fp16_size, 1.3),
        ("INT8 (Safari WASM)  ", int8_path, int8_size, 0.8),
    ]

    for label, path, size, budget in models:
        h = sha256_file(path)[:16]
        status = "PASSED" if size <= budget else "EXCEEDED"
        print(f"  {label:<22}: {size:5.2f} MB (Budget: <= {budget:.1f} MB) -> {status}  [SHA: {h}...]")

    # Run inference test
    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": dummy_input.numpy()})[0]
    print(f"\n  Inference Output Shape: {out.shape} (Format: [1, 18, 1050])")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Export BlazeFoot to ONNX.")
    parser.add_argument("--weights", default="outputs/stage_a/blazefoot_smoke_test/weights/best.pt")
    parser.add_argument("--output_dir", default="shoes-vto-ai-v2/models")
    parser.add_argument("--prefix", default="stage-a-320-blazefoot")
    args = parser.parse_args()

    export_blazefoot(args.weights, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()

