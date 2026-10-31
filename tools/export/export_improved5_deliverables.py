#!/usr/bin/env python3
"""
tools/export/export_improved5_deliverables.py
─────────────────────────────────────────────
Exports Stage A NanoDet (improved5) and Stage B RTMPose-13KP (improved5)
to ONNX (opset=12) across all 3 precision tiers: FP32, FP16, and INT8.
Verifies all 6 models load and run successfully in ONNX Runtime.
"""

import os
import sys
import shutil
import time
from pathlib import Path

import torch
import numpy as np
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.nanodet.model import NanoDetPlus
from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP


def export_nanodet_fp32(weights_path: str, output_path: str, img_size: int = 256):
    print(f"[*] Initializing NanoDet-Plus (improved5) for export...")
    model = NanoDetPlus(
        num_classes=1,
        input_size=(img_size, img_size),
        reg_max=16,
        strides=[8, 16, 32],
        neck_type="rep_pan",
        act_type="relu",
        pretrained=False
    )
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("ema_state", ckpt.get("model_state", ckpt.get("model", ckpt)))
    model.load_state_dict(state_dict)
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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Exporting NanoDet FP32 ONNX to {output_path} (opset 12)...")
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
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
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] NanoDet FP32 exported: {output_path} ({size_mb:.2f} MB)")


def export_rtmpose_fp32(weights_path: str, output_path: str, img_size: int = 256):
    print(f"[*] Initializing RTMPose-13KP Hand (improved5) for export...")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    backbone_name = ckpt.get("backbone_name", "cspnext")

    model = RTMPose13KP(
        num_keypoints=13,
        input_size=(img_size, img_size),
        simcc_split_ratio=2.0,
        backbone_name=backbone_name,
        pretrained=False,
        include_handedness=True
    )
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)
    output_names = ["simcc_x", "simcc_y", "vis_logits", "vis", "hand_logits", "hand"]
    dynamic_axes = {
        "images": {0: "batch"},
        "simcc_x": {0: "batch"},
        "simcc_y": {0: "batch"},
        "vis_logits": {0: "batch"},
        "vis": {0: "batch"},
        "hand_logits": {0: "batch"},
        "hand": {0: "batch"}
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Exporting RTMPose FP32 ONNX to {output_path} (opset 12)...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=12,
        dynamo=False
    )
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[+] RTMPose FP32 exported: {output_path} ({size_mb:.2f} MB)")


def convert_to_fp16(input_onnx: str, output_onnx: str):
    print(f"[*] Converting {input_onnx} to FP16...")
    m = onnx.load(input_onnx)
    m_fp16 = float16.convert_float_to_float16(m, keep_io_types=False)
    # Clear stale value_info to allow ONNX Runtime to infer clean types without Cast mismatch
    del m_fp16.graph.value_info[:]
    onnx.save(m_fp16, output_onnx)
    size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
    print(f"[+] FP16 model created: {output_onnx} ({size_mb:.2f} MB)")


def convert_to_int8(input_onnx: str, output_onnx: str):
    print(f"[*] Quantizing {input_onnx} to INT8 (QUInt8 dynamic)...")
    quantize_dynamic(
        model_input=input_onnx,
        model_output=output_onnx,
        weight_type=QuantType.QUInt8
    )
    size_mb = os.path.getsize(output_onnx) / (1024 * 1024)
    print(f"[+] INT8 model created: {output_onnx} ({size_mb:.2f} MB)")


def verify_model(onnx_path: str, expected_outputs: list):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_type = inp.type
    feed_data = np.random.randn(1, 3, 256, 256).astype(np.float16 if "float16" in in_type else np.float32)
    outs = sess.run(None, {inp.name: feed_data})
    out_names = [o.name for o in sess.get_outputs()]
    assert len(outs) == len(expected_outputs), f"Output count mismatch: got {len(outs)}, expected {len(expected_outputs)}"
    print(f"  [OK] Verified {Path(onnx_path).name}: Input={in_type}, Outputs={[o.name for o in sess.get_outputs()]}")


def main():
    print("=" * 80)
    print("  EXPORTING STAGE A & STAGE B IMPROVED5 DELIVERABLES (ONNX OPSET 12)")
    print("=" * 80)

    out_dir = Path("deliverables/qualcomm_experiments/models")
    out_dir.mkdir(parents=True, exist_ok=True)

    nanodet_ckpt = "checkpoints/nanodet_stage_a_improved5/best.pt"
    rtmpose_ckpt = "checkpoints/rtmpose_13kp_improved5/best_with_hand.pt"

    nano_fp32 = str(out_dir / "nanodet_stage_a_improved5_opset12.onnx")
    nano_fp16 = str(out_dir / "nanodet_stage_a_improved5_fp16.onnx")
    nano_int8 = str(out_dir / "nanodet_stage_a_improved5_int8.onnx")

    rtm_fp32 = str(out_dir / "rtmpose_13kp_hand_improved5_opset12.onnx")
    rtm_fp16 = str(out_dir / "rtmpose_13kp_hand_improved5_fp16.onnx")
    rtm_int8 = str(out_dir / "rtmpose_13kp_hand_improved5_int8.onnx")

    # 1. Export FP32 ONNX
    print("\n--- [Step 1: Export Master FP32 ONNX] ---")
    export_nanodet_fp32(nanodet_ckpt, nano_fp32)
    export_rtmpose_fp32(rtmpose_ckpt, rtm_fp32)

    # 2. Convert to FP16
    print("\n--- [Step 2: Convert to FP16] ---")
    convert_to_fp16(nano_fp32, nano_fp16)
    convert_to_fp16(rtm_fp32, rtm_fp16)

    # 3. Quantize to INT8
    print("\n--- [Step 3: Quantize to Dynamic INT8] ---")
    convert_to_int8(nano_fp32, nano_int8)
    convert_to_int8(rtm_fp32, rtm_int8)

    # 4. Verification in ONNX Runtime
    print("\n--- [Step 4: Verify ONNX Runtime Session Inference] ---")
    verify_model(nano_fp32, ["boxes", "scores"])
    verify_model(nano_fp16, ["boxes", "scores"])
    verify_model(nano_int8, ["boxes", "scores"])
    verify_model(rtm_fp32, ["simcc_x", "simcc_y", "vis_logits", "vis", "hand_logits", "hand"])
    verify_model(rtm_fp16, ["simcc_x", "simcc_y", "vis_logits", "vis", "hand_logits", "hand"])
    verify_model(rtm_int8, ["simcc_x", "simcc_y", "vis_logits", "vis", "hand_logits", "hand"])

    print("\n" + "=" * 80)
    print("  ALL 6 IMPROVED5 ONNX MODELS SUCCESSFULLY EXPORTED & VERIFIED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
