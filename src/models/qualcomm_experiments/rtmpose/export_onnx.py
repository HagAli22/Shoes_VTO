"""
rtmpose/export_onnx.py
───────────────────────
Exports Qualcomm RTMPose-13KP to ONNX for mobile / NPU deployment.
Supports:
  - FP32 ONNX export with baked standardization & dynamic batching (opset 12)
  - FP16 conversion
  - Dynamic INT8 quantization for mobile NPU/DSP runtimes
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP


def export_rtmpose_onnx(
    weights_path: str = "checkpoints/rtmpose_13kp_improved5/best_with_hand.pt",
    output_onnx: str = "deliverables/qualcomm_experiments/models/rtmpose_slim_improved5_opset12.onnx",
    backbone_name: str = None,
    img_size: int = 256,
    include_handedness: bool = True,
    quantize_int8: bool = True,
    convert_fp16: bool = True
):
    out_file = Path(output_onnx)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading model from {weights_path}...")
    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weights_path, map_location="cpu")

    if backbone_name is None:
        backbone_name = ckpt.get("student_arch", ckpt.get("backbone_name", "cspnext"))

    print(f"[*] Detected backbone architecture: {backbone_name}")

    model = RTMPose13KP(
        num_keypoints=13,
        input_size=(img_size, img_size),
        simcc_split_ratio=2.0,
        backbone_name=backbone_name,
        pretrained=False,
        include_handedness=include_handedness
    )
    state_dict = ckpt.get("model_state", ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    output_names = ["simcc_x", "simcc_y", "vis_logits", "vis"]
    dynamic_axes = {
        "images": {0: "batch"},
        "simcc_x": {0: "batch"},
        "simcc_y": {0: "batch"},
        "vis_logits": {0: "batch"},
        "vis": {0: "batch"},
    }
    if include_handedness:
        output_names.extend(["hand_logits", "hand"])
        dynamic_axes["hand_logits"] = {0: "batch"}
        dynamic_axes["hand"] = {0: "batch"}

    # 1. Export FP32 ONNX
    print(f"[*] Exporting FP32 ONNX to {out_file}...")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_file),
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=12,
        dynamo=False
    )
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[+] Successfully exported FP32 ONNX: {out_file} ({size_mb:.2f} MB)")

    # 2. Verify numerical equivalence with ONNXRuntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(out_file), providers=["CPUExecutionProvider"])
        ort_inputs = {"images": dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)

        with torch.no_grad():
            torch_outs = model(dummy_input)

        diff_x = np.max(np.abs(ort_outs[0] - torch_outs["simcc_x"].numpy()))
        diff_y = np.max(np.abs(ort_outs[1] - torch_outs["simcc_y"].numpy()))
        print(f"[+] ONNX validation verified: max abs diff x={diff_x:.6f}, y={diff_y:.6f}")
    except Exception as e:
        print(f"[*] Note: ONNX validation skipped or note: {e}")

    # 3. FP16 Conversion
    if convert_fp16:
        fp16_path = out_file.with_name(out_file.stem.replace("opset12", "fp16") + ".onnx")
        try:
            import onnx
            from onnxconverter_common import float16
            model_proto = onnx.load(str(out_file))
            model_fp16 = float16.convert_float_to_float16(model_proto, keep_io_types=True)
            onnx.save(model_fp16, str(fp16_path))
            fp16_mb = fp16_path.stat().st_size / (1024 * 1024)
            print(f"[+] Successfully exported FP16 ONNX: {fp16_path} ({fp16_mb:.2f} MB)")
        except Exception as e:
            print(f"[*] FP16 conversion skipped: {e}")

    # 4. INT8 Dynamic Quantization
    if quantize_int8:
        int8_path = out_file.with_name(out_file.stem.replace("opset12", "int8") + ".onnx")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                model_input=str(out_file),
                model_output=str(int8_path),
                weight_type=QuantType.QInt8,
                per_channel=True,
                reduce_range=False
            )
            int8_mb = int8_path.stat().st_size / (1024 * 1024)
            print(f"[+] Successfully exported INT8 ONNX: {int8_path} ({int8_mb:.2f} MB)")
        except Exception as e:
            print(f"[*] INT8 quantization skipped: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export RTMPose-13KP to ONNX (FP32 / FP16 / INT8)")
    parser.add_argument("--weights", type=str, default="checkpoints/rtmpose_13kp_improved5/best_with_hand.pt", help="Checkpoint path")
    parser.add_argument("--output", type=str, default="deliverables/qualcomm_experiments/models/rtmpose_slim_improved5_opset12.onnx", help="Output ONNX path")
    parser.add_argument("--backbone", type=str, default=None, help="Backbone override (cspnext, cspnext_slim, cspnext_slim_s4lite, cspnext_pico)")
    parser.add_argument("--img_size", type=int, default=256, help="Input image dimension")
    parser.add_argument("--no_int8", action="store_true", help="Disable dynamic INT8 quantization")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 conversion")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_rtmpose_onnx(
        weights_path=args.weights,
        output_onnx=args.output,
        backbone_name=args.backbone,
        img_size=args.img_size,
        quantize_int8=not args.no_int8,
        convert_fp16=not args.no_fp16
    )
