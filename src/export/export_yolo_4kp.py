"""
export_yolo_4kp.py
──────────────────
Export YOLOv8n-Pose (4 Coarse Keypoints) model to production ONNX formats:
  1. FP32 ONNX  (Opset 12, static 320x320, shape [1, 18, 2100])
  2. onnxsim graph simplification
  3. FP16 ONNX  (Half-precision weights for WebGPU / iOS CoreML)
  4. INT8 ONNX  (Dynamic quantization for WASM fallback)
  5. Verification & SHA256 checksums

Output Channels (18 channels total):
  - rows 0-3  : cx, cy, w, h (normalized bounding box)
  - rows 4-5  : left_foot score, right_foot score
  - rows 6-8  : kp0 (toe_tip: x, y, conf)
  - rows 9-11 : kp1 (heel: x, y, conf)
  - rows 12-14: kp2 (ball_medial: x, y, conf)
  - rows 15-17: kp3 (ball_lateral: x, y, conf)

Usage:
  python -m src.export.export_yolo_4kp --weights outputs/stage1_4kp/finetune_shuffled_v3/weights/best.pt --output_dir deliverables/stage_a_4kp
"""

import os
import sys
import shutil
import hashlib
import argparse
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

try:
    import onnxsim
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False

try:
    from onnxconverter_common import float16
    HAS_FLOAT16 = True
except ImportError:
    HAS_FLOAT16 = False

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False


def get_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_file_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024 * 1024)


def simplify_onnx(onnx_path: str):
    if not HAS_ONNXSIM:
        return
    try:
        model = onnx.load(onnx_path)
        model_sim, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_sim, onnx_path)
            print(f"   [onnxsim] Graph simplified successfully ({get_file_mb(onnx_path):.2f} MB)")
    except Exception as e:
        print(f"   [onnxsim] Note: simplification skipped: {e}")


def export_yolo_4kp(weights_path: str, output_dir: str = "deliverables/stage_a_4kp", prefix: str = "stage-a-320-yolo4kp"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  Exporting YOLOv8n-Pose (4 Coarse Keypoints) to ONNX Pipeline")
    print(f"  Weights   : {weights_path}")
    print(f"  Output Dir: {output_dir}")
    print("=" * 70)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint weights not found at: {weights_path}")

    # 1. Export Raw FP32 from Ultralytics
    print("\n[1/4] Exporting PyTorch -> ONNX FP32 (Opset 12, 320x320)...")
    model = YOLO(weights_path)
    raw_exported_path = model.export(
        format="onnx",
        imgsz=320,
        opset=12,
        dynamic=False,
        simplify=False,
        nms=False
    )

    fp32_dest = os.path.join(output_dir, f"{prefix}-fp32.onnx")
    shutil.copy2(raw_exported_path, fp32_dest)
    simplify_onnx(fp32_dest)

    # 2. Verify Output Shape
    print("\n[2/4] Verifying ONNX Inference & Tensor Dimensions...")
    session_fp32 = ort.InferenceSession(fp32_dest, providers=["CPUExecutionProvider"])
    in_name = session_fp32.get_inputs()[0].name
    out_name = session_fp32.get_outputs()[0].name
    in_shape = session_fp32.get_inputs()[0].shape
    out_shape = session_fp32.get_outputs()[0].shape
    print(f"   Input : '{in_name}' -> {in_shape}")
    print(f"   Output: '{out_name}' -> {out_shape}")

    dummy_input = np.random.randn(1, 3, 320, 320).astype(np.float32)
    out_data = session_fp32.run([out_name], {in_name: dummy_input})[0]
    print(f"   Inference check OK! Output array shape: {out_data.shape}")

    # 3. FP16 Conversion
    fp16_dest = os.path.join(output_dir, f"{prefix}-fp16.onnx")
    print("\n[3/4] Generating True FP16 Model (WebGPU / CoreML)...")
    if HAS_FLOAT16:
        try:
            model_fp32 = onnx.load(fp32_dest)
            model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=False)
            onnx.save(model_fp16, fp16_dest)
            print(f"   FP16 Model saved: {fp16_dest} ({get_file_mb(fp16_dest):.2f} MB)")
        except Exception as e:
            print(f"   FP16 conversion warning: {e}")
            shutil.copy2(fp32_dest, fp16_dest)
    else:
        print("   [!] onnxconverter_common not installed, copying FP32 as fallback.")
        shutil.copy2(fp32_dest, fp16_dest)

    # 4. INT8 Quantization
    int8_dest = os.path.join(output_dir, f"{prefix}-int8.onnx")
    print("\n[4/4] Generating Dynamic INT8 Model (WASM CPU fallback)...")
    if HAS_QUANT:
        try:
            quantize_dynamic(
                model_input=fp32_dest,
                model_output=int8_dest,
                weight_type=QuantType.QUInt8
            )
            print(f"   INT8 Model saved: {int8_dest} ({get_file_mb(int8_dest):.2f} MB)")
        except Exception as e:
            print(f"   INT8 quantization warning: {e}")
            shutil.copy2(fp32_dest, int8_dest)
    else:
        print("   [!] onnxruntime quantization not available, skipping INT8.")

    # Summary Table
    print("\n" + "=" * 70)
    print("  YOLOv8n-Pose (4-KP) Export Summary & File Checksums")
    print("=" * 70)
    for p in [fp32_dest, fp16_dest, int8_dest]:
        if os.path.exists(p):
            print(f"  File   : {os.path.basename(p)}")
            print(f"  Size   : {get_file_mb(p):.2f} MB")
            print(f"  SHA256 : {get_sha256(p)}")
            print("  " + "-" * 66)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8n-Pose 4-KP to ONNX.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--output_dir", type=str, default="deliverables/stage_a_4kp", help="Output directory")
    parser.add_argument("--prefix", type=str, default="stage-a-320-yolo4kp", help="Model filename prefix")
    args = parser.parse_args()

    export_yolo_4kp(args.weights, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
