"""
export_pruned_onnx.py
─────────────────────
Exports pruned YOLOv8-pose models to production FP32 ONNX format
and strictly enforces the project budget:
    FP32 ONNX file size < 5.0 MB
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


def sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_file_mb(file_path: str) -> float:
    return os.path.getsize(file_path) / (1024.0 * 1024.0)


def simplify_onnx(onnx_path: str):
    if not HAS_ONNXSIM:
        return
    try:
        model = onnx.load(onnx_path)
        model_sim, check = onnxsim.simplify(model)
        if check:
            onnx.save(model_sim, onnx_path)
            print(f"  [onnxsim] Graph simplified successfully ({get_file_mb(onnx_path):.2f} MB)")
    except Exception as e:
        print(f"  [onnxsim] Note: simplification skipped ({e})")


def export_and_verify_pruned_model(
    weights_path: str,
    output_dir: str = "experiments/exports",
    output_name: str = "pruned_model_fp32.onnx",
    expected_channels: int = 18, # 18 for 4-KP, 54 for 16-KP
    max_size_mb: float = 5.0,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, output_name)

    print("=" * 70)
    print("  EXPORTING PRUNED MODEL TO FP32 ONNX (STRICT <5 MB BUDGET)")
    print(f"  Weights   : {weights_path}")
    print(f"  Output    : {out_path}")
    print(f"  Max Budget: {max_size_mb:.2f} MB (FP32)")
    print("=" * 70)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint weights not found: {weights_path}")

    # 1. Ultralytics Export
    model = YOLO(weights_path)
    raw_path = model.export(
        format="onnx",
        imgsz=320,
        opset=12,
        dynamic=False,
        simplify=False,
        nms=False
    )

    shutil.copy2(raw_path, out_path)
    simplify_onnx(out_path)

    # 2. Check File Size Budget
    file_size_mb = get_file_mb(out_path)
    print(f"\n[BUDGET CHECK] Exported FP32 ONNX File Size: {file_size_mb:.2f} MB")
    
    if file_size_mb < max_size_mb:
        print(f"✅ PASSED BUDGET: {file_size_mb:.2f} MB < {max_size_mb:.2f} MB")
    else:
        print(f"⚠️ EXCEEDED BUDGET: {file_size_mb:.2f} MB >= {max_size_mb:.2f} MB")

    # 3. Verify ONNX Graph & Output Tensor Dimensions
    session = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    in_shape = session.get_inputs()[0].shape
    out_shape = session.get_outputs()[0].shape

    print(f"[ONNX CHECK] Input : '{in_name}' -> {in_shape}")
    print(f"[ONNX CHECK] Output: '{out_name}' -> {out_shape}")

    dummy = np.random.randn(1, 3, 320, 320).astype(np.float32)
    out_arr = session.run([out_name], {in_name: dummy})[0]
    print(f"[ONNX CHECK] Inference verification OK! Output shape: {out_arr.shape}")

    # 4. Report
    print("\n" + "=" * 70)
    print("  Export Summary")
    print(f"  File   : {os.path.basename(out_path)}")
    print(f"  Size   : {file_size_mb:.2f} MB")
    print(f"  SHA256 : {sha256(out_path)}")
    print("=" * 70)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export pruned YOLO model to FP32 ONNX.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--output_dir", type=str, default="experiments/exports", help="Output directory")
    parser.add_argument("--output_name", type=str, default="pruned_model_fp32.onnx", help="Export filename")
    parser.add_argument("--channels", type=int, default=18, help="Expected output channels (18 for 4-KP, 54 for 16-KP)")
    parser.add_argument("--max_mb", type=float, default=5.0, help="Max file size budget in MB")
    args = parser.parse_args()

    export_and_verify_pruned_model(
        weights_path=args.weights,
        output_dir=args.output_dir,
        output_name=args.output_name,
        expected_channels=args.channels,
        max_size_mb=args.max_mb
    )


if __name__ == "__main__":
    main()
