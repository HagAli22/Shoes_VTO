"""
export_shuffled_v3.py
─────────────────────
Export run_shuffled_v3/weights/best.pt to ONNX FP32 / FP16 / INT8
with every size-reduction step applied:
  1. Export raw ONNX (Opset 12, static 320×320)
  2. onnxsim simplification  → removes redundant ops, folds constants
  3. Path-A channel slicing  → [1,54,2100] → [1,18,2100]
  4. True FP16 conversion    → half-precision weights (WebGPU)
  5. INT8 dynamic quantization → WASM fallback
  6. Size + SHA-256 report

Usage:
    python -m src.export.export_shuffled_v3
    python -m src.export.export_shuffled_v3 --weights outputs/stage_a/run_shuffled_v3/weights/best.pt
"""

import argparse
import hashlib
import os
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType
from ultralytics import YOLO

try:
    import onnxsim
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False
    print("[WARN] onnxsim not found — skipping graph simplification step.")


# ─── Path-A keypoint channel indices ────────────────────────────────────────
# Raw YOLOv8 pose output: [1, 4+2+16*3, 2100] = [1, 54, 2100]
#   rows 0–3:   cx, cy, w, h
#   rows 4–5:   left_foot score, right_foot score
#   rows 6–53:  kp0_x, kp0_y, kp0_v, kp1_x, … (16 keypoints × 3)
# Path-A selects 4 coarse keypoints: kp0 (toe_tip), kp2 (heel_back),
#   kp4 (ball_medial), kp5 (ball_lateral)
PATH_A_INDICES = [0, 1, 2, 3, 4, 5,          # bbox + class scores
                  6, 7, 8,                     # kp0 toe_tip
                  12, 13, 14,                  # kp2 heel_back
                  18, 19, 20,                  # kp4 ball_medial
                  21, 22, 23]                  # kp5 ball_lateral


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def file_mb(path: str) -> float:
    return os.path.getsize(path) / 1_000_000


def simplify(onnx_path: str) -> str:
    """Run onnxsim to fold constants and remove dead nodes. Returns same path."""
    if not HAS_ONNXSIM:
        return onnx_path
    print("  [onnxsim] Simplifying graph …")
    model = onnx.load(onnx_path)
    model_sim, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(model_sim, onnx_path)
        print(f"  [onnxsim] OK → {file_mb(onnx_path):.2f} MB")
    else:
        print("  [onnxsim] simplification failed, keeping original.")
    return onnx_path


def slice_path_a(raw_onnx_path: str, out_path: str) -> str:
    """
    Insert a Gather node that selects PATH_A_INDICES along axis=1,
    reducing [1, 54, 2100] → [1, 18, 2100].
    """
    print(f"  [slice] Applying Path-A channel selection ({len(PATH_A_INDICES)} channels)…")
    model = onnx.load(raw_onnx_path)

    # Clear shape info that may conflict after insertion
    del model.graph.value_info[:]

    # Find the final output name
    original_output_name = model.graph.output[0].name
    gather_out_name = original_output_name + "_path_a"

    # Indices constant
    indices_init = numpy_helper.from_array(
        np.array(PATH_A_INDICES, dtype=np.int64), name="path_a_indices"
    )
    model.graph.initializer.append(indices_init)

    # Gather node: output[gather] = output[original][:, PATH_A_INDICES, :]
    gather_node = helper.make_node(
        "Gather",
        inputs=[original_output_name, "path_a_indices"],
        outputs=[gather_out_name],
        axis=1,
    )
    model.graph.node.append(gather_node)

    # Transpose: Gather reorders dims → [1, 18, 2100] via Transpose
    # Actually Gather on axis=1 gives [1, 18, 2100] directly — no Transpose needed.
    # Update the graph output to point to the new node
    model.graph.output[0].CopyFrom(
        helper.make_tensor_value_info(gather_out_name, TensorProto.FLOAT, [1, 18, 2100])
    )

    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, out_path)
    print(f"  [slice] Saved → {out_path}  ({file_mb(out_path):.2f} MB)")
    return out_path


def to_fp16(fp32_path: str, fp16_path: str) -> str:
    """True FP16 conversion with shape inference fix for Resize nodes."""
    print("  [fp16] Converting weights to float16 …")
    model_fp32 = onnx.load(fp32_path)
    # Clear value_info to avoid Resize node shape conflicts
    del model_fp32.graph.value_info[:]
    model_fp16 = float16.convert_float_to_float16(
        model_fp32, keep_io_types=True, disable_shape_infer=False
    )
    model_fp16 = onnx.shape_inference.infer_shapes(model_fp16)
    onnx.save(model_fp16, fp16_path)

    # Verify: count float16 initializers
    n_fp16 = sum(1 for init in model_fp16.graph.initializer if init.data_type == 10)
    print(f"  [fp16] {n_fp16} FP16 initializers → {fp16_path}  ({file_mb(fp16_path):.2f} MB)")
    return fp16_path


def to_int8(fp32_path: str, int8_path: str) -> str:
    """Dynamic INT8 quantization (QUInt8 — WASM compatible)."""
    print("  [int8] Dynamic INT8 quantization …")
    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QUInt8,
    )
    print(f"  [int8] Saved → {int8_path}  ({file_mb(int8_path):.2f} MB)")
    return int8_path


def verify_outputs(fp32_path: str, fp16_path: str, int8_path: str):
    """Run a dummy inference and print output shapes + basic accuracy delta."""
    print("\n  [verify] Running inference verification …")
    dummy = np.random.rand(1, 3, 320, 320).astype(np.float32)

    def infer(path, input_type=np.float32):
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        out = sess.run(None, {inp: dummy.astype(input_type)})[0]
        return out

    out_fp32 = infer(fp32_path)
    out_fp16 = infer(fp16_path)
    out_int8 = infer(int8_path)

    print(f"  FP32 output shape: {out_fp32.shape}")
    print(f"  FP16 output shape: {out_fp16.shape}")
    print(f"  INT8 output shape: {out_int8.shape}")
    print(f"  FP16 vs FP32 max delta: {np.abs(out_fp32 - out_fp16).max():.4f}")
    print(f"  INT8 vs FP32 max delta: {np.abs(out_fp32 - out_int8).max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Export run_shuffled_v3 to optimized ONNX.")
    parser.add_argument("--weights", default="outputs/stage_a/run_shuffled_v3/weights/best.pt")
    parser.add_argument("--output_dir", default="shoes-vto-ai-v2/models")
    parser.add_argument("--prefix", default="stage-a-320-shuffled",
                        help="Filename prefix, e.g. 'stage-a-320-shuffled'")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    weights = args.weights
    out_dir = args.output_dir
    prefix  = args.prefix
    os.makedirs(out_dir, exist_ok=True)

    # Intermediate raw ONNX (full 54-channel, will be deleted)
    raw_onnx = os.path.join(out_dir, f"{prefix}-raw.onnx")
    fp32_out = os.path.join(out_dir, f"{prefix}-fp32.onnx")
    fp16_out = os.path.join(out_dir, f"{prefix}-fp16.onnx")
    int8_out = os.path.join(out_dir, f"{prefix}-int8.onnx")

    print("\n" + "=" * 65)
    print(f"  Export Pipeline: {weights}")
    print(f"  Output dir:      {out_dir}/  (prefix: {prefix})")
    print("=" * 65)

    # ── Step 1: Export raw ONNX ───────────────────────────────────────────
    print(f"\n[1/5] Exporting PyTorch → raw ONNX (opset {args.opset}) …")
    model = YOLO(weights)
    exported = model.export(
        format="onnx",
        opset=args.opset,
        imgsz=args.imgsz,
        simplify=False,     # We apply onnxsim ourselves for full control
        dynamic=False,
        half=False,
    )
    # Ultralytics saves next to the .pt file — move it
    expected_raw = weights.replace(".pt", ".onnx")
    if os.path.exists(expected_raw) and expected_raw != raw_onnx:
        shutil.move(expected_raw, raw_onnx)
    elif not os.path.exists(raw_onnx):
        # Try finding it
        candidate = os.path.join(os.path.dirname(weights), "best.onnx")
        if os.path.exists(candidate):
            shutil.move(candidate, raw_onnx)
    print(f"  Raw ONNX: {raw_onnx}  ({file_mb(raw_onnx):.2f} MB)")

    # ── Step 2: onnxsim ──────────────────────────────────────────────────
    print("\n[2/5] Graph simplification (onnxsim) …")
    simplify(raw_onnx)

    # ── Step 3: Path-A slice → FP32 ──────────────────────────────────────
    print("\n[3/5] Path-A channel slicing → FP32 …")
    slice_path_a(raw_onnx, fp32_out)
    os.remove(raw_onnx)     # raw no longer needed

    # ── Step 4: FP16 ─────────────────────────────────────────────────────
    print("\n[4/5] FP16 conversion …")
    to_fp16(fp32_out, fp16_out)

    # ── Step 5: INT8 ─────────────────────────────────────────────────────
    print("\n[5/5] INT8 dynamic quantization …")
    to_int8(fp32_out, int8_out)

    # ── Verify ────────────────────────────────────────────────────────────
    verify_outputs(fp32_out, fp16_out, int8_out)

    # ── Size Report ───────────────────────────────────────────────────────
    models = [(fp32_out, "FP32"), (fp16_out, "FP16"), (int8_out, "INT8")]
    budgets = {"FP32": 14.0, "FP16": 7.0, "INT8": 4.0}  # standard (non-compact) budgets

    print("\n" + "=" * 65)
    print("  EXPORT COMPLETE — Size Report")
    print("=" * 65)
    print(f"  {'File':<40} {'Size':>8}   {'SHA-256[:16]'}")
    print("  " + "-" * 62)
    for path, prec in models:
        size_mb = file_mb(path)
        h = sha256(path)[:16]
        budget = budgets[prec]
        flag = "OK" if size_mb <= budget else "OVER"
        print(f"  {os.path.basename(path):<40} {size_mb:>7.2f}MB  [{flag}]  {h}…")
    print("=" * 65)

    # Also mirror to deliverables/
    mirror = "deliverables/models"
    os.makedirs(mirror, exist_ok=True)
    for path, _ in models:
        dst = os.path.join(mirror, os.path.basename(path))
        shutil.copy2(path, dst)
        print(f"  Mirrored → {dst}")


if __name__ == "__main__":
    main()
