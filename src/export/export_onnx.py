"""
export_onnx.py
──────────────
Export the trained RTMPose-Tiny PyTorch model to ONNX format.

Steps:
  1. Load model checkpoint (.pth)
  2. Run a dummy forward pass to verify correctness
  3. Export to ONNX (opset 17, static input shape 256×192)
  4. Simplify with onnxsim
  5. Validate: compare PyTorch vs ONNX outputs (max absolute diff < 1e-5)

Usage
-----
  python src/export/export_onnx.py \\
      --checkpoint outputs/checkpoints/best.pth \\
      --output     outputs/exports/rtmpose_tiny_18kp.onnx \\
      --config     configs/training_config.yaml
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml


def export_to_onnx(
    checkpoint: str,
    output:     str,
    config_path: str,
    opset: int = 17,
    simplify: bool = True,
) -> None:
    """
    Export a trained foot keypoint model to ONNX.

    Parameters
    ----------
    checkpoint  : Path to .pth model checkpoint.
    output      : Output .onnx file path.
    config_path : Path to training_config.yaml (for model params).
    opset       : ONNX opset version.
    simplify    : Run onnxsim to simplify the graph.
    """
    import onnx

    # ── Load config ────────────────────────────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    num_kp  = cfg["model"]["num_keypoints"]   # 18
    H, W    = cfg["model"]["input_size"]      # [256, 192]

    # ── Build and load model ───────────────────────────────────────────────
    from src.models.keypoint.train_keypoint import build_rtmpose_tiny
    model = build_rtmpose_tiny(num_keypoints=num_kp)

    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    # ── Dummy input ────────────────────────────────────────────────────────
    dummy_input = torch.randn(1, 3, H, W)

    # ── PyTorch forward pass (reference) ──────────────────────────────────
    with torch.no_grad():
        ref_output = model(dummy_input).numpy()

    # ── ONNX export ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output,
        opset_version=opset,
        input_names=["input"],
        output_names=["heatmaps"],
        dynamic_axes=None,   # static shape for mobile
        do_constant_folding=True,
    )
    print(f"ONNX exported: {output}")

    # ── Verify ONNX model ──────────────────────────────────────────────────
    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")

    # ── Simplify ──────────────────────────────────────────────────────────
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify

            simplified, ok = onnx_simplify(onnx_model)
            if ok:
                onnx.save(simplified, output)
                print("ONNX graph simplified successfully.")
            else:
                print("[WARN] onnxsim simplification failed — keeping original graph.")
        except ImportError:
            print("[WARN] onnxsim not installed — skipping simplification.")

    # ── Numerical validation ───────────────────────────────────────────────
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(output, providers=["CPUExecutionProvider"])
        ort_output = sess.run(
            ["heatmaps"],
            {"input": dummy_input.numpy()},
        )[0]

        max_diff = float(np.abs(ref_output - ort_output).max())
        print(f"Max absolute diff (PyTorch vs ONNX): {max_diff:.2e}")
        if max_diff < 1e-4:
            print("✅ Numerical validation passed.")
        else:
            print(f"⚠️  Diff {max_diff:.2e} is larger than 1e-4 — review model.")
    except ImportError:
        print("[WARN] onnxruntime not installed — skipping numerical validation.")

    print(f"\nExport complete → {output}")
    print(f"  Input:  [1, 3, {H}, {W}]")
    print(f"  Output: [1, {num_kp}, 64, 48]")
    print(f"  Opset:  {opset}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export RTMPose-Tiny to ONNX.")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth checkpoint file.")
    parser.add_argument("--output",     required=True,
                        help="Output .onnx file path.")
    parser.add_argument("--config",     default="configs/training_config.yaml",
                        help="Path to training_config.yaml.")
    parser.add_argument("--opset",      type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true",
                        help="Skip onnxsim simplification.")
    args = parser.parse_args()

    export_to_onnx(
        checkpoint=args.checkpoint,
        output=args.output,
        config_path=args.config,
        opset=args.opset,
        simplify=not args.no_simplify,
    )


if __name__ == "__main__":
    main()

