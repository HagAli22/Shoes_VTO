import os
import sys
import shutil
import hashlib
import torch
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

SRC_PT = "deliverables/stage_a_256_13kp/models/stage-a-256-13kp-v7.pt"
DST_MODELS = "deliverables/stage_a_256_13kp/models"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    os.makedirs(DST_MODELS, exist_ok=True)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    
    print("=" * 70)
    print("BUILDING STAGE A 256 MOBILE DELIVERABLES (OPSET 12)")
    print("=" * 70)

    # 1. Real FP16 for WebGPU (Native half on GPU)
    print("\n[1] Exporting Native FP16 ONNX (opset 12, device=0)...")
    m_gpu = YOLO(SRC_PT)
    fp16_exported = m_gpu.export(format="onnx", imgsz=256, opset=12, half=True, device=0)
    fp16_dst = os.path.join(DST_MODELS, "stage-a-256-13kp-fp16.onnx")
    if os.path.abspath(fp16_exported) != os.path.abspath(fp16_dst):
        shutil.move(fp16_exported, fp16_dst)
    print(f"  -> Generated FP16: {fp16_dst} ({os.path.getsize(fp16_dst)/1e6:.2f} MB)")

    # 2. Master FP32
    print("\n[2] Exporting Master FP32 ONNX (opset 12)...")
    m_cpu = YOLO(SRC_PT)
    fp32_exported = m_cpu.export(format="onnx", imgsz=256, opset=12, half=False, device="cpu")
    fp32_dst = os.path.join(DST_MODELS, "stage-a-256-13kp-fp32.onnx")
    if os.path.abspath(fp32_exported) != os.path.abspath(fp32_dst):
        shutil.move(fp32_exported, fp32_dst)
    print(f"  -> Generated FP32: {fp32_dst} ({os.path.getsize(fp32_dst)/1e6:.2f} MB)")

    # 3. Dynamic INT8 for WASM (< 5 MB)
    print("\n[3] Quantizing Dynamic INT8 for WASM (< 5 MB)...")
    int8_dst = os.path.join(DST_MODELS, "stage-a-256-13kp-int8.onnx")
    quantize_dynamic(
        model_input=fp32_dst,
        model_output=int8_dst,
        weight_type=QuantType.QUInt8,
    )
    print(f"  -> Generated INT8: {int8_dst} ({os.path.getsize(int8_dst)/1e6:.2f} MB)")

    # 4. Verify all 3 models load and run in ONNX Runtime
    print("\n[4] Verifying ONNX Runtime session creation and inference...")
    dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
    models_to_check = [
        ("FP32", fp32_dst),
        ("FP16", fp16_dst),
        ("INT8", int8_dst),
    ]

    for label, path in models_to_check:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        inp_type = sess.get_inputs()[0].type
        feed_dummy = dummy.astype(np.float16) if "float16" in inp_type else dummy
        out = sess.run(None, {inp_name: feed_dummy})
        assert out[0].shape == (1, 45, 1344), f"Unexpected shape {out[0].shape} for {label}"
        print(f"  [OK] {label:4s} loaded & verified! Shape: {out[0].shape}, Size: {os.path.getsize(path)/1e6:.2f} MB")

    # 5. Manifest and Hashes
    print("\n[5] Computing Deliverable Checksums:")
    manifest = {}
    for fname in [
        "stage-a-256-13kp-fp32.onnx",
        "stage-a-256-13kp-fp16.onnx",
        "stage-a-256-13kp-int8.onnx",
        "stage-a-256-13kp-v7.pt",
    ]:
        fpath = os.path.join(DST_MODELS, fname)
        size_mb = os.path.getsize(fpath) / 1e6
        sha = sha256_file(fpath)
        manifest[fname] = {"size_mb": round(size_mb, 2), "sha256": sha}
        print(f"  {fname:30s} | Size: {size_mb:5.2f} MB | SHA-256: {sha}")

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL DELIVERABLES EXPORTED & VERIFIED AT OPSET 12!")
    print("=" * 70)
    return manifest


if __name__ == "__main__":
    main()
