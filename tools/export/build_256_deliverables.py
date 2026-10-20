import os, sys, shutil, hashlib, onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType

SRC_ONNX = "deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.onnx"
SRC_PT = "deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.pt"
DST_MODELS = "deliverables/stage_a_256_13kp/models"

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    os.makedirs(DST_MODELS, exist_ok=True)
    
    # 1. FP32
    fp32_dst = os.path.join(DST_MODELS, "stage-a-256-13kp-fp32.onnx")
    shutil.copy2(SRC_ONNX, fp32_dst)
    print(f"[OK] Master FP32 copied: {fp32_dst} ({os.path.getsize(fp32_dst)/1e6:.2f} MB)")
    
    # 2. PyTorch
    pt_dst = os.path.join(DST_MODELS, "stage-a-256-13kp-v7.pt")
    shutil.copy2(SRC_PT, pt_dst)
    print(f"[OK] PyTorch weights copied: {pt_dst} ({os.path.getsize(pt_dst)/1e6:.2f} MB)")
    
    # 3. Real FP16 for WebGPU
    fp16_dst = os.path.join(DST_MODELS, "stage-a-256-13kp-fp16.onnx")
    m = onnx.load(fp32_dst)
    m_fp16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m_fp16, fp16_dst)
    print(f"[OK] Real FP16 exported: {fp16_dst} ({os.path.getsize(fp16_dst)/1e6:.2f} MB)")
    
    # 4. Dynamic INT8 for WASM
    int8_dst = os.path.join(DST_MODELS, "stage-a-256-13kp-int8.onnx")
    quantize_dynamic(
        model_input=fp32_dst,
        model_output=int8_dst,
        weight_type=QuantType.QUInt8
    )
    print(f"[OK] Dynamic INT8 quantized: {int8_dst} ({os.path.getsize(int8_dst)/1e6:.2f} MB)")
    
    # Compute Hashes
    manifest = {}
    for fname in ["stage-a-256-13kp-fp32.onnx", "stage-a-256-13kp-fp16.onnx", "stage-a-256-13kp-int8.onnx", "stage-a-256-13kp-v7.pt"]:
        fpath = os.path.join(DST_MODELS, fname)
        size_mb = os.path.getsize(fpath) / 1e6
        sha = sha256_file(fpath)
        manifest[fname] = {"size_mb": round(size_mb, 2), "sha256": sha}
        print(f"  {fname:32s} | Size: {size_mb:5.2f} MB | SHA-256: {sha}")
        
    return manifest

if __name__ == "__main__":
    main()
