import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch

_tlib = Path(torch.__file__).parent / "lib"
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(str(_tlib))
    except Exception:
        pass
os.environ["PATH"] = str(_tlib) + os.pathsep + os.environ.get("PATH", "")

import onnxruntime as ort

MODELS_DIR = Path("deliverables/qualcomm_experiments/models")
RTMO_FP32 = MODELS_DIR / "rtmpose_13kp_hand_improved5_opset12.onnx"
RTMO_FP16 = MODELS_DIR / "rtmpose_13kp_hand_improved5_fp16.onnx"

def profile_with_torch(model_path: Path, batch_size: int = 2, iters: int = 20):
    cuda_opts = {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "gpu_mem_limit": 3 * 1024 * 1024 * 1024,
        "cudnn_conv_algo_search": "DEFAULT",
        "do_copy_in_default_stream": True,
    }
    session = ort.InferenceSession(str(model_path), providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"])
    inp_info = session.get_inputs()[0]
    dtype = np.float32 if "float16" not in inp_info.type else np.float16
    arr = np.random.randn(batch_size, 3, 256, 256).astype(dtype)

    # Warmup
    for _ in range(25):
        session.run(None, {inp_info.name: arr})
    torch.cuda.synchronize()

    # PyTorch Profiler capturing CUDA kernels from native C++ calls
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False
    ) as prof:
        for _ in range(iters):
            session.run(None, {inp_info.name: arr})
        torch.cuda.synchronize()

    key_averages = prof.key_averages()
    table = key_averages.table(sort_by="cuda_time_total", row_limit=15)
    return str(table)

def main():
    print("=== PROFILING RTMPose FP32 CUDA KERNELS ===")
    print(profile_with_torch(RTMO_FP32, batch_size=2, iters=20))
    print("\n=== PROFILING RTMPose FP16 CUDA KERNELS ===")
    print(profile_with_torch(RTMO_FP16, batch_size=2, iters=20))

if __name__ == "__main__":
    main()

