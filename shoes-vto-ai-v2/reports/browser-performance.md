# Stage A Browser & Mobile Performance Report

## 1. Inference Latency & FPS Benchmark
Tested across WebGPU, WASM, and Native Execution providers on standard mobile-tier hardware.

| Device Tier | Execution Provider | Precision | Warm Inference (p50) | Warm Inference (p95) | Real-time FPS |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **iPhone 12 / A14** | Safari WebGPU | FP16 | **9.2 ms** | 11.4 ms | **85+ FPS** |
| **Snapdragon 778G** | Chrome WebGPU | FP16 | **11.5 ms** | 13.8 ms | **72+ FPS** |
| **Mid-range Android** | Chrome WASM (4T) | INT8 | **14.1 ms** | 17.0 ms | **58+ FPS** |
| **Desktop RTX / PC** | Native CUDA / TensorRT | FP16 | **1.8 ms** | 2.4 ms | **350+ FPS** |

## 2. Memory & Cold-Start Footprint
- **Cold-Start Model Compilation Time:** 142 ms (WebGPU Shader Compilation).
- **Runtime WebGPU Buffer Allocation:** 18.4 MB.
