# Stage A Browser & Mobile Performance Report

> **Measurement Status:** Partial — WASM measurements are real. WebGPU pending target-device testing.

---

## 1. Local Machine — Python ONNX Runtime (Measured ✅)

**Test Machine:** Intel Core i5-8300H 2.30 GHz | Windows 10 | ONNX Runtime 1.30.0  
**Method:** 100 warm iterations after 10 warmup runs | Input: `[1, 3, 320, 320]`

| Model | File Size | Provider | Cold Load | Cold Infer | Warm p50 | Warm p95 | FPS (p50) | RAM Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 (YOLOv8n 16-KP)** | 13.20 MB | CPUExecutionProvider | 108 ms | 11 ms | **16.1 ms** | 20.2 ms | **62 FPS** | +23 MB |
| **FP16 (YOLOv8n 16-KP)** | 6.65 MB | CPUExecutionProvider | 192 ms | 20 ms | **17.3 ms** | 21.2 ms | **58 FPS** | +18 MB |
| **INT8 (YOLOv8n 16-KP)** | 3.53 MB | CPUExecutionProvider | 155 ms | 35 ms | **27.3 ms** | 31.6 ms | **37 FPS** | +8 MB |

> **Note:** FP16 is slower than FP32 on CPU — expected. The i5-8300H has no FP16 hardware acceleration. On WebGPU/GPU hardware, FP16 runs at ~2× the speed of FP32.

---

## 2. Browser WASM Benchmark (Measured ✅)

**Test Machine:** Windows 10 (Win32) | Chrome 153.0.0.0 | ONNX Runtime Web 1.20  
**Method:** 50 warm iterations after 10 warmup runs | CORS Isolation headers enabled  
**WebGPU Status:** ❌ Not available — NVIDIA Quadro P2000 (Pascal, 2016) does not support D3D12 required by Chrome WebGPU on this driver version.

| Model / Provider | Cold Load | Cold Infer | Warm p50 | Warm p95 | FPS (p50) | ≤12 ms? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 — WASM (single-thread)** | 5120 ms | 111 ms | **113.2 ms** | 142.0 ms | **8.8 FPS** | ❌ |
| **INT8 — WASM (single-thread)** | 85 ms | 83 ms | **82.2 ms** | 92.7 ms | **12.2 FPS** | ❌ |
| **FP16 — WebGPU** | — | — | **N/A** | N/A | — | ❓ Pending |
| **FP32 — WebGPU** | — | — | **N/A** | N/A | — | ❓ Pending |

> **Why is WASM slow?** The 13.2 MB YOLOv8n model requires significant compute per frame. Single-threaded WASM (JavaScript) without GPU acceleration cannot approach the ≤12 ms target for a model of this size. This is the expected and documented limitation of running large models in WASM without WebGPU.

---

## 3. Browser WebGPU Benchmark (Pending ⏳)

WebGPU measurements require a device with D3D12 (Windows) or Vulkan/Metal (macOS/iOS/Android) support.

| Target Device | Browser | Provider | Expected p50 | Notes |
| :--- | :--- | :--- | :--- | :--- |
| iPhone 12 / A14 | Safari | WebGPU (Metal) | ~8–12 ms | To be measured |
| Snapdragon 778G | Chrome | WebGPU (Vulkan) | ~10–15 ms | To be measured |
| RTX 3060 (Desktop) | Chrome | WebGPU (Vulkan/DX12) | ~3–6 ms | To be measured |

> To measure: serve `tools/benchmark_browser/index.html` via `python tools/serve_benchmark.py` and open from a WebGPU-capable device on the same local network.

---

## 4. Target Requirements vs. Measured Results

| Requirement (AI_TEAM_ACTION_REQUIRED.md §7) | Target | Measured | Status |
| :--- | :--- | :--- | :--- |
| Stage A acquisition/recovery | ≤ 12 ms | 16 ms (CPU p50) | ❌ CPU-only |
| Stage A acquisition/recovery | ≤ 12 ms | 113 ms (WASM p50) | ❌ WASM-only |
| Stage A acquisition/recovery | ≤ 12 ms | TBD (WebGPU) | ⏳ Pending |
| Stage A cold start | Report only | 108 ms (CPU) / 5120 ms (WASM first load) | ℹ️ Noted |

> **Conclusion:** The ≤12 ms Stage A target is achievable only with WebGPU execution. WASM/CPU alone cannot meet this requirement for a 13 MB model. Measurement on a WebGPU-capable mobile device (iPhone 12+ or Snapdragon 865+) is required before final sign-off.

---

## 5. Path to ≤12 ms — Model Size Impact

| Model | FP32 Size | Expected WASM p50 | Expected WebGPU FP16 p50 |
| :--- | :--- | :--- | :--- |
| YOLOv8n 16-KP (this delivery) | 13.2 MB | ~113 ms | ~8–12 ms |
| Compact YOLOv8 (Path A alt.) | 4.9 MB | ~40–50 ms | ~3–6 ms |
| BlazeFoot 4-KP (in progress) | 2.3 MB | ~15–25 ms | ~2–4 ms |

> Smaller models benefit WASM proportionally more than WebGPU, since GPU compute is already fast even for large models.
