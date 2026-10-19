# Stage A 13-KP Foot Pose & Detection — Production Deliverables

Production-ready deliverables for **Stage A (Foot Detection & 13-Keypoint Pose Estimation)** designed for high-performance Virtual Try-On (Shoes VTO).

---

## 📌 Model Overview

- **Primary Checkpoint**: [`models/stage-a-320-13kp-v7.onnx`](models/stage-a-320-13kp-v7.onnx) (12.39 MB)
- **PyTorch Source**: [`models/stage-a-320-13kp-v7.pt`](models/stage-a-320-13kp-v7.pt) (6.63 MB)
- **Input Tensor**: `[1, 3, 320, 320]` (RGB normalized to $[0.0, 1.0]$, Letterbox padding)
- **Output Tensor**: `[1, 45, 2100]`
  - **Bounding Box (`4` channels)**: `cx, cy, w, h`
  - **Classification (`2` channels)**: `left_foot` (Class 0), `right_foot` (Class 1)
  - **Keypoints (`39` channels)**: 13 keypoints $\times$ 3 coordinates (`x`, `y`, `visibility_score`)
- **Number of Anchors**:
  - `320×320`: 2,100 anchors
  - `256×256`: 1,344 anchors
  - `192×192`: 756 anchors
  - `160×160`: 525 anchors

---

## 🦶 13 Anatomical Keypoints

| Index | Name | Description | Key Landmarks for VTO |
|:---:|:---|:---|:---|
| **0** | `toe_ground` | Toe ground contact point | Sole alignment |
| **1** | `heel_back` | Posterior edge of the calcaneus (heel) | Shoe counter fit |
| **2** | `heel_ground` | Lowest heel ground contact point | Foot grounding reference (origin) |
| **3** | `ball_medial` | 1st metatarsophalangeal joint (inner ball) | Forefoot width |
| **4** | `ball_lateral` | 5th metatarsophalangeal joint (outer ball) | Outer sole boundary |
| **5** | `instep_top` | Highest dorsal ridge of the midfoot | Tongue & collar placement |
| **6** | `arch_medial` | Medial longitudinal foot arch | Insole curvature |
| **7** | `midfoot_lateral` | Lateral midfoot boundary | Outer side profile |
| **8** | `malleolus_medial` | Inner ankle bone | Ankle collar fit |
| **9** | `malleolus_lateral` | Outer ankle bone | Ankle collar fit |
| **10** | `toe_tip` | Extreme anterior point of the toes | Shoe length & toe box |
| **11** | `throat` | Foot-leg anterior junction (instep crease) | Shoe opening |
| **12** | `achilles` | Posterior tendon insertion above calcaneus | Heel notch fit |

---

## 📁 Directory Structure

```
deliverables/stage_a_13kp/
├── README.md                      # This comprehensive documentation
│
├── models/                        # Production Checkpoints (PyTorch & ONNX)
│   ├── stage-a-320-13kp-v7.onnx   # Production V7 model (320x320 FP32, 12.39 MB)
│   ├── stage-a-320-13kp-v7.pt     # PyTorch weights (6.63 MB)
│   ├── stage-a-320-13kp-v7-fp16.onnx
│   ├── stage-a-320-13kp-v7-int8.onnx
│   ├── stage-a-320-13kp-v7-fp32.onnx
│   ├── stage-a-320-13kp-v6-freeze.onnx / .pt
│   ├── stage-a-320-13kp-v5-merged.onnx / .pt
│   └── stage-a-320-13kp.onnx / .pt
│
├── demo/                          # Desktop Python Video Evaluation Demo
│   ├── README.md                  # Video demo documentation
│   ├── run_video_demo.py          # OpenCV desktop inference & annotation script
│   ├── annotated_output_v7_unfreeze_noflip.mp4 # Benchmark video output (V7)
│   ├── annotated_output_v6_freeze.mp4
│   ├── annotated_output_v5_merged.mp4
│   └── test_annotated_preview.mp4
│
└── web_demo/                      # Zero-Latency Client-Side Web & Mobile Demo
    ├── README.md                  # Web demo architecture & deployment notes
    ├── serve_client.py            # Local HTTP server with Cross-Origin Isolation headers
    ├── requirements.txt           # Minimal Python dependencies
    ├── templates/
    │   └── client_side.html       # Responsive UI with 60 FPS camera & live comparison table
    ├── static/
    │   ├── app.js                 # Decoupled display loop & model switcher
    │   └── inference-worker.js    # Web Worker with multi-threaded WASM SIMD & WebGPU
    └── models/                    # Web-ready models across resolutions & quantizations
        ├── stage-a-320-13kp-v7.onnx        (12.39 MB, Original FP32)
        ├── stage-a-256-13kp-int8.onnx      (3.30 MB, Balanced INT8)
        ├── stage-a-192-13kp-int8.onnx      (3.28 MB, Turbo INT8)
        ├── stage-a-160-13kp-int8.onnx      (3.27 MB, Ultra-light INT8)
        ├── stage-a-256-13kp-fp16.onnx      (6.22 MB, WebGPU Half-Float)
        └── stage-a-256-13kp-fp32.onnx      (12.36 MB, FP32)
```

---

## ⚡ Measured Benchmark Comparison

### 1. Browser Client-Side Runtime (Mobile / Web via WASM 4T Multi-Core)
*Tested with real-time camera feed inside Chromium / Mobile Chrome via ONNX Runtime Web:*

| Model Variant | Resolution | Size | Precision | Latency | Inference FPS | Breakdown (Pre / Inf / Post) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **🚀 Turbo 192** | 192×192 | 3.28 MB | 8-bit Quantized | **76 – 83 ms** | **12.0 – 13.2 fps** | 1 ms / 80 ms / 2 ms |
| **⚡ INT8 (256×256)** | 256×256 | 3.30 MB | 8-bit Quantized | **88 – 99 ms** | **10.1 – 11.3 fps** | 2 ms / 93 ms / 4 ms |
| **🏛️ Original 320 (Unquantized)** | 320×320 | 12.39 MB | 32-bit Float | **114 – 117 ms** | **8.5 – 8.8 fps** | 2 ms / 108 ms / 4 ms |

### 2. Native Direct CPU Inference (4 Threads)
*Tested via native Python ONNX Runtime on 4-Core CPU:*

| Model Variant | Input Size | File Size | Native Latency | Native FPS |
|:---|:---:|:---:|:---:|:---:|
| **FP32 256** | 256×256 | 12.36 MB | **16.79 ms** | **59.6 fps** |
| **Turbo 192 INT8** | 192×192 | 3.28 MB | **20.03 ms** | **49.9 fps** |
| **Original 320 FP32** | 320×320 | 12.39 MB | **26.72 ms** | **37.4 fps** |
| **INT8 256** | 256×256 | 3.30 MB | **31.05 ms** | **32.2 fps** |

---

## 🚀 Quick Execution Guide

### Option A: Real-Time Mobile & Web Testing (Recommended)

Run the zero-latency client-side web application:

```powershell
# 1. Activate environment
conda activate yolo

# 2. Start the local cross-origin isolated server
python deliverables/stage_a_13kp/web_demo/serve_client.py --port 8000
```

- **Desktop**: Open [http://localhost:8000/templates/client_side.html](http://localhost:8000/templates/client_side.html)
- **Mobile Device**: Expose port `8000` with ngrok:
  ```powershell
  ngrok http 8000
  ```
  Open the HTTPS URL on your phone's browser, grant camera access, and tap any precision button (**Turbo 192**, **INT8**, **Original 320**) to switch models live.

### Option B: Desktop Video Annotation Demo

Run offline evaluation and video rendering using OpenCV:

```powershell
conda activate yolo

python deliverables/stage_a_13kp/demo/run_video_demo.py \
    --model deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.onnx \
    --video path/to/input.mp4 \
    --output deliverables/stage_a_13kp/demo/output_annotated.mp4
```

---

## 💡 Architecture & Optimization Highlights

1. **Decoupled 60 FPS Camera Display**:
   - The user's camera runs on an independent `requestAnimationFrame` loop at full 60 FPS with zero lag or stutter.
   - Inference frames are processed asynchronously in the background without blocking the UI thread.
2. **Zero-Copy Web Worker**:
   - Camera frames are converted to `ImageBitmap` and transferred to a dedicated Web Worker via zero-copy structured cloning.
3. **Hardware Acceleration & SIMD Multi-Threading**:
   - Multi-threading is unlocked in the browser using `SharedArrayBuffer` via `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: credentialless`.
   - WebGPU / WebGL execution providers are automatically negotiated for float models, with multi-core WASM SIMD for quantized INT8 models.
4. **Dynamic Resizing**:
   - The Web Worker automatically resizes its `OffscreenCanvas` and adjusts tensor strides on-the-fly when switching between 192×192, 256×256, and 320×320 models.
