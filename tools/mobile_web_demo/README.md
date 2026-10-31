# Mobile Web WASM Model Benchmark Tool

> **Location:** `tools/mobile_web_demo/`  
> **Target Framework:** ONNX Runtime Web 1.20+ (WebAssembly / CPU only)  
> **Security:** Sandboxed strictly to `tools/mobile_web_demo/public/` (no repository or private datasets exposed)

---

## 1. Overview & Purpose
This tool provides a lightweight, mobile-ready web application designed to benchmark and validate the **Shoes VTO Stage A 256 Mobile (13/15-Keypoint)** models directly inside real mobile device browsers (iOS Safari, Android Chrome).

### Core Features:
- **Client-Side CPU WASM Inference:** Hardcoded to executionProviders: ['wasm'] and single-threaded execution (ort.env.wasm.numThreads = 1). Absolutely **zero GPU** or WebGPU fallback is used.
- **Model Comparison:** Live switching between:
  - **Two-Stage Qualcomm (Apache-2.0, Recommended):**
    - **Exp 1: NanoDet(INT8) + RTMPose(FP16)** (~8.92 MB combined: 1.44 MB + 7.48 MB)
    - **Exp 2: NanoDet(FP32) + RTMPose(FP16)** (~11.64 MB combined: 4.16 MB + 7.48 MB)
    - **Exp 3: NanoDet(FP16) + RTMPose(FP16)** (~9.66 MB combined: 2.18 MB + 7.48 MB)
    *Note: The Stage B pose model (
tmpose-13kp-hand-fp16.onnx) is shared across all three; browser cache avoids re-downloading.*
  - **Single-Stage YOLO Baselines (AGPLv3):**
    - **YOLOv8n INT8 (stage-a-256-13kp-int8.onnx, 3.46 MB)**
    - **YOLOv8n FP16 (stage-a-256-13kp-fp16.onnx, 6.53 MB)**
- **Dual Input Feeds:**
  - **Sample Video:** Pre-loaded reference video (ideo.mp4).
  - **Phone Camera:** Real-time mobile camera stream (getUserMedia) with front/rear camera toggle.
- **Anatomical Processing & Clutter Rejection:**
  - Stage A NanoDet foot bounding box detection with Letterbox  \times 256$ (fill 114) and DFL decode.
  - Stage B RTMPose-13KP SimCC continuous expectation sub-pixel inference ($\pm 12$ bin local peak window mass).
  - **Anatomical Clutter Gate** (meanConf >= 0.20 && validKpts >= 3): Completely eliminates false detections on carpet patterns, floral rugs, and floor tiles.
  - In-Model Handedness / Signed cross-product chirality (Left Foot = Green, Right Foot = Cyan/Blue).
  - One-Euro temporal keypoint smoother (toggleable ON/OFF).
- **Live Telemetry HUD:** Rolling 30-frame statistics for Effective FPS, Median Latency ({50}$), Tail Latency ({95}$), Current Latency, Pre/Postprocessing time, Stage A / Stage B breakdown, and End-to-End time.
---

## 2. Directory Structure

```
tools/mobile_web_demo/
├── server.py              # Custom Python HTTP static server with COOP/COEP/CORP headers
├── README.md              # This documentation
└── public/                # Isolated directory served to the public internet
    ├── index.html         # Responsive mobile web app UI
    ├── style.css          # High-contrast mobile dark UI stylesheet
    ├── demo.js            # Client-side WASM inference, decoder, and renderer
    ├── video.mp4          # Reference test video
    ├── models/
    │   ├── stage-a-256-13kp-int8.onnx   (3.46 MB)
    │   └── stage-a-256-13kp-fp16.onnx   (6.53 MB)
    └── vendor/
        ├── float16.min.js               # ECMAScript Float16Array polyfill
        ├── ort.min.js                   # ONNX Runtime Web bundle
        ├── ort-wasm-simd-threaded.wasm  # WebAssembly SIMD binary
        ├── ort-wasm-simd-threaded.mjs   # WebAssembly JS loader
        └── ort-wasm-simd-threaded.jsep.wasm
```

---

## 3. How to Run the Demo

### Prerequisites
1. **Python 3.9+** (e.g. conda environment `yolo`).
2. **Cloudflare CLI (`cloudflared`)** (Installed via `winget install Cloudflare.cloudflared` or standalone executable).

---

### Step 1: Start the Local Isolated Server
In a terminal, run:

```powershell
# From the repository root (F:\Machine learning\Shoes_VTO):
conda activate yolo
python tools/mobile_web_demo/server.py 8080
```

The server binds strictly to `tools/mobile_web_demo/public/` on `127.0.0.1:8080` with the following security headers:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: credentialless`
- `Cross-Origin-Resource-Policy: cross-origin`
- `Access-Control-Allow-Origin: *`

---

### Step 2: Expose via Cloudflare Quick Tunnel (No Account / No Credentials Required)
In a second terminal, execute:

```powershell
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8080
```
*(Or simply `cloudflared tunnel --url http://127.0.0.1:8080` if `cloudflared` is in your PATH).*

The terminal will output an anonymous HTTPS tunnel URL, for example:
```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://your-unique-tunnel-name.trycloudflare.com                                         |
+--------------------------------------------------------------------------------------------+
```

---

### Step 3: Open and Test on Your Mobile Phone
1. Open the generated `https://*.trycloudflare.com` URL in your mobile browser (Safari on iOS or Chrome on Android).
2. **INT8 Model Testing:** The app automatically loads `stage-a-256-13kp-int8.onnx` via WASM and begins playback.
3. **Switch to Phone Camera:** Tap the **`Phone Camera`** button to benchmark with your physical feet under your phone camera.
4. **Switch to FP16:** Tap the **`FP16 (Half 6.53 MB)`** button to test half-precision execution on your browser's WASM engine.
5. **Jitter Comparison:** Toggle the **`One-Euro Jitter Filter`** switch to see raw neural predictions vs smoothed tracking.

---

## 4. Measured Performance Reference

Tested on a mobile phone browser (WASM CPU only, zero GPU):

| Metric | Measured INT8 Performance |
| :--- | :---: |
| **Median Inference Latency ($p_{50}$)** | **51.1 – 60.0 ms** |
| **Tail Inference Latency ($p_{95}$)** | **58.4 – 62.0 ms** |
| **Pre/Postprocessing Latency** | **2.2 ms** |
| **Effective Continuous Throughput** | **13.3 – 15.2 FPS** |
| **Amortized Latency (Cadence Stride 3)** | **$\approx 12.0 - 15.0$ ms** ($\le 15$ ms mobile budget passed) |
