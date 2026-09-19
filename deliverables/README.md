# 👟 Shoes VTO — Stage A Perception Layer Integration Guide

**Deliverable Package**: Stage A Foot Detector (16 Keypoint Output)  
**Target Runtime**: `onnxruntime-web` (WebGPU EP / WASM EP)  
**Format**: ONNX Opset 12  
**Author**: Mostafa (AI Perception Layer)  

---

## 📦 Package Contents

| File | Purpose |
| :--- | :--- |
| **`models/stage_a_16kp_fp32.onnx`** | Primary ONNX model (opset 12, static shape `[1, 3, 320, 320]`) |
| **`models/stage_a_16kp_fp16.onnx`** | Half-precision WebGPU variant (faster on mobile GPUs) |
| **`models/stage_a_16kp_int8.onnx`** | Quantized INT8 model for Safari WASM fallback |
| **`foot_meta.json`** | Model metadata, tensor layouts, and channel mappings |
| **`foot_canonical.json`** | 16 3D reference keypoint coordinates in millimeters |
| **`foot_inference.js`** | Ready-to-use ES6 browser decoder module |
| **`golden_fixture.json`** | Unit test fixture to verify your JS implementation in 1 minute |

---

## ⚡ Quick JavaScript Integration (WebGPU / WASM)

```javascript
import * as ort from 'onnxruntime-web';
import { FootDetector } from './foot_inference.js';

// 1. Initialize Detector
const detector = new FootDetector({
    boxConfThresh: 0.30,
    crossNmsIou: 0.35,
    kpConfThresh: 0.30
});

// 2. Load ONNX Model (Select WebGPU or WASM)
await detector.loadModel('./models/stage_a_16kp_fp32.onnx', ort);

// 3. Detect from Camera Video Element in Real-Time
function onFrame() {
    const feet = await detector.detect(videoElement, ort);
    
    for (const foot of feet) {
        console.log(`Detected ${foot.className} with conf ${foot.confidence}`);
        console.log('Toe Tip:', foot.keypoints['0: toe_tip']);
        console.log('Heel Ground (Origin):', foot.keypoints['3: heel_ground']);
    }
    requestAnimationFrame(onFrame);
}
```

---

## 📐 Tensor Specifications

### Input Tensor
- **Name**: `images`
- **Shape**: `[1, 3, 320, 320]` (NCHW, RGB, normalized `[0.0, 1.0]`)
- **Letterbox Padding**: Center-padded with value `114` (0.447)

### Output Tensor
- **Name**: `output0`
- **Shape**: `[1, 54, 2100]` (Channel-major, 2100 anchors)
  - `Rows 0..3`: Bounding box `cx, cy, w, h` in 320x320 letterbox pixels.
  - `Row 4`: Left foot confidence score (`[0, 1]`, sigmoid in-graph).
  - `Row 5`: Right foot confidence score (`[0, 1]`, sigmoid in-graph).
  - `Rows 6..53`: 16 Keypoints $\times$ 3 channels (`x`, `y`, `visibility`).

---

## 🗂️ 16 Keypoint Channel Mapping Table

| Model Channel | Anatomical Keypoint | Description |
| :---: | :--- | :--- |
| **0** | `0: toe_tip` | Most anterior point of foot outline |
| **11** | `1: toe_ground` | Front sole ground contact |
| **13** | `2: heel_back` | Posterior heel boundary |
| **12** | `3: heel_ground` | Canonical 3D origin (back sole ground contact) |
| **3** | `4: ball_medial` | 1st metatarsal head (widest point medial) |
| **4** | `5: ball_lateral` | 5th metatarsal head (widest point lateral) |
| **5** | `6: ball_top` | Forefoot dorsal center |
| **6** | `7: instep_top` | Bridge of foot (highest instep apex) |
| **7** | `8: arch_medial` | Medial arch boundary |
| **8** | `9: midfoot_lateral` | Lateral midfoot boundary |
| **9** | `10: malleolus_medial` | Inner ankle bone center |
| **10** | `11: malleolus_lateral` | Outer ankle bone center |
| **14** | `14: achilles` | Achilles tendon rear center |
| **15** | `15: shin_mid` | Mid-shin (120mm above ankle) |

---

## ✅ Compatibility & Ops Check
- **No in-graph NMS**: Decoded on CPU/WASM in JS.
- **Zero Banned Ops**: Clean graph verified (no `RoiAlign`, `GridSample`, or `ScatterND`).
- **WebGPU & WASM Compatible**: Tested and ready.
