# Shoes VTO Stage A — 16-Keypoint Production Delivery Package

**Version:** 2.0.0-16kp  
**Date:** 2026-09-22  
**Delivered By:** AI Perception Team  
**Delivered To:** NGM / Novagates AR SDK Team  

---

## 1. Executive Summary & Response to AI_TEAM_ACTION_REQUIRED.md

This delivery package provides the **Full 16-Keypoint Native Stage A Foot Detector** (`run_shuffled_v3`) strictly addressing all issues raised in `AI_TEAM_ACTION_REQUIRED.md`:

1. **Full 16-Keypoint Native Output Contract (`[1, 54, 2100]`):**
   - No missing channels or ambiguous mappings.
   - All 16 keypoints are fully identified, anatomically mapped, and documented according to Section 5 of the requirements document.
2. **Elimination of Cross-Class NMS Bug:**
   - In `reference-decoder.js`, NMS is executed **independently per class** (`left_foot` vs `right_foot`).
   - Valid overlapping or crossing feet are **never deleted** during walking or crossing sequences.
3. **Genuine WebGPU FP16 Export:**
   - Real FP16 initializers (`stage-a-320-16kp-fp16.onnx`, **6.65 MB**) with distinct SHA-256 (`001cf15e8a1b25d0...`).
4. **WASM INT8 Dynamic Quantization:**
   - Quantized INT8 candidate (`stage-a-320-16kp-int8.onnx`, **3.53 MB**) with distinct SHA-256 (`7177113ba05e24c1...`).
5. **Reconciled 3D Canonical Templates:**
   - `canonical-left.json` and `canonical-right.json` with matched heel-to-toe length ($265.0\text{ mm}$) and ball width ($98.0\text{ mm}$) in right-handed Y-up coordinates.
6. **Complete Test Fixtures & Reports:**
   - Ground truth test fixtures for 0, 1, and 2 overlapping feet.
   - Formal reports on accuracy (89.4% mAP), browser timing, dataset coverage, and ONNX operator compatibility.

---

## 2. Delivery Package File Structure

```text
deliverables/stage_a_16kp/
  README.md                     # This document
  model-contract.json           # Formal JSON schema & tensor contract
  canonical-left.json           # Reconciled 3D canonical left foot model (mm)
  canonical-right.json          # Reconciled 3D canonical right foot model (mm)
  reference-decoder.js          # JavaScript production decoder (Per-class NMS)
  crop-transforms.js            # ROI extraction & coordinate back-projection
  models/
    stage-a-320-16kp-fp32.onnx  # Primary FP32 ONNX (13.20 MB)
    stage-a-320-16kp-fp16.onnx  # WebGPU FP16 ONNX (6.65 MB)
    stage-a-320-16kp-int8.onnx  # WASM INT8 ONNX (3.53 MB)
  fixtures/
    expected-decoded.json       # Golden decoded output cases
    crop-transform-cases.json   # Crop test cases
  reports/
    accuracy.md                 # Validation metrics & mAP
    browser-performance.md      # Latency & FPS benchmarks
    dataset-coverage.md         # 80/10/10 stratified dataset report
    operator-compatibility.md   # ONNX Runtime Web compatibility
```

---

## 3. Model Specifications

| Artifact | File | Precision | Size | SHA-256 | Target Execution Provider |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FP32 Primary** | `models/stage-a-320-16kp-fp32.onnx` | Float32 | **13.20 MB** | `c6866cd07dd85175...` | ONNX Runtime Web / CPU |
| **FP16 WebGPU** | `models/stage-a-320-16kp-fp16.onnx` | Float16 | **6.65 MB** | `001cf15e8a1b25d0...` | ONNX Runtime Web / WebGPU |
| **INT8 WASM** | `models/stage-a-320-16kp-int8.onnx` | UInt8 | **3.53 MB** | `7177113ba05e24c1...` | ONNX Runtime Web / WASM |

---

## 4. Tensor IO Contract

- **Input `images`:** `[1, 3, 320, 320]`, Float32, NCHW, RGB `[0, 1]`, centered 114 padding.
- **Output `output0`:** `[1, 54, 2100]`, Float32.
  - Rows 0–3: `cx, cy, w, h` in 320x320 letterbox coordinates.
  - Rows 4–5: Class probabilities (`left_foot`, `right_foot`).
  - Rows 6–53: 16 Anatomical Keypoints $	imes (x, y, 	ext{visibility})$.

---

## 5. Frozen 16 Anatomical Keypoints Order

| Index | Name | Description |
| :---: | :--- | :--- |
| **0** | `toe_tip` | Anterior tip of distal phalanx of 1st/2nd toe |
| **1** | `toe_ground` | Inferior ground contact point under toe box |
| **2** | `heel_back` | Posterior-most prominence of calcaneus |
| **3** | `heel_ground` | Inferior ground contact of calcaneus |
| **4** | `ball_medial` | 1st metatarsal head prominence (medial ball) |
| **5** | `ball_lateral` | 5th metatarsal head prominence (lateral ball) |
| **6** | `ball_top` | Superior dorsal point over metatarsal heads |
| **7** | `instep_top` | Apex of the dorsal instep arch |
| **8** | `arch_medial` | Medial longitudinal arch apex |
| **9** | `midfoot_lateral` | Lateral midfoot outer boundary |
| **10** | `malleolus_medial` | Center of medial malleolus (inner ankle bone) |
| **11** | `malleolus_lateral` | Center of lateral malleolus (outer ankle bone) |
| **12** | `ankle_center` | Geometric center of talocrural joint |
| **13** | `throat` | Anterior shoe opening throat / lower instep flex point |
| **14** | `achilles` | Achilles tendon insertion onto posterior calcaneus |
| **15** | `shin_mid` | Anterior lower tibia midpoint for leg alignment |
