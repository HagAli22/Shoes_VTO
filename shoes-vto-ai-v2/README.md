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
   - All 16 keypoints are visually audited against ground-truth annotations across diverse foot poses and camera angles.
2. **Elimination of Cross-Class NMS Bug & Physical Foot Deduplication:**
   - In `reference-decoder.js` and `demo/run_video_demo.py`, NMS is executed **independently per class** (`left_foot` vs `right_foot`).
   - Valid overlapping or crossing feet are **never deleted** during walking or crossing sequences.
   - Physical duplicate bounding boxes on the same foot are resolved via geometric IoU gating.
3. **Anatomical Geometric Chirality Disambiguation:**
   - Deterministic 2D cross-product calculation resolves left vs right foot classification symmetry ambiguity from top-down angles.
4. **Genuine WebGPU FP16 & INT8 Exports:**
   - Real FP16 initializers (`stage-a-320-16kp-fp16.onnx`, **6.65 MB**) with distinct SHA-256 (`001cf15e8a1b25d0...`).
   - Quantized INT8 candidate (`stage-a-320-16kp-int8.onnx`, **3.53 MB**) with distinct SHA-256 (`7177113ba05e24c1...`).
   - Lightweight WASM fallback candidate (`stage-a-256-int8.onnx`, **3.51 MB**).
5. **Reconciled 3D Canonical Templates:**
   - `canonical-left.json` and `canonical-right.json` with matched heel-to-toe length ($265.0\text{ mm}$) and ball width ($98.0\text{ mm}$) in right-handed Y-up coordinates.
6. **Complete Test Fixtures, Reports & Live Video Demo:**
   - Ground truth test fixtures for 0, 1, and 2 overlapping feet.
   - Formal reports on accuracy (89.4% mAP), keypoint ordering audit, browser timing, and dataset coverage.
   - Standalone video demo script with One-Euro temporal smoothing filter.

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
    stage-a-256-int8.onnx       # Low-End WASM INT8 ONNX (3.51 MB)
  fixtures/
    expected-decoded.json       # Golden decoded output cases
    crop-transform-cases.json   # Crop test cases
  reports/
    keypoint-order-audit.md     # Ground-truth keypoint order visual audit report
    accuracy.md                 # Validation metrics & mAP
    browser-performance.md      # Latency & FPS benchmarks
    dataset-coverage.md         # 80/10/10 stratified dataset report
    operator-compatibility.md   # ONNX Runtime Web compatibility
  demo/
    run_video_demo.py           # Production-grade Python video inference demo
    README.md                   # Video demo user guide
```

---

## 3. Model Specifications

| Artifact | File | Input Shape | Precision | Size | SHA-256 | Target Execution Provider |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- |
| **FP32 Primary** | `models/stage-a-320-16kp-fp32.onnx` | `[1, 3, 320, 320]` | Float32 | **13.20 MB** | `c6866cd07dd85175...` | ONNX Runtime Web / CPU |
| **FP16 WebGPU** | `models/stage-a-320-16kp-fp16.onnx` | `[1, 3, 320, 320]` | Float16 | **6.65 MB** | `001cf15e8a1b25d0...` | ONNX Runtime Web / WebGPU |
| **INT8 WASM** | `models/stage-a-320-16kp-int8.onnx` | `[1, 3, 320, 320]` | UInt8 | **3.53 MB** | `7177113ba05e24c1...` | ONNX Runtime Web / WASM |
| **INT8 Fallback** | `models/stage-a-256-int8.onnx` | `[1, 3, 256, 256]` | UInt8 | **3.51 MB** | `c71f8964bada7a2c...` | ONNX Runtime Web / Low-End WASM |

---

## 4. Tensor IO Contract

- **Input `images`:** `[1, 3, 320, 320]` (or `[1, 3, 256, 256]`), Float32, NCHW, RGB `[0, 1]`, centered 114 padding.
- **Output `output0`:** `[1, 54, 2100]` for 320x320 (or `[1, 54, 1344]` for 256x256), Float32.
  - Rows 0–3: `cx, cy, w, h` in letterbox coordinates.
  - Rows 4–5: Class probabilities (`left_foot`, `right_foot`).
  - Rows 6–53: 16 Anatomical Keypoints $\times (x, y, \text{visibility})$.

---

## 5. Official Frozen 16 Anatomical Keypoints Order

| Index | Name | Tier | Description |
| :---: | :--- | :---: | :--- |
| **0** | `toe_ground` | 1 | Inferior contact point under toe box / lateral toe base |
| **1** | `heel_back` | 1 | Posterior-most prominence of calcaneus (heel back) |
| **2** | `heel_ground` | 1 | Inferior ground contact point of calcaneus |
| **3** | `ball_medial` | 1 | 1st metatarsal head prominence (medial ball) |
| **4** | `ball_lateral` | 1 | 5th metatarsal head prominence (lateral ball) |
| **5** | `ball_top` | 1 | Superior dorsal point over metatarsal heads |
| **6** | `instep_top` | 1 | Apex of the dorsal instep arch |
| **7** | `arch_medial` | 2 | Medial longitudinal arch apex |
| **8** | `midfoot_lateral` | 2 | Lateral midfoot outer boundary |
| **9** | `malleolus_medial` | 1 | Center of medial malleolus (inner ankle bone) |
| **10** | `malleolus_lateral` | 1 | Center of lateral malleolus (outer ankle bone) |
| **11** | `toe_tip` | 1 | Anterior tip of distal phalanx of 1st (big) toe |
| **12** | `ankle_center` | 1 | Geometric center of the talocrural joint |
| **13** | `throat` | 1 | Anterior shoe opening throat / lower instep flex point |
| **14** | `achilles` | 1 | Achilles tendon insertion onto posterior calcaneus |
| **15** | `shin_mid` | 1 | Anterior lower tibia midpoint for leg alignment |
