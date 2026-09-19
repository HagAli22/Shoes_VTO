# VTO Project — Version 1 Implementation Status

> **Project:** Real-Time Shoe AR Try-On — AI Perception Layer  
> **Engineer:** Mostafa (AI / Perception)  
> **Deliverable boundary:** `{bbox + 16 keypoints + presence + next_roi}` → handoff to AR (NGM)  
> **Last updated:** 2026-09-17  
> **Stage A status:** ✅ COMPLETE  
> **Stage B status:** 🔲 NOT STARTED

---

## Architecture Overview

```
Camera Frame (320×320)
        │
        ▼
┌──────────────────┐
│   STAGE A        │  YOLOv8n-pose
│   Foot Detector  │  ← runs only on re-acquisition
│                  │  → bbox [left/right] + 16 coarse KPs
└──────────────────┘
        │ rotated 224×224 crop
        ▼
┌──────────────────┐
│   STAGE B        │  MobileNetV3-Small + Multi-Head
│   Landmark Model │  ← runs EVERY frame
│                  │  → 16 KPs + presence + next_roi
└──────────────────┘
        │
        ▼
  {bbox + 16 keypoints + presence + next_roi}
        │
        ▼
   HANDOFF TO AR (NGM)
```

---

## 16-Keypoint Schema (FROZEN)

| Idx | Name | Zone | Tier |
|---|---|---|---|
| 0 | toe_tip | sole | 1 |
| 1 | toe_ground | sole | 1 |
| 2 | heel_back | sole | 1 |
| 3 | heel_ground | sole | 1 — canonical ORIGIN |
| 4 | ball_medial | ball | 1 |
| 5 | ball_lateral | ball | 1 |
| 6 | ball_top | ball | 1 |
| 7 | instep_top | mid | 1 |
| 8 | arch_medial | mid | 2 |
| 9 | midfoot_lateral | mid | 2 |
| 10 | malleolus_medial | ankle | 1 |
| 11 | malleolus_lateral | ankle | 1 |
| 12 | ankle_center | ankle | 1 |
| 13 | throat | ankle | 1 |
| 14 | achilles | ankle | 1 |
| 15 | shin_mid | leg | 1 |

---

## ✅ DONE — Stage A (Foot Detector)

### Step 1 — Environment Setup

**Status:** ✅ Complete  
**Conda env:** `yolo` (Python 3.10.18)

| Package | Version | Role |
|---|---|---|
| torch | 2.5.1+cu121 | Training (CUDA 12.1) |
| torchvision | 0.20.1+cu121 | Vision transforms |
| ultralytics | 8.3.189 | YOLOv8 training |
| onnx | 1.22.0 | Model export |
| onnxruntime | 1.23.2 | Inference validation |
| onnxslim | 0.1.96 | Graph optimization |
| albumentations | 2.0.8 | Augmentation |
| pycocotools | 2.0.11 | COCO metrics |

**Verification:**
```bash
conda activate yolo
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: Quadro P2000
```

---

### Step 2 — Dataset: `shoes_v1` Pilot (143 images)

**Status:** ✅ Complete  
**Source:** Roboflow `fingers_keypoint` v10 (exported 2026-09-16)  
**Format:** YOLOv8 pose  
**Schema:** 16 KP per foot, 2 classes  
**Split:** 100 train / 29 val / 14 test  

**Class fix applied:**
- OLD: `0=Foot_Right`, `1=Foot_left` (wrong order)
- NEW: `0=left_foot`, `1=right_foot` (matches AR SDK §5 frozen spec)

**Data location:** `data/stage_a/`

**Verification:**
```bash
conda activate yolo
python src/models/stage_a/prepare_stage_a_data.py
# Expected: 99 train images, balanced left/right
```

**Known issues:**
- 1 corrupt label (`not_palm1638...`) — detected and excluded by YOLO scanner
- 143 images is a pilot. Target is 15,000 frames for production (§9.1)
- Dataset covers mostly back-view and some front-view frames

---

### Step 3 — Stage A Training (YOLOv8n-pose, 200 epochs)

**Status:** ✅ Complete  
**Checkpoint:** `outputs/stage_a/run12/weights/best.pt` (6.8 MB)  
**Training time:** 10.9 min on Quadro P2000  
**GPU peak:** 0.691 GB / 4.0 GB  

**Final metrics (val set, 29 images):**

| Metric | Value | Threshold |
|---|---|---|
| Box mAP50 | **83.9%** | — |
| Box mAP50-95 | **49.8%** | — |
| left_foot mAP50 | **85.0%** | — |
| right_foot mAP50 | **82.9%** | — |
| Pose mAP50 | 38.1% | Expected low with 99 images |
| Pose mAP50-95 | 11.3% | Will improve with more data |

> [!NOTE]
> Box mAP50 = 83.9% is excellent for a 99-image pilot. Pose mAP will improve significantly
> with the target 15,000-frame dataset. The critical metric for Stage A is bounding box detection;
> keypoints from Stage A are only used for rough crop rotation (Stage B corrects within 1 frame).

**Training config:**
```
Base model   : yolov8n-pose.pt (COCO pretrained)
Input size   : 320 × 320
Epochs       : 200 (early stop patience=50)
Batch size   : 16
Optimizer    : AdamW, lr=1e-3
Augmentation : hsv, rotation±15°, scale, mosaic=0.5, copy_paste=0.1
fliplr       : 0.0  (disabled — class flip must be correct, §7)
flipud       : 0.0  (feet always gravity-down)
```

**Verification:**
```bash
conda activate yolo
python src/models/stage_a/train_stage_a.py --epochs 1 --batch 16 --device 0 --name test_run
# Should complete in ~1 min, mAP=0 expected at epoch 1
```

---

### Step 4 — Stage A ONNX Export (opset 12)

**Status:** ✅ Complete  
**ONNX file:** `outputs/stage_a/run12/weights/best.onnx` (12.6 MB)

**ONNX spec compliance:**

| Requirement | Expected | Actual | Status |
|---|---|---|---|
| Input shape | `[1, 3, 320, 320]` | `[1, 3, 320, 320]` | ✅ |
| Output shape | `[1, 54, 2100]` | `[1, 54, 2100]` | ✅ |
| ONNX opset | 12 | 12 | ✅ |
| Dynamic dims | None (static) | None | ✅ |
| In-graph NMS | Not allowed | Not present | ✅ |
| Banned ops | None | None | ✅ |
| Graph simplified | Yes (onnxslim) | Yes | ✅ |

> [!IMPORTANT]
> Output shape is `[1, 54, 2100]` because we train with all 16 KPs.
> The frozen AR SDK spec for Stage A requires `[1, 18, 2100]` (4 coarse KPs only).
> **This will be corrected before M2 handoff** by retraining with `kpt_shape: [4, 3]`.
> For the pilot, 54-row output is used and the 4 coarse KPs are extracted by index.

**Coarse KP extraction (indices in 16-KP schema):**
- `toe_tip` = KP index 0
- `heel_back` = KP index 2
- `ball_medial` = KP index 4
- `ball_lateral` = KP index 5

**Ops used (14 — all WebGPU/WASM safe):**
`Add, Concat, Conv, Div, MaxPool, Mul, Reshape, Resize, Sigmoid, Slice, Softmax, Split, Sub, Transpose`

**Verification:**
```bash
conda activate yolo
python -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('outputs/stage_a/run12/weights/best.onnx')
x = np.zeros((1,3,320,320), np.float32)
out = sess.run(None, {'images': x})
print('Shape:', out[0].shape)  # Expected: (1, 54, 2100)
"
```

---

### Step 5 — Stage A Inference Tests

**Status:** ✅ Scripts ready  
**Test video:** `data/test_video.mp4` (16.9 MB)  

```bash
# Test on video file
conda activate yolo
python tools/evaluation/test_stage_a_video.py

# Test live with webcam
python tools/evaluation/test_stage_a_live.py
```

**See results in:** `outputs/stage_a/test_results/`

---

### Step 6 — Annotation Reference Document

**Status:** ✅ Complete  
**File:** `ANNOTATION_PRECISION_REFERENCE_v2.md`

Key rules captured:
- `toe_tip` (0): silhouette point — always v=2 from front/top-down/side
- `toe_ground` (1): ground contact — v=2 from front/back-medial-side; v=1 elsewhere
- `heel_back` (2): posterior outline — v=2 from BACK VIEW ONLY; v=1 all others
- `ball_medial` (4) + `ball_lateral` (5): sole bumps — v=2 from BACK VIEW ONLY; v=1 all others
- `achilles` (14): posterior ankle — v=2 from back/side; v=1 from front/top-down
- **Golden rule:** Anatomical position never changes. Only visibility (v) changes with view angle.

---

## 🔲 REMAINING — Stage B and Beyond

### Step 7 — `foot_canonical.json` (M1 Deliverable)

**Status:** 🔲 Not started  
**Priority:** HIGH — unblocks AR/NGM team  
**File to create:** `deliverables/foot_canonical.json`

**Contents:**
- 16 KP positions in mm for a 265mm reference foot
- Canonical frame definition (origin = heel_ground KP3)
- Per-KP scaling rule (proportional to measured foot length)
- `flip_idx` = identity `[0..15]`

```bash
# To create:
python tools/create_foot_canonical.py
# Verify: check that all 16 KPs have anatomically valid mm positions
```

---

### Step 8 — Stage B: Landmark Model Architecture

**Status:** 🔲 Not started  
**Files to create:** `src/models/stage_b/`

**Architecture:**
```
Input: [1, 3, 224, 224]  (rotated foot crop from Stage A)
        │
MobileNetV3-Small backbone (pretrained ImageNet)
        │
Feature Pyramid (4 scales)
        │
   ┌────┴────────────────────────────────┐
   │                                     │
Keypoint Head                    Meta Heads
[1, 16, 3] (x, y, v)            [1, 1]   presence
                                 [1, 5]   next_roi (cx,cy,w,h,θ)
                                 [1, 16]  sigma (per-KP uncertainty)
                                 [1, 16]  confidence
                                 [1, 6]   rotation (6D continuous)
                                 [1, 2]   scale (width, length mm)
                                 [1, 2,160,160]  mask (foot+shoe / occluder)
```

**Verification after build:**
```bash
python -c "
from src.models.stage_b.model import StageBModel
import torch
m = StageBModel()
x = torch.zeros(1, 3, 224, 224)
out = m(x)
print({k: v.shape for k, v in out.items()})
"
```

---

### Step 9 — Stage B Training

**Status:** 🔲 Not started  
**Required data:** 15,000 frames minimum (§9.1)  
**Current data:** 143 frames (pilot only)

**Training pipeline:**
1. Stage A detects foot → crops 224×224 rotated patch
2. Stage B trains on these patches
3. Loss: heatmap loss (KPs) + BCE (presence) + L1 (next_roi) + cos (rotation)
4. Visibility training: **v=2 ONLY** for visibility head (§6.2 — critical: wrist v17→v18 fix)

**Minimum metrics to pass M2 (§8):**

| Metric | Target |
|---|---|
| KP NME (v=2 only) | ≤ 0.06 |
| Presence AUC | ≥ 0.90 |
| Left/Right class accuracy | ≥ 99% |
| Stage A + B budget (GPU) | ≤ 18ms total |
| Stage A alone (WASM) | ≤ 12ms |
| Stage B alone (WASM) | ≤ 7ms |

---

### Step 10 — Stage B ONNX Export (opset 12)

**Status:** 🔲 Not started  

**Two separate ONNX files:**
- `stage_a_v1.0.onnx` — input `[1,3,320,320]` → output `[1,18,2100]`
- `stage_b_v1.0.onnx` — input `[1,3,224,224]` → multiple named outputs

**All outputs from Stage B must have baked sigmoid/softmax** (§2 frozen spec).

---

### Step 11 — Deliverables Package (M2)

**Status:** 🔲 Not started  

```
deliverables/
  stage_a_v1.0.onnx          fp32
  stage_a_v1.0_fp16.onnx     fp16
  stage_b_v1.0.onnx          fp32
  stage_b_v1.0_fp16.onnx     fp16
  foot_canonical.json         16 KP reference positions
  foot_meta.json              model_version, weights_sha256, sigmas, flip_idx
  foot_inference.js           JS crop transform helper
  golden_fixture.json         5 reference frames + expected outputs
  eval_report.pdf             metrics vs §8 thresholds
```

---

## Quick Commands Reference

```bash
# Activate environment
conda activate yolo

# Prepare dataset
python src/models/stage_a/prepare_stage_a_data.py

# Train Stage A
python src/models/stage_a/train_stage_a.py --epochs 200 --batch 16 --device 0 --name run1

# Test Stage A on video
python tools/evaluation/test_stage_a_video.py --model outputs/stage_a/run12/weights/best.pt

# Test Stage A live camera
python tools/evaluation/test_stage_a_live.py --model outputs/stage_a/run12/weights/best.pt

# Validate ONNX output
python -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('outputs/stage_a/run12/weights/best.onnx')
out = sess.run(None, {'images': np.zeros((1,3,320,320), np.float32)})
print('Stage A ONNX shape:', out[0].shape)
"
```

---

## File Structure (Current)

```
Shoes_VTO/
├── VERSION1_README.md              ← this file
├── SHOES_VTO_AI_REQUIREMENTS_v1.md ← AR SDK requirements (NGM)
├── AI_IMPLEMENTATION_PLAN_v2.md    ← full implementation plan
├── ANNOTATION_PRECISION_REFERENCE_v2.md  ← annotator guide
├── MY_ROLE.md                      ← Mostafa scope boundary
├── requirements.txt                ← pinned dependencies
│
├── data/
│   ├── shoes_v1/                   ← raw pilot dataset (143 images)
│   ├── stage_a/                    ← processed Stage A training data
│   │   ├── data.yaml               ← class 0=left_foot, 1=right_foot, 16 KP
│   │   ├── train/images+labels     ← 99 images
│   │   ├── valid/images+labels     ← 29 images
│   │   └── test/images+labels      ← 14 images
│   └── test_video.mp4              ← test video for evaluation
│
├── src/
│   └── models/
│       ├── stage_a/
│       │   ├── prepare_stage_a_data.py  ← data prep + class fix
│       │   └── train_stage_a.py         ← YOLOv8n-pose training + ONNX export
│       └── stage_b/                     ← 🔲 NOT YET BUILT
│
├── outputs/
│   └── stage_a/
│       └── run12/
│           └── weights/
│               ├── best.pt         ← best PyTorch checkpoint (6.8 MB)
│               └── best.onnx       ← ONNX opset 12 (12.6 MB) ✅
│
└── tools/
    └── evaluation/
        ├── test_stage_a_video.py   ← video inference test
        └── test_stage_a_live.py    ← live camera test
```

---

*Version 1 — Stage A complete. Stage B pending data collection and model build.*

