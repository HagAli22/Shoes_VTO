# Two-Stage Foot Detection & 13-Keypoint Pose Pipeline (Status: NanoDet-Plus improved5 + RTMPose-13KP)

## 1. Executive Summary

This document formalizes the production-ready **Two-Stage Qualcomm / Apache-2.0 Foot Detection & 13-Keypoint Pose Estimation System**, updated with the **`improved5`** iteration fine-tuned on the comprehensive v6 dataset (including challenging camera perspectives and hard negative distractor scenes).

The system was designed from the ground up to overcome the commercial licensing limitations (AGPLv3) of YOLO and optimize edge NPU deployment on Qualcomm Snapdragon hardware. Both stages are exported and validated at **ONNX opset=12** across FP32, FP16, and INT8.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              TWO-STAGE ARCHITECTURE                                    │
│                                                                                        │
│   Input Frame [H, W, 3]                                                                │
│          │                                                                             │
│          ▼                                                                             │
│   ┌────────────────────────────────────────────────────────┐                           │
│   │ STAGE A: NanoDet-Plus (improved5)                      │                           │
│   │ Backbone: ShuffleNetV2 1.0x (Native ReLU)              │                           │
│   │ Neck: RepPAN | Head: NanoDetPlusHead (reg_max=16)      │                           │
│   │ Resolution: 256x256 | License: Apache-2.0              │                           │
│   │ Model Size: 3.97 MB FP32 (2.08 MB FP16 / 1.37 MB INT8) │                           │
│   └────────────────────────┬───────────────────────────────┘                           │
│                            │ Bounding Boxes [x1, y1, x2, y2]                           │
│                            ▼                                                           │
│   ┌────────────────────────────────────────────────────────┐                           │
│   │ DUAL-FOOT VTO & ANATOMICAL GATE                        │                           │
│   │ • Aspect ratio & scale normalization (1.25x crop)       │                           │
│   │ • Rejects carpet leaves, shadows, floor patterns       │                           │
│   └────────────────────────┬───────────────────────────────┘                           │
│                            │ Normalized Foot Crops [256x256]                           │
│                            ▼                                                           │
│   ┌────────────────────────────────────────────────────────┐                           │
│   │ STAGE B: RTMPose-13KP Hand (improved5)                 │                           │
│   │ Backbone: CSPNeXt-Tiny | Neck/Head: RTMCCHead (GAU)    │                           │
│   │ Coordinate Schema: 1D SimCC (Split ratio: 2.0)         │                           │
│   │ Heads: 13 Keypoints + Chirality (Left vs Right Foot)   │                           │
│   │ Resolution: 256x256 | License: Apache-2.0              │                           │
│   │ Model Size: 14.24 MB FP32 (7.14 MB FP16 / 3.70 MB INT8)│                           │
│   └────────────────────────┬───────────────────────────────┘                           │
│                            │ 13 Keypoints + Handedness Prob                            │
│                            ▼                                                           │
│   ┌────────────────────────────────────────────────────────┐                           │
│   │ TEMPORAL STABILIZATION & TRACKING                      │                           │
│   │ • One-Euro Vectorized Low-Pass Filter                  │                           │
│   │ • Dual-Foot Chirality-Locked Tracking (Max 2 Feet)     │                           │
│   │ • Smooth Box & Joint Jitter Elimination                │                           │
│   └────────────────────────────────────────────────────────┘                           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Checkpoint Details & Training Trajectory

### A. Stage A: NanoDet-Plus (Foot Detector)
- **Checkpoint**: `checkpoints/nanodet_stage_a_improved5/best.pt`
- **ONNX Model**: `deliverables/qualcomm_experiments/models/nanodet_stage_a_improved5_opset12.onnx`
- **ONNX Opset**: `12` | **File Sizes**: `3.97 MB` (FP32), `2.08 MB` (FP16), `1.37 MB` (INT8)
- **Input Specs**: `[1, 3, 256, 256]`, Standardized BGR in `[0, 1]`
- **Outputs**: `boxes: [1, 1344, 4]` (corner coords `[x1, y1, x2, y2]`), `scores: [1, 1344, 1]`

#### Convergence Evolution Across Iterations:
| Experiment | Checkpoint | Training Setup | Final Loss | F1@0.50 | F1@0.75 | Neg Precision | mIoU |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | `nanodet_stage_a_baseline` | 4-lvl FPN, leaky, TAL from scratch | 1.1240 | 72.1% | 46.8% | 52.0% | 71.4% |
| **Improved 1** | `nanodet_stage_a_improved` | 3-lvl RepPAN, native ReLU, reg_max=16 | 0.8120 | 83.4% | 59.2% | 68.5% | 76.8% |
| **Improved 2** | `nanodet_stage_a_improved2` | Warmup TAL + CIoU, lr 8e-4 | 0.7045 | 86.2% | 64.9% | 74.2% | 79.1% |
| **Improved 3** | `nanodet_stage_a_improved3` | Merged negative3 dataset, lr 5e-4 | 0.6641 | 87.8% | 67.8% | 79.8% | 80.5% |
| **Improved 4** | `nanodet_stage_a_improved4` | Fine-tuning continuation 40 ep, lr 3e-4 | 0.6233 | 88.6% | 70.34% (75.13% @ c35) | 83.3% | 81.3% |
| **Improved 5 (Current Production)** | **`nanodet_stage_a_improved5`** | Full v6 merged + 782 negatives (9,120 train / 1,138 val) | **0.5968** | **88.2%** | **69.23% (73.26% @ c35)** | **86.4%** | **81.94%** |

### B. Stage B: RTMPose-13KP (Pose + Chirality)
- **Checkpoint**: `checkpoints/rtmpose_13kp_improved5/best_with_hand.pt`
- **ONNX Model**: `deliverables/qualcomm_experiments/models/rtmpose_13kp_hand_improved5_opset12.onnx`
- **ONNX Opset**: `12` | **File Sizes**: `14.24 MB` (FP32), `7.14 MB` (FP16), `3.70 MB` (INT8)
- **Input Specs**: `[1, 3, 256, 256]`, Standardized BGR in `[0, 1]`
- **Outputs**:
  - `simcc_x`: `[1, 13, 512]` (1D horizontal bin logits, split ratio 2.0)
  - `simcc_y`: `[1, 13, 512]` (1D vertical bin logits, split ratio 2.0)
  - `vis_logits`: `[1, 13]` | `vis`: `[1, 13]` (keypoint visibility confidence)
  - `hand_logits`: `[1, 1]` | `hand`: `[1, 1]` (chirality probability: `<0.5` Left Foot, `≥0.5` Right Foot)
- **Chirality Head Accuracy**: **96.41%** (improved from 95.51% in improved4).
- **Keypoint Localization**: PCK@0.05: **64.27%** (up from 63.3%), PCK@0.10: **88.31%**, Mean Pixel Error: **10.28 px** (down from 10.59 px).

---

## 3. Post-Processing & Filtering Stack

1. **Aspect-Ratio & Scale-Normalized Cropping**:
   - Every detected box is expanded by $1.25\times$ padding to ensure the whole foot contour, ankle, and toes are centered inside the $256 \times 256$ crop.
   - Preserves high spatial detail without warping or aspect ratio distortion.

2. **Anatomical Clutter Gate**:
   - Rejects false alarm candidates from floor rugs, carpet leaf motifs, and shadows.
   - Criteria: Candidate must have $\text{mean\_kpt\_conf} \ge 0.15$ and at least 3 keypoints with confidence $\ge 0.20$.

3. **Dual-Foot Chirality Partitioning**:
   - Groups candidates by `hand_prob`: Left Foot candidates (`< 0.50`) and Right Foot candidates (`≥ 0.50`).
   - Limits active tracks to at most 1 Left Foot and 1 Right Foot (Active Tracks $\le 2$).
   - Horizontal fallback resolver resolves cases where feet cross or are momentarily labeled identical.

4. **One-Euro Vectorized Temporal Filter**:
   - Keypoint coordinates and bounding box corners are filtered via adaptive cutoff low-pass filters:
     - Low velocity $\to$ low cutoff frequency ($1.2\text{ Hz}$) eliminating jitter.
     - Fast movement $\to$ dynamic cutoff frequency ($>15\text{ Hz}$) eliminating lag.

---

## 4. Production Deployment Instructions

### Python Inference (PyTorch & ONNX):
```bash
# Video Inference with Two-Stage Pipeline (improved5)
python src/models/qualcomm_experiments/rtmpose/eval_video.py \
    --video data/test_video.mp4 \
    --detector checkpoints/nanodet_stage_a_improved5/best.pt \
    --rtmpose checkpoints/rtmpose_13kp_improved5/best_with_hand.pt \
    --output test_video_results/improved5_twostage.mp4 \
    --conf 0.40
```

### ONNX Runtime Inference:
```python
import onnxruntime as ort
import numpy as np

# Stage A: Foot Detection (improved5)
det_sess = ort.InferenceSession("deliverables/qualcomm_experiments/models/nanodet_stage_a_improved5_opset12.onnx")
boxes, scores = det_sess.run(None, {"images": det_input_tensor})

# Stage B: 13-Keypoint Pose & Chirality (improved5)
pose_sess = ort.InferenceSession("deliverables/qualcomm_experiments/models/rtmpose_13kp_hand_improved5_opset12.onnx")
simcc_x, simcc_y, _, vis, _, hand = pose_sess.run(None, {"images": crop_input_tensor})
```
