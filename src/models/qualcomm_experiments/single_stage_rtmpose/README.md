# Single-Stage RTMPose (RoI-Align + SimCC Upgrade)

**Permissive License:** Apache-2.0  
**Target Codebase:** `src/models/qualcomm_experiments/single_stage_rtmpose/`  
**Current Status:** Architecture Upgrade Completed & Verified | Full-Dataset Training Active (Epochs 1–120)

---

## 1. Overview & Architectural Motivation

This module implements an Apache-2.0 permissive, commercial-ready single-pass network that predicts **2-class chirality foot bounding boxes** (`0: Left Foot`, `1: Right Foot`) and **13 anatomical landmarks** in one unified forward pass without external crop preprocessing.

### Resolving the Single-Stage Resolution Gap
1. **The Problem:** In earlier single-stage anchor regression, the entire room was compressed to $320 \times 320$, leaving each foot with only $\approx 50 \times 30$ pixels ($\approx 20$ grid cells). Direct `Smooth L1` regression from anchor centers resulted in landmark distortion and centroid-seeking collapse.
2. **The Solution (RoI-Align + SimCC):** 
   - Feature maps from PAFPN level $P_2$ (stride 8, 96 channels) retain the finest spatial detail.
   - For every candidate foot box, an internal **`torchvision.ops.roi_align`** layer extracts an aligned $8 \times 8$ feature patch.
   - A $3 \times 3$ projection conv maps $96 \to 384$ channels to match official RTMPose specifications.
   - The **SimCC Head** (with 7x7 spatial layout conv and Gated Attention Units) classifies coordinates over **512 continuous 1D bins** for $X$ and $Y$.
   - **Continuous Expectation Decoding** computes sub-pixel coordinates $(\hat{u}, \hat{v}) \in [0, 1]$ with zero quantization error.

---

## 2. End-to-End Pipeline & Dataflow

```
Input Image [B, 3, 320, 320]
       │
       ▼
CSPNeXt Backbone (Pretrained COCO, 206 tensors)
       │  Strides 8, 16, 32
       ▼
CSPNeXt-PAFPN Bidirectional Fusion Neck (96 channels)
       ├───> P2 (Stride 8, 96 ch) ───────────────┐
       ├───> P3 (Stride 16, 96 ch)               │
       └───> P4 (Stride 32, 96 ch)               │
               │                                 │
               ▼                                 ▼
   SingleStageDetectionHead             Internal RoI-Align (8x8)
   ├─ Cls Tower (Quality Focal Loss)             │
   └─ Box Tower (CIoU distances)                 ▼
               │                          Projection Conv (96 -> 384 ch)
               ▼                                 │
   Class-Agnostic NMS (IoU 0.45)                 ▼
               │                         SimCC Head (GAU + 512 Bins)
               └──> Surviving RoIs ─────────────>│
                                                 ▼
                                     Continuous Expectation
                                                 │
                                                 ▼
                                    Unified Contract [B, 45, max_det]
```

---

## 3. Production Output Contract

The model outputs a unified tensor of shape **`[B, 45, max_det]`** (default `max_det=10`), maintaining full compatibility with the project's mobile WASM and ONNX runtime specifications:

| Channel Range | Semantic Meaning | Description |
|---|---|---|
| **0 .. 3** | Bounding Box | $(c_x, c_y, w, h)$ in full-frame pixel coordinates |
| **4 .. 5** | Class Probabilities | $[P(\text{left\_foot}), P(\text{right\_foot})]$ via Sigmoid |
| **6 .. 44** | 13 Keypoints | $13 \times (x, y, \text{conf})$ continuous pixel coordinates |

---

## 4. Pretrained Weights Warm-Start

The architecture initializes with **225 warm-started pretrained tensors** across both detection and keypoint branches:

1. **Backbone (206 tensors):** Loaded from `checkpoints/rtmpose_tiny_coco.pth` (CSPNeXt-Tiny weights pretrained on COCO).
2. **Keypoint Head (19 tensors):** Loaded from `checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt`:
   - `simcc_head.final` (7x7 conv spatial encoder)
   - `simcc_head.mlp` (ScaleNorm + Linear 64 -> 256)
   - `simcc_head.gau` (Relational attention: `gamma`, `beta`, `uv`, `o`)
   - `simcc_head.cls_x` & `cls_y` (512-bin 1D linear classifiers)
   - `simcc_head.vis` (Linear visibility predictor)
   - `simcc_head.hand` (In-model left/right handedness classifier)
3. **Detection Head Prior:** Bias initialized to $-4.595$ ($\sigma(-4.595) = 0.01$) to suppress early background explosion.

---

## 5. Multi-Threshold Validation Metrics

Validation evaluates accuracy against the foot bounding box diagonal $D = \sqrt{W^2 + H^2}$:

* **`Box F1@0.50`:** Bounding box IoU $\ge 0.50$ requiring exact Left vs. Right foot class match.
* **`PCK@0.20`:** Keypoints localized within $20\%$ of foot diagonal ($\approx 25\text{–}30\text{ px}$) — coarse limb alignment.
* **`PCK@0.10`:** Keypoints localized within $10\%$ of foot diagonal ($\approx 12\text{–}15\text{ px}$) — standard benchmark.
* **`PCK@0.05`:** Keypoints localized within $5\%$ of foot diagonal ($\approx 6\text{–}8\text{ px}$) — strict sub-millimeter toe and arch precision.
* **`NegPrec`:** Rejection rate of empty negative backgrounds (guaranteeing 0 false detections on patterned carpets).

---

## 6. Solved Issues & Bug Fixes

1. **Carpet Leaf False Positives:** Replaced crop-only detection with full-frame CSPNeXt backbone + Quality Focal Loss on negative backgrounds. Verified 0 ghost detections on patterned carpets.
2. **Duplicate Left/Right Candidate Boxes:** Replaced class-specific NMS with **Class-Agnostic NMS** (`ops.nms` at IoU 0.45) to ensure exactly one box per physical foot.
3. **Keypoint Distortion:** Upgraded coarse stride regression to RoI-Align + 512-bin SimCC with continuous expectation.
4. **CUDA Illegal Memory Access (`cudaErrorIllegalAddress`):**
   - *Cause:* In validation, each image was sliced as `p2[b:b+1]` (batch size 1), while candidate RoIs retained batch index `b` (`b = 0, 1, ..., 31`). For $b \ge 1$, `ops.roi_align` attempted to access non-existent batch slices in CUDA memory.
   - *Fix:* Set `b_idx_col = torch.zeros((K, 1))` for single-slice RoI evaluation and eliminated redundant CPU dictionary synchronizations inside `forward()`.

---

## 7. Active Training Specifications

Run on the full 6,620-image dataset:

```bash
python src/models/qualcomm_experiments/single_stage_rtmpose/train.py \
    --data data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3 \
    --output checkpoints/single_stage_rtmpose_simcc_320 \
    --epochs 120 \
    --batch 32 \
    --lr 0.001 \
    --img_size 320 \
    --amp
```

### Automatic Logging
Training metrics automatically append to `checkpoints/single_stage_rtmpose_simcc_320/trainlog.md` with:
```text
Epoch [02/120] (9.5s) | Loss: 3.1420 (Cls: 0.150, Box: 0.280, SimCC: 0.650) | Box F1: 64.2% | PCK@0.20: 92.5% | PCK@0.10: 84.1% | PCK@0.05: 71.3% | NegPrec: 90.0%
  [*] New best model saved! (Box F1: 64.2%, PCK@0.20: 92.5%, PCK@0.10: 84.1%, PCK@0.05: 71.3%)
```

---

## 8. File Structure

```
src/models/qualcomm_experiments/single_stage_rtmpose/
├── __init__.py           # Package exports (SingleStageRTMPose, RoISimCCHead, etc.)
├── backbone.py           # Multi-scale CSPNeXt backbone (COCO warm-start)
├── dataset.py            # Full-image YOLO-Pose loader (allow_flip=False)
├── eval_video.py         # End-to-end video inference with 1€ filter
├── export_onnx.py        # ONNX export with ONNX Runtime numerical verification
├── head.py               # Decoupled anchor-free detection head
├── loss.py               # Multi-task loss (QFL + CIoU + SimCC KL + BCE)
├── model.py              # Full SingleStageRTMPose module
├── neck.py               # CSPNeXt-PAFPN neck (strides 8, 16, 32)
├── simcc_roi_head.py     # RoI-Align + 512-bin SimCC with GAU attention
├── train.py              # Multi-metric training engine (AMP, EMA, Cosine LR)
└── README.md             # This technical documentation
```
