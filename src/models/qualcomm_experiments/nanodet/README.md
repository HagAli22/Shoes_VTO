# NanoDet-Plus (2-Class) Foot Detector

**Permissive License:** Apache-2.0  
**Target Codebase:** `src/models/qualcomm_experiments/nanodet/`  
**Latest Checkpoint:** `checkpoints/nanodet_foot_2class_320/best.pt` (Epoch 120, F1@0.75: 67.82%, mIoU: 83.26%, NegPrec: 83.3%, 1.17M params)

---

## 1. Overview & Evaluation Summary

NanoDet-Plus was trained for 120 epochs on 6,620 images with 2 distinct anatomical classes:
* `class_id 0`: Left Foot
* `class_id 1`: Right Foot

Paired with **RTMPose-13KP SimCC** (`checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt`) on `data/test_video.mp4` (574 frames, 1440x1440).

### What Succeeded
1. **Accurate Chirality on Real Feet:** 2-class supervision achieved $98\%\text{–}99\%$ confidence on real feet (Left Foot: $0.98\text{–}0.99$, Right Foot: $0.94\text{–}0.99$).
2. **Duplicate Box Removal:** Eliminated overlapping Left/Right candidate boxes on the same physical foot.
3. **Keypoint Quality:** Real feet retain crisp 13-point anatomical landmarks via dedicated $256 \times 256$ SimCC crops.

---

## 2. Critical Failure Modes & Root Causes

Visual analysis of evaluation frames (`nanodet_2class_demo.mp4`) revealed severe flaws:

### 1. Hardcoded 256px Bottom-Edge Truncation
* **Symptom:** In wide frames (e.g., Frame 147, 252, 464), feet at the bottom of the frame have their heels and ankles cut off horizontally at exactly $y = 1152\text{ px}$.
* **Root Cause:** In `model.py::_decode_predictions()`, `max_w` and `max_h` were hardcoded to `self.input_size = (256, 256)`. When running on $320 \times 320$, all bounding box bottom edges were clamped to 256.0! When scaled to original $1440 \times 1440$ video ($1440 / 320 = 4.5$), $256.0 \times 4.5 = 1152.0\text{ px}$, amputating heels and ankles.

### 2. Giant Phantom Bounding Boxes
* **Symptom:** In Frame 39, an enormous orange rectangle covers the entire top of the screen.
* **Root Cause:** NanoDet includes Stride 64 ($5 \times 5$ grid at 320x320). An anchor on Stride 64 has a reach of up to $512\text{ px}$, predicting a giant phantom box. In addition, `ops.batched_nms` evaluated classes independently, allowing the giant box to coexist with other detections.

### 3. Persistent Carpet Leaf False Positives
* **Symptom:** In Frames 39, 147, 252, and 464, the large orange/pink carpet leaf pattern continues to be detected as a `Left Foot` ($0.75\text{–}0.97$ confidence).
* **Root Cause:** 
  1. The 653 negative images in the training set are web-scraped socks and stockings, not empty crops of this room's carpet leaves.
  2. ShuffleNetV2 1.0x (1.17M params) has a small local receptive field. It views the leaf outline in isolation and confuses its shape with a foot sole.

### 4. Track ID Explosion (IDs Climbing into the 60s)
* **Symptom:** Over 574 frames, track IDs frequently died and re-spawned, reaching IDs in the 60s.
* **Root Cause:** Because the foot boxes were artificially clamped at $y=1152$, their IoU between frames dropped below the 0.25 tracking threshold during stepping motion, causing the tracker to kill the active track.

---

## 3. Required Improvements

### Phase 1: Immediate Inference Fixes (No Retraining Required)
1. **Dynamic Clamping:** In `NanoDetPlus._decode_predictions()`, dynamically compute `max_w` and `max_h` from the input feature tensor ($W \times \text{stride}$), removing the 256px ceiling so heels/ankles are never cut off.
2. **Class-Agnostic NMS:** Replace `ops.batched_nms` with `ops.nms` with IoU threshold 0.45.
3. **Anatomical Area Filter:** Reject detections where $(w \times h) > 0.30 \times \text{area}$ or $(w \times h) < 0.005 \times \text{area}$ to immediately kill giant phantom boxes.
4. **Spatial Tracking Prior:** Require confidence $\ge 0.55$ for unanchored detections in the upper half of the frame ($y < 0.6 \times H$), while keeping normal confidence ($0.35$) for lower-frame detections.

### Phase 2: Architectural Retraining (Model Level)
1. **Drop Stride 64:** Restrict GhostPAN and Head to Strides `[8, 16, 32]`. Feet in a 320px image never require Stride 64.
2. **Hard Carpet Negative Mining:** Extract empty carpet crops directly from the room video and add them to the negative training pool.
3. **Mosaic Augmentation:** Mix floor textures into the training images so the network explicitly learns that carpet leaves are background.

