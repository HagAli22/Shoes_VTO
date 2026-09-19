# 🎯 Mostafa — AI / Perception Engineer Role
### Shoe AR Try-On Project

---

## My Role in the Team

```
                   Mostafa
                AI / Perception
                      │
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
   Dataset          YOLO          RTMPose
   18 KP           Detector        18 KP
       │              │              │
       └──────────────┴──────────────┘
                      ↓
             AI Perception API
                      ↓
         { bbox + 18 keypoints }
                      │
                      ▼
             HANDOFF TO AR ENGINEER
```

> **My job ends** when I deliver a working model that outputs:
> `{ bbox: [x, y, w, h],  keypoints: [18 × (x, y, confidence)] }`
>
> The AR engineer takes this output and handles: 6DoF pose, 3D shoe rendering, and occlusion.

---

## My 3 Deliverables

### 1. 📦 Dataset — 18 Keypoints Per Image

Collect and annotate a dataset of foot images with **9 keypoints per foot** (18 total).

**Keypoints I use:**

| Index | Left Foot | Index | Right Foot |
|---|---|---|---|
| 0 | big_toe | 9 | big_toe |
| 1 | little_toe | 10 | little_toe |
| 2 | near_little_toe | 11 | near_little_toe |
| 3 | near_big_toe | 12 | near_big_toe |
| 4 | far_big_toe | 13 | far_big_toe |
| 5 | far_little_toe | 14 | far_little_toe |
| 6 | dorsum | 15 | dorsum |
| 7 | heel | 16 | heel |
| 8 | upper_heel | 17 | upper_heel |

**Dataset plan (3 stages, minimal manual work):**

| Stage | Source | Manual Annotation | KP Available |
|---|---|---|---|
| 1 — Pretraining | CMU + H3WB (public, free) | ❌ None | 3/foot (big_toe, little_toe, heel) |
| 2 — Synthetic | Foot3D + Blender renderer | ❌ None (auto-labeled) | 9/foot (all 18) |
| 3 — Fine-tuning | My real videos | ✅ 3,000 frames only | 9/foot (all 18) |

**Script to convert public data:**
```bash
python src/dataset/convert_to_schema.py \
    --dataset cmu \
    --input /path/to/cmu \
    --output data/annotations/cmu_18kp.json
```

**For annotation team:** see [`dataset_annotation_guidelines.md`](dataset_annotation_guidelines.md)

---

### 2. 🔍 YOLO Foot Detector

**What it does:** Takes a camera frame → outputs foot bounding boxes.

**Model:** YOLOv8n (smallest, fastest — runs real-time on mobile)

**My target:** mAP@50 > 0.90

**Train:**
```bash
python src/models/detector/train_detector.py \
    --data datasets/foot_detection/data.yaml \
    --epochs 100 \
    --batch 16 \
    --work-dir outputs/detector/
```

**What I need to prepare for training:**
```
datasets/foot_detection/
    images/
        train/   ← foot images
        val/     ← foot images
    labels/
        train/   ← YOLO format .txt (class cx cy w h)
        val/
    data.yaml
```

**Output after training:** `outputs/exports/foot_detector.onnx`

---

### 3. 🦶 RTMPose-Tiny — 18 Keypoint Model

**What it does:** Takes a cropped foot image (256×192) → outputs 18 keypoints (x, y, confidence).

**Model:** RTMPose-Tiny (3.51M params, 0.37 GFLOPs — very lightweight)

**My target:** OKS > 0.85 · PCK@5px > 80%

**Train (MMPose — preferred):**
```bash
mim train mmpose configs/rtmpose_tiny_18kp.py \
    --work-dir outputs/rtmpose_tiny_foot
```

**Train (pure PyTorch — fallback):**
```bash
python src/models/keypoint/train_keypoint.py \
    --config configs/training_config.yaml \
    --work-dir outputs/
```

**Export to ONNX (final deliverable format):**
```bash
python src/export/export_onnx.py \
    --checkpoint outputs/checkpoints/best.pth \
    --output outputs/exports/rtmpose_tiny_18kp.onnx \
    --config configs/training_config.yaml
```

**Output after export:** `outputs/exports/rtmpose_tiny_18kp.onnx`

---

## My Files (What I Work With)

```
shoes_vto/
│
├── src/dataset/
│   ├── foot_dataset.py          ← My dataset loader (PyTorch)
│   ├── augmentations.py         ← My augmentation pipeline
│   └── convert_to_schema.py     ← Convert public datasets to 18-KP format
│
├── src/models/detector/
│   ├── foot_detector.py         ← YOLO inference wrapper (for testing)
│   └── train_detector.py        ← YOLO training script
│
├── src/models/keypoint/
│   ├── keypoint_model.py        ← RTMPose inference wrapper (for testing)
│   └── train_keypoint.py        ← RTMPose training script
│
├── src/export/
│   └── export_onnx.py           ← Export .pth → .onnx (my final deliverable)
│
├── configs/
│   ├── rtmpose_tiny_18kp.py     ← MMPose training config
│   └── training_config.yaml     ← All hyperparameters
│
├── tools/evaluation/
│   ├── eval_keypoints.py        ← OKS + PCK metrics (I use to verify my model)
│   └── benchmark.py             ← FPS test (I verify real-time performance)
│
└── data/
    ├── annotations/             ← My COCO JSON files
    ├── processed/               ← My training images
    └── raw/                     ← My original videos
```

**Not my files** (AR engineer's responsibility):
- `src/models/geometry/` → PnP solver, 3D foot model
- `src/tracking/` → Alpha-Beta filter
- `src/pipeline/inference_pipeline.py` → full AR pipeline

---

## My Execution Order

```
Step 1 ─ Collect videos (my team)
         ↓
Step 2 ─ Convert public data (CMU + H3WB) → no manual work
         python src/dataset/convert_to_schema.py
         ↓
Step 3 ─ Annotate 3,000 real frames (annotation team)
         See: dataset_annotation_guidelines.md
         ↓
Step 4 ─ Train YOLO foot detector
         python src/models/detector/train_detector.py
         ↓
Step 5 ─ Train RTMPose-Tiny 18 KP
         mim train mmpose configs/rtmpose_tiny_18kp.py
         ↓
Step 6 ─ Evaluate my models
         python tools/evaluation/eval_keypoints.py
         ↓
Step 7 ─ Export to ONNX
         python src/export/export_onnx.py
         ↓
Step 8 ─ HANDOFF to AR engineer
         outputs/exports/foot_detector.onnx
         outputs/exports/rtmpose_tiny_18kp.onnx
```

---

## My Quality Gates (Before Handoff)

I do **not** hand off to the AR engineer until:

| Check | Metric | My Target |
|---|---|---|
| Foot Detector | mAP@50 | **> 0.90** |
| Keypoint Model | OKS | **> 0.85** |
| Keypoint Model | PCK @ 5px | **> 80%** |
| Speed (CPU) | Detector + Keypoints | **< 40ms total** |
| Export | ONNX numerical diff | **< 1e-4** |

---

## What I Hand Off to AR Engineer

Two ONNX model files + their input/output specs:

### foot_detector.onnx
```
Input:   image frame  [1, 3, 640, 640]   BGR float32
Output:  detections   [1, 5, N]          (cx, cy, w, h, conf)
```

### rtmpose_tiny_18kp.onnx
```
Input:   foot crop    [1, 3, 256, 192]   RGB float32, ImageNet normalised
Output:  heatmaps     [1, 18, 64, 48]    → decode to (x, y, confidence)
```

### Final API output per frame:
```python
{
  "left_foot": {
    "bbox":      [x, y, w, h],            # pixels in original frame
    "keypoints": [[x0,y0,c0], ..., [x8,y8,c8]]   # 9 points
  },
  "right_foot": {
    "bbox":      [x, y, w, h],
    "keypoints": [[x9,y9,c9], ..., [x17,y17,c17]] # 9 points
  }
}
```

---

## Key Numbers to Remember

| Item | Value |
|---|---|
| Keypoints per foot | **9** |
| Total keypoints | **18** |
| Model input size | **256 × 192** |
| Heatmap output size | **64 × 48** |
| Detector input size | **640 × 640** |
| RTMPose params | **3.51M** |
| RTMPose GFLOPs | **0.37G** |
| Real data needed | **3,000 frames** |
| Public data available | **~114K images** (free, no labeling) |

---

*Based on: Nguyen et al., "A real-time mobile solution for shoe try-on using foot pose estimation," Complex & Intelligent Systems, 2026.*

