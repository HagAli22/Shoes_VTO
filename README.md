# Shoe AR Try-On — AI Perception Layer

> **Real-time mobile foot pose estimation for virtual shoe try-on.**
> Based on: *"A real-time mobile solution for shoe try-on using foot pose estimation and 3D processing techniques"*, Nguyen et al., Complex & Intelligent Systems, 2026.

---

## System Overview

```
Camera Frame
     │
     ▼
Foot Detector (YOLOv8n)        ← detects foot bounding boxes
     │
     ▼
RTMPose-Tiny (18 KP)           ← estimates 9 keypoints per foot
     │
     ▼
EPnP / IPnP Solver             ← computes 6DoF pose (R, T)
     │
     ▼
Alpha-Beta Filter + IOU Gate   ← temporal stabilisation
     │
     ▼
Perception Output: {left_foot, right_foot} → (rvec, tvec, keypoints)
```

---

## Project Structure

```
shoes_vto/
├── src/
│   ├── dataset/
│   │   ├── foot_dataset.py         # PyTorch Dataset (18 KP, heatmaps)
│   │   ├── augmentations.py        # Foot-aware augmentation pipeline
│   │   └── convert_to_schema.py    # CMU / H3WB → unified 18-KP COCO JSON
│   ├── models/
│   │   ├── detector/
│   │   │   ├── foot_detector.py    # YOLOv8n ONNX inference wrapper
│   │   │   └── train_detector.py   # YOLOv8n training script
│   │   ├── keypoint/
│   │   │   ├── keypoint_model.py   # RTMPose-Tiny ONNX inference wrapper
│   │   │   └── train_keypoint.py   # Training: AdamW + MSE heatmap loss
│   │   └── geometry/
│   │       ├── foot_3d_model.py    # 3D reference foot (9 KP, mm coords)
│   │       ├── camera_model.py     # Camera intrinsics + calibration
│   │       └── pnp_solver.py       # EPnP + IPnP 6DoF solver
│   ├── tracking/
│   │   └── alpha_beta_filter.py    # Alpha-Beta filter + IOU gate
│   ├── pipeline/
│   │   └── inference_pipeline.py  # Full end-to-end AI perception pipeline
│   └── export/
│       └── export_onnx.py          # Export trained model to ONNX
├── configs/
│   ├── rtmpose_tiny_18kp.py        # MMPose training config
│   └── training_config.yaml        # All hyperparameters
├── tools/evaluation/
│   ├── eval_keypoints.py           # OKS + PCK metrics
│   ├── eval_6dof.py                # Rotation/translation error
│   └── benchmark.py                # FPS / latency benchmarking
├── data/                           # Dataset files (not committed)
├── outputs/                        # Checkpoints + ONNX exports
└── requirements.txt
```

---

## Keypoint Schema

18 keypoints total (9 per foot), matching the Springer 2026 paper:

| Index | Left Foot         | Index | Right Foot         |
|-------|-------------------|-------|--------------------|
| 0     | big_toe           | 9     | big_toe            |
| 1     | little_toe        | 10    | little_toe         |
| 2     | near_little_toe   | 11    | near_little_toe    |
| 3     | near_big_toe      | 12    | near_big_toe       |
| 4     | far_big_toe       | 13    | far_big_toe        |
| 5     | far_little_toe    | 14    | far_little_toe     |
| 6     | dorsum            | 15    | dorsum             |
| 7     | heel              | 16    | heel               |
| 8     | upper_heel        | 17    | upper_heel         |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Convert public datasets to unified schema

```bash
# CMU Foot Keypoint Dataset
python src/dataset/convert_to_schema.py \
    --dataset cmu \
    --input /path/to/cmu_dataset \
    --output data/annotations/cmu_18kp.json

# H3WB (extract foot KPs)
python src/dataset/convert_to_schema.py \
    --dataset h3wb \
    --input /path/to/h3wb \
    --output data/annotations/h3wb_18kp.json
```

### 3. Train foot detector

```bash
python src/models/detector/train_detector.py \
    --data datasets/foot_detection/data.yaml \
    --epochs 100 \
    --batch 16 \
    --work-dir outputs/detector/
```

### 4. Train keypoint model

```bash
# Option A: MMPose (preferred)
mim train mmpose configs/rtmpose_tiny_18kp.py \
    --work-dir outputs/rtmpose_tiny_foot

# Option B: Pure PyTorch fallback
python src/models/keypoint/train_keypoint.py \
    --config configs/training_config.yaml \
    --work-dir outputs/
```

### 5. Export to ONNX

```bash
python src/export/export_onnx.py \
    --checkpoint outputs/checkpoints/best.pth \
    --output outputs/exports/rtmpose_tiny_18kp.onnx \
    --config configs/training_config.yaml
```

### 6. Evaluate

```bash
# Keypoint accuracy
python tools/evaluation/eval_keypoints.py \
    --gt data/annotations/test_annotations.json \
    --model outputs/exports/rtmpose_tiny_18kp.onnx \
    --image-dir data/processed/

# 6DoF accuracy
python tools/evaluation/eval_6dof.py \
    --gt-poses data/gt_poses.json \
    --pred-poses outputs/pred_poses.json

# FPS benchmark
python tools/evaluation/benchmark.py \
    --detector outputs/exports/foot_detector.onnx \
    --keypoint outputs/exports/rtmpose_tiny_18kp.onnx \
    --camera 0
```

### 7. Run live pipeline (webcam test)

```bash
python src/pipeline/inference_pipeline.py \
    --detector outputs/exports/foot_detector.onnx \
    --keypoint outputs/exports/rtmpose_tiny_18kp.onnx \
    --camera 0
```

---

## Quality Targets

| Metric              | Target      |
|---------------------|-------------|
| Foot Detector mAP@50| > 0.90      |
| Keypoint OKS        | > 0.85      |
| PCK @ 5px           | > 80%       |
| Rotation error      | < 5°        |
| Translation error   | < 10 mm     |
| FPS (iPhone 11)     | ≥ 30 FPS    |
| FPS (mid Android)   | ≥ 25 FPS    |

---

## Dataset Strategy

| Stage | Data Source | KP/foot | Manual Work |
|-------|-------------|---------|-------------|
| 1 — Pretraining | CMU + H3WB + MOOF | 3 | None |
| 2 — Synthetic | Foot3D + Blender renderer | 9 | None |
| 3 — Fine-tuning | Your real videos | 9 | 3,000 frames |

See [`dataset_annotation_guidelines.md`](dataset_annotation_guidelines.md) for annotator handbook.

---

## Configuration

All hyperparameters are in [`configs/training_config.yaml`](configs/training_config.yaml).
MMPose-specific settings are in [`configs/rtmpose_tiny_18kp.py`](configs/rtmpose_tiny_18kp.py).

