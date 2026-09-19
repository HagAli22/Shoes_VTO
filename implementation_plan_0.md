# 🥿 Real-Time Mobile Shoe AR Try-On — Full Production Plan

## Executive Summary

This plan covers the complete production system for a **real-time mobile shoe AR try-on** application. It is based on the 2026 Springer paper *"A real-time mobile solution for shoe try-on using foot pose estimation and 3D processing techniques"* and the architectural analysis in `idea.md`. The system runs on-device (iOS/Android) at real-time frame rates using a lightweight deep learning pipeline.

---

## System Architecture (Full Pipeline)

```
Camera Frame
     │
     ▼
┌─────────────────┐
│  Foot Detector  │  ← YOLOv8n / YOLO-NAS-S  (full frame 640×480)
└────────┬────────┘
         │  Left ROI + Right ROI (bounding boxes)
         ▼
┌─────────────────────┐
│  RTMPose-Tiny       │  ← 256×192 crop, 3.51M params, 0.37 GFLOPs
│  (9 KP × 2 feet)    │
└────────┬────────────┘
         │  18 × (x, y, confidence)
         ▼
┌─────────────────────┐
│  6DoF Estimation    │  ← EPnP / IPnP + intrinsic camera matrix
│  (PnP Algorithm)    │
└────────┬────────────┘
         │  Rotation R + Translation T
         ▼
┌─────────────────────┐
│  Alpha-Beta Filter  │  ← Temporal stabilization (jitter elimination)
│  + IOU Gate         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3D Post-Process    │  ← 3D shoe model transform + Ray-Casting occlusion
│  + Rendering        │
└────────┬────────────┘
         │
         ▼
      Output AR Frame
```

---

## Paper Key Findings (What Worked in Production)

| Finding | Value |
|---|---|
| Keypoints per foot | **9** |
| Total keypoints | **18** |
| Model architecture | Single MobilePose (MobileNetV2 backbone) |
| Input resolution | **256 × 192** |
| Dataset size (original) | 6,655 images / 98 videos / 22 subjects |
| Augmented training set | **33,274 images** |
| 6DoF algorithm | **EPnP + IPnP** |
| Stabilization | **Alpha-Beta filter + IOU gate (threshold 0.96)** |
| Real-time target (iPhone 11) | **~30 FPS** |
| Loss function | **L2 (MSE on heatmaps)** |
| Optimizer | **AdamW**, lr=1e-4, Exponential Decay |

---

## Our Upgraded Model vs. Paper

| Aspect | Paper (Baseline) | Our Production Model |
|---|---|---|
| Backbone | MobileNetV2 | **RTMPose-Tiny** |
| Params | 2.80M–5.53M | **3.51M** |
| GFLOPs | 0.796G–1.426G | **0.37G** |
| Export | TFLite | **ONNX → CoreML / NCNN** |
| Keypoints | 9/foot (18 total) | **9/foot (18 total)** ✓ |
| Training framework | Custom | **MMPose** |
| Detector | Not specified | **YOLOv8n foot detector** |

> [!IMPORTANT]
> We adopt the **same 9-keypoint schema** from the paper because it is directly validated for 6DoF shoe AR try-on. We improve the backbone and training framework.

---

## The 9 Keypoints Per Foot (Exact Schema)

```
Index   Name              Description
────────────────────────────────────────
  0     big_toe           Tip of the big toe
  1     little_toe        Tip of the little (5th) toe
  2     near_little_toe   Outer lateral side, near little toe
  3     near_big_toe      Inner medial side, near big toe
  4     far_big_toe       Inner medial side, far from toe
  5     far_little_toe    Outer lateral side, far from toe
  6     dorsum            Top center of the foot (instep)
  7     heel              Bottom back of foot
  8     upper_heel        Back upper part of heel

  ── Repeated for left_foot and right_foot ──
  Total: 18 keypoints
```

These 9 points define a complete 3D bounding "shell" of the foot, enabling stable 6DoF (rotation + translation) estimation via PnP.

---

## Dataset Strategy (3-Stage Pipeline)

### Stage 1 — Public Pretraining (6 KP, NO manual work)

| Dataset | KP/foot | Images | License | Action |
|---|---|---|---|---|
| CMU Foot Keypoint | 3 (big toe, small toe, heel) | ~14K annotations | CC BY 4.0 | ✅ Use directly |
| H3WB | 3 (foot subset of 133-KP) | 100K images | Research | ✅ Extract foot crops |
| MOOF | 3 | 14,589 frames | Non-commercial | ⚠️ Research only — use for pretraining only |

**Goal**: Pretrain RTMPose-Tiny on 3 KP/foot → gives the model a strong foot detection prior.

### Stage 2 — Synthetic 9-KP Generation (NO manual work)

Use **Foot3D** (118 scanned 3D feet) + **Blender/PyBullet renderer** to generate synthetic images with all 9 keypoints automatically labeled.

```
3D Foot Mesh (Foot3D)
    ↓
Random: camera angle, rotation, scale, lighting, background
    ↓
Render → synthetic image
    +
Project 9 3D points → 2D keypoints (ground truth automatic)
    ↓
~20,000 synthetic images with 18-KP annotations
```

### Stage 3 — Real Data Fine-Tuning (1,000–3,000 manually annotated)

**This is where your annotation team works.**

Distribution of 3,000 real frames:
```
750   (25%) top-down / first-person view
750   (25%) rotated / angled feet  
600   (20%) one foot raised or partially off-ground
450   (15%) partial occlusion (feet crossing)
450   (15%) difficult lighting / cluttered background

── Foot state distribution ──
1,000  barefoot
800    socks
700    shoes (different styles)
300    sandals / other
200    different foot sizes (children, large)
```

> [!IMPORTANT]
> Include **shoes-on-foot** images in real fine-tuning data. The paper used barefoot/socks only — this limits keypoint inference when shoes partially occlude anatomical landmarks. Our model must generalize to **visible shoe + inferred foot geometry**.

---

## Dataset Annotation Guidelines Document

> *(See the separate `dataset_annotation_guidelines.md` artifact for the complete annotator handbook)*

---

## Full Code Structure

```
shoes_vto/
├── data/
│   ├── raw/                    # Original videos + images
│   ├── processed/              # Extracted frames
│   ├── annotations/            # COCO-format JSON files
│   ├── synthetic/              # Blender-generated images
│   └── augmented/              # Final augmented training set
│
├── src/
│   ├── dataset/
│   │   ├── download_public.py      # Download CMU, H3WB datasets
│   │   ├── convert_to_schema.py    # Unify all datasets to our 18-KP schema
│   │   ├── foot_dataset.py         # PyTorch Dataset class
│   │   └── augmentations.py        # Albumentations pipeline
│   │
│   ├── synthetic/
│   │   ├── blender_renderer.py     # Blender Python API for synthetic gen
│   │   ├── foot3d_loader.py        # Load Foot3D 3D meshes
│   │   └── generate_synthetic.py   # Main synthetic generation script
│   │
│   ├── models/
│   │   ├── detector/
│   │   │   ├── train_detector.py   # YOLOv8n foot detector training
│   │   │   └── foot_detector.py    # Inference wrapper
│   │   │
│   │   ├── keypoint/
│   │   │   ├── rtmpose_config.py   # RTMPose-Tiny custom 18-KP config
│   │   │   ├── train_keypoint.py   # MMPose training script
│   │   │   ├── keypoint_model.py   # Inference wrapper
│   │   │   └── custom_head.py      # Custom 18-KP output head
│   │   │
│   │   └── geometry/
│   │       ├── pnp_solver.py       # EPnP + IPnP implementation
│   │       ├── camera_model.py     # Camera intrinsics / calibration
│   │       └── foot_3d_model.py    # 3D reference foot model keypoints
│   │
│   ├── tracking/
│   │   ├── alpha_beta_filter.py    # Alpha-Beta filter (R, T smoothing)
│   │   ├── iou_gate.py             # IOU-based motion detection gate
│   │   └── kalman_filter.py        # Optional Kalman alternative
│   │
│   ├── rendering/
│   │   ├── shoe_renderer.py        # 3D shoe model renderer
│   │   ├── ray_casting.py          # Ray-casting occlusion algorithm
│   │   └── ar_compositor.py        # Merge AR frame with camera feed
│   │
│   ├── pipeline/
│   │   ├── inference_pipeline.py   # Full end-to-end pipeline (Python)
│   │   └── benchmark.py            # FPS / latency benchmarking
│   │
│   └── export/
│       ├── export_onnx.py          # Export RTMPose to ONNX
│       ├── export_coreml.py        # ONNX → CoreML (iOS)
│       ├── export_ncnn.py          # ONNX → NCNN (Android)
│       └── quantize_int8.py        # INT8 quantization
│
├── mobile/
│   ├── ios/                        # Swift / Objective-C integration
│   │   ├── ShoeVTOKit/
│   │   │   ├── FootDetector.swift
│   │   │   ├── KeypointEstimator.swift
│   │   │   ├── PnPSolver.swift
│   │   │   ├── StabilizationFilter.swift
│   │   │   └── ARRenderer.swift
│   │   └── ShoeVTOApp/             # Sample iOS app
│   │
│   └── android/                    # Kotlin / Java integration
│       ├── ShoeVTOKit/
│       │   ├── FootDetector.kt
│       │   ├── KeypointEstimator.kt
│       │   ├── PnPSolver.kt
│       │   ├── StabilizationFilter.kt
│       │   └── ARRenderer.kt
│       └── ShoeVTOApp/             # Sample Android app
│
├── tools/
│   ├── annotation_tool/            # Custom labeling tool (3D Render Tool)
│   │   ├── annotation_gui.py       # PyQt5/Tkinter GUI
│   │   ├── foot_3d_overlay.py      # 3D foot overlay for assisted labeling
│   │   └── export_coco.py          # Export annotations to COCO JSON
│   │
│   └── evaluation/
│       ├── eval_keypoints.py       # PCK / OKS metrics
│       ├── eval_6dof.py            # R, T error metrics
│       └── eval_fps.py             # Runtime benchmarking
│
├── configs/
│   ├── rtmpose_tiny_18kp.py        # MMPose config file
│   ├── yolov8n_foot.yaml           # YOLO detector config
│   └── training_config.yaml        # All hyperparameters
│
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_synthetic_generation.ipynb
│   ├── 03_training_keypoints.ipynb
│   ├── 04_pnp_validation.ipynb
│   └── 05_mobile_benchmark.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Execution Roadmap (Phases)

### Phase 0 — Setup (Week 1)
- [ ] Setup Python environment (PyTorch, MMPose, Ultralytics)
- [ ] Download CMU Foot + H3WB datasets
- [ ] Convert public data to unified 18-KP schema (6 KP available, 12 KP = `None`)
- [ ] Setup annotation tool for your team

### Phase 1 — Detector Training (Week 1–2)
- [ ] Collect/label 500–1000 foot bounding box images
- [ ] Train YOLOv8n foot detector
- [ ] Target: **mAP@50 > 0.90** on validation set
- [ ] Export to ONNX + TFLite

### Phase 2 — Keypoint Pretraining (Week 2–3)
- [ ] Pretrain RTMPose-Tiny on public 6-KP data
- [ ] Verify foot crop pipeline works end-to-end

### Phase 3 — Synthetic Data Generation (Week 3–4)
- [ ] Setup Blender renderer with Foot3D meshes
- [ ] Generate 20,000 synthetic images with 18 KP
- [ ] Fine-tune RTMPose-Tiny on synthetic 18-KP data

### Phase 4 — Real Data Collection & Annotation (Week 4–6)
- [ ] Collect 300–400 videos per annotation guidelines
- [ ] Annotate 3,000 frames using assisted annotation tool
- [ ] Quality check: PCK@0.05 inter-rater agreement > 90%

### Phase 5 — Final Fine-Tuning (Week 6–7)
- [ ] Fine-tune RTMPose-Tiny on real 3,000 frames
- [ ] Augment to ~15,000 training images
- [ ] Target: **OKS > 0.85** on test set

### Phase 6 — Geometry & Stabilization (Week 7–8)
- [ ] Implement EPnP + IPnP solver
- [ ] Implement Alpha-Beta filter (R, T smoothing)
- [ ] Implement IOU gate (threshold 0.96)
- [ ] End-to-end pipeline validation

### Phase 7 — Mobile Export & Integration (Week 8–10)
- [ ] Export RTMPose-Tiny → ONNX → CoreML (iOS) / NCNN (Android)
- [ ] INT8 quantization
- [ ] Integrate into iOS Swift SDK
- [ ] Integrate into Android Kotlin SDK
- [ ] Target: **≥ 30 FPS** on iPhone 11 / mid-range Android

### Phase 8 — AR Rendering (Week 10–12)
- [ ] Implement Ray-Casting occlusion
- [ ] 3D shoe model pipeline
- [ ] Full AR compositor
- [ ] User study / qualitative evaluation

---

## Key Technical Decisions

### Why RTMPose-Tiny over Single MobilePose?

| | Single MobilePose | RTMPose-Tiny |
|---|---|---|
| Params | 2.80M | 3.51M |
| GFLOPs | 0.796G | **0.37G** |
| Framework | Custom TFLite | **MMPose (active)** |
| Export | TFLite | **ONNX / CoreML / NCNN** |
| Community | Limited | **Large, well-maintained** |
| Training tooling | Custom | **Full MMPose ecosystem** |

### Why EPnP over deep 6DoF estimation?
- O(n) complexity → real-time safe
- No additional training data needed
- Works with 9 sparse keypoints perfectly
- Paper validated this approach in production

### Alpha-Beta Filter Parameters
```python
# Recommended starting values (tune on your data)
alpha = 0.85   # Position smoothing (0 = ignore measurement, 1 = raw)
beta  = 0.15   # Velocity smoothing
IOU_threshold = 0.96   # Foot stability gate
```

---

## Metrics & Quality Gates

| Stage | Metric | Target |
|---|---|---|
| Foot Detector | mAP@50 | > 0.90 |
| Keypoint Model | OKS (COCO) | > 0.85 |
| Keypoint Model | PCK@0.05 | > 80% |
| 6DoF — Rotation | Mean Angular Error | < 5° |
| 6DoF — Translation | Mean Distance Error | < 10mm |
| End-to-end | FPS (iPhone 11) | ≥ 30 FPS |
| End-to-end | FPS (mid Android) | ≥ 25 FPS |

---

## Open Questions (Need Your Input)

> [!IMPORTANT]
> **Q1 — Target platform priority**: Do you need iOS first, Android first, or both simultaneously? This affects the export format priority (CoreML vs NCNN).

> [!IMPORTANT]  
> **Q2 — Shoe state in camera**: Will the user be wearing shoes when trying on virtual ones (shoe-on-shoe)? Or will they be barefoot/socks? This critically affects annotation requirements.

> [!IMPORTANT]
> **Q3 — 3D shoe models availability**: Do you have 3D shoe models (.glb/.obj) for the AR rendering step? This is needed for Phase 8.

> [!NOTE]
> **Q4 — Annotation team size**: How many people do you have for annotation? This determines how fast Phase 4 can complete.

> [!NOTE]
> **Q5 — Target devices**: What are the minimum device specs? (e.g., iPhone X and above, Android with Snapdragon 660+?)
