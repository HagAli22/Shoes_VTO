# Shoes VTO Stage A — 13-Keypoint Production Delivery Package

**Version:** 3.0.0-13kp (v7 Production Champion)  
**Output Tensor Shape:** `[1, 45, 2100]` (Batch, Channels, Anchors)  
**Primary Deliverable:** [`deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.onnx`](models/stage-a-320-13kp-v7.onnx) (12.4 MB)  
**PyTorch Checkpoint:** [`deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt`](models/stage-a-320-13kp-v7.pt) (6.6 MB)  
**Demo Video:** [`deliverables/stage_a_13kp/demo/annotated_output_v7_unfreeze_noflip.mp4`](demo/annotated_output_v7_unfreeze_noflip.mp4) (41.8 MB)  

---

## 1. Executive Summary & Purpose

The **Stage A 13-Keypoint (13-KP)** pipeline represents the major anatomical refinement of the Shoes VTO foot perception system. 

In the legacy 16-keypoint model, three keypoints (`ball_top`, `ankle_center`, and `shin_mid`) were identified as severely noisy and anatomically ill-defined across varied foot shapes, causing high temporal jitter and degraded 3D virtual shoe alignment in Stage B.

By surgically excising these 3 points, standardizing the network output contract to **13 keypoints**, and scaling the training dataset from 1,000 images to **5,161 diverse real-world images**, the detector achieved:
- **89.15% Pose mAP@50** and **93.43% Box mAP@50**.
- **64.46% Box mAP@50-95** and **52.13% Pose mAP@50-95** (All-time project records).
- **92.5% Pose Precision** with zero Left/Right foot chirality inversion.
- **2.1 ms inference latency (~475 FPS)** on NVIDIA Tesla T4 / ~30 ms on standard CPU.

---

## 2. 13-Keypoint Anatomical Schema

The 13 surviving keypoints provide a rigid, mathematically consistent envelope covering the sole, metatarsal arch, ankle bones, and Achilles tendon:

```
                  [12] achilles
                     \
         [8] malleolus_medial  ──  [9] malleolus_lateral
                     |                 |
          [6] arch_medial     ──  [7] midfoot_lateral
                     |      [5] instep_top
                     |         /       \
          [3] ball_medial   ──          [4] ball_lateral
                     \                 /
               [1] heel_back ── [2] heel_ground ── [0] toe_ground ── [10] toe_tip
```

| 13-KP Index | Keypoint Name | Anatomical Region | Legacy 16-KP Index | Description |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `toe_ground` | Sole Contour | 0 | Ground contact point directly beneath the big toe |
| **1** | `heel_back` | Sole Contour | 1 | Posterior curve of the calcaneus (heel boundary) |
| **2** | `heel_ground` | Sole / Origin | 2 | Ground contact base of calcaneus (origin of foot axis) |
| **3** | `ball_medial` | Metatarsal Ball | 3 | Medial prominence of the 1st metatarsal joint |
| **4** | `ball_lateral` | Metatarsal Ball | 4 | Lateral prominence of the 5th metatarsal joint |
| **5** | `instep_top` | Dorsum / Midfoot | 6 | Highest dorsal point on the cuneiform bridge |
| **6** | `arch_medial` | Midfoot / Arch | 7 | Apex of medial longitudinal foot arch |
| **7** | `midfoot_lateral`| Lateral Edge | 8 | Tuberosity of the 5th metatarsal / lateral sole edge |
| **8** | `malleolus_medial` | Ankle Bone | 9 | Medial ankle protrusion (distal tibia) |
| **9** | `malleolus_lateral`| Ankle Bone | 10 | Lateral ankle protrusion (distal fibula) |
| **10** | `toe_tip` | Sole Contour | 11 | Foremost anterior tip of the toe box |
| **11** | `throat` | Dorsum / Ankle | 13 | Anterior junction between tibia stem and foot dorsum |
| **12** | `achilles` | Posterior Ankle | 14 | Insertion point of Achilles tendon into calcaneus |

*Excised Legacy Points:* `ball_top` (Old 5), `ankle_center` (Old 12), and `shin_mid` (Old 15).

---

## 3. Training & Fine-Tuning History (v4 $\rightarrow$ v7)

The table and breakdown below illustrate the full chronological trajectory from the initial 13-KP surgery up to the current production checkpoint:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE A 13-KP EVOLUTION TIMELINE                                │
└────────────────────────────────────────────────────────────────────────────────────────┘

  [16-KP Legacy] ──► [Tensor Surgery] ──► [v4: 13-KP Baseline] (~1,000 imgs)
                                                      │
                                                      ▼
                                         [v5: Multi-Dataset Merge] (2,021 imgs: v3+v4)
                                                      │
                                                      ▼
                                         [v6: Backbone Freezing] (1,608 imgs: v4.1)
                                                      │
                                                      ▼
                                         [v7 Initial: Chirality Conflict] (5,161 imgs)
                                                      │
                                                      ▼
                                         [v7 Final: No-Flip + Full Unfreeze] (12.4 MB ONNX)
```

### Detailed Version Log

#### Version 4: `run_v4_13kp` — Initial 13-KP Architecture & Tensor Surgery
- **Base Checkpoint:** 16-KP legacy weights (`stage-a-320.pt`) modified via `tools/adapt_checkpoint_keypoints.py`. Sliced convolution output channels from 48 down to 39 ($13 \times 3$) to eliminate keypoint reinitialization.
- **Dataset:** `shuffled_v3_13kp` (~1,000 images, mostly barefoot).
- **Training Config:** 100 epochs, batch 16, imgsz 320, `lr0=0.001`, AdamW optimizer.
- **Metrics:** Box mAP@50: **94.98%** | Pose mAP@50: **93.26%** | Pose mAP@50-95: **51.81%**.
- **Deliverables:** [`stage-a-320-13kp.pt`](models/stage-a-320-13kp.pt), [`stage-a-320-13kp.onnx`](models/stage-a-320-13kp.onnx).
- **Outcome:** Validated that excising the 3 unstable points produces significantly crisper keypoint convergence.

#### Version 5: `run_v5_merged_13kp` — Multi-Dataset Merge (v3 + v4)
- **Goal:** Generalize the detector across new background types and varied foot angles.
- **Tooling:** Built `tools/merge_datasets.py` with SHA-256 deduplication and collision prefixing.
- **Dataset:** `merged_v3_v4_13kp` — **2,021 unique deduplicated images** (1,616 train, 202 val, 203 test).
- **Base Checkpoint:** Initialized from `stage-a-320-13kp.pt` (v4).
- **Training Config:** 100 epochs, batch 16, imgsz 320, gentle `lr0=0.0008`.
- **Metrics:** Box mAP@50: **92.69%** | Pose mAP@50: **85.63%** | Pose Precision: **90.3%**.
- **Deliverables:** [`stage-a-320-13kp-v5-merged.pt`](models/stage-a-320-13kp-v5-merged.pt), [`stage-a-320-13kp-v5-merged.onnx`](models/stage-a-320-13kp-v5-merged.onnx).
- **Outcome:** Absorbed the new dataset without catastrophic forgetting of earlier geometries.

#### Version 6: `run_v6_v4_1_freeze_lr3e4` — Backbone Freezing on Cleaned Annotations
- **Goal:** Adapt to cleaned and stabilized ground truth keypoints in `v4_1`.
- **Dataset:** `shuffled_v4_1_13kp` (1,608 unique images).
- **Strategy:** Frozen backbone (`--freeze 10`) to lock low-level convolutional feature extractors and prevent parameter drift, updating only the neck and pose head.
- **Base Checkpoint:** Initialized from `stage-a-320-13kp-v5-merged.pt` (v5).
- **Training Config:** 60 epochs, batch 16, imgsz 320, `lr0=0.0003`, `--freeze 10`.
- **Metrics:** Box mAP@50: **93.78%** | Box mAP@50-95: **60.05%** | Pose mAP@50: **85.72%** | Pose Recall: **83.6%**.
- **Deliverables:** [`stage-a-320-13kp-v6-freeze.pt`](models/stage-a-320-13kp-v6-freeze.pt), [`stage-a-320-13kp-v6-freeze.onnx`](models/stage-a-320-13kp-v6-freeze.onnx).

#### Version 7 (First Iteration): Scale-up & Chirality Collision
- **Goal:** Scale training to **5,161 total images** by merging `v4_1` (1,608 images) + `v4_2` (3,734 images). Added complex carpets, bare feet, diverse socks, and children's feet.
- **First Run Settings:** Batch 32, `--freeze 4`, `lr0=0.0005`, `fliplr=0.5`.
- **Observed Failure Modes:**
  - Recall was high (91%), but Precision collapsed to ~60% and Pose mAP@50 dropped to 76.37%.
  - Visual audits exposed Left/Right class reversals on sneakers, dual overlapping boxes (both Left Foot & Right Foot detected on the same foot), and class confusion on baby feet.
- **The Chirality Discovery:**
  Human feet are **chiral (asymmetric)**. When YOLO performs horizontal flipping (`fliplr: 0.5`), a left foot visually mirrors into a right foot, but YOLO **never swaps class IDs (0 $\leftrightarrow$ 1)**. Half of the augmented training samples trained the network that a left foot looks like a right foot, poisoning classifier certainty.

#### Version 7 (Final Production): `run_v7_merged_full_unfreeze_noflip` 🏆
- **The Three Critical Fixes:**
  1. **`--fliplr 0.0`:** Completely disabled horizontal flipping to preserve true anatomical foot chirality.
  2. **`--agnostic_nms`:** Enabled class-agnostic NMS to strictly discard dual overlapping candidate boxes on the same physical foot.
  3. **`--freeze 0`:** Unfroze all 144 layers so low-level edge filters could fully adapt to the 5,161 diverse images.
- **Dataset:** `merged_v4_1_v4_2_raw_13kp` (**5,161 unique images**; 4,128 train, 516 val, 517 test).
- **Training Config:** 50 epochs, batch 32, imgsz 320, `lr0=0.0003`, `--freeze 0`, `--fliplr 0.0`, `--agnostic_nms`.
  - Finished in **11.2 minutes** on Tesla T4.
- **Deliverables:** [`stage-a-320-13kp-v7.pt`](models/stage-a-320-13kp-v7.pt), [`stage-a-320-13kp-v7.onnx`](models/stage-a-320-13kp-v7.onnx), [`annotated_output_v7_unfreeze_noflip.mp4`](demo/annotated_output_v7_unfreeze_noflip.mp4).

---

## 4. Comprehensive Performance Comparison Matrix

| Version | Dataset | Total Images | Key Strategy | Pose mAP@50 | Pose mAP@50-95 | Pose Precision | Box mAP@50 | Box mAP@50-95 | Primary Deliverable |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **v4** | `shuffled_v3_13kp` | ~1,000 | 16 $\rightarrow$ 13 KP surgery | 93.26% | 51.81% | ~88% | 94.98% | 60.33% | [`stage-a-320-13kp.onnx`](models/stage-a-320-13kp.onnx) |
| **v5** | `merged_v3_v4_13kp` | 2,021 | Deduplicated dataset merge | 85.63% | 49.49% | 90.3% | 92.69% | 59.38% | [`stage-a-320-13kp-v5-merged.onnx`](models/stage-a-320-13kp-v5-merged.onnx) |
| **v6** | `shuffled_v4_1_13kp` | 1,608 | Frozen backbone (`freeze=10`) | 85.72% | 49.68% | 80.4% | 93.78% | 60.05% | [`stage-a-320-13kp-v6-freeze.onnx`](models/stage-a-320-13kp-v6-freeze.onnx) |
| **v7** | `merged_v4_1_v4_2_13kp` | **5,161** | **No-Flip (`fliplr=0`), Agnostic NMS, Full Unfreeze** | **89.15%** | **52.13%** *(Best)* | **92.5%** *(Best)* | **93.43%** | **64.46%** *(Best)* | [`stage-a-320-13kp-v7.onnx`](models/stage-a-320-13kp-v7.onnx) |

---

## 5. Package Directory Structure

```text
deliverables/stage_a_13kp/
├── README.md                                  # This document
├── models/
│   ├── stage-a-320-13kp-v7.onnx               # PRODUCTION CHAMPION ONNX (12.4 MB)
│   ├── stage-a-320-13kp-v7.pt                 # PRODUCTION CHAMPION PyTorch (6.6 MB)
│   ├── stage-a-320-13kp-v6-freeze.onnx        # Version 6 ONNX (12.4 MB)
│   ├── stage-a-320-13kp-v6-freeze.pt          # Version 6 PyTorch (6.6 MB)
│   ├── stage-a-320-13kp-v5-merged.onnx        # Version 5 ONNX (12.4 MB)
│   ├── stage-a-320-13kp-v5-merged.pt          # Version 5 PyTorch (6.6 MB)
│   ├── stage-a-320-13kp.onnx                  # Version 4 ONNX (12.4 MB)
│   └── stage-a-320-13kp.pt                    # Version 4 PyTorch (6.6 MB)
└── demo/
    ├── README.md                              # Standalone demo guide & CLI arguments
    ├── run_video_demo.py                      # Production inference & HUD visualization runner
    ├── annotated_output_v7_unfreeze_noflip.mp4 # Production H.264 demo video (41.8 MB)
    ├── annotated_output_v6_freeze.mp4         # v6 comparison video
    ├── annotated_output_v5_merged.mp4         # v5 comparison video
    └── collection defferent videos/           # Unannotated test raw video benchmarks
```

---

## 6. Quick Start & Execution

### Running Inference via PyTorch Checkpoint
```bash
python deliverables/stage_a_13kp/demo/run_video_demo.py \
    --weights deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt \
    --video data/test_video.mp4 \
    --output deliverables/stage_a_13kp/demo/annotated_output_v7_unfreeze_noflip.mp4
```

### Running Inference via Exported ONNX Runtime
```bash
python deliverables/stage_a_13kp/demo/run_video_demo.py \
    --weights deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.onnx \
    --video data/test_video.mp4 \
    --output deliverables/stage_a_13kp/demo/annotated_output_v7_unfreeze_noflip.mp4
```

### Advanced Demo Features Built-in:
1. **13-Point Anatomical Skeleton:** Renders the sole contour, dorsal bridge, ankle bones, and Achilles tendon.
2. **Occluded Keypoint Styling:** Solid filled circles for high-visibility points ($v \ge 0.35$); elegant hollow rings for occluded/inferred points ($0.15 \le v < 0.35$).
3. **One-Euro Temporal Filter:** Adaptive smoothing eliminates micro-jitter on stationary feet while maintaining zero lag during rapid foot movement.
4. **Deterministic Geometric Chirality:** Uses 2D cross-product ($\vec{V}_{\text{axis}} \times \vec{V}_{\text{lateral}}$) to mathematically verify Left vs Right foot identity.
5. **Real-time Telemetry HUD:** Displays instantaneous and rolling FPS, latency (ms), active foot count, and frame progress.
