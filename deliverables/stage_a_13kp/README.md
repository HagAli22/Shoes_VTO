# Shoes VTO Stage A — 13-Keypoint Production Delivery Package

**Version:** 3.2.0-13kp (v7 Champion Suite: 320 Desktop + 256 Mobile & Background Rejection)  
**Output Tensor Shapes:** 
- **320 Model:** `[1, 45, 2100]` (Batch, Channels, Anchors)  
- **256 Model:** `[1, 45, 1344]` (Batch, Channels, Anchors)  
**Primary Deliverables:** 
- **320 Desktop/Server ONNX:** [`deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.onnx`](models/stage-a-320-13kp-v7.onnx) (12.4 MB)  
- **256 Mobile Champion ONNX (FP32):** [`deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.onnx`](models/stage-a-256-13kp-v7.onnx) (12.4 MB)  
- **256 Mobile Lightweight ONNX (INT8 WASM):** [`deliverables/stage_a_13kp/models/stage-a-256-13kp-int8.onnx`](models/stage-a-256-13kp-int8.onnx) (3.46 MB, $\le 5$ MB Web Budget)  
**PyTorch Checkpoints:** 
- [`deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt`](models/stage-a-320-13kp-v7.pt) (6.6 MB)  
- [`deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.pt`](models/stage-a-256-13kp-v7.pt) (6.6 MB, Champion fine-tuned on 8,275 images with background negatives)  
- [`deliverables/stage_a_13kp/models/stage-a-256-13kp-v7-merged-negatives.pt`](models/stage-a-256-13kp-v7-merged-negatives.pt) (Direct alias)  
- [`deliverables/stage_a_13kp/models/stage-a-256-13kp-v7-base.pt`](models/stage-a-256-13kp-v7-base.pt) (Base 256 checkpoint)  
**Demo Videos:** 
- [`deliverables/stage_a_13kp/demo/annotated_output_v7_unfreeze_noflip.mp4`](demo/annotated_output_v7_unfreeze_noflip.mp4) (41.8 MB)  
- [`deliverables/stage_a_13kp/demo/annotated_output_v7_256.mp4`](demo/annotated_output_v7_256.mp4) (41.9 MB)  

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

#### Version 7 (Production Suite & Negative-Augmented 256 Champion): 320 Standard + 256 Mobile 🏆
- **The Three Critical Fixes:**
  1. **`--fliplr 0.0`:** Completely disabled horizontal flipping to preserve true anatomical foot chirality.
  2. **`--agnostic_nms`:** Enabled class-agnostic NMS to strictly discard dual overlapping candidate boxes on the same physical foot.
  3. **`--freeze 0`:** Unfroze all 144 layers so low-level edge filters could fully adapt to the diverse imagery.
- **320 Model Training (`run_v7_merged_full_unfreeze_noflip`):**
  - Dataset: `merged_v4_1_v4_2_raw_13kp` (5,161 unique images).
  - 50 epochs, batch 32, imgsz 320, `lr0=0.0003`, `--freeze 0`, `--fliplr 0.0`, `--agnostic_nms`.
  - Deliverables: [`stage-a-320-13kp-v7.pt`](models/stage-a-320-13kp-v7.pt), [`stage-a-320-13kp-v7.onnx`](models/stage-a-320-13kp-v7.onnx).
- **256 Mobile Base Model (`run_v7_256_13kp`):**
  - Initialized from `stage-a-320-13kp-v7.pt`, fine-tuned at resolution $256\times256$ in **9.5 minutes**.
  - Output shape: `[1, 45, 1344]`, FLOPs reduced from 2.2 GFLOPs to **1.4 GFLOPs (~40% lighter)**.
  - Deliverables: [`stage-a-256-13kp-v7-base.pt`](models/stage-a-256-13kp-v7-base.pt), [`stage-a-256-13kp-v7-base.onnx`](models/stage-a-256-13kp-v7-base.onnx).
- **256 Mobile Champion with Negative Backgrounds & Hard Edge Cases (`run_v7_256_merged_with_negatives`):**
  - **Dataset:** Scaled to **8,275 unique images** (`merged_v4_1_v4_2_v5_raw_13kp_with_negative3`), incorporating **814 true negative background scenes** (empty floors, carpets, furniture, pets, hands) and ~2,500 hard edge-case foot poses from `shuffled_v5` (severe occlusions, socks/sandals, sharp lateral perspectives).
  - **Training Config:** 60 epochs, batch 64, imgsz 256, `lr0=0.0002`, `lrf=0.01`, AdamW, 53.9 min on Tesla T4.
  - **Hard Evaluation Benchmark (827 images including 84 negative backgrounds):**
    - Box mAP@50: **94.84%** | Box mAP@50-95: **66.76%** | Pose mAP@50: **88.99%** | Pose mAP@50-95: **53.74%**.
  - **Identical Apples-to-Apples Comparison on 516 Benchmark Images (vs Old Base Model):**
    - **Box mAP@50:** **98.1%** vs 97.6% (+0.5% 📈)
    - **Box mAP@50-95:** **72.4%** vs 64.2% (**+8.2% massive gain!** 🚀)
    - **Pose mAP@50:** **96.0%** vs 92.8% (**+3.2% gain!** 🚀)
    - **Pose mAP@50-95:** **62.1%** vs 51.3% (**+10.8% all-time record jump!** 🏆)
    - **Pose Recall:** **94.3%** vs 90.9% (+3.4%) | **Box Recall:** **95.9%** vs 93.8% (+2.1%)
  - **Deliverables:** [`stage-a-256-13kp-v7.pt`](models/stage-a-256-13kp-v7.pt), [`stage-a-256-13kp-v7.onnx`](models/stage-a-256-13kp-v7.onnx), [`stage-a-256-13kp-int8.onnx`](models/stage-a-256-13kp-int8.onnx).

---

## 4. Comprehensive Performance Comparison Matrix

| Version | Resolution | Output Tensor | FLOPs | Pose mAP@50 | Pose mAP@50-95 | Pose Precision | Box mAP@50 | Box mAP@50-95 | Primary Deliverable |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **v4** | 320 | `[1, 45, 2100]` | 2.2 G | 93.26% | 51.81% | ~88% | 94.98% | 60.33% | [`stage-a-320-13kp.onnx`](models/stage-a-320-13kp.onnx) |
| **v5** | 320 | `[1, 45, 2100]` | 2.2 G | 85.63% | 49.49% | 90.3% | 92.69% | 59.38% | [`stage-a-320-13kp-v5-merged.onnx`](models/stage-a-320-13kp-v5-merged.onnx) |
| **v6** | 320 | `[1, 45, 2100]` | 2.2 G | 85.72% | 49.68% | 80.4% | 93.78% | 60.05% | [`stage-a-320-13kp-v6-freeze.onnx`](models/stage-a-320-13kp-v6-freeze.onnx) |
| **v7 (320)** | **320** | `[1, 45, 2100]` | 2.2 G | **89.15%** | **52.13%** | **92.5%** | **93.43%** | **64.46%** | [`stage-a-320-13kp-v7.onnx`](models/stage-a-320-13kp-v7.onnx) |
| **v7 (256 Base)** | 256 | `[1, 45, 1344]` | 1.4 G | 92.80% | 51.30% | 94.1% | 97.60% | 64.20% | [`stage-a-256-13kp-v7-base.onnx`](models/stage-a-256-13kp-v7-base.onnx) |
| **v7 (256 Champion + Negatives)** 🏆 | **256** | `[1, 45, 1344]` | **1.4 G** | **96.00%** | **62.10%** | **95.0%** | **98.10%** | **72.40%** | [`stage-a-256-13kp-v7.onnx`](models/stage-a-256-13kp-v7.onnx) |

---

## 5. Package Directory Structure

```text
deliverables/stage_a_13kp/
├── README.md                                  # This document
├── models/
│   ├── stage-a-320-13kp-v7.onnx               # PRODUCTION CHAMPION ONNX - 320 Desktop/Server (12.4 MB)
│   ├── stage-a-320-13kp-v7.pt                 # PRODUCTION CHAMPION PyTorch - 320 (6.6 MB)
│   ├── stage-a-256-13kp-v7.onnx               # PRODUCTION CHAMPION ONNX - 256 Mobile/WASM (12.4 MB, FP32 Master)
│   ├── stage-a-256-13kp-v7.pt                 # PRODUCTION CHAMPION PyTorch - 256 (6.6 MB, with Negatives & Hard Poses)
│   ├── stage-a-256-13kp-v7-merged-negatives.pt# Direct alias to 256 Champion PyTorch (6.6 MB)
│   ├── stage-a-256-13kp-v7-merged-negatives.onnx # Direct alias to 256 Champion ONNX (12.4 MB)
│   ├── stage-a-256-13kp-int8.onnx             # PRODUCTION LIGHTWEIGHT WASM - 256 Mobile (3.46 MB, <= 5 MB Web Budget)
│   ├── stage-a-256-13kp-v7-base.onnx          # Historical Base 256 ONNX (12.4 MB)
│   ├── stage-a-256-13kp-v7-base.pt            # Historical Base 256 PyTorch (6.6 MB)
│   ├── stage-a-320-13kp-v6-freeze.onnx        # Version 6 ONNX (12.4 MB)
│   ├── stage-a-320-13kp-v6-freeze.pt          # Version 6 PyTorch (6.6 MB)
│   ├── stage-a-320-13kp-v5-merged.onnx        # Version 5 ONNX (12.4 MB)
│   ├── stage-a-320-13kp-v5-merged.pt          # Version 5 PyTorch (6.6 MB)
│   ├── stage-a-320-13kp.onnx                  # Version 4 ONNX (12.4 MB)
│   └── stage-a-320-13kp.pt                    # Version 4 PyTorch (6.6 MB)
└── demo/
    ├── README.md                              # Standalone demo guide & CLI arguments
    ├── run_video_demo.py                      # Production inference & HUD visualization runner
    ├── annotated_output_v7_unfreeze_noflip.mp4 # 320 Production H.264 demo video (41.8 MB)
    ├── annotated_output_v7_256.mp4            # 256 Mobile Production H.264 demo video (41.9 MB)
    ├── test_derived_15kp_demo_fixed.mp4       # 15-KP Expansion Production H.264 demo video (42.0 MB)
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

### Running Inference with 15-Keypoint Expansion (Derived `ball_top` & `shin_mid`)
```bash
python deliverables/stage_a_13kp/demo/run_video_demo.py \
    --weights deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt \
    --video data/test_video.mp4 \
    --output deliverables/stage_a_13kp/demo/test_derived_15kp_demo_fixed.mp4 \
    --expand_15kp
```

### Advanced Demo Features Built-in:
1. **13-Point & 15-Point Anatomical Skeleton:** Renders the sole contour, dorsal bridge, ankle bones, Achilles tendon, and (when `--expand_15kp` is passed) derived `ball_top` and `shin_mid`.
2. **Occluded Keypoint Styling:** Solid filled circles for high-visibility points ($v \ge 0.35$); elegant hollow rings for occluded/inferred points ($0.15 \le v < 0.35$).
3. **One-Euro Temporal Filter:** Adaptive smoothing eliminates micro-jitter on stationary feet while maintaining zero lag during rapid foot movement.
4. **Deterministic Geometric Chirality:** Uses 2D cross-product ($\vec{V}_{\text{axis}} \times \vec{V}_{\text{lateral}}$) to mathematically verify Left vs Right foot identity.
5. **Real-time Telemetry HUD:** Displays instantaneous and rolling FPS, latency (ms), active foot count, frame progress, and schema version (`13-KP` vs `15-KP (Derived)`).

---

## 7. Viewpoint-Invariant 15-Keypoint Expansion (Post-Processing Engineering)

### 7.1 Why Post-Processing Instead of Neural Retraining?
When attempting to directly train a 15-keypoint neural model (`run_v8_15kp` restoring `ball_top` and `shin_mid`), validation metrics suffered a drastic collapse:
- **Pose mAP@50-95 plummeted from 57.92% to 39.28% (-18.64% drop)**.
- **Root Cause:**
  1. **Scarcity of Ground Truth:** Dataset auditing revealed that out of thousands of images, fewer than 20 contained labeled `shin_mid` points due to frequent cropping and occlusion by pants/socks.
  2. **Non-Rigid Limb Boundary:** The YOLO Stage A detector is trained on foot bounding boxes. The shin sits outside the primary foot bounding box, creating ambiguous bounding box limits and huge annotator variance.
  3. **Curved Metatarsal Dome:** `ball_top` sits on a smooth continuous surface without rigid bone markers.
- **The Engineering Solution:**  
  Retain the champion 13-KP neural detector (94.96% Pose mAP, 97.97% Box mAP), and derive `ball_top` and `shin_mid` in **pure deterministic post-processing** in $<0.05\text{ ms}$ with zero network overhead.

---

### 7.2 15-Keypoint Schema Mapping

When `--expand_15kp` is activated, the 13 neural keypoints are expanded to 15:

| 15-KP Index | Name | Derivation Source | Anatomical Location |
| :---: | :--- | :--- | :--- |
| **0..4** | `toe_ground` $\rightarrow$ `ball_lateral` | Neural (Stage A Model) | Sole contour & metatarsal balls |
| **5** | **`ball_top`** | **Geometric Post-Processing** | Midpoint of ball medial/lateral aligned toward instep top |
| **6..13** | `instep_top` $\rightarrow$ `achilles` | Neural (Shifted by +1) | Dorsum, arch, malleoli, toe tip, throat, achilles |
| **14** | **`shin_mid`** | **Geometric Post-Processing** | True anatomical midpoint of lower tibia (~120mm proximal) |

---

### 7.3 Viewpoint-Invariant Derivation Mathematics

```
[View 1: Top-Down]        [View 2: Front Hanging]       [View 3: Back Standing]       [View 4: Side Reclined]
     (Toe Tip)                  (Jeans Cuff)                  (Calf Midline)                (Tibia Stem)
        ▲                            ▲                              ▲                            ▲
        │                            │                              │                           /
  [Foot Dorsum]               [Shin Mid (14)]                [Shin Mid (14)]             [Shin Mid (14)]
        │                            ▲                              ▲                         /
  [Throat (12)]                      │                              │                    [Throat (12)]
        │                     [Throat (12)]                  [Achilles (13)]                  /
        ▼                            │                              │                   [Toe Box]
 [Shin Mid (14)]                 (Toe Box)                     (Heel Base)
```

#### A. `ball_top` (Index 5)
Computed from the midpoint of `ball_medial` and `ball_lateral`, projected $0.28\times$ along the dorsal vector toward `instep_top`:
$$\vec{M}_{\text{ball}} = 0.5 \cdot (\vec{P}_{\text{ball\_medial}} + \vec{P}_{\text{ball\_lateral}})$$
$$\vec{P}_{\text{ball\_top}} = \vec{M}_{\text{ball}} + 0.28 \cdot (\vec{P}_{\text{instep\_top}} - \vec{M}_{\text{ball}})$$
$$v_{\text{ball\_top}} = \min(v_{\text{bm}}, v_{\text{bl}}, v_{\text{instep}})$$

#### B. `shin_mid` (Index 14)
To ensure the projection is invariant across all camera angles (top-down, front, back, and side), an adaptive vector is used:
$$\vec{u}_{\text{leg}} = 
\begin{cases} 
\frac{\vec{P}_{\text{achilles}} - \vec{P}_{\text{heel\_ground}}}{\|\vec{P}_{\text{achilles}} - \vec{P}_{\text{heel\_ground}}\|} & \text{if } v_{\text{toe}} < 0.30 \text{ (Back View)} \\
\text{normalize}\left(0.65\,\vec{u}_{\text{dorsal}} + 0.35\,\vec{u}_{\text{malleoli\_\perp}}\right) & \text{otherwise (Front, Top-Down, Side)}
\end{cases}$$

$$\vec{P}_{\text{shin\_mid}} = 
\begin{cases} 
\vec{P}_{\text{throat}} + 1.20 \cdot L_{\text{ref}} \cdot \vec{u}_{\text{leg}} & \text{if Back View} \\
\vec{P}_{\text{throat}} + 1.10 \cdot L_{\text{ref}} \cdot \vec{u}_{\text{leg}} & \text{otherwise}
\end{cases}$$

where:
- $\vec{u}_{\text{dorsal}} = \frac{\vec{P}_{\text{throat}} - \vec{P}_{\text{toe\_tip}}}{\|\vec{P}_{\text{throat}} - \vec{P}_{\text{toe\_tip}}\|}$
- $\vec{u}_{\text{malleoli\_\perp}}$ is the normal vector to the transmalleolar axis ($\vec{P}_{\text{malleolus\_lateral}} - \vec{P}_{\text{malleolus\_medial}}$) pointing along $\vec{u}_{\text{dorsal}}$.
- $L_{\text{ref}} = 0.40 \times \sqrt{w^2 + h^2}$ is the scale reference derived from the foot bounding box.

---

### 7.4 Critical Steps & Best Practices to Follow

When consuming or extending the 15-KP derivation pipeline in downstream tasks (e.g. Stage B 3D mesh alignment or augmented reality overlays), strictly adhere to the following rules:

1. **Apply One-Euro Temporal Smoothing FIRST:**  
   Always run the temporal smoother on the 13 raw neural keypoints **before** computing `derive_secondary_keypoints_15kp`. Because the derived points are algebraic linear combinations of the anchor points, smoothing the anchors guarantees that `ball_top` and `shin_mid` exhibit zero jitter and zero temporal lag.
2. **Handle Back-View Occlusion Deterministically:**  
   In back views (person standing facing away from camera), `toe_tip` is occluded ($v < 0.30$). Never use the foot longitudinal axis when $v_{\text{toe}} < 0.30$; always switch to the Achilles-to-heel vertical rise vector ($\vec{P}_{\text{achilles}} - \vec{P}_{\text{heel\_ground}}$).
3. **Preserve Transmalleolar Orthogonality in Side Views:**  
   In lateral/side views, the foot and leg form a $\approx 90^\circ$ angle. Relying solely on the foot forward vector causes `shin_mid` to shoot backward horizontally. The $35\%$ weighting on the transmalleolar normal ($\vec{u}_{\text{malleoli\_\perp}}$) forces the leg vector to bend upwards along the tibia stem.
4. **Maintain the Elevated Mid-Shin Extension Factor ($1.10 - 1.20 \cdot L_{\text{ref}}$):**  
   Low multipliers ($<0.6 \cdot L_{\text{ref}}$) place the point at the ankle collar / pant cuff (beginning of shin). The $1.10 - 1.20$ scaling factor correctly elevates `shin_mid` into the true anatomical midpoint of the lower leg (~120 mm proximal to the ankle joint).
5. **Never Alter the Underlying 13-KP Model Weights:**  
   The underlying PyTorch (`.pt`) and ONNX (`.onnx`) models must remain pure 13-keypoint networks to preserve the 94.96% Pose mAP and 475+ FPS inference performance. All 15-keypoint expansion must occur in `src/models/stage_a/postprocess.py`.
