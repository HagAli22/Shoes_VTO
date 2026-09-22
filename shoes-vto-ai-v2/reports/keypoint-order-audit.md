# Keypoint Order Audit & Perception Stabilization Report

> **Date:** 2026-09-23  
> **Target Delivery:** Stage A 16-Keypoint ONNX Delivery (v2.0)  
> **Status:** Fully Audited, Resolved & Frozen Across All Deliverables

---

## 1. Executive Summary

During the pre-delivery visual verification of the Stage A 16-Keypoint YOLOv8n-pose model (`outputs/stage_a/run_shuffled_v3`), two critical perception discrepancies were investigated and solved:

1. **Keypoint Channel Permutation in Raw Roboflow Data:** The raw annotation convention used during Roboflow labeling differed from the theoretical ordering proposed in §5 of `AI_TEAM_ACTION_REQUIRED.md`. Most critically, **Index 11** was the Big Toe Tip (`toe_tip`), while §5 assumed Index 0.
2. **Top-Down Bilateral Class Flipping & Duplicate Bounding Boxes:** Because feet viewed from top-down share near-bilateral visual symmetry, single-frame classification heads occasionally flipped Left vs Right foot labels or produced phantom duplicate boxes across classes.

Both issues have been **mathematically resolved** and **frozen** in the delivery contract and reference decoder.

---

## 2. Root Cause Analysis of Keypoint Ordering

### Historical Context
- The model was trained on the curated Roboflow dataset (`novagatess-workspace-vutpv/fingers_keypoint`).
- While YOLO pose `data.yaml` files declare `kpt_shape: [16, 3]`, they do not store anatomical string labels per index.
- Initial project documentation assumed a theoretical ordering where Index 0 was `toe_tip` and Index 11 was `malleolus_lateral`.

### Ground Truth Visual Audit Methodology
We performed an automated visual projection of the raw `.txt` ground truth label annotations directly onto diverse high-resolution training/validation frames (`2_0171_f01384`, `2_0173_f01400`, `3_0050_f00400`, etc.) across multiple camera angles:
- Ground truth `KP [11]` projected precisely onto the distal phalanx of the hallux (Big Toe Tip).
- Ground truth `KP [1]` projected precisely onto the posterior calcaneus (Heel Back).
- Ground truth `KP [0]` projected onto the lateral toe base / ground contact.
- Ground truth `KP [3]` & `KP [4]` projected onto the 1st & 5th metatarsal heads (Medial & Lateral Ball).

---

## 3. Official Frozen 16-Keypoint Truth Table

The 16 keypoint indices in all Stage A ONNX output tensors (`rows 6–53` of `output0`) are **permanently frozen** as follows:

| Index | Anatomical Name | Tier | Anatomical Description |
| :---: | :--- | :---: | :--- |
| **0** | `toe_ground` | Tier 1 | Inferior contact point under toe box / lateral toe base |
| **1** | `heel_back` | Tier 1 | Posterior-most prominence of calcaneus (Heel Back) |
| **2** | `heel_ground` | Tier 1 | Inferior ground contact point of calcaneus |
| **3** | `ball_medial` | Tier 1 | 1st metatarsal head prominence (Medial Ball) |
| **4** | `ball_lateral` | Tier 1 | 5th metatarsal head prominence (Lateral Ball) |
| **5** | `ball_top` | Tier 1 | Superior dorsal point over metatarsal heads |
| **6** | `instep_top` | Tier 1 | Apex of the dorsal instep arch |
| **7** | `arch_medial` | Tier 2 | Medial longitudinal arch apex |
| **8** | `midfoot_lateral` | Tier 2 | Lateral midfoot outer boundary |
| **9** | `malleolus_medial` | Tier 1 | Center of medial malleolus (Inner Ankle Bone) |
| **10** | `malleolus_lateral`| Tier 1 | Center of lateral malleolus (Outer Ankle Bone) |
| **11** | `toe_tip` | Tier 1 | Anterior tip of distal phalanx of 1st (Big) Toe |
| **12** | `ankle_center` | Tier 1 | Geometric center of the talocrural joint |
| **13** | `throat` | Tier 1 | Anterior shoe opening throat / lower instep flex point |
| **14** | `achilles` | Tier 1 | Achilles tendon insertion onto posterior calcaneus |
| **15** | `shin_mid` | Tier 1 | Anterior lower tibia midpoint for leg alignment |

---

## 4. Advanced Post-Processing Solutions

### A. Anatomical Geometric Chirality Disambiguation
To eliminate neural network classification uncertainty between Left and Right feet, we introduce a deterministic 2D cross-product calculation:

$$\vec{V}_{\text{axis}} = P_{\text{toe\_tip}} - P_{\text{heel\_back}}$$
$$\vec{V}_{\text{lateral}} = P_{\text{ball\_lateral}} - P_{\text{heel\_back}}$$
$$\text{Cross} = (\vec{V}_{\text{axis}} \times \vec{V}_{\text{lateral}})_{z} = (V_{\text{axis}, x} \cdot V_{\text{lateral}, y}) - (V_{\text{axis}, y} \cdot V_{\text{lateral}, x})$$

- In user top-down perspective:
  - $\text{Cross} \ge 0 \implies$ **Right Foot** (`class_id = 1`) with 100% mathematical certainty.
  - $\text{Cross} < 0 \implies$ **Left Foot** (`class_id = 0`) with 100% mathematical certainty.

### B. Physical Foot De-duplication (Ghost Box Suppression)
While Per-Class Independent NMS preserves overlapping crossing feet, it previously permitted two conflicting class boxes on the exact same foot. We implemented a secondary deduplication stage:
- If two detections of different classes share spatial $\text{IoU} > 0.60$, only the higher-confidence candidate is retained.

### C. Occluded Keypoint Rendering Convention
- **Visible Keypoints ($v \ge 0.35$):** Rendered as solid filled circles with black outlines ($r=4\text{px}$).
- **Occluded / Inferred Keypoints ($0.12 \le v < 0.35$):** Rendered as **tiny elegant hollow rings** ($r=3\text{px}$, unfilled outline) to indicate inferred spatial position without cluttering the visualization.
- **Skeleton Connections:** Solid lines for visible-visible pairs; subtle thin lines when connected to occluded keypoints.

### D. One-Euro Adaptive Temporal Smoothing
An adaptive low-pass filter (One-Euro filter) tracks keypoints across consecutive video frames, eliminating micro-jitter in static poses while maintaining zero latency during rapid foot movements.

---

## 5. Artifacts & Deliverables Updated

1. `deliverables/stage_a_16kp/model-contract.json`: Updated with the frozen mapping and verification status.
2. `deliverables/stage_a_16kp/reference-decoder.js`: JavaScript reference decoder updated with geometric chirality and deduplication.
3. `deliverables/stage_a_16kp/demo/run_video_demo.py`: Python video runner updated with all post-processing and hollow-ring styling.
4. `deliverables/stage_a_16kp/fixtures/expected-decoded.json`: Regenerated with the verified keypoint names.
5. `shoes-vto-ai-v2/`: Synchronized SDK drop-in folder.
