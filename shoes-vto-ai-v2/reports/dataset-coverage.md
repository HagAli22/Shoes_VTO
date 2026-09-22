# Dataset Coverage & Coordinate Integrity Report

## 1. Dataset Characteristics
- **Dataset Source:** Roboflow Foot Pose v3.
- **Split Strategy:** Strict 80/10/10 Stratified Random Shuffle (1,291 Train / 161 Val / 162 Test).
- **Coordinate Normalization:** Clamped strictly to $[0.0, 1.0]$.

- **Dataset Source:** Roboflow Foot Pose v3 (project: `fingers_keypoint`, workspace: `novagatess-workspace-vutpv`, versions 10–12)
- **Split Strategy:** Strict 80/10/10 Stratified Random Shuffle (1,291 Train / 161 Val / 162 Test)
- **Coordinate Normalization:** Clamped strictly to `[0.0, 1.0]`
- **Class Balance:**
  - `left_foot`: 49.6%
  - `right_foot`: 50.4%
- **Anatomical Diversity:** Barefoot, athletic sneakers, formal shoes, sandals, high socks, trousers/jeans cuff coverage.

---

## 2. Footwear Coverage

> **IMPORTANT — Existing Shoe Removal Capability**

The current training dataset contains **barefoot and socked feet only**. No images with existing footwear (sneakers, sandals, boots, or dress shoes) are included in this training set.

**Consequences:**
- The model's keypoint predictions are calibrated to **bare foot anatomy**, not to footwear outlines. This is correct behavior for a VTO system where the virtual shoe must fit the actual foot.
- The model has **not been trained** to detect or distinguish an existing shoe from the foot. Existing-shoe removal is **not supported** in this delivery.
- Users attempting try-on while wearing existing shoes may see degraded keypoint accuracy, particularly for occluded points (heel_back, ball_medial, ball_lateral).

**Recommendation to NGM SDK team:** For production deployment, the AR pipeline should prompt users to remove existing footwear or use bare feet/socks for optimal accuracy. A dedicated "existing shoe detection" model is a separate workstream not covered by this Stage A delivery.

---

## 3. Scene & Condition Coverage

| Condition | Present in Dataset |
| :--- | :--- |
| Bare feet | ✅ Yes |
| Socks (thin/ankle) | ✅ Yes |
| Existing shoes/sneakers | ❌ No |
| Sandals / open footwear | ❌ No |
| Ankle boots / tall boots | ❌ No |
| Trouser/jeans hem partial occlusion | ✅ Partial |
| Indoor floor (flat, neutral) | ✅ Yes |
| Outdoor / textured floor | ✅ Partial |
| Multiple feet in frame | ✅ Yes |
| Foot crossing / overlap | ✅ Partial |
| Motion blur | ❌ No |
| Low light | ❌ No |
| Top-down view | ✅ Yes |
| Side / lateral view | ✅ Yes |
| Rear view | ✅ Partial |

---

## 4. Keypoint Order — Source of Truth & Verification Status

> ⚠️ **Critical Notice for Integration**

The 16 keypoint channels in the model output (`rows 6–53` of `output0`) are ordered according to the **Roboflow annotation order** used during labeling. The `data.yaml` file does not specify keypoint names — only `kpt_shape: [16, 3]`.

The frozen order in `model-contract.json` and §5 of `AI_TEAM_ACTION_REQUIRED.md` was defined as the **target annotation convention** and reflects the Roboflow labeling guide used during this project. However, **no automated per-keypoint identity verification** has been performed to confirm that each channel's learned output matches its declared anatomical name.

**What has been verified:**
- ✅ Class mapping (0=left_foot, 1=right_foot) — confirmed via `prepare_shuffled_dataset.py` class remapping
- ✅ `kpt_shape: [16, 3]` — correct number of keypoints
- ✅ Keypoint coordinates fall within reasonable anatomical positions in decoded fixtures

**What has NOT been verified:**
- ❌ Per-channel identity proof that index 0 = `toe_tip`, index 1 = `toe_ground`, etc.
- ❌ Visual ground-truth overlay matching decoded keypoints to anatomical landmarks

**Recommended verification step (before production sign-off):**
Run `tools/generate_fixtures.py` output frames through a visualization overlay and confirm each decoded keypoint's position matches its declared anatomical name.

---

## 5. Class Remapping History

The raw Roboflow download used reversed class IDs:
```
Roboflow raw:  0 = Foot_Right,  1 = Foot_left
Our contract:  0 = left_foot,   1 = right_foot
```

This was corrected during dataset preparation in `prepare_shuffled_dataset.py` (line 29–30):
```python
# Roboflow raw: 0=Foot_Right, 1=Foot_left -> Remap to 0=left_foot, 1=right_foot
cls = 1 if cls == 0 else 0
```
All labels in `data/shuffled_v3/` are already remapped to the correct convention.
