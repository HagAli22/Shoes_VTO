# 📋 Dataset Collection & Annotation Guidelines
### Shoe AR Try-On — Foot Keypoint Dataset
**Version 1.0 — For Annotators & Data Collection Teams**

---

## 🎯 Purpose

This document is the official handbook for people collecting video data and annotating foot keypoints for the Shoe AR Try-On system. Every annotator must read this document fully before starting work.

The goal is to build a **high-quality dataset of 9 foot keypoints per foot** (18 total per image) that will train a deep learning model capable of powering a real-time AR shoe try-on app.

---

## Part 1: Video Collection Guidelines

### 1.1 Equipment

| Item | Minimum Requirement | Preferred |
|---|---|---|
| Device | Any modern smartphone | iPhone 11+ / Samsung Galaxy S10+ |
| Camera resolution | 720p | 1080p |
| Frame rate | 30 FPS | 60 FPS |
| Lighting | Daylight or bright indoor | Well-lit, no harsh shadows |

---

### 1.2 What to Capture (Subject Distribution)

#### Foot State (CRITICAL — collect all 4)
| Category | Target % | Description |
|---|---|---|
| Barefoot | 30% | No socks, no shoes |
| Socks | 25% | Any type of socks |
| Shoes (various styles) | 30% | Sneakers, dress shoes, boots |
| Sandals / flip-flops | 15% | Open-toe footwear |

> [!IMPORTANT]
> **Do NOT collect only barefoot videos.** The model must work when the user is already wearing shoes. Shoe-on-foot images are critical.

#### Camera Perspective
| Perspective | Target % | Description |
|---|---|---|
| First-person (top-down) | 70% | Camera above looking down at own feet |
| Third-person | 30% | Camera in front or side |

#### Special Scenarios (must include)
| Scenario | Target % |
|---|---|
| Rotated feet (toes pointing sideways) | 25% |
| One foot raised | 20% |
| Feet crossing / partial occlusion | 15% |
| Difficult backgrounds | 15% |
| Difficult lighting | 15% |
| Normal standing (baseline) | 10% |

#### Diversity Requirements
- **Gender**: At least 40% female participants
- **Age groups**: 18–30, 31–50, 50+
- **Foot sizes**: Small (EU 35–38), medium (EU 39–42), large (EU 43–46)
- **Skin tones**: Diverse (dark, medium, light)

---

### 1.3 Recording Protocol

**Step 1 — Environment**
- Real indoor or outdoor setting (not studio backdrop)
- Floor clearly visible
- No strong glare

**Step 2 — Video Duration**
- Each video: **5–10 seconds**
- Move feet naturally — small rotations, small lifts
- Do NOT make fast movements (causes blur)

**Step 3 — Naming Convention**
```
{subject_id}_{foot_state}_{perspective}_{scenario}_{take}.mp4

Examples:
  S001_barefoot_firstperson_normal_01.mp4
  S002_shoes_thirdperson_rotated_01.mp4
  S003_socks_firstperson_crossed_02.mp4
```

**Step 4 — Quality Check Before Submission**
- ✅ Video is not blurry
- ✅ Feet are clearly visible for at least 80% of the video
- ✅ At least one foot is fully in frame
- ✅ Video is well-lit
- ✅ Named correctly per convention

---

## Part 2: Keypoint Annotation Schema

### 2.1 The 9 Keypoints Per Foot

```
                    6 (dorsum / instep)
                  ╱              ╲
     4 (far_big_toe)              5 (far_little_toe)
    ╱                                              ╲
3 (near_big_toe)                           2 (near_little_toe)
|                                                   |
0 (big_toe) ─────────────────────── 1 (little_toe)

               7 (heel)   ← back-bottom
               8 (upper_heel) ← back-top
```

#### Keypoint Descriptions

| Index | Name | Anatomical Location | How to Find It |
|---|---|---|---|
| **0** | `big_toe` | Tip of the big toe (hallux) | Very tip of the largest toe |
| **1** | `little_toe` | Tip of the 5th (little/pinky) toe | Very tip of the smallest toe |
| **2** | `near_little_toe` | Outer lateral edge of foot at toe-base level | Where outer foot edge meets the toe row |
| **3** | `near_big_toe` | Inner medial edge at toe-base level | Where inner foot edge meets big toe knuckle |
| **4** | `far_big_toe` | Inner medial edge, mid-foot level (arch side) | Inner edge of the arch, midway along foot |
| **5** | `far_little_toe` | Outer lateral edge, mid-foot level | Outer foot bulge at mid-foot |
| **6** | `dorsum` | Top center of foot (instep) | Highest point of the foot's upper surface |
| **7** | `heel` | Bottom/back of the heel | Back-most point of heel from above |
| **8** | `upper_heel` | Upper/back surface of heel | Top of the heel bump (Achilles area) |

---

### 2.2 Visibility Labels (REQUIRED for every point)

| Label | Code | Meaning |
|---|---|---|
| **Visible** | `2` | Clearly visible — you can see exactly where it is |
| **Occluded** | `1` | Partially hidden BUT you can reasonably estimate location |
| **Not Visible** | `0` | Completely off-screen or totally hidden with NO way to estimate |

> [!IMPORTANT]
> **Occluded (1) ≠ Not Visible (0)**. If a shoe covers the heel but you can see the approximate heel shape, mark it **Occluded (1)** and place the point at your best estimate. Only use Not Visible (0) when there is truly no way to locate the point.

---

## Part 3: Annotation Process (Step-by-Step)

### 3.1 Using the 3D Render Tool

1. **Auto-initialize**: Click "Auto Detect" — a 3D foot model appears overlaid on the image
2. **Adjust the 3D model**: Use sliders to rotate/translate until it fits the real foot:
   - Rotate X: Tilt forward/backward
   - Rotate Y: Rotate left/right
   - Rotate Z: Roll clockwise/counterclockwise
3. **Confirm keypoints**: Click "Apply" — all 9 keypoints placed automatically
4. **Manual correction**: Fine-tune any misplaced points
5. **Set visibility**: For each point, set visibility label (2/1/0)
6. **Save and next**: Click "Save & Next"

### 3.2 Manual Annotation (If 3D Tool Fails)

1. Start with `big_toe` (point 0) — easiest to find
2. Then `little_toe` (point 1)
3. Then `heel` (point 7)
4. These 3 form a triangle. Estimate the rest relative to this triangle.
5. Work through all 9 points.

---

## Part 4: Quality Standards

### 4.1 Accuracy Requirements

| Keypoint Type | Max Allowed Error |
|---|---|
| Clear visible points (vis=2) | ≤ 5 pixels |
| Occluded points (vis=1) | ≤ 15 pixels |

### 4.2 Common Mistakes to Avoid

| Mistake | How to Avoid |
|---|---|
| Placing big_toe on the nail instead of toe tip | Place at the very tip of the toe, not the nail |
| Placing dorsum (6) too far forward | Dorsum = middle of the top surface, not toe area |
| Confusing heel (7) and upper_heel (8) | heel = back-bottom, upper_heel = back-top (Achilles) |
| Marking visible points as occluded | If you can clearly see it, it's visible (2) |
| All points Not Visible when foot is partially in frame | Only use 0 when truly impossible to estimate |

---

## Part 5: Edge Cases

### 5.1 When Shoe Covers Foot
- Place keypoints at the **anatomical foot position under the shoe**, not on the shoe surface
- Use the shoe shape to estimate where the foot anatomy is inside
- Set visibility = `1` (Occluded) for points hidden by shoe

### 5.2 When Feet Are Crossing
- Annotate each foot separately
- Points blocked by the other foot → Occluded (1)
- The foot in front has visual priority

### 5.3 Only One Foot Visible
- Annotate only the visible foot
- For the invisible foot: all 9 points → visibility = `0` (Not Visible)
- Do NOT guess positions for a foot that is completely off-screen

### 5.4 Blurry Images
- If blurry but estimable: annotate with Occluded (1)
- If too blurry to place any point accurately: **Reject** the image
- Motion blur affecting < 30% of keypoints is acceptable

### 5.5 Side/Angled Views
- Many points occluded from the foot's own body
- Typical occluded from side: big_toe (0), near_big_toe (3), far_big_toe (4)
- Use 3D model overlay to get estimates
- Mark occluded points as visibility = 1

---

## Part 6: Quality Control

### 6.1 Annotator Self-Check
Before submitting batch:
- [ ] Visible keypoints within 5px of anatomical landmark
- [ ] Visibility labels correctly set
- [ ] No images with all points = Not Visible if a foot is partially in frame
- [ ] Batch count matches assigned count

### 6.2 Peer Review (2-person check)
- Each image reviewed by a second annotator
- Disagreement > 10px on any keypoint → flagged for re-annotation
- Visibility label disagreement → senior reviewer resolves

### 6.3 Rejection Criteria
Reject an image if:
- More than 50% of expected keypoints cannot be estimated at all
- Image is heavily blurred / too dark
- Foot is smaller than 30×30 pixels
- Annotation conflicts cannot be resolved

---

## Part 7: Annotation Output Format (COCO JSON)

```json
{
  "annotations": [{
    "id": 1,
    "image_id": 1,
    "category_id": 1,
    "keypoints": [
      x0, y0, v0,    // left big_toe
      x1, y1, v1,    // left little_toe
      x2, y2, v2,    // left near_little_toe
      x3, y3, v3,    // left near_big_toe
      x4, y4, v4,    // left far_big_toe
      x5, y5, v5,    // left far_little_toe
      x6, y6, v6,    // left dorsum
      x7, y7, v7,    // left heel
      x8, y8, v8,    // left upper_heel
      x9, y9, v9,    // right big_toe
      x10,y10,v10,   // right little_toe
      x11,y11,v11,   // right near_little_toe
      x12,y12,v12,   // right near_big_toe
      x13,y13,v13,   // right far_big_toe
      x14,y14,v14,   // right far_little_toe
      x15,y15,v15,   // right dorsum
      x16,y16,v16,   // right heel
      x17,y17,v17    // right upper_heel
    ],
    "num_keypoints": 18,
    "bbox": [x, y, width, height]
  }]
}
```

---

## Part 8: Dataset Split Target

| Split | Original Images | Augmented |
|---|---|---|
| Train | 2,400 | ~12,000 |
| Validation | 300 | 300 (no augmentation) |
| Test | 300 | 300 (no augmentation) |
| **Total** | **3,000** | **~12,600** |

**Augmentations applied to training only:**
- Horizontal flip (swaps left/right foot labels automatically)
- Random brightness/contrast ±20%
- Random scale 0.8–1.2×
- Random rotation ±15°
- Gaussian noise
- Random crop (keeping foot in frame)

---

## Quick Reference Card (Print & Keep)

```
╔══════════════════════════════════════════════════╗
║         FOOT KEYPOINT QUICK REFERENCE            ║
╠══════════════════════════════════════════════════╣
║  0  big_toe       → TIP of big toe               ║
║  1  little_toe    → TIP of little toe            ║
║  2  near_little   → Outer edge, toe-base level   ║
║  3  near_big      → Inner edge, toe-base level   ║
║  4  far_big       → Inner edge, mid-foot (arch)  ║
║  5  far_little    → Outer edge, mid-foot bulge   ║
║  6  dorsum        → TOP CENTER of foot (instep)  ║
║  7  heel          → BACK-BOTTOM of heel          ║
║  8  upper_heel    → BACK-TOP of heel (Achilles)  ║
╠══════════════════════════════════════════════════╣
║  VISIBILITY: 2=Visible  1=Occluded  0=NotVisible ║
╠══════════════════════════════════════════════════╣
║  ACCURACY: Visible ≤5px  |  Occluded ≤15px       ║
╚══════════════════════════════════════════════════╝
```

---

## Appendix: Inter-Annotator Calibration

Weekly calibration: 20 pre-annotated "gold standard" images given to all annotators.

| PCK@5px Score | Status |
|---|---|
| > 85% | ✅ Qualified — continue annotating |
| 70–85% | ⚠️ Re-read guidelines, review with senior |
| < 70% | ❌ Pause — senior re-training required |
