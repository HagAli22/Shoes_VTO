# 📐 Keypoint Annotation Precision Reference
### Shoe AR Try-On — Version 2.0
**For annotators and model designers.**
**This document supersedes the keypoint section of annotation_guidelines v1.0.**

---

> [!IMPORTANT]
> **Core principle: Anatomical position never moves. Only visibility changes.**
>
> A keypoint marks a fixed anatomical location on the body.
> The camera can change angle but the point stays where it is anatomically.
> When a point is not directly visible from the camera, mark it at its estimated
> anatomical location with `v=1` (occluded). **Never move it to the visible surface.**

---

## Visibility Labels — Used for Every Keypoint

| Label | Code | Meaning | What to do |
|---|---|---|---|
| **Visible** | `v=2` | Keypoint is clearly visible and unoccluded | Mark at exact anatomical location |
| **Occluded** | `v=1` | Keypoint exists but is hidden (sole, other foot, sock, etc.) | Mark at estimated anatomical location. Position is still supervised. |
| **Unlabeled** | `v=0` | Point is outside the frame or impossible to estimate | Skip — excluded from all training loss |

---

## The 16 Keypoints — Precise Definitions

---

### KP 0 — `toe_tip`

**Category: Silhouette / Outline Point**

The most anterior (forward-most) point of the **foot or shoe outline**.
This is a **silhouette point** — it lives on the outer boundary of the foot shape.

```
SIDE VIEW                      TOP-DOWN VIEW
─────────────────              ─────────────────
          ● KP0                ────── ●────────
         /  ← most             foot top surface
        /    forward
       /      point
──────────────────             toe_tip is visible
Ground                         from above ✅
```

| View | Visibility | Notes |
|---|---|---|
| Side view | `v=2` | Marked at the very front tip of the toe silhouette |
| Top-down | `v=2` | Visible — it's the forward-most boundary of the foot from above |
| Back view | `v=0` | Cannot be seen or estimated; mark as not labeled |
| Shoe on | `v=2` | Mark at the tip of the shoe, not the hidden toe inside |

**❌ Common mistakes:**
- Placing it where the toe touches the ground → that is `toe_ground` (KP 1), not this
- Placing it on the toe nail → place it at the very outermost tip of the full toe/shoe outline
- Moving it inward when the foot is foreshortened in perspective

---

### KP 1 — `toe_ground`

**Category: Ground Contact Point**

The point on the **sole** where the **front of the foot makes contact with the floor**.
This is on the big-toe side, at the bottom of the forefoot, where sole meets ground.

```
SIDE VIEW                       TOP-DOWN VIEW
─────────────────               ─────────────────
KP0 ●                           ─────────────────
     \                          foot top surface,
      \                         toe_ground is SOLE
       ●  KP1                   side → OCCLUDED (v=1)
────────────────
Ground ←────── KP1 is here

KP1 is slightly behind and      Mark at estimated sole position
below KP0.                      with v=1 (not v=0)
```

| View | Visibility | Notes |
|---|---|---|
| Side view (from big-toe side) | `v=2` | Visible at ground level, below KP0 |
| Top-down | `v=1` | Sole is not visible from above; mark estimated anatomical position |
| Back view | `v=1` | Not visible; mark estimated position at ground level under forefoot |
| Wearing shoe | `v=1` | Hidden under shoe sole; estimate and mark with v=1 |

**❌ Common mistakes:**
- Marking it at the same position as `toe_tip` → they are different points
- Marking `v=0` in top-down views → it should be `v=1` (position known even if occluded)
- Placing it on the little-toe side → it is on the **big-toe side** (medial forefoot contact)

---

### KP 4 — `ball_medial`

**Category: Sole / Metatarsal Point**

The **1st metatarsal head** — the bump on the **big-toe (medial) side** of the forefoot,
at the widest point on the medial edge of the foot.

```
SOLE VIEW (from below)         TOP-DOWN VIEW
─────────────────────          ─────────────────
    big toe side                anatomical position
         ● ← KP4               is the same. But this
    1st metatarsal head         point is on the SOLE,
    widest medial edge          not the dorsum. So in
                                top-down → v=1 (occluded)

    ────── SOLE ──────
                ● KP5           Mark at the estimated
         5th metatarsal         position even when v=1.
         widest lateral         DO NOT move it inward.
```

| View | Visibility | Notes |
|---|---|---|
| Sole (from below) | `v=2` | The metatarsal-head prominence is directly visible |
| **Back view** | `v=2` | Both metatarsal heads visible from behind at forefoot widest edge |
| Top-down | `v=1` | The sole is face-down; point is occluded but position is known |
| Side view (medial) | `v=1` | The bump is not directly visible from the medial profile — mark estimated position |
| Side view (lateral) | `v=1` | Hidden behind foot; mark estimated position |
| Front view | `v=1` | Sole-side point; mark estimated position |

**❌ Critical mistakes to avoid:**
- Moving the point **inward** because the prominence is not visible from top → **wrong**. Keep it at the true widest medial edge even if v=1
- Placing it on the dorsum (top surface) → it is a SOLE-level point
- Confusing it with the toe knuckle → it is the **widest point** of the foot, not the toe area

---

### KP 5 — `ball_lateral`

**Category: Sole / Metatarsal Point**

The **5th metatarsal head** — the bump on the **little-toe (lateral) side** of the forefoot,
at the widest point on the lateral edge of the foot.

**Same rules as KP 4 but on the opposite side.**

| View | Visibility | Notes |
|---|---|---|
| Sole (from below) | `v=2` | Directly visible as the widest lateral bump |
| **Back view** | `v=2` | Both metatarsal heads visible from behind at forefoot widest edge |
| Top-down | `v=1` | Occluded — sole is face-down |
| Side view (lateral) | `v=1` | The bump is not directly visible from the lateral profile — mark estimated position |
| Side view (medial) | `v=1` | Hidden; mark estimated position |
| Front view | `v=1` | Sole-side point; mark estimated position |

**❌ Critical mistakes to avoid:**
- Moving it inward from the true widest point → **wrong**. The point does not move; only v changes.
- Confusing it with the little toe knuckle → it is the widest bump of the **forefoot body**, not the toe

```
CROSS-SECTION AT WIDEST FOREFOOT (from front):

        dorsum (top)
            │
  KP4 ●────┼────● KP5
big-toe    ●    little-toe
 side      │     side
        sole (bottom)

The ● marks show true anatomical positions.
From top-down, both KP4 and KP5 are on the SOLE side → v=1.
Their positions do NOT move inward.
```

---

### KP 14 — `achilles`

**Category: Posterior Ankle Point**

The **back of the ankle** at the Achilles tendon.
This point is at the **same horizontal level** as `ankle_center` (KP 12).
It is approximately midway vertically between `malleolus_medial` (KP 10) and
`malleolus_lateral` (KP 11), but located **on the posterior (back) surface**.

```
BACK VIEW (achilles visible)    FRONT/DORSAL VIEW
────────────────────────        ────────────────────────
                                ankle_center ●
malleolus   ●     ● malleolus               │ (visible from front)
lateral         medial                      │
              ● ← KP14                      │
          (Achilles tendon,     KP14 ●──────┘
           back of ankle,       (is BEHIND the ankle,
           level with KP12)      therefore occluded: v=1)
                                DO NOT move to front surface.
```

| View | Visibility | Notes |
|---|---|---|
| Back view | `v=2` | Directly visible at back of ankle, level with KP 12 |
| Front view | `v=1` | The point is behind the ankle — mark estimated position behind it |
| Side view | `v=1` or `v=2` | Visible from lateral/medial side; confirm it is level with KP 12 |
| Top-down | `v=1` | Estimate position; it is behind the ankle joint |
| Wearing boot | `v=1` | Hidden by boot shaft; estimate level with KP 12 |

**❌ Critical mistakes to avoid:**
- Moving the point to the **front** of the ankle when annotating from a front-view image → **wrong**. The point is ALWAYS at the back.
- Placing it higher up on the Achilles tendon → it should be at the **level of ankle_center (KP 12)**, not higher
- Marking `v=0` when it's occluded by camera angle → use `v=1`, the position is estimable

---

## View-Angle Decision Table

**Quick reference for annotators. For each view angle, here's what to expect:**

| Keypoint | Top-Down | Front View | Back View | Side View (medial) | Side View (lateral) |
|---|---|---|---|---|---|
| `toe_tip` (0) | v=2 ✅ | v=2 ✅ | v=0 ❌ skip | v=2 ✅ | v=2 ✅ |
| `toe_ground` (1) | **v=1** ⚠️ | v=2 ✅ | v=2 ✅ | v=2 ✅ (big-toe side) | **v=1** ⚠️ |
| `heel_back` (2) | **v=1** ⚠️ | **v=1** ⚠️ | v=2 ✅ | **v=1** ⚠️ | **v=1** ⚠️ |
| `heel_ground` (3) | v=2 ✅ | **v=1** ⚠️ | v=2 ✅ | v=2 ✅ | v=2 ✅ |
| `ball_medial` (4) | **v=1** ⚠️ | **v=1** ⚠️ | v=2 ✅ | **v=1** ⚠️ | **v=1** ⚠️ |
| `ball_lateral` (5) | **v=1** ⚠️ | **v=1** ⚠️ | v=2 ✅ | **v=1** ⚠️ | **v=1** ⚠️ |
| `ball_top` (6) | v=2 ✅ | v=2 ✅ | **v=1** ⚠️ | v=2 ✅ | v=2 ✅ |
| `instep_top` (7) | v=2 ✅ | v=2 ✅ | **v=1** ⚠️ | v=2 ✅ | v=2 ✅ |
| `arch_medial` (8) | v=2 ✅ | v=2 ✅ | **v=1** ⚠️ | v=2 ✅ (big-toe) | **v=1** ⚠️ |
| `midfoot_lateral` (9) | v=2 ✅ | v=2 ✅ | **v=1** ⚠️ | **v=1** ⚠️ | v=2 ✅ |
| `malle_medial` (10) | v=2 ✅ | v=2 ✅ | **v=1** ⚠️ | v=2 ✅ | **v=1** ⚠️ |
| `malle_lateral` (11) | v=2 ✅ | v=2 ✅ | **v=1** ⚠️ | **v=1** ⚠️ | v=2 ✅ |
| `ankle_center` (12) | v=2 ✅ | v=2 ✅ | **v=1** ⚠️ | v=2 ✅ | v=2 ✅ |
| `throat` (13) | v=2 ✅ | v=2 ✅ | v=0 ❌ skip | v=2 ✅ | v=2 ✅ |
| **`achilles` (14)** | **v=1** ⚠️ | **v=1** ⚠️ | v=2 ✅ | v=2 ✅ | v=2 ✅ |
| `shin_mid` (15) | v=2 ✅ | v=2 ✅ | v=2 ✅ | v=2 ✅ | v=2 ✅ |

> **v=2** = visible, mark precisely  
> **v=1 ⚠️** = occluded, mark estimated anatomical position (still supervised in training)  
> **v=0 ❌** = cannot estimate at all, skip

---

## Design Implications for Model Training

### Stage A — Coarse Keypoints (toe_tip, heel_back, ball_medial, ball_lateral)

These 4 points are used only to build the **rotated crop** for Stage B.

| Coarse KP | Views where v=2 | All other views | Stage A behavior |
|---|---|---|---|
| `toe_tip` (0) | Top-down, Front, Side | Back view (skip) | High confidence from most views |
| `heel_back` (2) | **Back view only** | All others → v=1 | **Low confidence** from front/top-down/side — model must infer from box context and toe_tip direction |
| `ball_medial` (4) | **Back view only** | All others → v=1 | **Low confidence** from all non-back views |
| `ball_lateral` (5) | **Back view only** | All others → v=1 | **Low confidence** from all non-back views |

**Critical consequence for crop rotation:**

In any view that is NOT a back view (i.e., front, top-down, side):
- `toe_tip` is the **only reliable coarse KP** for heading estimation
- `heel_back`, `ball_medial`, `ball_lateral` are all occluded → Stage A predicts them from context with low confidence

**How Stage A still works:**

```
Crop rotation angle from:
  PRIMARY:   toe_tip ↔ bounding box center → coarse heading ✅ (always works)
  SECONDARY: heel_back prediction (low-confidence from non-back views)
  FALLBACK:  box aspect ratio long-axis if other signals fail
```

The crop rotation from Stage A will be approximate in front/top-down views.
**Stage B corrects this within 1 frame via `next_roi`** — this is by design.

**Training implication:** Stage A needs 30–40% back-view images so it learns
to predict `heel_back`, `ball_medial`, `ball_lateral` accurately where they ARE visible.
In non-back views, the loss on these KPs is down-weighted (v=1 supervision only).

### Stage B — Landmark Model

The model must learn the **viewpoint-invariant anatomical prior**:
- `achilles` is always POSTERIOR — never predict it at the front surface in front views
- `ball_medial/lateral` are always at the SOLE-level widest points — never predict them inward just because the prominence is not visible from the current angle
- `toe_ground` is always BELOW and BEHIND `toe_tip` — never merge them to the same position

This is achieved by training on data with correct `v=1` labels at true anatomical positions — the model learns that occluded predictions should follow anatomy, not the visible surface.

---

## Golden Rule for Annotators

```
Ask yourself: "If I could see through the foot/shoe from any direction,
               where would this point be?"

Mark it there.
Then set v=2 if it IS directly visible from the current camera angle,
or v=1 if the camera angle hides it but you know where it is anatomically.
```

---

*Version 2.0 — Updated with reference images for KP 0, 1, 4, 5, and 14.*
*Supersedes the keypoint section of annotation_guidelines_v1.0.*

