# Shoes VTO — AI Model Requirements

**Version** 1.0 ·

**Date** 2026-09-13 ·

**From** NGM - AR side at the SDK ·

**To** AI Team ·

**Status** Awaiting your review. Reply per section number.

---

## 0. How to read this

This is the output contract. Everything marked **FROZEN** cannot change after M1 sign-off
without a new `model_version` and new `weights_sha256` — our decoder binds to it.

Same house contract as `wrist_keypoint` and `neck_keypoint`: ONNX opset 12, letterboxed
input, channel-major output, no in-graph NMS, activations baked in-graph, plus a
`*_meta.json` sidecar. If something here conflicts with how you shipped `wrist v20`,
this document wins — say so and we will reconcile.

Nothing in §5–§9 is optional. §12 is the delivery order that unblocks us earliest.

---

## 1. Agreed scope

Unconstrained live capture, both feet, in motion. Confirmed by you, approved by us.

The model must hold up across:

| Axis          | Range it must cover                                                  |
| ------------- | -------------------------------------------------------------------- |
| Distance      | 0.3 m to 2.5 m+ — the user may push a foot at the lens or stand back |
| Camera pitch  | steep top-down (~20° from vertical) through near-horizontal          |
| Foot heading  | full 360° yaw, including toe-toward-camera and heel-toward-camera    |
| Motion        | walking, lifting a leg, rotating the ankle, crossing feet            |
| Blur          | rolling-shutter smear and motion blur at 1/30 s exposure             |
| Occlusion     | trouser hem, sock cuff, the other foot, frame edge, held objects     |
| Feet in frame | 0, 1, or 2                                                           |

**Out of scope for v1:** bystanders' feet (see §13.4), feet below the knee of a seated
third party, non-human feet.

---

## 2. Runtime constraints — non-negotiable

We run in a **web browser**, not on a server and not in a native app.

| Constraint    | Value                                                                                                                                                                                                                                  |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Runtime       | `onnxruntime-web`, WebGPU EP with WASM EP fallback                                                                                                                                                                                     |
| Format        | ONNX, **opset 12**, fp32 primary                                                                                                                                                                                                       |
| Shapes        | Fully static except batch. **No dynamic spatial dims.**                                                                                                                                                                                |
| Ops           | Must execute on **both** the WebGPU EP and the WASM EP with **no CPU fallback**. Ship the op list. `NonMaxSuppression`, `RoiAlign`, `GridSample`, `ScatterND` are known to break or silently fall back — avoid or clear with us first. |
| NMS           | **Not in-graph.** We pick the best anchor per class.                                                                                                                                                                                   |
| Activations   | sigmoid/softmax **baked inside the graph** on visibility, presence, confidence, and mask. We will not re-apply them.                                                                                                                   |
| Target device | iPhone 12 / Snapdragon 778G class, Safari and Chrome. Safari is **WASM only** — it must hit the budget without a GPU.                                                                                                                  |

### 2.1 Compute budget

Frame budget at 30 fps is 33 ms. Our renderer needs ~12 ms. **You get ≤ 18 ms for
everything, both feet included.**

That is only reachable if the detector does **not** run every frame — see §3.

| Stage                  | Cadence                    | Budget (WASM, mid device)                            |
| ---------------------- | -------------------------- | ---------------------------------------------------- |
| A — Detector           | on acquisition / loss only | ≤ 12 ms                                              |
| B — Landmark+Pose+Mask | **every frame, per foot**  | **≤ 7 ms**                                           |
| Steady state (2 feet)  |                            | **≤ 16 ms** (2 × 7 ms + ~2 ms crop prep on our side) |

### 2.2 Download budget

| Artifact             | fp32        | fp16 (WebGPU) | int8-dynamic (WASM) |
| -------------------- | ----------- | ------------- | ------------------- |
| Stage A              | ≤ 5 MB      | ≤ 2.5 MB      | ≤ 1.5 MB            |
| Stage B              | ≤ 14 MB     | ≤ 7 MB        | ≤ 4 MB              |
| **Total first load** | **≤ 19 MB** | ≤ 10 MB       | ≤ 6 MB              |

fp16 and int8 variants are required deliverables, not extras. On WASM, fp16 is _slower_ —
ship int8 for that path (this is what we learned on `neck_256_int8`).

---

## 3. Pipeline shape we expect

Two stages, MediaPipe-style. This is a requirement, not a suggestion — it is the only
shape that fits §2.1 while staying accurate at 2.5 m.

```
frame ──► [A] Foot Detector ──► box + class + 4 coarse kpts ──┐
            (only when tracking is lost)                      │
                                                              ▼
         ┌──────────────────────── rotated crop per foot ─────┤
         │                                                    │
         └──► [B] Landmark + Pose + Mask + Scale ──► keypoints, rotation,
                  (every frame, per foot)              mask, mm, sigma,
                                                       presence, next_roi
                          │
                          └── next_roi feeds the NEXT frame's crop directly.
                              Stage A re-runs only when presence < threshold.
```

**Therefore Stage B must be self-sustaining:** it outputs both a `presence` score and the
ROI for the next frame. If it cannot track without re-detection, the budget in §2.1 is
unreachable and we need to talk before you start training.

---

## 4. What we feed you

| Property      | Value                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------- |
| Source        | `getUserMedia`, **rear camera** (`facingMode: 'environment'`)                            |
| Resolution    | 1920×1080 ideal, negotiated down on weak devices                                         |
| Frame rate    | 30 fps ideal                                                                             |
| **Mirroring** | **None by default** (rear camera, raw frames). The product spec keeps a **Flip camera** control and front-camera use (PRD CV-09), so the class-flip rule in §7 must be implemented and documented — it just does not fire in the normal path. |
| Color         | sRGB, 8-bit, RGB order after our preprocess                                              |
| Orientation   | Portrait and landscape both occur. Assume no fixed up-vector.                            |

### 4.1 Single-image mode

The product spec also ships **uploaded photo / model try-on** in V1 (PRD §2). The same
models must produce a usable result from **one still frame**, with no temporal history:

- Stage A runs normally; Stage B runs once. `next_roi` is ignored.
- No reliance on smoothing or on a previous frame to reach the §8.1 bars.
- Stills are higher resolution and sharper than video frames — do not let the dataset
  become video-only, or single-image accuracy will quietly lag (§9.2).

Preprocess we apply, identical to `wrist_meta.json` — confirm it matches your training:
letterbox to square, pad `rgb(114,114,114)`, divide by 255, NCHW, RGB, fp32, `[0,1]`.

We will send you real capture clips from this exact path (§11) — please validate on those,
not on stock photography. We have been bitten before by evaluating on synthetic crops that
did not match the deployed frame source.

---

## 5. Stage A — Foot Detector

**FROZEN**

|         |                                                                           |
| ------- | ------------------------------------------------------------------------- |
| Family  | YOLOv8n-pose or equivalent single-shot                                    |
| Input   | `images`, `[1, 3, 320, 320]`, fp32, NCHW, RGB, `[0,1]`                    |
| Output  | `output0`, `[1, 18, N]`, channel-major, N = anchor count for 320 (= 2100) |
| Classes | **2** — class `0 = left_foot`, class `1 = right_foot`                     |
| max_det | 2                                                                         |

Row layout, `value(c, i) = data[c*N + i]`:

| Rows | Meaning                                                                                                 |
| ---- | ------------------------------------------------------------------------------------------------------- |
| 0–3  | box `cx, cy, w, h` — letterbox pixels                                                                   |
| 4    | class score, `left_foot`                                                                                |
| 5    | class score, `right_foot`                                                                               |
| 6–17 | 4 coarse keypoints × 3 (`x`, `y`, `v`), in order: `toe_tip`, `heel_back`, `ball_medial`, `ball_lateral` |

The 4 coarse keypoints exist for one reason: to build the **rotated** crop for Stage B.
A rotation-normalized crop is what makes Stage B work at 360° foot heading.

Also ship a 256 variant for the WASM fallback path.

---

## 6. Stage B — Landmark + Pose + Mask + Scale

Runs on one rotation-normalized crop of a single foot. All tensors named, all **FROZEN**.

| Tensor            | Shape              | Dtype | Notes                                                                                                |
| ----------------- | ------------------ | ----- | ---------------------------------------------------------------------------------------------------- |
| in `crop`         | `[1, 3, 224, 224]` | fp32  | NCHW, RGB, `[0,1]`, no letterbox pad (the crop is already square)                                    |
| out `keypoints`   | `[1, 16, 3]`       | fp32  | `x`, `y` crop-normalized `[0,1]`; `v` visibility `[0,1]`, sigmoid in-graph                           |
| out `rotation`    | `[1, 6]`           | fp32  | 6-D continuous rotation, §6.3                                                                        |
| out `scale`       | `[1, 2]`           | fp32  | `foot_length_mm`, `foot_width_mm`, §6.5                                                              |
| out `scale_sigma` | `[1, 2]`           | fp32  | mm, §6.5                                                                                             |
| out `sigma`       | `[1, 16]`          | fp32  | per-keypoint positional σ, crop units                                                                |
| out `confidence`  | `[1, 16]`          | fp32  | `[0,1]`, baked `1 - smoothstep(σ, σ_lo, σ_hi)`, §6.6                                                 |
| out `presence`    | `[1, 1]`           | fp32  | `[0,1]`, "this crop still contains a foot"                                                           |
| out `next_roi`    | `[1, 5]`           | fp32  | `cx, cy, w, h, theta` for the next frame's crop, in full-frame normalized coords; `theta` in radians |
| out `mask`        | `[1, 2, 160, 160]` | fp32  | `[0,1]`, sigmoid in-graph, §6.4                                                                      |

Give us the exact inverse transform from crop-normalized to source-frame pixels in
`foot_inference.js`. Do not make us reverse-engineer it.

### 6.1 Keypoint schema — **FROZEN**

16 keypoints. Order below is the tensor order and never changes.

| #   | Name                | Zone  | Tier | Anatomical definition                                                          |
| --- | ------------------- | ----- | ---- | ------------------------------------------------------------------------------ |
| 0   | `toe_tip`           | sole  | 1    | Most anterior point of the foot/shoe outline                                   |
| 1   | `toe_ground`        | sole  | 1    | Where the front of the sole meets the floor, on the long axis                  |
| 2   | `heel_back`         | sole  | 1    | Most posterior point of the heel outline                                       |
| 3   | `heel_ground`       | sole  | 1    | Where the back of the sole meets the floor — **origin of the canonical frame** |
| 4   | `ball_medial`       | ball  | 1    | 1st metatarsal head — the bump on the big-toe side, at the foot's widest       |
| 5   | `ball_lateral`      | ball  | 1    | 5th metatarsal head — the bump on the little-toe side, at the foot's widest    |
| 6   | `ball_top`          | ball  | 1    | Top surface of the forefoot, directly above the midpoint of 4–5                |
| 7   | `instep_top`        | mid   | 1    | Apex of the instep (highest point of the top of the foot)                      |
| 8   | `arch_medial`       | mid   | 2    | Sole line at the deepest point of the medial arch                              |
| 9   | `midfoot_lateral`   | mid   | 2    | Sole line at the lateral midfoot, opposite 8                                   |
| 10  | `malleolus_medial`  | ankle | 1    | Centre of the inner ankle bone                                                 |
| 11  | `malleolus_lateral` | ankle | 1    | Centre of the outer ankle bone                                                 |
| 12  | `ankle_center`      | ankle | 1    | Midpoint of the ankle joint, between 10 and 11                                 |
| 13  | `throat`            | ankle | 1    | Front ankle crease where foot meets leg — where a shoe tongue tops out         |
| 14  | `achilles`          | ankle | 1    | Back of the ankle at the Achilles tendon, level with 12                        |
| 15  | `shin_mid`          | leg   | 1    | Leg centreline, 120 mm proximal to 12                                          |

Tier 2 is now only indices **8 and 9**. Index 15 (`shin_mid`) is **Tier 1** because the
product spec makes Boots and High Boots MUST in V1 (PRD §3) and the shaft needs a leg axis.

Tier 2 points may be labeled `v=0` in the first dataset pass. **The layout does not change** —
the tensor is always `[1, 16, 3]`.

Every keypoint has a consumer. None is decorative:

| Zone         | Indices        | What we build from it                                    |
| ------------ | -------------- | -------------------------------------------------------- |
| `sole_line`  | 0,1,2,3,8,9    | Floor plane, foot long axis, shoe length, contact shadow |
| `ball_ring`  | 4,5,6          | Foot width, forefoot volume, shoe flex line              |
| `ankle_ring` | 10,11,12,13,14 | Shoe collar/opening, ankle axis, tongue height           |
| `leg_axis`   | 12,15          | Boot shaft direction                                     |

### 6.2 Visibility convention — identical to `wrist_meta.json`

| Label | Meaning                                                                                    | Head target            |
| ----- | ------------------------------------------------------------------------------------------ | ---------------------- |
| `v=0` | Not labeled / outside frame                                                                | excluded from all loss |
| `v=1` | Present but **covered** (sock, trouser, other foot, worn shoe). Position still supervised. | **negative**           |
| `v=2` | Visible and unoccluded                                                                     | **positive**           |

Train the visibility head on `v == 2` only. This is the exact `wrist v17 → v18` fix —
using `v != 0` gave chance-level occlusion AUC (0.492). Do not repeat it.

**Required:** occlusion AUC ≥ **0.80**.

### 6.3 Canonical foot frame and 6-DoF — **FROZEN**

Right-handed, Y-up, matching our GLB convention (real metres, sole at y=0, front +Z).

```
O   = keypoint 3 (heel_ground)
+Z  = unit( toe_tip - heel_back ), projected onto the floor plane   [foot forward]
+Y  = floor normal, pointing up                                      [foot up]
+X  = +Y x +Z                                                        [right-handed]
```

For a **right** foot, `+X` points to the **medial** (big-toe) side.
For a **left** foot, `+X` points to the **lateral** (little-toe) side.
The construction rule is identical for both feet, so `R` is continuous across the
left/right class. **Do not negate an axis to "fix" a mirrored foot.**

**Representation:** 6-D continuous rotation (Zhou et al.) — the first two columns of `R`,
recovered by Gram–Schmidt. Not quaternions (double cover), not Euler (gimbal lock).

**Division of labour — please read, it reduces your work:**

We solve the final 6-DoF pose ourselves with PnP, using your 2-D keypoints + the canonical
3-D template (§10.5) + `foot_length_mm`. Camera intrinsics are our problem, not yours.

Your `rotation` head is used as a **prior and as the fallback** when too few keypoints are
visible for a stable PnP (toe-on views, heavy occlusion). So it must be good, but it does
not have to carry the placement alone.

**We do not want a translation head.** Do not regress camera-space translation.

### 6.4 Segmentation — `mask`, 2 channels

| Ch  | Class                                                                                                    | We use it to                                                                                                                                              |
| --- | -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | **foot_or_shoe** — the real foot including any sock or shoe already worn, up to the ankle line at kpt 12 | Know exactly what the virtual shoe has to cover. Where the real shoe is _larger_ than the virtual one, we edge-fill the remainder against the background. |
| 1   | **occluder / lower leg** — trouser leg, hem, sock cuff, the calf and shin, the other foot, hands, held objects | Two jobs: draw those pixels **in front of** the virtual shoe, **and** give boot shafts a leg to wrap (PRD §3 makes Boots and High Boots MUST, PRD §7 requires lower-leg occlusion)                                                                                                        |

Notes:

- Crop space, `160×160`, sigmoid baked in-graph.
- Channel 0 must include an already-worn shoe. A barefoot-only mask is useless to us —
  most users will be wearing shoes when they try shoes on.
- **Edge quality is what we ship, not IoU.** A mask that is 0.95 IoU with a ragged
  boundary looks worse than 0.90 IoU with a clean one. See the boundary-IoU bar in §8.
- **Temporal stability is an acceptance gate.** A mask edge that crawls by 3 px between
  frames is instantly visible as a shimmering halo, even when every single frame is
  "correct" in isolation.

### 6.5 Metric scale — `scale`, `scale_sigma`

`foot_length_mm` = heel_back to toe_tip along the long axis, foot flat and loaded.
`foot_width_mm` = ball_medial to ball_lateral.

Ground truth must be **physically measured** (Brannock device or ruler), not derived from
a label, on ≥ 100 real subjects spanning EU 34–48.

> **Priority — read first.** The product spec puts *exact shoe-size measurement from camera*
> in **MVP Exclusions** (PRD §20) and states plainly: **do not claim exact shoe sizing in V1**
> (PRD §6). So this head is **NOT a V1 blocker.** Treat it as V1.1 R&D.
>
> What **is** required in V1 is the **relative** scale below — AR placement depends on it.
> Do not spend M1–M4 effort on millimetres.

**V1 requirement — relative scale.** `foot_length` in units of the detection-box diagonal.
Well-posed, no calibration needed, and it is what our fit solver actually consumes.

**V1.1 target — metric scale.** Length median absolute error ≤ **5 mm**, P90 ≤ **10 mm**.
One **full** EU size step is ~6.7 mm (one Paris point), so 5 mm median is about ±0.75 of a
size — the loosest we can hedge in the UI and still be useful. Until this is validated,
nothing measured here may be presented to a customer as a shoe size (PRD §6, §12).

`scale_sigma` is mandatory. We gate whether the size recommendation is shown at all on it.

**Honesty clause.** Monocular metric scale with no reference object is ill-posed and we
know it. If you judge the 5 mm bar unreachable without a reference object, say so at
**M1** — we would then add a guided calibration step (A4 sheet or bank card on the floor),
and that is a product change with lead time. It is not a reason to delay M1–M4.

Sizing is a **separately gated head**. If it misses, try-on still ships.

### 6.6 Uncertainty head

Same contract as `neck-sigma-head.onnx`: per-keypoint σ plus a baked `[0,1]` confidence,
`confidence = 1 - smoothstep(σ, σ_lo, σ_hi)`, with `σ_lo` / `σ_hi` published in the sidecar.

We fade the shoe on low confidence rather than blinking it off. Required calibration:
Spearman(σ, actual error) ≥ **+0.40**, correctly signed, no confidence collapse.

May be a head on the shared backbone or a separate tiny model — your call, but it counts
against the §2 budget either way.

---

## 7. Mirroring and class flip

Frames are **not** mirrored (§4), so no flip handling is needed in normal operation.

Documented for completeness, in case a front camera is ever used:

- `flip_idx` is **identity** `[0..15]`. Keypoint indices do **not** swap — they are
  anatomical, and mirroring the image moves the same physical point across the frame.
- Only the **class label** flips: `left_foot ↔ right_foot`.

State this explicitly in the sidecar. It is the single most common decode bug in this class
of model.

---

## 8. Acceptance criteria

All measured on the **held-out** set of §9.4, which includes our capture footage and which
you must never train on.

### 8.1 Detection and keypoints

| Metric                                | Bar                          |
| ------------------------------------- | ---------------------------- |
| Detection rate, all distance buckets  | ≥ 0.97                       |
| Detection rate, 2.0–2.5 m bucket      | ≥ 0.92                       |
| Left/right class accuracy             | ≥ 0.99                       |
| Pose mAP@50                           | ≥ 0.80                       |
| Pose mAP@50-95                        | ≥ 0.62                       |
| NME, visible kpts, clean frames       | ≤ 3.0 % of foot-box diagonal |
| NME, motion blur or partial occlusion | ≤ 6.0 %                      |
| Occlusion AUC (§6.2)                  | ≥ 0.80                       |

### 8.2 Rotation

| Metric                          | Bar   |
| ------------------------------- | ----- |
| Geodesic rotation error, median | ≤ 6°  |
| Geodesic rotation error, P90    | ≤ 15° |

Yaw is by far the most visible axis — a 10° yaw error reads as the shoe pointing the wrong
way. Report per-axis error separately.

### 8.3 Segmentation

| Metric                                | Bar                                                   |
| ------------------------------------- | ----------------------------------------------------- |
| Mask IoU, channel 0                   | ≥ 0.90                                                |
| **Boundary IoU @ 3 px, channel 0**    | ≥ **0.75**                                            |
| Mask IoU, channel 1                   | ≥ 0.85                                                |
| Boundary temporal jitter, static clip | ≤ 1.5 px mean displacement between consecutive frames |

### 8.4 Temporal stability — **the AR-quality gate**

Per-frame accuracy does not predict how AR looks. These do. Measure on video, not stills.

| Metric                                                 | Bar                          |
| ------------------------------------------------------ | ---------------------------- |
| Keypoint jitter, static foot, per-kpt std              | ≤ 1.0 % of foot-box diagonal |
| Spike rate (frame-to-frame jump > 8 % of box diagonal) | ≤ 2 per 100 frames           |
| **Left/right class-flip rate**                         | ≤ 1 per 500 frames           |
| Re-acquisition after full occlusion                    | ≤ 3 frames                   |
| ID stability through a foot-crossing event             | ≥ 0.98                       |

A class flip swaps the two shoes on screen. It is the most damaging single failure in this
product — worse than losing tracking entirely. Please treat the 1-in-500 bar as hard.

We apply One-Euro smoothing on our side and will publish the constants back to you. Do not
build heavy temporal smoothing into the graph — it fights ours and adds latency.

### 8.5 Annotation-geometry gate — **read this one carefully**

We lost a full model cycle on `wrist v17–v19` to this exact failure, and the defect is still
documented in `wrist_meta.json`: the three `wrist_ring` keypoints were annotated **along**
the forearm axis instead of **across** it. Every per-frame metric looked healthy. But the
cross-wrist separation had collapsed to near zero, the derived ring geometry was unusable,
and the geometry gate passed on **1 %** of predictions. We could not size a bracelet from
the keypoints at all and had to fall back to box width.

The identical trap exists here: `ball_medial` (4) and `ball_lateral` (5) must be annotated
**across** the foot at its widest, not drifted along it. Same for the two malleoli (10, 11).

Run these gates on **both ground-truth labels and model predictions**, and put the numbers
in the validation report:

| Gate                                                         | Bar                       |
| ------------------------------------------------------------ | ------------------------- |
| Angle between ball line (4→5) and sole long axis (1→3)       | median in **[75°, 105°]** |
| `\|kpt4 - kpt5\| / foot_length`, near-top-down frames only   | median ≥ **0.30**         |
| `\|kpt10 - kpt11\| / foot_length`, near-top-down frames only | median ≥ **0.18**         |
| Overall geometry-gate pass rate                              | ≥ **85 %**                |

If the pass rate comes back low, the fix is in the annotation guide, not in the loss
function. Catch it at the 200-frame pilot (M1), not after 15,000 frames are labeled.

---

## 9. Dataset requirements

You own sourcing (Roboflow, as with `wrist_keypoint`). We own the schema and sign-off.

You wrote that you will "cover all different data scenarios as much as possible". We agree
with the intent — the table below is that intent made checkable, so both sides can tell when
coverage is actually done rather than estimated.

### 9.1 Volume

|                      | Minimum    |
| -------------------- | ---------- |
| Annotated frames     | **15,000** |
| Distinct subjects    | **400**    |
| Distinct video clips | **800**    |

For calibration: `wrist v20` needed 4,651 frames for 6 keypoints, 2 classes and a narrow
pose range. This target has 16 keypoints, full 360° yaw, an 8× distance range, a
segmentation mask and a metric head. 15,000 is the floor, not the goal.

### 9.2 Coverage matrix — minimum share of the annotated set

| Axis                        | Buckets and minimum share                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------ |
| Distance                    | 0.3–0.6 m **25 %** · 0.6–1.2 m **35 %** · 1.2–2.0 m **25 %** · 2.0 m+ **15 %**                         |
| Camera pitch from vertical  | 0–30° **30 %** · 30–60° **40 %** · 60–90° **30 %**                                                     |
| Foot yaw                    | 8 buckets of 45°, **none below 8 %**                                                                   |
| Footwear                    | barefoot 20 % · socks 15 % · sneakers 25 % · boots 15 % · sandals 10 % · heels/formal 10 % · other 5 % |
| Skin tone                   | Fitzpatrick I–VI, **each ≥ 10 %**                                                                      |
| Motion                      | ≥ **30 %** from walking / lifting / rotating clips                                                     |
| Motion blur                 | ≥ **15 %** with visible blur                                                                           |
| Occlusion                   | ≥ **20 %** partially occluded (hem, other foot, object, frame edge)                                    |
| Frame edge                  | ≥ **10 %** with the foot clipped by the frame boundary                                                 |
| Two feet in frame           | ≥ **35 %**                                                                                             |
| Feet crossing / overlapping | ≥ **8 %**                                                                                              |
| Floor surface               | ≥ 8 distinct types (carpet, tile, wood, concrete, grass, sand, patterned, reflective)                  |
| Lighting                    | indoor / outdoor / low-light, low-light ≥ **15 %**                                                     |

### 9.3 Hygiene

- **Split by subject and by clip, never by random frame.** Random-frame splits leak near-
  duplicates into validation and every number becomes fiction.
- Perceptual-hash dedup at **phash ≤ 6**, same as `wrist v20`.
- Sample video clips, do not take consecutive frames.
- Publish the realized coverage matrix against §9.2 — not the plan, the actual counts.

### 9.4 Held-out test set

~500 frames, supplied by us from the real SDK capture path (§11), **never trained on and
never tuned against**. All §8 numbers are reported on this set. You may report your own
validation split alongside, but ours is the one that decides acceptance.

### 9.5 Annotation guide

Before bulk labeling, deliver a written guide with a reference photo per keypoint, showing
the point placed correctly and at least one common mistake. We review it at M1. The §8.5
failure is an annotation-guide failure, and the guide is where it gets prevented.

---

## 10. Deliverables

| #    | Artifact                             | Notes                                                                                                                                                                                                                                                                                    |
| ---- | ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 10.1 | `foot_det_*.onnx`, `foot_lmk_*.onnx` | fp32 + fp16 + int8-dynamic variants                                                                                                                                                                                                                                                      |
| 10.2 | `foot_meta.json`                     | Sidecar, key-for-key mirroring `wrist_meta.json`: `model_version`, `weights_sha256` per file, input/output spec, keypoint table, zones, thresholds, `flip_idx`, `frozen`, `warnings`, `changelog`                                                                                        |
| 10.3 | `foot_inference.js`                  | Reference decoder for `onnxruntime-web`. Plain JS, no framework. Must include the crop transform and its exact inverse.                                                                                                                                                                  |
| 10.4 | `README.md`                          | Integration notes + changelog + metric table                                                                                                                                                                                                                                             |
| 10.5 | **`foot_canonical.json`**            | The 16 keypoints in **millimetres**, in the canonical frame of §6.3, for a reference foot of `foot_length = 265 mm`, plus the scaling rule to other lengths. **This is the single artifact that lets the AI model, the 3D shoe models and our renderer agree.** Without it nothing fits. |
| 10.6 | Validation report                    | Every table in §8, on §9.4. Include the §8.5 geometry gates on GT _and_ predictions.                                                                                                                                                                                                     |
| 10.7 | **Golden fixture**                   | A 300-frame clip plus your decoder's exact per-frame JSON output. We diff our decoder against it. This catches decode mismatches in an hour instead of a week.                                                                                                                           |
| 10.8 | ONNX op list                         | Per model, with confirmation that every op runs on the WebGPU EP and the WASM EP without CPU fallback (§2).                                                                                                                                                                              |
| 10.9 | Realized coverage matrix             | §9.3                                                                                                                                                                                                                                                                                     |

`weights_sha256` must be filled for every shipped file. Placeholders block integration —
we bind to the hash to detect a silent retrain.

---

## 11. What Novagates provides

|                                                                         | Owner       | When                      |
| ----------------------------------------------------------------------- | ----------- | ------------------------- |
| Real capture clips from the live SDK frame source, across the §9.2 axes | AR/SDK      | on M1 kickoff             |
| The 500-frame held-out test set (§9.4)                                  | AR/SDK      | before M3                 |
| Canonical frame diagram + our GLB origin convention                     | AR/SDK + 3D | on M1 kickoff             |
| Review of the annotation guide and the 200-frame pilot                  | AR/SDK      | M1, within 3 working days |
| One-Euro smoothing constants we settle on                               | AR/SDK      | after M2                  |
| Integration harness + measured on-device latency feedback               | AR/SDK      | continuous from M2        |

---

## 12. Delivery order

Staged so we can build in parallel instead of waiting for one big drop.

| Milestone | You deliver                                                                                                                                      | Unblocks                                                                                                       |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **M1**    | Frozen keypoint schema sign-off, canonical-frame sign-off, annotation guide, `foot_canonical.json`, 200-frame pilot annotation. **No training.** | The **3D team starts immediately** — every shoe GLB can be aligned to the template. This is the critical path. |
| **M2**    | Detector + landmarks only. No mask, no rotation, no mm, no sigma. Accuracy can be rough.                                                         | We wire the full pipeline end to end and return real on-device latency                                         |
| **M3**    | Rotation head + first full acceptance run against §8.1/§8.2/§8.4                                                                                 | Real placement, real look                                                                                      |
| **M4**    | Segmentation (§6.4)                                                                                                                              | Occlusion — the jump from "sticker" to "worn"                                                                  |
| **M5**    | Uncertainty head (§6.6) + relative scale (§6.5)                                                                                                   | Confidence-driven fallback and fit solving — **this completes V1**                                             |
| **V1.1**  | Metric `foot_length_mm` (§6.5), separately scoped                                                                                                | Size recommendation. **Not a V1 blocker** (PRD §20).                                                           |

M1 is small and it unblocks the most people. Please do not fold it into M2.

---

## 13. Open questions — please answer with your M1 reply

1. **Rotation head.** Can you hit §8.2 with a direct 6-D head, or do you want us to rely on
   PnP and treat your rotation purely as a fallback? Either is fine — we need to know which
   before we tune the blend.
2. **Mask placement.** Head on the shared Stage-B backbone, or a separate model? Give us the
   measured ms either way, since it lands in the same §2.1 budget.
3. **Metric scale.** Pure monocular, or do you need a reference object? Answer at M1 even
   though the work is V1.1 — a reference object is a UX change with lead time (§6.5).
4. **`max_det`.** We spec 2. If a bystander's feet enter the frame, what does the model do —
   and do you want us to pick by box area, by confidence, or by proximity to frame centre?
5. **Crop resolution.** We spec 224×224 for Stage B. If 192 hits §8 and buys headroom in
   §2.1, we would rather have the headroom. What do your experiments say?
6. **Boots and high-tops.** `shin_mid` (15) is Tier 2. Are boot-shaft frames in the dataset
   plan? If not, boots are a v2 category and we will scope the catalogue accordingly.

---

## 14. Sign-off

Nothing marked **FROZEN** changes after M1 without a new `model_version` and new
`weights_sha256`. If a retrain changes the keypoint order, the tensor layout, the class
order or the canonical frame, our placement breaks silently — the sidecar hash is how we
find out before our customers do.

Reply per section number. Anything you disagree with, say so at M1 — it is far cheaper to
change this document than to change 15,000 annotations.

---

### Appendix A — `foot_meta.json` skeleton

```jsonc
{
  "meta_schema_version": 1,
  "model_name": "foot_pose",
  "model_version": "v1",
  "framework": "...",
  "opset": 12,
  "trained_date": "...",
  "trained_on": "...",
  "changelog": "...",

  "num_classes": 2,
  "class_names": ["left_foot", "right_foot"],
  "coordinate_convention": "Stage A: letterbox pixels (0..imgsz). Stage B: crop-normalized [0,1]. Inverse transform in foot_inference.js.",

  "stage_a": { "input": {}, "output": {}, "variants": [] },
  "stage_b": { "input": {}, "outputs": {} },

  "num_keypoints": 16,
  "keypoints": [{ "index": 0, "name": "toe_tip", "zone": "sole_line", "tier": 1, "definition": "..." }],
  "zones": { "sole_line": [0, 1, 2, 3, 8, 9], "ball_ring": [4, 5, 6], "ankle_ring": [10, 11, 12, 13, 14], "leg_axis": [12, 15] },

  "canonical_frame": {
    "origin_keypoint": 3,
    "forward": "+Z = unit(kpt0 - kpt2) on floor plane",
    "up": "+Y = floor normal",
    "right": "+X = +Y x +Z",
    "rotation_repr": "6d-gram-schmidt",
  },

  "visibility_convention": {
    "v0": "unlabeled",
    "v1": "covered (negative)",
    "v2": "visible (positive)",
    "output_range": "[0,1], sigmoid in-graph",
    "occlusion_auc": 0.0,
  },

  "horizontal_mirror": {
    "flip_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    "class_flip": { "class_0_becomes": 1, "class_1_becomes": 0 },
    "note": "indices are identity; only the class label flips",
  },

  "thresholds": {
    "detection_conf": 0.0,
    "presence_gate": 0.0,
    "visibility_show": 0.0,
    "visibility_hide": 0.0,
    "reacquire_jump_fraction": 0.0,
    "grace_hold_ms": 0,
    "recommended_max_det": 2,
  },

  "scale_head": { "sigma_lo": 0.0, "sigma_hi": 0.0, "length_mae_mm": 0.0, "length_p90_mm": 0.0, "calibration_subjects": 0 },

  "performance": { "stage_a_ms_wasm": 0.0, "stage_b_ms_wasm": 0.0, "stage_b_ms_webgpu": 0.0, "device": "..." },

  "variants": [{ "file": "...", "imgsz": 0, "dtype": "fp32", "size_mb": 0.0, "backend": "...", "role": "..." }],
  "weights_sha256": {},

  "frozen": { "keypoint_order_0_15": true, "class_order": true, "stage_b_tensor_names": true, "canonical_frame": true },
  "warnings": [],
}
```
