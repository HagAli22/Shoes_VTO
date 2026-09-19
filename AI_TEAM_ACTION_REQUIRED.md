# Shoes VTO — AI team action required

**Date:** 2026-09-18  
**From:** NGM / Novagates AR SDK team  
**To:** AI Perception team  
**Priority:** Production blocker  
**Please reply per numbered section and do not rename tensors or change signed-off layouts without a new model/contract version.**

## 1. Why we need another delivery

Thank you for the Stage A package. We integrated it as an engineering-preview detector and verified that its FP32 model runs through our browser ONNX pipeline. It is not sufficient for the agreed production scope: live rear camera, both feet, walking, crossing, motion blur, partial/full occlusion, one foot passing in front of the other, still-image mode, open footwear, ankle boots, and high boots.

The current SDK foundation already provides camera acquisition, product selection, Stage A preprocessing/decode, bounded candidates, temporal association, persistent track IDs, and a perspective Three.js scene. The production blocker is the missing perception output needed for accurate pose, self-sustained tracking, fitting, and occlusion.

## 2. Verified status of the 2026-09-17 delivery

| Item | Verified result | Consequence |
| --- | --- | --- |
| Stage A output | `[1,54,2100]`, not the requested `[1,18,2100]` | We added a version-specific preview decoder; this must not silently replace the production contract. |
| Keypoints | 16 raw triplets, but metadata identifies only 14 anatomical points | Raw channels 1 and 2 are unknown; `ankle_center` and `throat` are absent from the mapping. |
| Reference decoder | Cross-class NMS at IoU 0.35 | It deletes a valid second foot during overlap. Our SDK removed this behavior, but the official decoder/fixtures must also be corrected. |
| FP16 file | Byte-identical to FP32; SHA-256 for both is `140067c8d9a32809441f078483cadbb598c611830fdc57eb514ce2fda16b87d8` | This is not an FP16 export and must not be advertised as one. |
| INT8 file | Dynamic quantization; SHA-256 `7cdac5ee75a7595dd316e1fe2e61b2cb5a2f860b27a4cc74af37dc2129c60625` | Requires real browser/WASM compatibility, accuracy, and latency evidence before deployment. |
| Stage A FP32 size | 13,229,560 bytes | Exceeds the requested 5 MB Stage A budget. |
| Canonical length | Metadata says 265 mm; `heel_back.z=-15`, `toe_tip.z=265` | The defined heel-back-to-toe distance is 280 mm, so the geometry and metadata disagree. |
| Canonical width | Metadata says 98 mm; ball-point separation is about 94.76 mm | The measurement definition must be made explicit and consistent. |
| Fixture | Five decoded JSON frames, only one containing two detections | There are no source images/video, raw tensors, loss/recovery sequence, or reproducible browser benchmark. |
| Missing production outputs | No Stage B, masks, rotation, uncertainty, confidence, presence, next ROI, or scale | Accurate walking fit and occlusion cannot be completed on the AR side without them. |

## 3. Required decision on Stage A

Please choose and document one of these paths. We prefer **A** for the production two-stage pipeline.

### A. Deliver the frozen compact detector

- Input `images`: `[1,3,320,320]`, FP32, NCHW, RGB `[0,1]`, centered `114` padding.
- Output `output0`: `[1,18,2100]`, channel-major.
- Rows 0–3: `cx, cy, w, h` in letterbox pixels.
- Rows 4–5: sigmoid class scores for `left_foot`, `right_foot`.
- Rows 6–17: four coarse keypoints × `(x,y,visibility)` in this order:
  `toe_tip`, `heel_back`, `ball_medial`, `ball_lateral`.
- No in-graph NMS and no cross-class NMS in the reference decoder.
- Also provide a static 256-input fallback variant with its exact anchor count and output shape.

### B. Keep the current 16-keypoint Stage A as a separately versioned contract

If this is intentional, identify raw channels 1 and 2 with exact anatomical definitions and training labels, or explicitly mark them reserved/unknown. Do not map them to `ankle_center` or `throat` without evidence. Explain why Stage B is still needed and how this larger detector meets acquisition latency and download budgets.

For either path, output bounded raw candidates. Anatomical side is evidence, not a track ID; the AR tracker owns persistent identity.

## 4. Required Stage B model

Stage B runs every processed frame on one rotation-normalized crop for one foot. It must support a batch of one and two independent calls without hidden temporal state.

### 4.1 Input

| Tensor | Shape | Type | Contract |
| --- | --- | --- | --- |
| `crop` | `[1,3,224,224]` | FP32 | NCHW, RGB `[0,1]`, square rotated crop, no letterbox padding |

Please provide exact crop construction, padding, angle sign, pixel-center, resize-rounding, and inverse-transform rules with executable fixtures.

### 4.2 Required outputs

| Tensor | Shape | Required meaning |
| --- | --- | --- |
| `keypoints` | `[1,16,3]` | Crop-normalized `(x,y,visibility)` in the frozen order in §5 |
| `rotation` | `[1,6]` | Continuous 6-D rotation: first two columns of the object-to-camera rotation matrix, recovered by Gram–Schmidt |
| `relative_scale` | `[1,2]` | Dimensionless foot length and width relative to the canonical template |
| `relative_scale_sigma` | `[1,2]` | Uncertainty in the same relative units |
| `sigma` | `[1,16]` | Per-keypoint positional sigma in crop-normalized units |
| `confidence` | `[1,16]` | Calibrated `[0,1]` confidence, with the exact sigma-to-confidence mapping documented |
| `presence` | `[1,1]` | Probability that the target foot remains inside this crop |
| `next_roi` | `[1,5]` | `(cx,cy,w,h,theta)` in **current crop-normalized coordinates**, not full-frame coordinates |
| `foot_mask` | `[1,1,160,160]` | Target foot or currently worn shoe, sigmoid in graph |
| `foreground_mask` | `[1,1,160,160]` | Pixels in front of the target virtual shoe: garment, leg/cuff, hand/object, or the other foot, sigmoid in graph |
| `depth_order` | `[1,1]` or documented equivalent | Confidence that the detected foreground/other-foot region is in front of this target foot |

Important contract corrections:

1. A crop-only model cannot output a full-frame ROI unless it receives the crop-to-frame transform. Output `next_roi` in current crop coordinates; the SDK composes it with the recorded crop transform. If you require full-frame output, add the explicit transform tensor as an input and specify it.
2. Do not label monocular visual scale as millimeters. Use `relative_scale`. A separate `metric_scale_mm` output may be added only if you can provide calibration methodology and held-out metric error.
3. A union mask does not establish which foot is in front. We require per-target foreground semantics plus depth-order confidence for foot-on-foot crossing.
4. `visibility`, positional `confidence`, and track `presence` are different quantities and must not share one undocumented score.

If you propose a different tensor design, send it for written sign-off before training/export.

## 5. Frozen 16-point anatomical order

| Index | Name | Tier |
| ---: | --- | ---: |
| 0 | `toe_tip` | 1 |
| 1 | `toe_ground` | 1 |
| 2 | `heel_back` | 1 |
| 3 | `heel_ground` | 1 |
| 4 | `ball_medial` | 1 |
| 5 | `ball_lateral` | 1 |
| 6 | `ball_top` | 1 |
| 7 | `instep_top` | 1 |
| 8 | `arch_medial` | 2 |
| 9 | `midfoot_lateral` | 2 |
| 10 | `malleolus_medial` | 1 |
| 11 | `malleolus_lateral` | 1 |
| 12 | `ankle_center` | 1 |
| 13 | `throat` | 1 |
| 14 | `achilles` | 1 |
| 15 | `shin_mid` | 1 |

Visibility labels:

- `v=0`: not labeled or outside frame; excluded from loss.
- `v=1`: anatomically present but covered; position may be supervised, visibility target is negative.
- `v=2`: visible and unoccluded; visibility target is positive.

Please state whether coordinates follow underlying foot anatomy or the visible outline of an existing shoe. Training on a large worn-shoe outline as anatomical foot length would make the replacement shoe too large.

## 6. Canonical geometry corrections required before sign-off

Deliver a versioned right and left canonical template plus an illustrated axis guide.

- Right-handed coordinate system, Y up, sole plane at `y=0`, foot forward `+Z`.
- State which side every template represents.
- Define the left/right reflection rule and preserve proper rotation matrices; do not fix handedness by negating one rotation-matrix column.
- Reconcile `reference_foot_length_mm` with the actual heel-to-toe coordinates.
- Reconcile `reference_foot_width_mm` with the named measurement points.
- Define whether each measurement is anatomical, footwear-outline, or last-based.
- Separate foot pose, lower-leg pose, and ground-contact state. A walking foot's local up axis is not always the world floor normal.

Do not change canonical coordinates after asset calibration without a new canonical version.

## 7. Required model variants and browser evidence

For every production model, deliver:

- FP32 primary ONNX, opset 12, static spatial shapes.
- A real FP16 WebGPU export with a different hash and verified FP16 initializers.
- An INT8 WASM candidate only if it passes accuracy parity and operator compatibility.
- Complete input/output names, shapes, dtypes, activation locations, opset, operator list, and SHA-256 hashes.
- Proof that each graph executes through `onnxruntime-web` WebGPU and WASM execution providers without unexpected graph partitioning.

Requested size budgets:

| Artifact | FP32 | FP16 | INT8 |
| --- | ---: | ---: | ---: |
| Stage A | ≤5 MB | ≤2.5 MB | ≤1.5 MB |
| Stage B | ≤14 MB | ≤7 MB | ≤4 MB |

Requested timing evidence on iPhone 12-class and Snapdragon 778G-class devices or the closest available named devices:

- Stage A acquisition/recovery: ≤12 ms per invocation.
- Stage B: ≤7 ms per foot.
- Two Stage B calls plus crop preparation: ≤16 ms steady-state target.
- Report cold start, warm p50/p95, preprocessing, inference, output readback, provider, browser, device, temperature/test duration, and whether cross-origin isolation was enabled.
- Test both WebGPU where supported and forced single-thread WASM. Do not use a blanket “Safari = WASM” rule; report the exact browser/device/provider tested.

If a budget cannot be met, report the measured result before changing shapes or dropping outputs.

## 8. Required reproducible delivery package

Use lowercase kebab-case filenames and include:

```text
shoes-vto-ai-v2/
  README.md
  model-contract.json
  canonical-left.json
  canonical-right.json
  reference-decoder.js
  crop-transforms.js
  models/
    stage-a-320-fp32.onnx
    stage-a-320-fp16.onnx
    stage-a-256-int8.onnx
    stage-b-224-fp32.onnx
    stage-b-224-fp16.onnx
    stage-b-224-int8.onnx
  fixtures/
    source-frames-or-clips/
    raw-output-tensors/
    expected-decoded.json
    crop-transform-cases.json
  reports/
    accuracy.md
    browser-performance.md
    dataset-coverage.md
    operator-compatibility.md
```

Fixtures must cover:

- 0, 1, and 2 feet.
- Overlapping boxes and both feet receiving the same top class.
- Slow and fast crossings in both front/back orders.
- One foot hidden for 0.25, 0.5, and 1 second, followed by re-entry.
- Walking toward, away from, and across the camera.
- Heel strike, toe-off, foot lift, and ankle rotation.
- Top-down, toe-on, heel-on, portrait, landscape, and frame clipping.
- Motion blur, low light, patterned backgrounds, reflective floors.
- Bare feet, socks, existing shoes, sandals/open footwear, trouser hems, hands/objects, and boot-height lower-leg coverage.
- Still-image inference without temporal history.

Every expected output must identify the source frame ID and timestamp. Provide at least one contiguous 300-frame golden sequence with raw outputs so identity, loss, and reacquisition are reproducible.

## 9. Required held-out evaluation report

Report pooled and per-bucket denominators, subject/clip-separated splits, and confidence intervals where practical.

| Measure | Acceptance gate |
| --- | ---: |
| Detection rate | ≥97% overall; ≥92% at 2.0–2.5 m |
| Anatomical side accuracy | ≥99% |
| Pose mAP | ≥0.80 at 50; ≥0.62 at 50–95 |
| Visible-keypoint NME | ≤3% box diagonal clean; ≤6% blur/partial occlusion |
| Rotation | median ≤6°; p90 ≤15° |
| Occlusion AUC | ≥0.80 |
| Foot/existing-shoe mask IoU | ≥0.90 |
| Foreground/leg mask IoU | ≥0.85 |
| Boundary IoU at 3 px | ≥0.75 |
| Static stability | keypoint std ≤1% box diagonal; mask edge displacement ≤1.5 px |
| Temporal jumps | ≤2 jumps over 8% diagonal per 100 frames |
| Class flips | ≤1 per 500 frames |
| Crossing identity | ≥98% events retain identity; no visible swap in mandatory scenarios |
| Reacquisition | ≤3 processed frames after the foot becomes sufficiently visible; also report milliseconds |
| Uncertainty calibration | Spearman(sigma, actual error) ≥+0.40 without confidence collapse |

Define visibility eligibility, confidence thresholds, pose metric, rotation frame, and missing-prediction handling before presenting results.

## 10. Existing-shoe removal and tall-boots decision

Please answer explicitly:

1. Can your V1 models distinguish the target foot/existing shoe, foreground garment, and the other foot with reliable depth ownership?
2. Can you reconstruct skin/floor hidden by a larger existing shoe? A mask alone cannot reveal hidden pixels. If not, we will require bare feet/compatible socks and mark broad shoe removal unsupported.
3. Does training include `shin_mid`, calf/lower-leg masks, and sufficient vertical crop coverage for ankle and high boots?
4. Is a separate lower-leg/boot-shaft model required? If yes, provide its proposed contract and budget before implementation.

## 11. What is not assigned to the AI team

The following remain with Novagates SDK, 3D/content, backend, product, and QA teams:

- Splitting display-pair GLBs into correct left/right runtime meshes.
- Shoe fit anchors, heel/sole/collar metadata, authored transforms, LODs, material optimization, and per-SKU calibration.
- Camera lifecycle, crop composition, persistent identity logic, smoothing, PnP, Three.js projection, rendering, compositing, capture, commerce UI, analytics, and SDK events.
- Dashboard entitlement and publishing workflows.
- Final browser/device integration, thermal testing, merchant iframe validation, accessibility, and staged rollout.

We will provide real capture-path clips, SDK transform fixtures, target-device measurements, and 3D sign-off inputs. Production release requires both teams' deliverables; the AI package alone does not complete the product.

## 12. Reply checklist

Please return:

1. Stage A path A or B, with final shape/version.
2. Meaning of current raw keypoint channels 1 and 2.
3. Agreement or proposed revision for every Stage B tensor.
4. Corrected canonical geometry and side rules.
5. Mask/depth ownership design.
6. Existing-shoe removal and tall-boot capability decision.
7. Dataset coverage and held-out split plan.
8. Model variant/export plan with delivery dates.
9. Named-device/browser benchmark plan.
10. Any acceptance gate you cannot meet, with measured evidence and a proposed revision.

Please do not send another model-only archive. We need the versioned contract, reproducible fixtures, hashes, browser compatibility report, and held-out metrics with the model files.
