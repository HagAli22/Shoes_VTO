# Stage A v2 (16-keypoint) — integration feedback and requests

Reviewed and integrated 2026-09-27 by NGM against the delivery package `stage_a_16kp` (2.0.0-16kp, 2026-09-22).
Everything below was verified by running the delivered models on the delivered fixture frames and on the SDK's
shoes try-on; nothing here is inferred from the reports alone.

## What works and is now in production

- `stage-a-320-16kp-fp32.onnx` (sha `c6866cd0…`) and the real fp16 export (sha `001cf15e…`) run in the SDK as
  `public/ai-models/onnx/shoes/stage-a-16kp-v2-fp{32,16}.onnx` (fp16 on the WebGPU path, fp32 on wasm).
  The int8 exports are held back until they are measured on a phone (see §4).
- Both feet are detected on the fixture frames (v1 lost one foot or both on 4 of the 8). Detection is stable
  enough to track two feet through the whole `test_video.mp4`.
- The frozen 16-keypoint order (`model-contract.json` §5) is correct: index 11 lands on the big toe and
  indices 3/4 on the medial/lateral ball on every fixture frame we overlaid.
- The canonical templates' length/width inconsistencies from the v1 review are resolved (265 / 98 mm, sides
  mirror exactly).

## Defects found — please fix in the next drop

1. **`fixtures/expected-decoded.json` uses the old keypoint names.** It labels index 0 `toe_tip`, 1
   `toe_ground`, 2 `heel_back`, 3 `heel_ground`, … while `model-contract.json` and the audit report freeze
   index 0 = `toe_ground`, 1 = `heel_back`, 2 = `heel_ground`, 3 = `ball_medial`, 11 = `toe_tip`. On frame 0 the
   point the fixture calls `heel_ground` (832, 820) sits on the medial ball. Regenerate the fixture from the
   frozen table so the golden file cannot be integrated with the wrong mapping.
2. **`reference-decoder.js` does not parse.** `resolvedClassId` and `finalDetections` are each declared twice
   (`const` redeclaration → SyntaxError), and the two `resolvedClassId` lines carry opposite chirality signs.
   We kept the file as `reference-decoder.js.txt`. Ship one decoder that runs, with a single, tested sign.
3. **Chirality sign.** With image y down, `cross = axis × (ball_lateral − heel)`: on the fixture frames the
   person's left foot gives `cross < 0` and the right foot `cross > 0`. That is the second of the two
   contradictory lines in the decoder. The SDK uses the signed difference `cross(lateral) − cross(medial)`
   with `heel_ground`/`ankle_center` as a fallback rear anchor, because `heel_back` is invisible (0.04) from
   straight above on every fixture frame — the decoder's `heel_back`-only formula returns nothing there.
4. **Class head is unreliable top-down** (both feet classed `left_foot` at 0.61 / 0.53 on frame 0, and again
   on frame 30). The SDK ignores the class for the side and uses geometry; keep training the head, but do
   not rely on it in your own demo either.
5. **Canonical left/right files are mirrored.** `canonical-left.json` has `ball_medial` at +X with +Y up and
   +Z toe. In a right-handed frame (`coordinate_frame.system: right_handed`) +X is the person's LEFT side
   when facing +Z, so a medial ball at +X is a RIGHT foot. Confirmed empirically: fitting the detector's
   apparent-left feet with the file named "left" gives 75–80 px RMS, with the file named "right" 35–41 px
   (frames 0/60/250, fx = 0.9 × width). The SDK swaps the two files. Either relabel the files or change
   `coordinate_frame` to say what the frame really is.
6. **Ankle group vs template.** After the side fix the largest residuals are `throat`, `ankle_center` and
   `achilles` (85–100 px on 1440 px frames, i.e. a quarter of the foot length). Their canonical heights
   (68–95 mm) or the model's placement of them do not agree. The SDK down-weights them; please check what
   the annotation guide meant by these three points and either move the template or retrain.
7. **`dataset-coverage.md` contradicts itself:** §1 lists "athletic sneakers, formal shoes, sandals", §2 says
   barefoot and socks only; §4 says per-channel identity was NOT verified while `keypoint-order-audit.md`
   says it was. Make the reports agree with each other and with the contract.
8. **The two demo videos are the same file** (`VIDEO-2026-09-23-01-59-42.mp4` and `…02-07-28.mp4`,
   byte-identical). Send the second one.

## Requests for the next delivery

- **WebGPU timing.** `browser-performance.md` measured wasm only (82–113 ms per frame, 8–12 fps); WebGPU is
  "pending". Measure fp16 on an iPhone 12+ and a Snapdragon 865+ before calling the ≤12 ms target met.
- **Int8 on device.** `stage-a-320-16kp-int8.onnx` and `stage-a-256-int8.onnx` are dynamically quantised;
  give us wasm timings on a mid-range Android plus an accuracy delta vs fp32 on the test split.
- **Cross-scene fixtures.** All fixtures and the demo are one person, one carpet, one camera. Add clips from
  at least three other scenes (plain floor, outdoor, low light) with both feet, including the side-view and
  heel-first poses in the five model photos we were sent (`public/media/assets/model-photos/feet-models/`),
  which the current model handles poorly (its training data is top-down).
- **Socks and shoes.** Per your note the model is barefoot/thin-sock only. The try-on will mostly be used on
  people wearing shoes; a detector that finds a shod foot is the next milestone, as agreed.
- **Stage B / pose head.** The SDK now solves a 6-DoF pose from the 16 points with the canonical template
  (see `src/components/vto-tech/shoes/FootPose.ts`). A trained rotation or pose head would replace the
  priors that solve the near-planar ambiguity today.
