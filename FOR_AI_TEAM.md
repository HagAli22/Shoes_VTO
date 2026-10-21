# For the AI team — every model must run on WASM (CPU)

The new SDK runs in the shopper's browser, inside the brand's page. We do not choose the device. The rule below
comes from what browsers support today and from what we measured.

## The rule

| runtime        | status in the SDK  | why                                                                                                                                                                                                                       |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **WASM (CPU)** | **required**       | runs everywhere: every iPhone, every Android, every desktop browser. It is the version that ships.                                                                                                                        |
| **WebGPU**     | optional fast path | not on iOS before 26, not on Android before Chrome 121 / Android 12, off on many mid-range GPUs. Used when present, never relied on.                                                                                      |
| **WebGL**      | **not accepted**   | ONNX Runtime Web has deprecated its WebGL backend (v1.29, [release notes](https://newreleases.io/project/github/microsoft/onnxruntime/release/v1.29.0)); it is in maintenance mode with limited ops. Do not build for it. |

So: **do not send us WebGL versions. Send a WASM version of every model, plus the WebGPU one if you have it.**

What we measured (face landmarker, MediaPipe, Mac, 30 runs): CPU median 16.2 ms with a 12.8–13.5 ms spread; GPU
17.3 ms with a 12.9–25.3 ms spread. The GPU was no faster and its spikes break the 33 ms frame budget. Competitors
ships CPU-only inference (`tfjs-backend-wasm-simd`) for the same reason. A steady 20 ms beats a jumpy 12 ms.

## Your list, answered

| model                      | you sent               | we need                                                                                  |
| -------------------------- | ---------------------- | ---------------------------------------------------------------------------------------- |
| ring landmarks (MediaPipe) | WASM, WebGPU preferred | WASM is the shipped one. WebGPU as an extra file if you have it.                         |
| neck YOLO                  | WebGPU                 | **the lighter WASM version you offered** — this is the shipping one. WebGPU as an extra. |
| wrist YOLO                 | WebGPU                 | same as neck                                                                             |
| nails segmenter            | GPU ("a bit heavy")    | a WASM version that fits the budget below (smaller input, int8 or fp16, fewer channels)  |

"The previous 3 models can run in parallel" — in a browser they share one CPU budget per frame. Parallel is fine as
long as the sum fits the budget below.

## The budget (per model, on WASM, phone)

| item           | limit                                                                          |
| -------------- | ------------------------------------------------------------------------------ |
| inference time | ≤ 15 ms on a Pixel 6a / iPhone 12 class phone, measured                        |
| input size     | ≤ 256 px on the short side unless the task needs more                          |
| file size      | ≤ 5 MB per model (the shopper downloads it on 4G)                              |
| ops            | only ops that exist in ONNX Runtime Web WASM / MediaPipe tasks — no custom ops |
| precision      | fp16 or int8 where it does not hurt accuracy; say which                        |

One frame is 33 ms at 30 fps. The face model already spends ~15 ms of it. Any second model in the same feature
(e.g. hair + face) shares what is left.

## What to deliver per model

1. The model file (`.onnx` or `.task` / `.tflite`), WASM-runnable, plus the WebGPU one if different.
2. Input: size, colour order, normalisation, letterbox or stretch.
3. Output: tensor names, shapes, what each channel / index means, the decode (anchors, NMS threshold) if any.
4. A timing table: ms on WASM and on WebGPU, on one phone and one laptop, with the device names.
5. A 5-line JS or Python snippet that runs one image through it end to end, so we can check we decode it the same way.

## The old SDK's models, one by one

An `.onnx` file is not "a WebGPU version" or "a WASM version": the same file runs on both in ONNX Runtime Web, and
the old SDK already fell back to WASM. What changes between the two is **speed**. So the ask is not a new format —
it is a model light enough to meet the budget on WASM, with the timing to prove it.

| model                                                                                     | what it is                                  | runtime today             | ask                                                                                                        |
| ----------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `face-landmarker.task`, `hand-landmarker.task`                                            | Google MediaPipe                            | WASM                      | nothing — keep                                                                                             |
| `hair-segmenter.nova`, `selfie-multiclass.nova`                                           | Google models in the NOVA wrapper           | tfjs-tflite WASM          | nothing — keep                                                                                             |
| skin-analysis `*.nova` (15), `model-age-q.nova`, find-the-look, personality-finder        | your tflite classifiers, one shot per photo | tfjs-tflite WASM          | nothing — keep (not per frame)                                                                             |
| `nail-keypoint.nova` 1.2 MB                                                               | your tflite                                 | tfjs-tflite WASM          | the new M1 + M3 pipeline replaces it — send those as `.onnx` with timings                                  |
| `fingers-v17.onnx` 0.5 MB, `head-kp-regressor.onnx` 0.4 MB, `neck-sigma-head.onnx` 0.1 MB | small MLP heads (Gemm only)                 | ORT WASM                  | nothing — trivial on CPU                                                                                   |
| `ear-yolo-v8n.onnx` 12.7 MB, **640 × 640**, fp32                                          | YOLOv8n detector                            | ORT WebGPU, WASM fallback | **too heavy for WASM** (~8.7 GFLOPs). 256 px input, int8 or fp16, ≤ 5 MB, timed                            |
| `neck-256.onnx` 12.9 MB, 256 × 256, fp32                                                  | YOLOv8 pose (12 keypoints)                  | same                      | fp16/int8 for size (12.9 MB is 3× too big to download); timed on WASM; a 192 px variant if it misses 15 ms |
| `wrist-256.onnx` 12.5 MB, 256 × 256, fp32                                                 | YOLOv8 pose                                 | same                      | same as neck                                                                                               |
| `neck-416.onnx` 13.0 MB                                                                   | the 416 px neck variant                     | same                      | drop — the 256 one is the WASM path                                                                        |
| `stage-a-16kp-v2-fp16.onnx` 6.6 MB, 320 × 320                                             | YOLOv8 pose, shoes (16 keypoints)           | same                      | fp16 already; 256 px + int8 if it misses 15 ms on WASM; timed                                              |

All the YOLO files export opset 12 with standard ops only (Conv, Mul, Sigmoid, Concat, Reshape) — they will _load_ on
WASM today; the question is only the milliseconds. Neck, wrist and shoes move slowly, so those detectors may also run
every 2nd or 3rd frame with the face tracker between — say if your model tolerates that.

## How we test what you send

Every model is run in the SDK with a fake camera (a fixed clip) so the result is reproducible, and timed in the
browser, not in Python. If it misses the budget we send the numbers back with the device name.
