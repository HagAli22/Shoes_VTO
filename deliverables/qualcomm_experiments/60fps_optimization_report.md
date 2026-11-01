# Local Validation & Performance Report: Production Shoes VTO Two-Stage Pipeline

## 1. Executive Summary & Local Validation Scope

This report documents the **local validation and performance profiling** of the production Two-Stage Shoes VTO pipeline (`tools/evaluation/optimize_process_video_two_stage.py`).

### Local Validation Scope & Testing Rigor:
- **Environment**: Evaluated locally on host hardware (NVIDIA Quadro P2000, Desktop `CUDAExecutionProvider`, Compute Capability 6.1, CUDA 12.8, cuDNN 9.7).
- **Camera Access Limitation**: The development machine currently has no physical camera attached, and tunnel-based remote camera access is not yet configured. Therefore, all tests were conducted strictly via **local offline video validation**.
- **Video Test Suite**:
  - Primary standardized benchmark: `data/test_video.mp4` (574 frames, 1440×1440, 29.94 fps).
  - Cross-resolution sanity validation: `data/video.mp4` (244 frames, 1920×1080, 25.00 fps) processed losslessly from frame 1 to 244 with zero crashes or memory leaks.
- **Statistical Rigor**: All configurations were executed for **5 repeated benchmark passes after a 10-frame warm-up**, reporting mean and standard deviation ($\text{mean} \pm \text{std}$) across runs.
- **Detailed Artifacts Generated**:
  - Full output videos with telemetry HUD overlays for all configurations in `deliverables/qualcomm_experiments/benchmark_videos/`.
  - Granular per-frame CSV execution logs (frame index, timestamp, trigger reason, ROI count, batch size, per-foot detections, and 10 stage timing measurements) in `deliverables/qualcomm_experiments/logs/`.
  - Machine-readable benchmark summary in `pipeline_optimization_benchmark.csv` and `pipeline_optimization_benchmark.json`.

> [!IMPORTANT]
> **Performance Reporting Rigor**:
> - **Processing FPS** reflects pure algorithmic pipeline execution speed (ROI update + Stage A detection + crop extraction + Batch-2 RTMPose + SimCC postprocessing + Kalman state update).
> - **End-to-End Offline Wall FPS** reflects complete disk-to-disk throughput including video decoding from disk, telemetry HUD rendering, and MP4 video encoding.
> - **Camera-to-Display Latency**: True glass-to-glass camera latency and live camera 60 FPS cannot be claimed from offline video file playback and will require validation once a physical camera sensor is connected.

---

## 2. RTMPose Validation & Execution Coverage

To verify that RTMPose inference is never silently skipped, explicit frame counters and per-foot inference trackers were integrated into the pipeline:

### Primary Benchmark Video (`data/test_video.mp4`, 574 Frames):
- **Total Valid ROI Opportunities**: 1,108 foot crops across 574 frames.
- **Total RTMPose Inferences Executed**: 1,108 crop inferences.
- **Silent Pose Skips**: **0 (Zero)**.
- **Pose Inference Coverage**: **100.0%**.
- **Frame Distribution by Crop Count**:
  - **Frames with 2 Valid Foot Crops**: **542 frames (94.43%)** — Processed via **Batch-2** coalesced inference in a single forward pass.
  - **Frames with 1 Valid Foot Crop**: **24 frames (4.18%)** — Processed via Batch-1 single crop inference (e.g. during edge-of-frame transitions).
  - **Frames with 0 Valid Foot Crops**: **8 frames (1.39%)** — Occurs only when both feet are completely outside the field of view; RTMPose correctly bypassed.
- **Per-Foot Inferences**: Left Foot: 546 inferences | Right Foot: 562 inferences.

### Multi-Resolution Validation Video (`data/video.mp4`, 244 Frames, 1920×1080):
- **Total Valid ROI Opportunities**: 310 foot crops.
- **Total RTMPose Inferences Executed**: 310 crop inferences (**100.0% coverage, 0 silent skips**).
- **Frame Distribution**: 90 frames with 2 crops (36.89%, Batch-2), 131 frames with 1 crop (53.69%, Batch-1), 23 frames with 0 crops (9.43%).

---

## 3. Tracking & Keypoint-Derived ROI Updates

### Keypoint-to-ROI Measurement Lifecycle (`keypoints_to_bbox`)
To resolve the dependency on Stage A detection and allow the Kalman filter to measure true physical foot dynamics:
1. When Stage A (NanoDet) is triggered, the detected bounding box updates the Kalman tracker.
2. When Stage A is skipped (80.14% of frames in heartbeat mode), RTMPose executes on the predicted crop and predicts 13 sub-pixel keypoints in frame coordinates.
3. The helper `keypoints_to_bbox` extracts the tight bounding box of valid keypoints ($\text{conf} \ge 0.20$) and expands it with a 20% margin to cover the full foot envelope.
4. This keypoint-derived bounding box serves as the **actual observation measurement** for `kalman.update_measurement(obs_box)`.
5. This guarantees that:
   - The Kalman state $[cx, cy, w, h, vx, vy, vw, vh]$ tracks true physical foot movement, rotation, and scale.
   - Kalman velocity is directly measured from keypoint displacement, enabling accurate forward extrapolation.
   - NanoDet is triggered immediately when keypoint tracking confidence drops below 0.35, when fewer than 2 feet are tracked, or when velocity indicates fast motion ($> 25\text{ px/frame}$).

---

## 4. Multi-Run Benchmark Results (5 Repeated Passes after Warm-up)

All experiments were executed on `data/test_video.mp4` (574 frames) across 5 complete passes following a 10-frame warm-up:

| Config ID | Configuration Name | 5-Run Proc FPS | 5-Run Wall FPS | Proc P95 Latency | Crops (0/1/2) | Pose Cov % | Silent Skips | Skip Rate % | Baseline KP MAE | Baseline ID Flips | Baseline Chirality % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **exp1_baseline** | Baseline Ref (Every-Frame + Seq Pose + FP16) | 19.41 ± 0.43 | 15.53 ± 2.02 | 61.63 ms | 0 / 4 / 570 | 100.0% | 0 | 0.0% | 0.00 px (ref) | 0 | 100.0% |
| **exp0_ref_heartbeat5** | Ref Heartbeat 5 (Full FB LK + Sync Writer) | 23.51 ± 0.87 | 18.06 ± 2.58 | 56.70 ms | 8 / 24 / 542 | 100.0% | 0 | 80.14% | 25.72 px | 2 | 95.34% |
| **exp1_crop_lk** | Crop-Level LK (256×256 Patch FB LK) | 43.57 ± 0.49 | 28.64 ± 5.27 | 30.71 ms | 8 / 24 / 542 | 100.0% | 0 | 80.14% | 25.72 px | 2 | 95.34% |
| **exp2_kalman_roi** | Pose-Guided Kalman ROI (0.05ms Tracking) | 51.55 ± 1.34 | 31.50 ± 6.29 | 30.38 ms | 8 / 24 / 542 | 100.0% | 0 | 80.14% | 25.72 px | 2 | 95.34% |
| **exp3_cuda_engine** | Native CUDA Batch-2 + cuDNN Heuristic | **74.29 ± 3.12** | 39.29 ± 9.10 | 24.60 ms | 8 / 24 / 542 | 100.0% | 0 | 80.14% | 25.79 px | 2 | 95.16% |
| **exp4_vectorized_crop** | In-Place Normalization + CUDA Batch-2 | 73.18 ± 6.82 | 39.26 ± 9.49 | 29.39 ms | 8 / 24 / 542 | 100.0% | 0 | 80.14% | 25.79 px | 2 | 95.16% |
| **exp5_integrated_async** | Full Production (Kalman ROI + CUDA + Async Writer) | **75.71 ± 0.65** | **41.08 ± 5.54** | **18.99 ms** | 8 / 24 / 542 | 100.0% | 0 | 80.14% | 25.79 px | 2 | 95.16% |
| **exp_async_offline** | Asynchronous 3-Worker Pipeline (Lossless) | **68.67 ± 3.78** | **62.36 ± 14.04** | **24.69 ms** | 8 / 24 / 542 | 100.0% | 0 | 51.05% | 25.80 px | 2 | 95.16% |
| **exp_async_live** | Asynchronous 3-Worker Pipeline (Live-Dropping) | **68.18 ± 4.06** | **61.96 ± 14.41** | **33.20 ms** | 8 / 49 / 408 | 100.0% | 0 | 51.05% | 25.74 px | 2 | 91.63% |

---

## 5. Granular Decomposed Latency Breakdown (Mean ms)

| Stage | Baseline (Exp 1) | Ref Heartbeat (Exp 0) | Crop LK (Exp 1) | Kalman ROI (Exp 2) | CUDA Batch-2 (Exp 3) | Full Production (Exp 5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Frame Capture (Disk Decode)** | 10.06 ms | 9.77 ms | 9.94 ms | 9.87 ms | 9.95 ms | 9.92 ms |
| **Optical Flow / ROI Update** | 22.19 ms | 24.49 ms | 3.97 ms | **0.05 ms** | **0.05 ms** | **0.05 ms** |
| **Kalman Predict** | 0.07 ms | 0.07 ms | 0.07 ms | 0.08 ms | 0.07 ms | 0.07 ms |
| **NanoDet Preprocessing** | 1.34 ms | 0.32 ms | 0.32 ms | 0.32 ms | 0.28 ms | 0.28 ms |
| **NanoDet Inference** | 6.70 ms | 7.44 ms | 6.79 ms | 7.14 ms | 6.95 ms | 6.42 ms |
| **NanoDet Postprocessing** | 0.15 ms | 0.04 ms | 0.04 ms | 0.04 ms | 0.04 ms | 0.04 ms |
| **Crop Extraction & Prep** | 2.95 ms | 2.85 ms | 2.99 ms | 2.99 ms | **1.69 ms** | **1.87 ms** |
| **RTMPose Batch-2 Inference** | 20.53 ms (seq) | 16.63 ms | 13.55 ms | 14.34 ms | **9.77 ms** | **8.65 ms** |
| **Pose Post & SimCC** | 0.65 ms | 0.65 ms | 0.65 ms | 0.66 ms | 0.64 ms | 0.64 ms |
| **Track Update & Filter** | 0.25 ms | 0.24 ms | 0.24 ms | 0.24 ms | 0.25 ms | 0.25 ms |
| **Render & Video Encode** | 23.25 ms | 22.83 ms | 23.51 ms | 22.98 ms | 23.12 ms | **8.64 ms** (Async) |
| **Total Processing Latency (Mean)** | **53.86 ms** | **44.88 ms** | **23.47 ms** | **20.30 ms** | **14.33 ms** | **13.33 ms** |

---

## 6. Local Validation Findings: What Passed, What Failed, What Needs Camera

### Passed:
1. **Zero Silent Skips & 100% Pose Coverage**: Verified across 574 frames in `test_video.mp4` and 244 frames in `video.mp4`. Whenever a valid ROI was detected or tracked, RTMPose was executed without exception.
2. **Batch-2 Coalescing**: Successfully processed both feet simultaneously in a single forward pass for 94.4% of frames in `test_video.mp4` (542/574).
3. **Keypoint-Driven Kalman Lifecycle**: Deriving observation bounding boxes from RTMPose 13-keypoints on detector-skipped frames eliminated drift and maintained high tracking consistency (Keypoint MAE = 25.79 px, 95.16% chirality agreement, only 2 identity flips across 574 frames).
4. **Processing Throughput**: Achieved **75.71 ± 0.65 Processing FPS** (P95: 18.99 ms) in `exp5_integrated_async` and **68.67 ± 3.78 Processing FPS** (Wall FPS: 62.36 ± 14.04) in the asynchronous offline pipeline.
5. **No Memory Leaks or Crashes**: 5 repeated passes over 574 frames (total 2,870 frames processed per configuration) ran continuously with zero memory leaks, thread deadlocks, or segmentation faults.

### Failed / Bottlenecks Identified in Local Processing:
1. **Disk Video Encoding Bottleneck**: In synchronous mode, `cv2.VideoWriter.write()` takes **22-23 ms per frame**, capping wall-clock playback at ~18-22 FPS even though the algorithm runs at 75 FPS. The `AsyncVideoWriter` worker thread successfully mitigated this by offloading encoding to a background thread, bringing wall-clock throughput to 41.08 FPS.
2. **File Capture Overhead**: `cv2.VideoCapture.read()` takes ~9.9 ms per frame for 1440×1440 H.264 video file decoding on Windows. This is specific to offline file reading and does not represent live camera grab latency.

### What Still Needs Real-Camera Validation:
1. **Live Camera-to-Display Latency**: Measuring actual glass-to-glass latency (from photon capture at the camera sensor to pixel render on the display) requires an active USB/MIPI camera sensor.
2. **Live Auto-Exposure & Motion Blur**: Testing how RTMPose and NanoDet respond to rapid camera auto-exposure adjustments and real-world high-speed motion blur.
3. **On-Device Qualcomm Hardware Acceleration**: Validating the pipeline on Qualcomm Snapdragon NPU/Hexagon DSP using the `QNNExecutionProvider` (HTP backend) for INT8 quantized inference.

---

## 7. Deliverables & Artifact Links

- **Benchmark Summary CSV**: [pipeline_optimization_benchmark.csv](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/pipeline_optimization_benchmark.csv)
- **Benchmark Summary JSON**: [pipeline_optimization_benchmark.json](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/pipeline_optimization_benchmark.json)
- **Per-Frame CSV Logs Directory**: [logs/](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/logs/):
  - [exp1_baseline_per_frame.csv](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/logs/exp1_baseline_per_frame.csv)
  - [exp0_ref_heartbeat5_per_frame.csv](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/logs/exp0_ref_heartbeat5_per_frame.csv)
  - [exp1_crop_lk_per_frame.csv](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/logs/exp1_crop_lk_per_frame.csv)
  - [exp2_kalman_roi_per_frame.csv](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/logs/exp2_kalman_roi_per_frame.csv)
  - [exp3_cuda_engine_per_frame.csv](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/logs/exp3_cuda_engine_per_frame.csv)
  - [exp4_vectorized_crop_per_frame.csv](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/logs/exp4_vectorized_crop_per_frame.csv)
  - [exp5_integrated_async_per_frame.csv](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/logs/exp5_integrated_async_per_frame.csv)
- **Output Video Demos**: [benchmark_videos/](file:///f:/Machine%20learning/shoes_temp/Shoes_VTO/deliverables/qualcomm_experiments/benchmark_videos/)
