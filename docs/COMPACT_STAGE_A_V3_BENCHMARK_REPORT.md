# Compact Stage A (v3 Shuffled) Benchmark & Delivery Report

**Date:** September 22, 2026  
**Model Architecture:** Compact YOLOv8-Pico (`width_multiple=0.16`, 81 layers, 1.19M parameters, 5.1 GFLOPs)  
**Dataset:** `data/shuffled_v3/data.yaml` (1,109 total clean images: 880 Train / 110 Valid / 112 Test)  
**Training Pipeline:** Warm-start Knowledge Transfer from `run_shuffled_v3` teacher $\rightarrow$ 150-epoch retraining with balanced augmentations.

---

## 1. Executive Summary

The new **Compact Stage A (`compact_shuffled_v3`)** detector resolves previous pose degradation while staying strictly within mobile and web browser delivery budgets ($\le 5.0\text{ MB}$ FP32, $\le 2.5\text{ MB}$ FP16, $\le 1.5\text{ MB}$ INT8).

- **Pose mAP@50:** Increased from **$\sim 5\%$** (earlier compact runs) to **$62.0\%$**.
- **Box mAP@50:** Reached **$82.7\%$**.
- **Video Both-Feet Detection Rate:** Jumped from **$41.4\%$** (Standard) $\rightarrow$ **$71.3\%$** (Compact FP16).
- **Inference Speed:** Runs at **$78\text{--}85\text{ FPS}$** ($11.7\text{--}12.8\text{ ms}$ latency).

---

## 2. Model Size & Budget Verification

All models use the Path A output tensor contract `[1, 18, 2100]` with 4 coarse keypoints (`toe_tip`, `heel_back`, `ball_medial`, `ball_lateral`), bounding box coordinates, and left/right foot class probabilities.

| Deliverable Model | Precision | Target Platform | File Size | Budget Limit | Status | SHA-256 Hash |
|---|---|---|---|---|---|---|
| `stage-a-320-compact-fp32.onnx` | FP32 | Desktop / WebGL | **4.93 MB** | $\le 5.0\text{ MB}$ | **PASSED** | `69132b815ee2d5aace9f0e7d328b340faac92a494d017862f6aed22deb4a14c9` |
| `stage-a-320-compact-fp16.onnx` | FP16 | WebGPU / Mobile | **2.49 MB** | $\le 2.5\text{ MB}$ | **PASSED** | `e7dac77940c8f70fae6efbcc046a66dd31eb0eee1123a113a41463bc6583e79a` |
| `stage-a-320-compact-int8.onnx` | INT8 | WASM / Safari | **1.46 MB** | $\le 1.5\text{ MB}$ | **PASSED** | `94a2afa76f212c3832d6d483717f82c86e6c72844a34dba1c68b550256de7428` |

**Active Deliverable Paths:**
- `shoes-vto-ai-v2/models/`
- `deliverables/models/`
- `shoes-vto-ai-v2/model-contract.json` (synchronized)

---

## 3. Validation Set Metrics Comparison

Evaluated on the held-out validation set (`data/shuffled_v3/valid`, 110 images):

| Model | Checkpoint Size | Box Precision | Box Recall | Box mAP@50 | Pose Precision | Pose Recall | Pose mAP@50 |
|---|---|---|---|---|---|---|---|
| **Standard Teacher (`run_shuffled_v3`)** | 6.78 MB | 0.852 | 0.853 | **0.922** | 0.880 | 0.788 | **0.828** |
| **New Compact (`compact_shuffled_v3`)** | **2.62 MB** | 0.709 | 0.773 | **0.827** | 0.754 | 0.577 | **0.620** |
| Old Compact Baseline (`compact_run2`) | 2.62 MB | 0.434 | 0.352 | 0.434 | 0.120 | 0.080 | 0.051 |

---

## 4. Real-World Video Benchmark on `data/test_video.mp4` (515 Frames)

| Model Name | Precision | Model Size | Latency (ms) | Inference FPS | $\ge 1$ Foot Detected | Both Feet Detected | Output Annotated Video Path |
|---|---|---|---|---|---|---|---|
| **Stage A Compact 320 (New)** | **FP32** | **4.93 MB** | **11.74 ms** | **85.2 FPS** | **98.6%** | **71.1%** | `data/test_video_results/stage_a_320_compact_fp32.mp4` |
| **Stage A Compact 320 (New)** | **FP16** | **2.49 MB** | **12.83 ms** | **77.9 FPS** | **98.8%** | **71.3%** | `data/test_video_results/stage_a_320_compact_fp16.mp4` |
| **Stage A Compact 320 (New)** | **INT8** | **1.46 MB** | **22.63 ms** | **44.2 FPS** | **97.1%** | **65.6%** | `data/test_video_results/stage_a_320_compact_int8.mp4` |
| Stage A Standard 320 | FP32 | 13.23 MB | 17.30 ms | 57.8 FPS | 98.8% | 41.4% | `data/test_video_results/stage_a_320_standard_fp32.mp4` |
| Stage A Standard 320 | FP16 | 6.64 MB | 18.57 ms | 53.8 FPS | 98.8% | 41.4% | `data/test_video_results/stage_a_320_standard_fp16.mp4` |
| Stage A Standard 320 | INT8 | 3.55 MB | 30.34 ms | 33.0 FPS | 98.8% | 40.4% | `data/test_video_results/stage_a_320_standard_int8.mp4` |

---

## 5. Generated Video Artifacts Locations

All annotated evaluation videos containing live latency counters, detection boxes, and skeleton overlays are located on disk at:

1. **New Compact FP32 Video:**
   `F:\Machine learning\Shoes_VTO\data\test_video_results\stage_a_320_compact_fp32.mp4`
2. **New Compact FP16 Video:**
   `F:\Machine learning\Shoes_VTO\data\test_video_results\stage_a_320_compact_fp16.mp4`
3. **New Compact INT8 Video:**
   `F:\Machine learning\Shoes_VTO\data\test_video_results\stage_a_320_compact_int8.mp4`
4. **Benchmark Summary JSON:**
   `F:\Machine learning\Shoes_VTO\data\test_video_results\benchmark_summary.json`

---

## 6. Next Optimization Pathways for Smaller & Stronger Models

1. **Soft Knowledge Distillation (KD):**
   Train the compact student using full probability distribution outputs from the teacher model alongside ground-truth labels to boost mAP by an additional $+3\%\text{--}6\%$.
2. **GhostNet / RepVGG Backbone:**
   Incorporate re-parameterized linear blocks for inference-time structural fusion, reducing computation by $35\%$.
3. **Structured Channel Pruning:**
   Prune $20\%$ of low-magnitude filters from `compact_shuffled_v3` and perform a 20-epoch recovery fine-tune to bring FP32 footprint under $\sim 3.8\text{ MB}$.
4. **Multi-Scale Augmentation (320px $\rightarrow$ 384px fine-tune):**
   Enhance detection robustness on small distant feet during extreme perspective angles.

