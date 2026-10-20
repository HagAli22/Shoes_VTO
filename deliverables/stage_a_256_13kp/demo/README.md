# Stage A 256 Mobile Video Inference & HUD Demo

This directory contains the standalone video demo runner for the **Shoes VTO Stage A 256 Mobile (13-Keypoint)** perception model.

---

## ⚡ Quick Start

### 1. Requirements
```bash
pip install ultralytics onnxruntime opencv-python numpy
```

### 2. Run Demo (ONNX or PyTorch)
```bash
# Run ONNX inference (default: stage-a-256-13kp-fp32.onnx):
python deliverables/stage_a_256_13kp/demo/run_video_demo.py \
    --weights deliverables/stage_a_256_13kp/models/stage-a-256-13kp-fp32.onnx \
    --video data/test_video.mp4 \
    --output deliverables/stage_a_256_13kp/demo/annotated_output_256.mp4

# Run with optional 15-Keypoint Expansion (derived ball_top and shin_mid):
python deliverables/stage_a_256_13kp/demo/run_video_demo.py \
    --weights deliverables/stage_a_256_13kp/models/stage-a-256-13kp-fp32.onnx \
    --video data/test_video.mp4 \
    --output deliverables/stage_a_256_13kp/demo/annotated_output_256_15kp_derived.mp4 \
    --expand_15kp
```

---

## 🛠️ CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--video` | `data/test_video.mp4` | Path to input MP4 video file |
| `--weights` | `models/stage-a-256-13kp-fp32.onnx` | Path to ONNX or PyTorch weights |
| `--output` | `demo/annotated_output_256.mp4` | Destination path for annotated MP4 video |
| `--conf` | `0.25` | Foot confidence threshold `[0.0, 1.0]` |
| `--nms` | `0.45` | NMS IoU threshold |
| `--show_indices` | `False` | Display numeric keypoint indices `[0..12]` |
| `--show_labels` | `False` | Display full anatomical landmark names |
| `--max_frames` | `0` (all) | Limit execution to first $N$ frames |
| `--no_smooth` | `False` | Disable One-Euro temporal smoothing |
| `--expand_15kp` | `False` | Deterministically derive `ball_top` and `shin_mid` |

---

## 📦 Delivered Video Benchmarks

This directory contains two pre-rendered, verified demonstration videos:
1. `annotated_output_256.mp4`: Native 13-Keypoint mobile perception with telemetry HUD.
2. `annotated_output_256_15kp_derived.mp4`: Viewpoint-invariant 15-Keypoint expansion demonstrating deterministic geometric derivation of `ball_top` and `shin_mid`.
