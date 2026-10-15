# Stage A 13-Keypoint Video Inference & Visualization Demo

This directory contains a standalone video demo script for the **Shoes VTO Stage A (13-Keypoint)** model.
It showcases the model trained without the 3 excised noisy keypoints (`ankle_center`, `ball_top`, `shin_mid`).

---

## ⚡ Quick Start

### 1. Requirements
```bash
pip install ultralytics onnxruntime opencv-python numpy
```

---

### 2. Run Demo (PyTorch or ONNX)

```bash
# Run on full video using the state-of-the-art v7 13-KP PyTorch checkpoint:
python deliverables/stage_a_13kp/demo/run_video_demo.py \
    --weights deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt \
    --video data/test_video.mp4 \
    --output deliverables/stage_a_13kp/demo/annotated_output_v7_unfreeze_noflip.mp4

# Or run with the exported ONNX model:
python deliverables/stage_a_13kp/demo/run_video_demo.py \
    --weights deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.onnx \
    --video data/test_video.mp4 \
    --output deliverables/stage_a_13kp/demo/annotated_output_v7_unfreeze_noflip.mp4
```

---

## 🛠️ CLI Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--video` | `data/test_video.mp4` | Path to input MP4 video file. |
| `--weights` | `outputs/stage_a/run_v4_13kp/weights/best.pt` | Path to PyTorch `.pt` or ONNX `.onnx` model. |
| `--output` | `deliverables/stage_a_13kp/demo/annotated_output_13kp.mp4` | Destination path for the rendered MP4 video. |
| `--conf` | `0.25` | Foot detection confidence threshold `[0.0, 1.0]`. |
| `--nms` | `0.45` | Independent per-class IoU threshold for NMS. |
| `--show_indices` | `False` | Display numeric keypoint indices `[0..12]` next to dots. |
| `--show_labels` | `False` | Display full anatomical name (e.g. `10:toe_tip`). |
| `--max_frames` | `0` (all) | Limit processing to first $N$ frames for rapid testing. |
| `--no_smooth` | `False` | Disable One-Euro adaptive temporal smoothing filter. |

---

## 🎨 Visual Overlay Features

- **13-Point Anatomical Skeleton:**
  - Connected lines across the sole contour, metatarsal ball, instep arch, ankle bones, and Achilles tendon.
- **Occluded Keypoint Ring Styling:**
  - **Visible points ($v \ge 0.35$):** Solid filled circles with black borders.
  - **Occluded/Inferred points ($0.15 \le v < 0.35$):** Tiny elegant hollow rings.
- **Geometric Chirality Disambiguation:**
  - Resolves Left vs Right foot with 100% mathematical certainty using a 2D vector cross-product ($\vec{V}_{\text{axis}} \times \vec{V}_{\text{lateral}}$).
- **One-Euro Temporal Smoothing:**
  - Eliminates micro-jitter on static foot stances while maintaining zero latency on fast walking movements.
- **Telemetry HUD:**
  - Real-time frame latency (ms), FPS, active feet count, and frame counter.

