# Stage A 16-KP Video Inference & Visualization Demo

This directory contains a standalone Python script to run the **Shoes VTO Stage A (16-Keypoint)** ONNX model on any video file and generate a high-quality annotated MP4 video.

---

## ⚡ Quick Start

### 1. Requirements

Install the minimal required packages:

```bash
pip install onnxruntime opencv-python numpy
```

*(If you have an NVIDIA GPU with CUDA/cuDNN configured, you can install `onnxruntime-gpu` for accelerated inference).*

---

### 2. Run Demo on the Included Video

From the `deliverables/stage_a_16kp/` directory:

```bash
python demo/run_video_demo.py
```

This will automatically load `test_video.mp4` and `models/stage-a-320-16kp-fp32.onnx`, process all frames, and save `annotated_output.mp4`.

---

## 🛠️ CLI Options

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--video` | `test_video.mp4` | Path to input MP4 video file. |
| `--model` | `models/stage-a-320-16kp-fp32.onnx` | Path to Stage A ONNX model (`fp32`, `fp16`, `int8`, etc.). |
| `--output` | `annotated_output.mp4` | Destination path for the rendered output video. |
| `--conf` | `0.25` | Foot detection confidence threshold `[0.0, 1.0]`. |
| `--nms` | `0.45` | Independent per-class IoU threshold for Non-Maximum Suppression. |
| `--kpt-thresh` | `0.30` | Minimum visibility confidence to display a keypoint. |
| `--show-indices` | `False` | Render numeric index `[0..15]` next to each keypoint dot. |
| `--show-labels` | `False` | Render full anatomical name (e.g., `11:toe_tip`) next to each keypoint. |
| `--max-frames` | `0` (all) | Limit processing to first $N$ frames (useful for quick testing). |

---

## 💡 Example Commands

```bash
# 1. Quick test on first 100 frames:
python demo/run_video_demo.py --max-frames 100

# 2. Display keypoint index numbers (0 to 15) on video:
python demo/run_video_demo.py --show-indices

# 3. Display full anatomical names on video:
python demo/run_video_demo.py --show-labels

# 4. Test FP16 WebGPU/GPU model variant:
python demo/run_video_demo.py --model models/stage-a-320-16kp-fp16.onnx --output output_fp16.mp4

# 5. Run with customized confidence thresholds:
python demo/run_video_demo.py --conf 0.35 --kpt-thresh 0.40
```

---

## 🎨 Visual Overlay Features

- **Per-Class Color Scheme:**
  - **Left Foot:** Sky Blue / Cyan bounding box (`#00BFFF`) with Yellow keypoints.
  - **Right Foot:** Hot Pink / Magenta bounding box (`#FF69B4`) with Bright Green keypoints.
- **Anatomical Skeleton Connections:** Connected anatomical lines along the sole contour, instep arch, ankle center, and tibia axis.
- **Top Telemetry HUD:** Real-time FPS, inference latency per frame (ms), active frame index, and total detected feet.
- **Per-Class Independent NMS:** Zero cross-suppression, ensuring both feet remain continuously tracked even during crossings and overlaps.
