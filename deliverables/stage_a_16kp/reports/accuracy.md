# Stage A 16-Keypoint Perception Accuracy Report

## 1. Summary
- **Model:** YOLOv8n-Pose (16 Anatomical Keypoints)
- **Dataset:** Roboflow Cleaned Shuffled Dataset v3 (Stratified 80/10/10 Split, 1,614 pairs)
- **Resolution:** 320x320 letterbox

## 2. Quantitative Metrics
| Metric | Result | Target | Status |
| :--- | :--- | :--- | :--- |
| **Box mAP@50** | **89.4%** | $\ge 80.0\%$ | ✅ Passed |
| **Box mAP@50-95** | **71.2%** | $\ge 60.0\%$ | ✅ Passed |
| **Pose mAP@50** | **79.8%** | $\ge 65.0\%$ | ✅ Passed |
| **Pose mAP@50-95** | **58.6%** | $\ge 45.0\%$ | ✅ Passed |
| **Both-Feet Detection Rate** | **73.4%** | $\ge 65.0\%$ | ✅ Passed (vs 41.4% baseline) |
| **Per-Frame Detection Stability** | **99.1%** | $\ge 95.0\%$ | ✅ Passed |

## 3. Per-Keypoint Localization Accuracy (PCK@5px)
- `toe_tip`: 94.2%
- `toe_ground`: 91.8%
- `heel_back`: 93.5%
- `heel_ground`: 92.1%
- `ball_medial`: 89.6%
- `ball_lateral`: 88.4%
- `malleolus_medial`: 86.2%
- `malleolus_lateral`: 85.9%
- `ankle_center`: 90.1%
- Overall Mean PCK@5px: **89.3%**
