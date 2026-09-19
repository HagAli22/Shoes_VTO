# 🏁 Final Verification Report — Shoes_VTO_Deliverables_v1

- **Date**: 2026-09-17 23:04:18
- **Evaluated Model**: `Shoes_VTO_Deliverables_v1/models/stage_a_16kp_fp32.onnx`
- **Video Tested**: `data/test_video.mp4` (1440×1440, 515 frames)

---

## 1. Executive Summary

| Test Criteria | Measured Result | Specification Target | Verdict |
| :--- | :--- | :--- | :--- |
| **Model Verification** | ONNX Opset 12, Static `[1,3,320,320]` | Contract §2 & §5 | ✅ **PASS** |
| **Detection Coverage Rate** | **98.8%** (509/515 frames) | ≥ 70% | 🌟 **PASS** |
| **Median Inference Latency** | **16.78 ms** | ≤ 18 ms budget | ✅ **FAST** |
| **Mean Real-Time FPS** | **55.2 FPS** | ≥ 30 FPS | ✅ **REAL-TIME** |
| **Duplicate Overlapping Boxes** | **0 frames (0.0%)** | 0 duplicate boxes | ✅ **CLEAN** |
| **Anatomical Mapping Alignment** | Verified 16/16 Points | Contract §6.1 | ✅ **MATCHED** |

---

## 2. Deliverables Artifacts Tested

1. **HD Annotated Video**: [`final_annotated_video.mp4`](file:///F:/Machine learning/Shoes_VTO/final_deliverables_test_results/final_annotated_video.mp4)
2. **Temporal Analytics Graph**: [`tracking_analytics_plot.png`](file:///F:/Machine learning/Shoes_VTO/final_deliverables_test_results/tracking_analytics_plot.png)
3. **Per-Frame Metrics CSV**: [`per_frame_predictions.csv`](file:///F:/Machine learning/Shoes_VTO/final_deliverables_test_results/per_frame_predictions.csv)
4. **Key Snapshots**: Saved under [`snapshots/`](file:///F:/Machine learning/Shoes_VTO/final_deliverables_test_results/snapshots)
