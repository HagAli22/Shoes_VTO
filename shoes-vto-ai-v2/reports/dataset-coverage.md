# Dataset Coverage & Coordinate Integrity Report

## 1. Dataset Characteristics
- **Dataset Source:** Roboflow Foot Pose v3.
- **Split Strategy:** Strict 80/10/10 Stratified Random Shuffle (1,291 Train / 161 Val / 162 Test).
- **Coordinate Normalization:** Clamped strictly to $[0.0, 1.0]$.
- **Class Balance:**
  - `left_foot`: 49.6%
  - `right_foot`: 50.4%
- **Anatomical Diversity:** Barefoot, athletic sneakers, formal shoes, sandals, high socks, trousers/jeans cuff coverage.
