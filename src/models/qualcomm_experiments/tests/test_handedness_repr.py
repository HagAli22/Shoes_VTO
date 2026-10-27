"""
Test Handedness Head Architecture:
Compares:
1) Flattened tokens: [B, 13 * 256] -> Linear / MLP
2) Geometric cross-product from keypoints:
   Medial vs Lateral cross with Heel-to-Toe axis.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP
from src.models.qualcomm_experiments.common.dataset import FootPoseDataset13KP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_ds = FootPoseDataset13KP(
    images_dir="data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid/images",
    img_size=256,
    mode="crop",
    augment=False,
    include_negatives=False
)

print(f"Loaded {len(val_ds)} validation crops.")

# Test geometric accuracy on GT keypoints:
# 2: heel_ground, 0: toe_ground, 3: ball_medial, 4: ball_lateral, 6: arch_medial, 7: midfoot_lateral
correct_geom = 0
total_geom = 0

for item in val_ds:
    cls_id = int(item["class_id"])
    if cls_id < 0:
        continue
    kps = item["keypoints"].numpy()  # [13, 3]
    # Heel to toe vector
    v_axis = kps[0, :2] - kps[2, :2]
    # Medial vector
    v_med = kps[3, :2] - kps[2, :2]
    # 2D cross product: vx * my - vy * mx
    cross = v_axis[0] * v_med[1] - v_axis[1] * v_med[0]

    # Lateral vector check
    v_lat = kps[4, :2] - kps[2, :2]
    cross_lat = v_axis[0] * v_lat[1] - v_axis[1] * v_lat[0]

    # Also arch medial vs midfoot lateral
    v_arch = kps[6, :2] - kps[2, :2]
    cross_arch = v_axis[0] * v_arch[1] - v_axis[1] * v_arch[0]

    vote = (cross > 0) + (cross_lat < 0) + (cross_arch > 0)
    pred_cls = 0 if vote >= 2 else 1  # 0: Left, 1: Right

    # Let's count alignment
    if pred_cls == cls_id:
        correct_geom += 1
    total_geom += 1

print(f"Geometric orientation on GT keypoints: {correct_geom}/{total_geom} = {correct_geom/total_geom*100:.2f}% (or inverse: {(total_geom-correct_geom)/total_geom*100:.2f}%)")

