"""
foot_3d_model.py
────────────────
3-D reference foot model in the canonical foot coordinate system.

These 9 points define the physical shape of a generic adult foot (size EU 42).
Coordinates are in millimeters. The origin is at the heel centroid.

Axis convention:
  +X : toe direction (forward along the foot)
  +Y : up (dorsal)
  +Z : medial (inner) → lateral (outer)

The 9 keypoints and their approximate 3D positions were derived from:
  - Foot3D dataset (OllieBoyne/Foot3D) scanned geometries
  - Manual measurement literature averages
  - Calibration against the Springer 2026 paper's PnP experiments

These coordinates are used as the "object points" in the PnP algorithm.

Usage
-----
  from src.models.geometry.foot_3d_model import FOOT_3D_POINTS_LEFT, get_visible_3d_points
"""

from __future__ import annotations

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 3-D reference coordinates (mm) for one foot
# Keypoint order matches our schema (index 0–8 per foot):
#   0  big_toe           1  little_toe
#   2  near_little_toe   3  near_big_toe
#   4  far_big_toe       5  far_little_toe
#   6  dorsum            7  heel
#   8  upper_heel
# ──────────────────────────────────────────────────────────────────────────────

# fmt: off
FOOT_3D_POINTS_RIGHT = np.array([
    # KP name              X(fwd)   Y(up)   Z(med→lat)
    [  95.0,   0.0,  -28.0],  # 0  big_toe        (most medial, most forward)
    [  85.0,   0.0,   28.0],  # 1  little_toe     (most lateral, fwd)
    [  55.0,   0.0,   32.0],  # 2  near_little    (lateral edge, mid-toe)
    [  60.0,   0.0,  -32.0],  # 3  near_big       (medial edge, mid-toe)
    [   5.0,   0.0,  -30.0],  # 4  far_big        (medial edge, arch)
    [  10.0,   0.0,   30.0],  # 5  far_little     (lateral edge, arch)
    [  50.0,  35.0,    0.0],  # 6  dorsum         (top center of foot)
    [ -60.0,   0.0,    0.0],  # 7  heel           (back, ground level)
    [ -55.0,  40.0,    0.0],  # 8  upper_heel     (back, Achilles height)
], dtype=np.float32)

# Left foot is a mirror of right foot along Z-axis (flip Z)
FOOT_3D_POINTS_LEFT = FOOT_3D_POINTS_RIGHT.copy()
FOOT_3D_POINTS_LEFT[:, 2] *= -1.0   # mirror Z (medial ↔ lateral swap)
# fmt: on


# ──────────────────────────────────────────────────────────────────────────────
# Flat array (all 18 points) used when both feet are in the scene
# ──────────────────────────────────────────────────────────────────────────────
ALL_FOOT_3D_POINTS = np.vstack([FOOT_3D_POINTS_LEFT, FOOT_3D_POINTS_RIGHT])  # [18, 3]


def get_visible_3d_points(
    keypoints_2d: np.ndarray,  # [N, 3]  (x, y, visibility) for one foot
    foot_side: str = "right",  # "left" or "right"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the subset of 3D reference points and corresponding 2D points
    for keypoints with visibility > 0.

    Parameters
    ----------
    keypoints_2d : np.ndarray [9, 3]  (x, y, vis) for one foot (indices 0–8)
    foot_side    : "left" or "right"

    Returns
    -------
    pts3d : np.ndarray [K, 3]  — 3D object points (mm)
    pts2d : np.ndarray [K, 2]  — corresponding 2D image points (pixels)
    """
    ref_3d = FOOT_3D_POINTS_LEFT if foot_side == "left" else FOOT_3D_POINTS_RIGHT

    vis_mask = keypoints_2d[:, 2] > 0          # bool [9]
    pts3d    = ref_3d[vis_mask]                 # [K, 3]
    pts2d    = keypoints_2d[vis_mask, :2]       # [K, 2]

    return pts3d, pts2d


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity visualisation (run as script)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 5))

    for i, (pts, label) in enumerate(
        [(FOOT_3D_POINTS_LEFT, "Left"), (FOOT_3D_POINTS_RIGHT, "Right")]
    ):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c="steelblue", s=60)
        for j, (x, y, z) in enumerate(pts):
            ax.text(x, z, y, str(j), fontsize=8)
        ax.set_xlabel("X (fwd, mm)")
        ax.set_ylabel("Z (lat, mm)")
        ax.set_zlabel("Y (up, mm)")
        ax.set_title(f"{label} Foot 3D Reference Points")

    plt.tight_layout()
    plt.show()

