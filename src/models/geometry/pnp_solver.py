"""
pnp_solver.py
─────────────
6-DoF foot pose estimation using the Perspective-n-Point (PnP) algorithm.

Implements both:
  - EPnP  (cv2.SOLVEPNP_EPNP)       O(n) — preferred, used when ≥ 6 visible points
  - IPnP  (cv2.SOLVEPNP_ITERATIVE)  Levenberg-Marquardt refinement — used as fallback

Pipeline
--------
  2D keypoints (x, y, vis)
      │
      ▼
  Filter visible points (vis > 0)
      │
      ▼
  Match to 3D reference foot model
      │
      ▼
  EPnP (or IPnP if < 6 pts)
      │
      ▼
  RANSAC outlier rejection (optional)
      │
      ▼
  Reprojection error validation
      │
      ▼
  (R rotation vec, T translation vec)

Based on:
  Nguyen et al., "A real-time mobile solution for shoe try-on using foot pose
  estimation and 3D processing techniques," Complex & Intelligent Systems, 2026.
  https://doi.org/10.1007/s40747-025-02188-x
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.models.geometry.foot_3d_model import get_visible_3d_points
from src.models.geometry.camera_model import CameraModel


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PnPResult:
    """Encapsulates the output of a PnP solve."""
    success:          bool
    rvec:             Optional[np.ndarray]  # [3, 1] rotation vector (Rodrigues)
    tvec:             Optional[np.ndarray]  # [3, 1] translation vector (mm)
    reprojection_err: float                 # mean reprojection error (pixels)
    method_used:      str                   # "epnp" | "ipnp" | "ransac"
    num_points:       int                   # number of 2D-3D correspondences used

    @property
    def R(self) -> Optional[np.ndarray]:
        """Rotation matrix [3, 3] from the rotation vector."""
        if self.rvec is None:
            return None
        R, _ = cv2.Rodrigues(self.rvec)
        return R

    @property
    def rotation_matrix(self) -> Optional[np.ndarray]:
        return self.R

    def __bool__(self) -> bool:
        return self.success


# ──────────────────────────────────────────────────────────────────────────────
# PnP Solver
# ──────────────────────────────────────────────────────────────────────────────

class FootPnPSolver:
    """
    6-DoF foot pose estimation from 2D keypoints and 3D reference model.

    Parameters
    ----------
    camera          : CameraModel with intrinsics (K, dist_coeffs).
    reprojection_threshold : Max acceptable mean reprojection error in pixels.
                             Results above this threshold are marked as failed.
    min_visible_points     : Minimum number of visible keypoints to attempt PnP.
    use_ransac             : If True, run RANSAC to handle outlier keypoints.
    ransac_threshold       : Inlier threshold for RANSAC (pixels).
    """

    def __init__(
        self,
        camera: CameraModel,
        reprojection_threshold: float = 8.0,
        min_visible_points: int = 4,
        use_ransac: bool = False,
        ransac_threshold: float = 8.0,
    ) -> None:
        self.camera                  = camera
        self.reprojection_threshold  = reprojection_threshold
        self.min_visible_points      = min_visible_points
        self.use_ransac              = use_ransac
        self.ransac_threshold        = ransac_threshold

    def solve(
        self,
        keypoints_2d: np.ndarray,   # [9, 3]  (x, y, visibility) for ONE foot
        foot_side: str = "right",   # "left" or "right"
    ) -> PnPResult:
        """
        Estimate the 6-DoF pose of one foot.

        Parameters
        ----------
        keypoints_2d : np.ndarray [9, 3]  — each row is (x_pixel, y_pixel, vis)
                       where vis: 0=not visible, 1=occluded, 2=visible.
        foot_side    : Which foot ("left" or "right").

        Returns
        -------
        PnPResult with rvec [3,1], tvec [3,1], and validity flag.
        """
        # 1. Get matching 3D ↔ 2D points (only visible ones)
        pts3d, pts2d = get_visible_3d_points(keypoints_2d, foot_side)

        if len(pts3d) < self.min_visible_points:
            return PnPResult(
                success=False, rvec=None, tvec=None,
                reprojection_err=float("inf"),
                method_used="none",
                num_points=len(pts3d),
            )

        pts3d_f64 = pts3d.astype(np.float64)
        pts2d_f64 = pts2d.reshape(-1, 1, 2).astype(np.float64)
        K         = self.camera.K
        dist      = self.camera.dist_coeffs

        # 2. Choose solve method
        if self.use_ransac:
            return self._solve_ransac(pts3d_f64, pts2d_f64, K, dist)

        if len(pts3d) >= 6:
            return self._solve_epnp(pts3d_f64, pts2d_f64, K, dist)
        else:
            return self._solve_ipnp(pts3d_f64, pts2d_f64, K, dist)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal solve methods
    # ──────────────────────────────────────────────────────────────────────────

    def _solve_epnp(
        self,
        pts3d: np.ndarray,
        pts2d: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> PnPResult:
        """EPnP — O(n) complexity, best for ≥ 6 points."""
        success, rvec, tvec = cv2.solvePnP(
            pts3d, pts2d, K, dist,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success:
            # Fall back to iterative if EPnP fails
            return self._solve_ipnp(pts3d, pts2d, K, dist)

        # Refine with Levenberg-Marquardt
        rvec, tvec = cv2.solvePnPRefineLM(pts3d, pts2d, K, dist, rvec, tvec)

        return self._build_result(rvec, tvec, pts3d, pts2d.reshape(-1, 2), K, dist, "epnp")

    def _solve_ipnp(
        self,
        pts3d: np.ndarray,
        pts2d: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> PnPResult:
        """Iterative PnP (Levenberg-Marquardt). Handles as few as 4 points."""
        success, rvec, tvec = cv2.solvePnP(
            pts3d, pts2d, K, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return PnPResult(
                success=False, rvec=None, tvec=None,
                reprojection_err=float("inf"),
                method_used="ipnp",
                num_points=len(pts3d),
            )
        return self._build_result(rvec, tvec, pts3d, pts2d.reshape(-1, 2), K, dist, "ipnp")

    def _solve_ransac(
        self,
        pts3d: np.ndarray,
        pts2d: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
    ) -> PnPResult:
        """RANSAC PnP — robust to outlier keypoints."""
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, K, dist,
            reprojectionError=self.ransac_threshold,
            confidence=0.99,
            iterationsCount=100,
        )
        if not success or inliers is None or len(inliers) < self.min_visible_points:
            return PnPResult(
                success=False, rvec=None, tvec=None,
                reprojection_err=float("inf"),
                method_used="ransac",
                num_points=len(pts3d),
            )

        # Refine on inliers only
        inlier_mask = inliers.flatten()
        pts3d_in    = pts3d[inlier_mask]
        pts2d_in    = pts2d.reshape(-1, 2)[inlier_mask]

        rvec, tvec = cv2.solvePnPRefineLM(
            pts3d_in, pts2d_in.reshape(-1, 1, 2), K, dist, rvec, tvec
        )
        return self._build_result(rvec, tvec, pts3d_in, pts2d_in, K, dist, "ransac")

    def _build_result(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        pts3d: np.ndarray,
        pts2d: np.ndarray,
        K: np.ndarray,
        dist: np.ndarray,
        method: str,
    ) -> PnPResult:
        """Compute reprojection error and package the result."""
        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
        proj     = proj.reshape(-1, 2)
        err      = float(np.linalg.norm(proj - pts2d, axis=1).mean())

        success  = err < self.reprojection_threshold
        return PnPResult(
            success=success,
            rvec=rvec,
            tvec=tvec,
            reprojection_err=err,
            method_used=method,
            num_points=len(pts3d),
        )

    def solve_both_feet(
        self,
        left_kps:  np.ndarray,   # [9, 3]  (x, y, vis) for left foot
        right_kps: np.ndarray,   # [9, 3]  (x, y, vis) for right foot
    ) -> Tuple[PnPResult, PnPResult]:
        """
        Estimate 6-DoF pose for both feet independently.

        Returns
        -------
        (left_result, right_result) : Tuple[PnPResult, PnPResult]
        """
        left_result  = self.solve(left_kps,  foot_side="left")
        right_result = self.solve(right_kps, foot_side="right")
        return left_result, right_result

