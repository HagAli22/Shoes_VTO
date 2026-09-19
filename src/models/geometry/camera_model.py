"""
camera_model.py
───────────────
Camera intrinsics helper for the PnP 6DoF solver.

Provides:
  - Default intrinsic matrices for common mobile cameras.
  - A utility to estimate intrinsics from image dimensions (when no calibration file).
  - Functions to load/save calibration from JSON.

The camera matrix K is:
    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

where (cx, cy) is the principal point (usually image center) and
fx, fy are the focal lengths in pixels.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Default / estimated intrinsics
# ──────────────────────────────────────────────────────────────────────────────

# HFOV ≈ 70° is a reasonable default for modern smartphone rear cameras.
DEFAULT_HFOV_DEG = 70.0


def estimate_camera_matrix(
    image_width: int,
    image_height: int,
    hfov_deg: float = DEFAULT_HFOV_DEG,
) -> np.ndarray:
    """
    Estimate a pinhole camera matrix from image dimensions and approximate HFOV.

    Parameters
    ----------
    image_width  : Width of the camera frame in pixels.
    image_height : Height of the camera frame in pixels.
    hfov_deg     : Horizontal field-of-view in degrees (default 70°).

    Returns
    -------
    K : np.ndarray shape (3, 3), dtype float64 — camera intrinsic matrix.
    """
    fx = (image_width / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    fy = fx  # assume square pixels
    cx = image_width  / 2.0
    cy = image_height / 2.0

    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return K


# Pre-computed matrices for common resolutions
KNOWN_MATRICES: dict[str, np.ndarray] = {
    "1080p":  estimate_camera_matrix(1920, 1080),
    "720p":   estimate_camera_matrix(1280, 720),
    "480p":   estimate_camera_matrix(640,  480),
    "iphone_12_wide": np.array([
        [1594.0,    0.0, 960.0],
        [   0.0, 1594.0, 540.0],
        [   0.0,    0.0,   1.0],
    ], dtype=np.float64),
}


# ──────────────────────────────────────────────────────────────────────────────
# CameraModel class
# ──────────────────────────────────────────────────────────────────────────────

class CameraModel:
    """
    Encapsulates camera intrinsics and distortion coefficients.

    Parameters
    ----------
    K            : 3×3 camera intrinsic matrix.
    dist_coeffs  : Distortion coefficients (default: zero — assume undistorted).
    image_size   : (width, height) of the camera frame.
    """

    def __init__(
        self,
        K: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.K           = K.astype(np.float64)
        self.dist_coeffs = (
            dist_coeffs.astype(np.float64)
            if dist_coeffs is not None
            else np.zeros((4, 1), dtype=np.float64)
        )
        self.image_size  = image_size  # (W, H)

    @classmethod
    def from_image_size(
        cls,
        width: int,
        height: int,
        hfov_deg: float = DEFAULT_HFOV_DEG,
    ) -> "CameraModel":
        """Create a CameraModel with estimated intrinsics from image size."""
        K = estimate_camera_matrix(width, height, hfov_deg)
        return cls(K, image_size=(width, height))

    @classmethod
    def from_json(cls, path: str) -> "CameraModel":
        """Load a CameraModel from a JSON calibration file."""
        with open(path) as f:
            data = json.load(f)
        K = np.array(data["K"], dtype=np.float64)
        dist = np.array(data.get("dist_coeffs", [0, 0, 0, 0]), dtype=np.float64)
        size = tuple(data.get("image_size", [None, None]))
        return cls(K, dist, image_size=size)

    def to_json(self, path: str) -> None:
        """Save this CameraModel to a JSON calibration file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "K":            self.K.tolist(),
            "dist_coeffs":  self.dist_coeffs.flatten().tolist(),
            "image_size":   list(self.image_size) if self.image_size else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Undistort 2D image points using the distortion model.

        Parameters
        ----------
        points : np.ndarray [N, 2] in pixel coordinates.

        Returns
        -------
        np.ndarray [N, 2] undistorted pixel coordinates.
        """
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        undist = cv2.undistortPoints(pts, self.K, self.dist_coeffs, P=self.K)
        return undist.reshape(-1, 2)

    def project_points(
        self,
        points_3d: np.ndarray,   # [N, 3]
        rvec: np.ndarray,         # [3, 1] or [3]
        tvec: np.ndarray,         # [3, 1] or [3]
    ) -> np.ndarray:
        """
        Project 3D object points to 2D image coordinates.

        Returns
        -------
        np.ndarray [N, 2] projected pixel coordinates.
        """
        proj, _ = cv2.projectPoints(
            points_3d.astype(np.float64),
            rvec.astype(np.float64),
            tvec.astype(np.float64),
            self.K,
            self.dist_coeffs,
        )
        return proj.reshape(-1, 2)

    def reprojection_error(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
    ) -> float:
        """
        Mean reprojection error (pixels).

        Parameters
        ----------
        points_3d : [N, 3]
        points_2d : [N, 2]
        rvec, tvec: from PnP solver
        """
        proj = self.project_points(points_3d, rvec, tvec)
        errors = np.linalg.norm(proj - points_2d, axis=1)
        return float(errors.mean())

    def __repr__(self) -> str:
        return (
            f"CameraModel(fx={self.fx:.1f}, fy={self.fy:.1f}, "
            f"cx={self.cx:.1f}, cy={self.cy:.1f})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def get_default_camera(frame_width: int = 1920, frame_height: int = 1080) -> CameraModel:
    """
    Return the default estimated CameraModel for a given frame resolution.
    Use this when no calibration file is available.
    """
    return CameraModel.from_image_size(frame_width, frame_height)

