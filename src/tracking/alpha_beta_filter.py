"""
alpha_beta_filter.py
────────────────────
Temporal stabilisation for real-time 6DoF foot pose estimation.

Two components:
  1. AlphaBetaFilter — smooths rotation (rvec) and translation (tvec) over frames.
  2. IOUGate         — detects when the foot is stationary and freezes R, T.

Based on:
  Nguyen et al., Complex & Intelligent Systems, 2026.
  Alpha-beta filter with IOU-based motion detection gate.
  IOU stability threshold = 0.96, comparison interval = 10 frames.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Alpha-Beta Filter
# ──────────────────────────────────────────────────────────────────────────────

class AlphaBetaFilter:
    """
    Alpha-Beta filter for smoothing 6DoF pose parameters (rvec, tvec).

    Implements the state-update equations:
        x_n  = x_(n-1) + v_(n-1) * dt + alpha * (z_n - x_pred_n)
        v_n  = v_(n-1)             + (beta / dt) * (z_n - x_pred_n)

    where:
        x_n   = filtered state (position of rvec or tvec component)
        v_n   = estimated velocity
        z_n   = raw measurement (network output)
        alpha = position smoothing factor  (0 = ignore measurement, 1 = raw)
        beta  = velocity smoothing factor  (smaller = smoother but laggier)

    Parameters
    ----------
    alpha : Position smoothing weight. Higher = faster, less smooth.
    beta  : Velocity smoothing weight. Higher = more responsive.
    dt    : Time step (default 1 frame = 1).
    dim   : Dimensionality of the state vector.
            rvec and tvec each have 3 components → dim=3 each filter.
    """

    def __init__(
        self,
        alpha: float = 0.85,
        beta:  float = 0.15,
        dt:    float = 1.0,
        dim:   int   = 3,
    ) -> None:
        self.alpha = alpha
        self.beta  = beta
        self.dt    = dt
        self.dim   = dim

        self._x: Optional[np.ndarray] = None   # filtered state   [dim]
        self._v: np.ndarray = np.zeros(dim)    # estimated velocity [dim]
        self._initialised = False

    def reset(self) -> None:
        """Reset filter state (call when tracking is lost or restarted)."""
        self._x = None
        self._v = np.zeros(self.dim)
        self._initialised = False

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Feed a new raw measurement and return the filtered state.

        Parameters
        ----------
        measurement : np.ndarray shape [dim] — raw rvec or tvec from PnP.

        Returns
        -------
        np.ndarray shape [dim] — filtered (smoothed) state.
        """
        z = measurement.flatten().astype(np.float64)

        if not self._initialised:
            # First measurement — initialise directly
            self._x = z.copy()
            self._v = np.zeros(self.dim, dtype=np.float64)
            self._initialised = True
            return self._x.copy()

        # Predict
        x_pred = self._x + self._v * self.dt

        # Residual (innovation)
        residual = z - x_pred

        # Update
        self._x = x_pred + self.alpha * residual
        self._v = self._v + (self.beta / self.dt) * residual

        return self._x.copy()

    def is_initialised(self) -> bool:
        return self._initialised


# ──────────────────────────────────────────────────────────────────────────────
# Pose stabiliser (wraps two AlphaBetaFilters — one for rvec, one for tvec)
# ──────────────────────────────────────────────────────────────────────────────

class PoseStabiliser:
    """
    Smooths both rotation vector (rvec) and translation vector (tvec)
    independently using separate Alpha-Beta filters.

    Parameters
    ----------
    alpha, beta : Filter weights (see AlphaBetaFilter).
    """

    def __init__(self, alpha: float = 0.85, beta: float = 0.15) -> None:
        self._r_filter = AlphaBetaFilter(alpha=alpha, beta=beta, dim=3)
        self._t_filter = AlphaBetaFilter(alpha=alpha, beta=beta, dim=3)

    def reset(self) -> None:
        self._r_filter.reset()
        self._t_filter.reset()

    def update(
        self,
        rvec: np.ndarray,  # [3, 1] or [3]
        tvec: np.ndarray,  # [3, 1] or [3]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (filtered_rvec, filtered_tvec) in [3, 1] shape.
        """
        r_filt = self._r_filter.update(rvec.flatten())
        t_filt = self._t_filter.update(tvec.flatten())
        return r_filt.reshape(3, 1), t_filt.reshape(3, 1)


# ──────────────────────────────────────────────────────────────────────────────
# IOU Gate — motion detection
# ──────────────────────────────────────────────────────────────────────────────

def compute_iou(
    box_a: np.ndarray,   # [4]  (x, y, w, h)
    box_b: np.ndarray,   # [4]  (x, y, w, h)
) -> float:
    """Compute Intersection-over-Union of two axis-aligned bounding boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]

    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a  = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b  = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union   = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return float(inter / union)


class IOUGate:
    """
    Detects whether the tracked foot is stationary by comparing current bbox
    to a bbox *history_frames* frames in the past.

    If IOU ≥ stability_threshold → foot is stable → freeze R, T.

    Parameters
    ----------
    stability_threshold : IOU value above which foot is deemed stable (default 0.96).
    history_frames      : How many frames back to compare (default 10).
    """

    def __init__(
        self,
        stability_threshold: float = 0.96,
        history_frames: int = 10,
    ) -> None:
        self.threshold      = stability_threshold
        self.history_frames = history_frames
        self._history: Deque[np.ndarray] = deque(maxlen=history_frames + 1)

    def reset(self) -> None:
        self._history.clear()

    def update(self, bbox: np.ndarray) -> bool:
        """
        Add current bbox to history and decide if foot is stable.

        Parameters
        ----------
        bbox : np.ndarray [4]  (x, y, w, h) — foot bounding box this frame.

        Returns
        -------
        bool : True if foot is stable (IOU with past bbox ≥ threshold).
        """
        self._history.append(bbox.astype(np.float32))

        if len(self._history) <= self.history_frames:
            # Not enough history yet — treat as moving
            return False

        past_bbox  = self._history[0]
        iou        = compute_iou(bbox, past_bbox)
        return iou >= self.threshold


# ──────────────────────────────────────────────────────────────────────────────
# Combined per-foot tracker
# ──────────────────────────────────────────────────────────────────────────────

class FootTracker:
    """
    Combines PoseStabiliser + IOUGate for a single foot track.

    Usage
    -----
        tracker = FootTracker()

        # Per-frame:
        is_stable   = tracker.iou_gate.update(bbox)
        if not is_stable:
            r_smooth, t_smooth = tracker.pose_stabiliser.update(rvec, tvec)
        # else: reuse previous r_smooth, t_smooth
    """

    def __init__(
        self,
        alpha: float = 0.85,
        beta:  float = 0.15,
        iou_threshold: float = 0.96,
        iou_history_frames: int = 10,
    ) -> None:
        self.pose_stabiliser = PoseStabiliser(alpha=alpha, beta=beta)
        self.iou_gate        = IOUGate(iou_threshold, iou_history_frames)

        self._last_rvec: Optional[np.ndarray] = None
        self._last_tvec: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.pose_stabiliser.reset()
        self.iou_gate.reset()
        self._last_rvec = None
        self._last_tvec = None

    def update(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        bbox: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Update the tracker with new PnP results and bbox.

        Returns
        -------
        (rvec_filtered, tvec_filtered, is_stable)
        """
        is_stable = self.iou_gate.update(bbox)

        if is_stable and self._last_rvec is not None:
            # Foot is not moving — return frozen pose
            return self._last_rvec, self._last_tvec, True

        # Foot is moving — smooth with alpha-beta filter
        r_filt, t_filt = self.pose_stabiliser.update(rvec, tvec)
        self._last_rvec = r_filt
        self._last_tvec = t_filt
        return r_filt, t_filt, False

