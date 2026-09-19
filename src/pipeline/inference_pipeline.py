"""
inference_pipeline.py
─────────────────────
Full AI Perception Layer — end-to-end inference pipeline.

Processes one camera frame and returns structured foot pose data:
  1. Foot Detection      (YOLOv8n ONNX)
  2. Keypoint Estimation (RTMPose-Tiny ONNX, 18 KP)
  3. 6DoF Estimation     (EPnP / IPnP via PnP solver)
  4. Temporal Smoothing  (Alpha-Beta filter + IOU gate)

Output dict per frame:
  {
    "left_foot": {
      "detected":    bool,
      "bbox":        [x, y, w, h],
      "keypoints":   np.ndarray [9, 3]   (x, y, confidence),
      "rvec":        np.ndarray [3, 1]   or None,
      "tvec":        np.ndarray [3, 1]   or None,
      "pose_valid":  bool,
      "stable":      bool,
      "reproj_err":  float,
    },
    "right_foot": { ... same ... },
    "fps":   float,
    "frame_id": int,
  }
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.models.detector.foot_detector     import FootDetector
from src.models.keypoint.keypoint_model    import FootKeypointEstimator
from src.models.geometry.pnp_solver        import FootPnPSolver
from src.models.geometry.camera_model      import CameraModel, get_default_camera
from src.tracking.alpha_beta_filter        import FootTracker


# ──────────────────────────────────────────────────────────────────────────────
# Per-foot result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FootResult:
    detected:   bool          = False
    bbox:       Optional[List[float]] = None
    keypoints:  Optional[np.ndarray]  = None   # [9, 3]
    rvec:       Optional[np.ndarray]  = None   # [3, 1]
    tvec:       Optional[np.ndarray]  = None   # [3, 1]
    pose_valid: bool          = False
    stable:     bool          = False
    reproj_err: float         = float("inf")

    def to_dict(self) -> Dict:
        return {
            "detected":   self.detected,
            "bbox":       self.bbox,
            "keypoints":  self.keypoints.tolist() if self.keypoints is not None else None,
            "rvec":       self.rvec.tolist()      if self.rvec      is not None else None,
            "tvec":       self.tvec.tolist()      if self.tvec      is not None else None,
            "pose_valid": self.pose_valid,
            "stable":     self.stable,
            "reproj_err": self.reproj_err,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    detector_model:     str   = "outputs/exports/foot_detector.onnx"
    keypoint_model:     str   = "outputs/exports/rtmpose_tiny_18kp.onnx"
    conf_threshold:     float = 0.40
    iou_threshold:      float = 0.45
    kp_conf_threshold:  float = 0.30
    reprojection_thresh:float = 8.0
    min_visible_pts:    int   = 4
    alpha:              float = 0.85
    beta:               float = 0.15
    iou_stability_thresh: float = 0.96
    iou_history_frames: int   = 10
    use_ransac:         bool  = False
    providers:          List[str] = field(default_factory=lambda: ["CPUExecutionProvider"])


# ──────────────────────────────────────────────────────────────────────────────
# ShoeVTOPipeline
# ──────────────────────────────────────────────────────────────────────────────

class ShoeVTOPipeline:
    """
    Full AI Perception Layer for the Shoe AR Try-On system.

    Parameters
    ----------
    config  : PipelineConfig with all model paths and hyperparameters.
    camera  : CameraModel for PnP. If None, estimated from first frame.

    Thread-safety: `process_frame()` is protected by a threading.Lock.
                   Safe to call from a camera-thread while main thread reads results.
    """

    def __init__(
        self,
        config: PipelineConfig,
        camera: Optional[CameraModel] = None,
    ) -> None:
        self.config = config
        self.camera = camera  # may be None until first frame

        # ── Load models ──────────────────────────────────────────────────────
        self.detector  = FootDetector(
            model_path=config.detector_model,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            providers=config.providers,
        )
        self.keypoint_estimator = FootKeypointEstimator(
            model_path=config.keypoint_model,
            conf_threshold=config.kp_conf_threshold,
            providers=config.providers,
        )

        # PnP solver — created after camera is known
        self._pnp_solver: Optional[FootPnPSolver] = None

        # ── Per-foot trackers ─────────────────────────────────────────────────
        self._trackers: Dict[str, FootTracker] = {
            "left":  FootTracker(config.alpha, config.beta,
                                 config.iou_stability_thresh, config.iou_history_frames),
            "right": FootTracker(config.alpha, config.beta,
                                 config.iou_stability_thresh, config.iou_history_frames),
        }

        # ── FPS counter ────────────────────────────────────────────────────────
        self._frame_id      = 0
        self._t_last        = time.perf_counter()
        self._fps_window    = []
        self._lock          = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Run the full AI Perception pipeline on one BGR frame.

        Parameters
        ----------
        frame : np.ndarray [H, W, 3] BGR camera frame.

        Returns
        -------
        dict with keys: "left_foot", "right_foot", "fps", "frame_id"
        """
        with self._lock:
            return self._process(frame)

    # ──────────────────────────────────────────────────────────────────────────

    def _process(self, frame: np.ndarray) -> Dict:
        self._frame_id += 1
        h, w = frame.shape[:2]

        # ── Initialise camera on first frame ──────────────────────────────────
        if self.camera is None:
            self.camera = get_default_camera(w, h)

        if self._pnp_solver is None:
            self._pnp_solver = FootPnPSolver(
                camera=self.camera,
                reprojection_threshold=self.config.reprojection_thresh,
                min_visible_points=self.config.min_visible_pts,
                use_ransac=self.config.use_ransac,
            )

        # ── Step 1: Foot detection ─────────────────────────────────────────────
        detections = self.detector.detect(frame)

        # ── Step 2: Assign left / right ───────────────────────────────────────
        assigned = self._assign_feet(detections, frame_width=w)

        # ── Step 3–4: Keypoints + PnP + Smoothing per foot ───────────────────
        results = {}
        for side in ["left", "right"]:
            det = assigned.get(side)
            results[side] = self._process_one_foot(frame, det, side)

        # ── FPS ────────────────────────────────────────────────────────────────
        now = time.perf_counter()
        dt  = now - self._t_last
        self._t_last = now
        self._fps_window.append(1.0 / max(dt, 1e-6))
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)
        fps = float(np.mean(self._fps_window))

        return {
            "left_foot":  results["left"].to_dict(),
            "right_foot": results["right"].to_dict(),
            "fps":        fps,
            "frame_id":   self._frame_id,
        }

    # ──────────────────────────────────────────────────────────────────────────

    def _assign_feet(
        self,
        detections: List[Dict],
        frame_width: int,
    ) -> Dict[str, Optional[Dict]]:
        """
        Simple heuristic: foot with lower centroid-x → left foot.
        Assigns at most one box per side.
        """
        if not detections:
            return {"left": None, "right": None}

        # Sort by centroid x
        sorted_dets = sorted(
            detections,
            key=lambda d: d["bbox"][0] + d["bbox"][2] / 2,
        )

        assigned: Dict[str, Optional[Dict]] = {"left": None, "right": None}

        if len(sorted_dets) == 1:
            cx = sorted_dets[0]["bbox"][0] + sorted_dets[0]["bbox"][2] / 2
            side = "left" if cx < frame_width / 2 else "right"
            assigned[side] = sorted_dets[0]
        else:
            # Use the two most-confident detections, assign leftmost → left
            top2 = sorted(sorted_dets, key=lambda d: d["confidence"], reverse=True)[:2]
            top2 = sorted(top2, key=lambda d: d["bbox"][0])
            assigned["left"]  = top2[0]
            assigned["right"] = top2[1]

        return assigned

    # ──────────────────────────────────────────────────────────────────────────

    def _process_one_foot(
        self,
        frame:    np.ndarray,
        det:      Optional[Dict],
        side:     str,             # "left" | "right"
    ) -> FootResult:
        """Process keypoints + PnP + smoothing for one foot."""
        result = FootResult()

        if det is None:
            self._trackers[side].reset()
            return result

        bbox = det["bbox"]  # [x, y, w, h]
        result.detected = True
        result.bbox     = bbox

        # Step 2: Keypoint estimation
        all_kps = self.keypoint_estimator.predict(frame, tuple(bbox))
        # Split into per-foot (each is [9, 3])
        left_kps, right_kps = self.keypoint_estimator.split_feet(all_kps)
        kps9 = left_kps if side == "left" else right_kps
        result.keypoints = kps9

        # Check if enough visible keypoints for PnP
        visible_count = int((kps9[:, 2] > 0).sum())
        if visible_count < self.config.min_visible_pts:
            return result

        # Step 3: PnP
        pnp_out = self._pnp_solver.solve(kps9, foot_side=side)
        result.reproj_err = pnp_out.reprojection_err

        if not pnp_out.success:
            return result

        # Step 4: Temporal smoothing + IOU gate
        bbox_arr = np.array(bbox, dtype=np.float32)
        r_filt, t_filt, is_stable = self._trackers[side].update(
            pnp_out.rvec, pnp_out.tvec, bbox_arr
        )

        result.rvec       = r_filt
        result.tvec       = t_filt
        result.pose_valid = True
        result.stable     = is_stable

        return result

    # ──────────────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset all trackers (call when switching subjects or restarting)."""
        for tracker in self._trackers.values():
            tracker.reset()
        self._frame_id   = 0
        self._fps_window = []

    # ──────────────────────────────────────────────────────────────────────────

    def run_webcam(self, camera_index: int = 0) -> None:
        """
        Quick-test: run pipeline on webcam and display FPS + keypoints overlay.
        Press 'q' to quit.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame)

            # Draw keypoints
            for side in ["left_foot", "right_foot"]:
                foot = result[side]
                if foot["keypoints"] is not None:
                    kps = np.array(foot["keypoints"])
                    for kp in kps:
                        x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                        if conf > 0:
                            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

            cv2.putText(
                frame, f"FPS: {result['fps']:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
            )
            cv2.imshow("Shoe VTO — AI Perception", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run AI Perception pipeline.")
    parser.add_argument("--detector",  required=True, help="Path to foot_detector.onnx")
    parser.add_argument("--keypoint",  required=True, help="Path to rtmpose_tiny_18kp.onnx")
    parser.add_argument("--camera",    type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    cfg = PipelineConfig(
        detector_model=args.detector,
        keypoint_model=args.keypoint,
    )
    pipeline = ShoeVTOPipeline(cfg)
    pipeline.run_webcam(args.camera)

