"""
keypoint_model.py
─────────────────
RTMPose-Tiny ONNX inference wrapper for foot keypoint estimation.

Input:   BGR image crop of a single foot ROI (any size).
Output:  18 keypoints [(x_orig, y_orig, confidence)] in original image coordinates.

Pipeline
--------
  Input image (any size)
      │
      ▼
  Crop to foot bbox
      │
      ▼
  Resize → 256×192 (letterbox with padding)
      │
      ▼
  Normalise (ImageNet mean/std)
      │
      ▼
  RTMPose-Tiny ONNX forward pass
      │
      ▼
  Heatmap → argmax → keypoints in crop space
      │
      ▼
  Rescale back to original image coordinates
      │
      ▼
  [18 × (x, y, confidence)]
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

INPUT_H, INPUT_W   = 256, 192
HEATMAP_H, HEATMAP_W = 64, 48
NUM_KEYPOINTS      = 18

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

KEYPOINT_NAMES = [
    "left_big_toe", "left_little_toe", "left_near_little_toe",
    "left_near_big_toe", "left_far_big_toe", "left_far_little_toe",
    "left_dorsum", "left_heel", "left_upper_heel",
    "right_big_toe", "right_little_toe", "right_near_little_toe",
    "right_near_big_toe", "right_far_big_toe", "right_far_little_toe",
    "right_dorsum", "right_heel", "right_upper_heel",
]


# ──────────────────────────────────────────────────────────────────────────────
# Pre/post processing helpers
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(image_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize BGR image to (INPUT_H, INPUT_W) with zero padding, normalise.

    Returns
    -------
    blob  : np.ndarray [1, 3, INPUT_H, INPUT_W]  float32
    scale : float — resize scale applied (max of h/INPUT_H, w/INPUT_W)
    pad   : (pad_w_half, pad_h_half) — pixels of padding added each side
    """
    h, w   = image_bgr.shape[:2]
    scale  = min(INPUT_H / h, INPUT_W / w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2.resize(image_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)

    pad_h = (INPUT_H - nh) // 2
    pad_w = (INPUT_W - nw) // 2
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

    # RGB normalise
    img_rgb  = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_rgb - MEAN) / STD
    blob     = img_norm.transpose(2, 0, 1)[None]  # [1, 3, H, W]

    return blob.astype(np.float32), scale, (pad_w, pad_h)


def heatmaps_to_keypoints(
    heatmaps: np.ndarray,  # [18, 64, 48]
) -> np.ndarray:
    """
    Extract keypoint locations from heatmaps using soft-argmax.

    Applies 1-D soft-argmax (weighted average over each heatmap) for
    sub-pixel accuracy.

    Returns
    -------
    kps : np.ndarray [18, 3]  (x_hm, y_hm, confidence) in heatmap coordinates.
    """
    _, hm_h, hm_w = heatmaps.shape
    kps = np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

    for k in range(NUM_KEYPOINTS):
        hm  = heatmaps[k]                                    # [H, W]
        conf = float(hm.max())
        kps[k, 2] = conf

        if conf < 1e-4:
            continue

        # Soft-argmax: probability-weighted average of coordinates
        # Flatten, softmax, reshape and compute expected coordinate
        flat   = hm.flatten()
        exp_f  = np.exp(flat - flat.max())
        prob   = exp_f / (exp_f.sum() + 1e-9)

        xs     = np.arange(hm_w)
        ys     = np.arange(hm_h)
        prob_2d = prob.reshape(hm_h, hm_w)

        x_coord = float((prob_2d.sum(axis=0) * xs).sum())
        y_coord = float((prob_2d.sum(axis=1) * ys).sum())

        kps[k, 0] = x_coord
        kps[k, 1] = y_coord

    return kps


def kps_heatmap_to_image(
    kps_hm:  np.ndarray,         # [18, 3]  in heatmap space
    bbox:    Tuple[float, ...],  # (x, y, w, h) in original image
    scale:   float,
    pad:     Tuple[int, int],    # (pad_w, pad_h)
) -> np.ndarray:
    """
    Rescale keypoints from heatmap space back to original image pixel coordinates.

    Returns
    -------
    kps_img : np.ndarray [18, 3]  (x_img, y_img, confidence)
    """
    bx, by, bw, bh = bbox
    hm_to_input_x  = INPUT_W / HEATMAP_W
    hm_to_input_y  = INPUT_H / HEATMAP_H

    kps_img = kps_hm.copy()
    for k in range(NUM_KEYPOINTS):
        # heatmap → input (256×192) coordinates
        x_inp = kps_hm[k, 0] * hm_to_input_x
        y_inp = kps_hm[k, 1] * hm_to_input_y

        # Remove padding
        x_inp -= pad[0]
        y_inp -= pad[1]

        # Unscale to bbox-local
        x_local = x_inp / scale
        y_local = y_inp / scale

        # Translate to original image space
        kps_img[k, 0] = bx + x_local
        kps_img[k, 1] = by + y_local

    return kps_img


# ──────────────────────────────────────────────────────────────────────────────
# FootKeypointEstimator
# ──────────────────────────────────────────────────────────────────────────────

class FootKeypointEstimator:
    """
    RTMPose-Tiny ONNX inference wrapper for foot 2D keypoint estimation.

    Parameters
    ----------
    model_path : Path to the RTMPose-Tiny ONNX file (18 KP output).
    conf_threshold : Minimum heatmap confidence to accept a keypoint.
    providers      : ONNX Runtime execution providers.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.3,
        providers: Optional[List[str]] = None,
    ) -> None:
        import onnxruntime as ort

        self.conf_threshold = conf_threshold
        ep = providers or ["CPUExecutionProvider"]
        self._session      = ort.InferenceSession(model_path, providers=ep)
        self._input_name   = self._session.get_inputs()[0].name
        self._output_name  = self._session.get_outputs()[0].name

    def predict(
        self,
        frame: np.ndarray,               # BGR full frame
        bbox:  Tuple[float, float, float, float],  # (x, y, w, h) in frame coords
    ) -> np.ndarray:
        """
        Estimate 18 foot keypoints from a detected foot region.

        Parameters
        ----------
        frame : np.ndarray [H, W, 3]  BGR — full camera frame.
        bbox  : (x, y, w, h) bounding box of the foot in frame coordinates.

        Returns
        -------
        np.ndarray [18, 3]  (x_pixel, y_pixel, confidence)
        """
        bx, by, bw, bh = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # Clamp and crop
        x1 = max(0, bx);   y1 = max(0, by)
        x2 = min(w, bx + bw); y2 = min(h, by + bh)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return np.zeros((NUM_KEYPOINTS, 3), dtype=np.float32)

        # Pre-process
        blob, scale, pad = preprocess(crop)

        # Inference
        raw = self._session.run(
            [self._output_name], {self._input_name: blob}
        )[0]  # [1, 18, 64, 48]

        heatmaps = raw[0]  # [18, 64, 48]

        # Decode heatmaps
        kps_hm  = heatmaps_to_keypoints(heatmaps)

        # Rescale to original image coordinates
        # bbox offset: the crop started at (x1, y1), not (bx, by)
        effective_bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
        kps_img = kps_heatmap_to_image(kps_hm, effective_bbox, scale, pad)

        # Apply confidence threshold
        kps_img[kps_img[:, 2] < self.conf_threshold, 2] = 0.0

        return kps_img  # [18, 3]

    def split_feet(
        self,
        kps: np.ndarray,  # [18, 3]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the 18-KP output into left foot (indices 0–8) and right foot (9–17).

        Returns
        -------
        (left_kps, right_kps) : each np.ndarray [9, 3]
        """
        return kps[:9], kps[9:]

    def benchmark(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        n_runs: int = 100,
    ) -> Dict:
        """Latency benchmark."""
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.predict(frame, bbox)
            latencies.append((time.perf_counter() - t0) * 1000)

        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms":  float(np.std(latencies)),
            "min_ms":  float(np.min(latencies)),
            "max_ms":  float(np.max(latencies)),
            "fps":     float(1000 / np.mean(latencies)),
        }

