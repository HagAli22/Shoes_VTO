"""
foot_detector.py
────────────────
YOLOv8n-based foot bounding box detector — ONNX inference wrapper.

Responsibilities:
  - Accept a BGR frame.
  - Run YOLOv8n forward pass (ONNX via onnxruntime).
  - Return per-foot bounding boxes with confidence scores.
  - Post-process: NMS + confidence threshold.

The detector outputs a list of dicts:
  {
    "bbox":       [x, y, w, h],   # in original frame pixels
    "confidence": float,
    "foot_side":  "left" | "right" | "unknown",   # side label if model provides it
  }
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_CONF   = 0.40
_DEFAULT_IOU    = 0.45
_DEFAULT_SIZE   = 640


# ──────────────────────────────────────────────────────────────────────────────
# Helper: letterbox resize
# ──────────────────────────────────────────────────────────────────────────────

def letterbox(
    image: np.ndarray,
    new_shape: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with unchanged aspect ratio using padding.

    Returns
    -------
    image_lb : padded image [new_shape, new_shape, 3]
    scale    : scale factor applied to original image
    pad      : (pad_w, pad_h) — padding added (half-value each side)
    """
    h, w    = image.shape[:2]
    scale   = new_shape / max(h, w)
    nh, nw  = int(round(h * scale)), int(round(w * scale))
    image_r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h   = (new_shape - nh) // 2
    pad_w   = (new_shape - nw) // 2

    image_lb = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    image_lb[pad_h:pad_h + nh, pad_w:pad_w + nw] = image_r
    return image_lb, scale, (pad_w, pad_h)


def scale_boxes(
    boxes_xywh: np.ndarray,  # [N, 4] in detection-input space
    scale: float,
    pad: Tuple[int, int],    # (pad_w, pad_h)
) -> np.ndarray:
    """
    Rescale YOLO output boxes back to original image coordinates.

    Boxes are expected in (cx, cy, w, h) format.
    Returns (x1, y1, w_orig, h_orig) format.
    """
    boxes = boxes_xywh.copy().astype(np.float32)
    # Subtract padding and unscale
    boxes[:, 0] = (boxes[:, 0] - pad[0]) / scale  # cx
    boxes[:, 1] = (boxes[:, 1] - pad[1]) / scale  # cy
    boxes[:, 2] = boxes[:, 2] / scale              # w
    boxes[:, 3] = boxes[:, 3] / scale              # h
    # Convert cx,cy,w,h → x,y,w,h
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    return boxes


# ──────────────────────────────────────────────────────────────────────────────
# FootDetector
# ──────────────────────────────────────────────────────────────────────────────

class FootDetector:
    """
    YOLOv8n foot detector backed by ONNX Runtime.

    Parameters
    ----------
    model_path       : Path to the ONNX model file.
    conf_threshold   : Minimum confidence to keep a detection.
    iou_threshold    : NMS IoU threshold.
    input_size       : Square input size for the YOLO model (default 640).
    providers        : ONNX Runtime execution providers.
                       E.g. ["CUDAExecutionProvider", "CPUExecutionProvider"]
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = _DEFAULT_CONF,
        iou_threshold:  float = _DEFAULT_IOU,
        input_size:     int   = _DEFAULT_SIZE,
        providers: Optional[List[str]] = None,
    ) -> None:
        import onnxruntime as ort

        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.input_size     = input_size

        ep = providers or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(model_path, providers=ep)
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    # ──────────────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect feet in a BGR frame.

        Parameters
        ----------
        frame : np.ndarray [H, W, 3] BGR — camera frame.

        Returns
        -------
        List of dicts:
          {
            "bbox":       [x, y, w, h],   # original pixel space
            "confidence": float,
            "foot_side":  "left" | "right" | "unknown",
          }
        """
        # 1. Pre-process
        img_lb, scale, pad = letterbox(frame, self.input_size)
        img_rgb  = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_chw  = img_rgb.transpose(2, 0, 1)[None]  # [1, 3, H, W]

        # 2. Forward pass
        raw_output = self._session.run(
            [self._output_name], {self._input_name: img_chw}
        )[0]  # [1, 5+num_classes, num_anchors] for YOLOv8

        # 3. Post-process
        detections = self._postprocess(raw_output, scale, pad, frame.shape[:2])
        return detections

    # ──────────────────────────────────────────────────────────────────────────

    def _postprocess(
        self,
        raw:     np.ndarray,             # [1, 5 or (5+C), N]
        scale:   float,
        pad:     Tuple[int, int],
        orig_hw: Tuple[int, int],
    ) -> List[Dict]:
        """
        Decode YOLOv8 raw output to bounding boxes.

        YOLOv8 output shape: [1, 4 + num_classes, num_anchors]
        """
        pred = raw[0]  # [4+C, N]

        if pred.ndim == 1:
            return []

        if pred.shape[0] < 5:
            return []

        # Transpose to [N, 4+C]
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # [N, 4+C]

        # Box coords (cx, cy, w, h) + class scores
        boxes_cxcywh = pred[:, :4]        # [N, 4]
        scores_all   = pred[:, 4:]        # [N, C]

        # Class confidence
        class_confs  = scores_all.max(axis=1)   # [N]
        class_ids    = scores_all.argmax(axis=1) # [N]

        # Confidence filter
        mask = class_confs >= self.conf_threshold
        if not mask.any():
            return []

        boxes_cxcywh = boxes_cxcywh[mask]
        class_confs  = class_confs[mask]
        class_ids    = class_ids[mask]

        # Scale boxes to original image space
        boxes_xywh   = scale_boxes(boxes_cxcywh, scale, pad)

        # NMS
        boxes_xyxy   = np.column_stack([
            boxes_xywh[:, 0],
            boxes_xywh[:, 1],
            boxes_xywh[:, 0] + boxes_xywh[:, 2],
            boxes_xywh[:, 1] + boxes_xywh[:, 3],
        ])
        nms_indices  = cv2.dnn.NMSBoxes(
            boxes_xywh.tolist(),
            class_confs.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )
        if len(nms_indices) == 0:
            return []

        nms_indices = np.array(nms_indices).flatten()

        orig_h, orig_w = orig_hw
        results = []
        for i in nms_indices:
            x, y, w, h = boxes_xywh[i]
            # Clamp to image bounds
            x = float(max(0, x))
            y = float(max(0, y))
            w = float(min(w, orig_w - x))
            h = float(min(h, orig_h - y))

            results.append({
                "bbox":       [x, y, w, h],
                "confidence": float(class_confs[i]),
                "foot_side":  "unknown",   # upgraded to "left"/"right" by pipeline
            })

        return results

    # ──────────────────────────────────────────────────────────────────────────

    def benchmark(self, frame: np.ndarray, n_runs: int = 50) -> Dict:
        """Run N forward passes and return latency statistics."""
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.detect(frame)
            latencies.append((time.perf_counter() - t0) * 1000)

        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms":  float(np.std(latencies)),
            "min_ms":  float(np.min(latencies)),
            "max_ms":  float(np.max(latencies)),
            "fps":     float(1000 / np.mean(latencies)),
        }

