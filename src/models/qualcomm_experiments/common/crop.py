"""
crop.py
───────
Shared crop geometry, affine transformation, and coordinate projection functions.
Ensures 100% mathematical parity between training augmentation and inference pipelines.

Key Features:
  - compute_crop_box: Adaptive aspect-ratio padding (default padding = 0.25)
  - get_crop_affine_transform: 2x3 affine matrix & exact inverse matrix
  - warp_image_crop: Constant (114, 114, 114) border (zero ghost reflection)
  - transform_keypoints: Forward projection into crop space with boundary checks
  - invert_keypoints: Inverse projection back to full image/video space
  - box_from_kps: MediaPipe-style bounding box estimation from keypoints via calibrated K-ratio
  - calibrate_keypoint_box_ratio: Computes empirical median(size / extent) on training data
"""

from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np


def compute_crop_box(
    cx: float,
    cy: float,
    bw: float,
    bh: float,
    crop_padding: float = 0.25
) -> Tuple[float, float, float]:
    """
    Computes square crop bounding box center and size with adaptive padding.
    Elongated feet (high aspect ratio) receive an additional 0.15 margin to prevent
    toe or heel clipping during rapid rotation.

    Args:
        cx: Box center X (pixels)
        cy: Box center Y (pixels)
        bw: Box width (pixels)
        bh: Box height (pixels)
        crop_padding: Base margin expansion ratio (default 0.25)

    Returns:
        (cx, cy, size): Center coordinates and square crop side length in pixels
    """
    bw_safe = max(1.0, float(bw))
    bh_safe = max(1.0, float(bh))
    ar = max(bw_safe / bh_safe, bh_safe / bw_safe)
    extra_margin = 0.15 if ar > 1.3 else 0.05
    size = max(bw_safe, bh_safe) * (1.0 + crop_padding + extra_margin)
    return float(cx), float(cy), float(size)


def get_crop_affine_transform(
    cx: float,
    cy: float,
    size: float,
    out_size: int = 256,
    rot: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2x3 affine transformation matrix M and its inverse M_inv.
    Maps (cx, cy) to the center of the output crop (out_size/2, out_size/2).

    Args:
        cx: Center X in source coordinates
        cy: Center Y in source coordinates
        size: Source crop size (side length)
        out_size: Target square image dimension (e.g., 256)
        rot: Rotation angle in degrees (clockwise)

    Returns:
        (M, M_inv): 2x3 float64 affine matrices
    """
    size_safe = max(1.0, float(size))
    scale = float(out_size) / size_safe
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(rot), scale)
    M[0, 2] += (out_size / 2.0) - float(cx)
    M[1, 2] += (out_size / 2.0) - float(cy)
    M_inv = cv2.invertAffineTransform(M)
    return M, M_inv


def warp_image_crop(
    img: np.ndarray,
    M: np.ndarray,
    out_size: int = 256,
    border_value: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """
    Warps source image into standard square crop using affine matrix M.
    Uses constant gray border to prevent reflection mirror artifacts.
    """
    return cv2.warpAffine(
        img,
        M,
        (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )


def transform_keypoints(
    kps: np.ndarray,
    M: np.ndarray,
    out_size: int = 256,
    invalidate_out_of_bounds: bool = True
) -> np.ndarray:
    """
    Applies forward affine transform M to keypoints [N, 2] or [N, 3].
    If input is [N, 3] and invalidate_out_of_bounds is True, marks points
    outside [0, out_size) as invisible (v = 0.0).

    Args:
        kps: Array of keypoints [N, 2] or [N, 3] in source pixel coordinates
        M: 2x3 affine matrix
        out_size: Crop resolution
        invalidate_out_of_bounds: Whether to zero visibility for out-of-bounds points

    Returns:
        Transformed keypoints with matching shape [N, 2] or [N, 3]
    """
    kps_out = np.array(kps, dtype=np.float32).copy()
    n_kps = len(kps_out)
    if n_kps == 0:
        return kps_out

    # Homogeneous coordinate multiplication: [N, 3] @ M.T -> [N, 2]
    homo = np.ones((n_kps, 3), dtype=np.float64)
    homo[:, :2] = kps_out[:, :2]
    transformed_xy = homo @ M.T

    kps_out[:, :2] = transformed_xy

    if invalidate_out_of_bounds and kps_out.shape[1] >= 3:
        out = (
            (kps_out[:, 0] < 0) |
            (kps_out[:, 0] >= out_size) |
            (kps_out[:, 1] < 0) |
            (kps_out[:, 1] >= out_size)
        )
        kps_out[out, 2] = 0.0

    return kps_out


def invert_keypoints(
    kps: np.ndarray,
    M_inv: np.ndarray
) -> np.ndarray:
    """
    Projects predicted keypoints from crop space back to full image/video frame space
    using the exact inverse affine matrix M_inv.

    Args:
        kps: Predicted keypoints [N, 2] or [N, 3] in crop coordinates [0, out_size)
        M_inv: 2x3 inverse affine matrix

    Returns:
        Keypoints projected to original image space
    """
    kps_out = np.array(kps, dtype=np.float32).copy()
    n_kps = len(kps_out)
    if n_kps == 0:
        return kps_out

    homo = np.ones((n_kps, 3), dtype=np.float64)
    homo[:, :2] = kps_out[:, :2]
    orig_xy = homo @ M_inv.T
    kps_out[:, :2] = orig_xy

    return kps_out


def box_from_kps(
    kps: np.ndarray,
    K: float = 1.594,
    conf_thresh: float = 0.30,
    min_kps: int = 6,
    prev_size: Optional[float] = None,
    max_shrink_ratio: float = 0.15,
    size_ema_alpha: float = 0.70,
    min_size_px: float = 30.0
) -> Optional[Tuple[float, float, float]]:
    """
    Derives square crop box (cx, cy, size) from keypoint extents for MediaPipe-style tracking.
    When confidence drops, too few keypoints are visible, or box collapses below min_size_px,
    returns None to trigger re-detection.
    
    Includes tracking feedback safeguards:
      - Limits per-frame size shrinkage to <= max_shrink_ratio (default 15%) so occluded
        keypoint dropouts do not collapse the crop into a tight clipping box.
      - Smooths size via exponential moving average (EMA) when prev_size is provided.
    """
    if kps is None or len(kps) == 0:
        return None

    visible_mask = kps[:, 2] >= conf_thresh
    v_pts = kps[visible_mask, :2]
    if len(v_pts) < min_kps:
        return None

    min_xy = v_pts.min(axis=0)
    max_xy = v_pts.max(axis=0)

    cx = float((min_xy[0] + max_xy[0]) / 2.0)
    cy = float((min_xy[1] + max_xy[1]) / 2.0)

    ext_x = float(max_xy[0] - min_xy[0])
    ext_y = float(max_xy[1] - min_xy[1])
    ext = max(ext_x, ext_y, 10.0)

    raw_size = float(K * ext)

    # Tracking feedback safeguards against occlusion-induced collapse
    if prev_size is not None and prev_size > 0:
        min_allowed_size = prev_size * (1.0 - max_shrink_ratio)
        clamped_size = max(raw_size, min_allowed_size)
        size = float(size_ema_alpha * clamped_size + (1.0 - size_ema_alpha) * prev_size)
    else:
        size = raw_size

    if size < min_size_px:
        return None  # Box collapsed below threshold -> trigger re-detection

    return cx, cy, size


def calibrate_keypoint_box_ratio(
    instances: List[Dict[str, Any]],
    img_w: int,
    img_h: int,
    crop_padding: float = 0.25,
    conf_thresh: float = 0.50,
    min_kps: int = 6
) -> float:
    """
    Measures empirical ratio K = median(crop_size / keypoint_extent) across
    ground truth training annotations to ensure tracked crops match training crops.
    """
    ratios: List[float] = []

    for inst in instances:
        cx_norm, cy_norm, bw_norm, bh_norm = inst["bbox"]
        cx = cx_norm * img_w
        cy = cy_norm * img_h
        bw = bw_norm * img_w
        bh = bh_norm * img_h

        _, _, size = compute_crop_box(cx, cy, bw, bh, crop_padding=crop_padding)

        kps = inst["keypoints"]  # [13, 3] normalized or pixel
        pts = kps[kps[:, 2] >= conf_thresh, :2].copy()
        if len(pts) < min_kps:
            continue

        # Convert to pixel space if normalized
        if pts[:, 0].max() <= 1.05 and pts[:, 1].max() <= 1.05:
            pts[:, 0] *= img_w
            pts[:, 1] *= img_h

        ext = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        if ext > 5.0:
            ratios.append(size / ext)

    if not ratios:
        return 1.35  # sensible default fallback

    return float(np.median(ratios))

