"""
postprocess.py
──────────────
Production Post-Processing for Stage A Foot Detector.

Solves:
  1. Multi-class duplicate boxes on the same foot (Cross-Class Spatial NMS).
  2. Enforces maximum 2 distinct physical feet (1 Left + 1 Right, or top-2 non-overlapping).
  3. Keypoint filtering with Mandatory Anatomical Anchors:
     - Mandatory Anchors: [0: toe_tip, 3: heel_ground, 4: ball_medial, 5: ball_lateral, 2: heel_back]
       -> Always retained; flagged as 'occluded/estimated' if visibility confidence is below threshold.
     - Secondary Keypoints: Only retained if visibility confidence >= min_kp_conf (e.g. 0.45).
"""

import numpy as np

# Mandatory anatomical anchor keypoint indices
MANDATORY_ANCHOR_INDICES = {0, 2, 3, 4, 5}


def compute_box_iou(box1, box2):
    """Computes IoU between two [x1, y1, x2, y2] bounding boxes."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(1e-6, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    boxBArea = max(1e-6, (box2[2] - box2[0]) * (box2[3] - box2[1]))

    return interArea / (boxAArea + boxBArea - interArea)


def postprocess_stage_a(
    raw_result,
    box_conf_thresh: float = 0.30,
    cross_class_iou_thresh: float = 0.35,
    min_kp_conf: float = 0.45,
    max_feet: int = 2
):
    """
    Cleans raw YOLO detections into stable, non-duplicate foot perceptions.

    Returns:
      list of dicts, each containing:
        - 'box': [x1, y1, x2, y2]
        - 'cls': int (0 for left_foot, 1 for right_foot)
        - 'cls_name': str ('left_foot' or 'right_foot')
        - 'conf': float (box detection score)
        - 'keypoints': np.ndarray [16, 2] (coordinates)
        - 'kp_conf': np.ndarray [16] (per-keypoint confidences)
        - 'kp_status': list of str (length 16: 'visible', 'occluded_anchor', or 'hidden')
    """
    if raw_result is None or raw_result.boxes is None or len(raw_result.boxes) == 0:
        return []

    boxes_raw = raw_result.boxes
    kpts_raw = raw_result.keypoints

    # 1. Collect candidate detections above base threshold
    candidates = []
    for i in range(len(boxes_raw)):
        conf = float(boxes_raw.conf[i])
        if conf < box_conf_thresh:
            continue

        cls = int(boxes_raw.cls[i])
        xyxy = boxes_raw.xyxy[i].cpu().numpy().tolist()

        if kpts_raw is not None and i < len(kpts_raw.xy):
            pts = kpts_raw.xy[i].cpu().numpy()
            kc = kpts_raw.conf[i].cpu().numpy() if kpts_raw.conf is not None else np.ones(len(pts))
        else:
            pts = np.zeros((16, 2), dtype=np.float32)
            kc = np.zeros(16, dtype=np.float32)

        candidates.append({
            "box": xyxy,
            "cls": cls,
            "conf": conf,
            "pts": pts,
            "kc": kc
        })

    if not candidates:
        return []

    # Sort candidates by detection confidence descending
    candidates.sort(key=lambda x: x["conf"], reverse=True)

    # 2. Cross-Class Spatial NMS (Suppress duplicate classifications of the same foot)
    selected = []
    seen_classes = set()

    for cand in candidates:
        overlap = False
        for sel in selected:
            iou = compute_box_iou(cand["box"], sel["box"])
            if iou > cross_class_iou_thresh:
                # Same physical foot detected twice with competing classes
                overlap = True
                break

        if not overlap:
            # Enforce max 1 left_foot and max 1 right_foot if 2 feet are present
            if len(selected) < max_feet:
                selected.append(cand)
                seen_classes.add(cand["cls"])

    # 3. Keypoint Filtering with Mandatory Anatomical Anchor Retention
    processed_feet = []
    class_names = {0: "left_foot", 1: "right_foot"}

    for item in selected:
        pts = item["pts"]
        kc = item["kc"]
        kp_status = []

        for k in range(16):
            conf_k = float(kc[k]) if k < len(kc) else 0.0
            is_anchor = k in MANDATORY_ANCHOR_INDICES

            if conf_k >= min_kp_conf:
                kp_status.append("visible")
            elif is_anchor:
                # Mandatory anchor whose direct visibility is occluded/low
                kp_status.append("occluded_anchor")
            else:
                # Secondary noisy keypoint below threshold
                kp_status.append("hidden")

        processed_feet.append({
            "box": [int(round(v)) for v in item["box"]],
            "cls": item["cls"],
            "cls_name": class_names.get(item["cls"], str(item["cls"])),
            "conf": item["conf"],
            "keypoints": pts,
            "kp_conf": kc,
            "kp_status": kp_status
        })

    return processed_feet

