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

# 15-Keypoint Anatomical Mapping (Derived Post-Processing)
KEYPOINT_NAMES_15KP = [
    "toe_ground",        # Index 0
    "heel_back",         # Index 1
    "heel_ground",       # Index 2
    "ball_medial",       # Index 3
    "ball_lateral",      # Index 4
    "ball_top",          # Index 5  (Derived: midpoint ball_medial/lateral aligned with instep_top)
    "instep_top",        # Index 6
    "arch_medial",       # Index 7
    "midfoot_lateral",   # Index 8
    "malleolus_medial",  # Index 9
    "malleolus_lateral", # Index 10
    "toe_tip",           # Index 11
    "throat",            # Index 12
    "achilles",          # Index 13
    "shin_mid"           # Index 14 (Derived: projected upward along tibia axis)
]


def derive_secondary_keypoints_15kp(keypoints_13, bbox=None):
    """
    Deterministically derives 'ball_top' (index 5) and 'shin_mid' (index 14)
    from high-precision 13-keypoint detections using anatomical geometry.

    Viewpoint-invariant:
      - Top-down / selfie perspectives
      - Front views (feet hanging down)
      - Back views (standing calves / heels)
      - Side views (lateral ankle / reclined)

    Supports both:
      - List of dicts: [{'index': 0, 'name': '...', 'x': float, 'y': float, 'visibility': float}, ...]
      - Numpy ndarray: shape (13, 2) or (13, 3)

    Returns:
      15 keypoints in the same format as input.
    """
    is_dict = isinstance(keypoints_13, list) and len(keypoints_13) > 0 and isinstance(keypoints_13[0], dict)

    if is_dict:
        k = {item["index"]: item for item in keypoints_13}

        # 1. Derive ball_top (midpoint of ball_medial [3] & ball_lateral [4], aligned towards instep_top [5])
        p_bm = np.array([k[3]["x"], k[3]["y"]], dtype=np.float32)
        p_bl = np.array([k[4]["x"], k[4]["y"]], dtype=np.float32)
        p_instep = np.array([k[5]["x"], k[5]["y"]], dtype=np.float32)

        m_ball = 0.5 * (p_bm + p_bl)
        v_dorsal_ball = p_instep - m_ball
        p_ball_top = m_ball + 0.28 * v_dorsal_ball
        v_ball_top = min(k[3]["visibility"], k[4]["visibility"], k[5]["visibility"])

        # 2. Derive shin_mid (viewpoint-invariant anatomical lower leg axis)
        p_throat = np.array([k[11]["x"], k[11]["y"]], dtype=np.float32)
        p_ach = np.array([k[12]["x"], k[12]["y"]], dtype=np.float32)
        p_toe = np.array([k[10]["x"], k[10]["y"]], dtype=np.float32)
        p_hg = np.array([k[2]["x"], k[2]["y"]], dtype=np.float32)
        p_mm = np.array([k[8]["x"], k[8]["y"]], dtype=np.float32)
        p_ml = np.array([k[9]["x"], k[9]["y"]], dtype=np.float32)

        v_toe = k[10]["visibility"]
        v_hg = k[2]["visibility"]
        v_ach = k[12]["visibility"]
        v_thr = k[11]["visibility"]

        # Scale reference: use bbox diagonal or foot keypoints extent
        if bbox is not None:
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            ref_len = float(np.sqrt(bw * bw + bh * bh)) * 0.40
        else:
            all_x = [kp["x"] for kp in keypoints_13]
            all_y = [kp["y"] for kp in keypoints_13]
            bw = max(all_x) - min(all_x)
            bh = max(all_y) - min(all_y)
            ref_len = float(np.sqrt(bw * bw + bh * bh)) * 0.40

        # CASE A: Back view (toes occluded, heel and achilles prominent)
        if v_toe < 0.30 and (v_hg > 0.40 or v_ach > 0.50):
            v_leg = p_ach - p_hg
            norm_leg = float(np.linalg.norm(v_leg))
            u_leg = (v_leg / norm_leg) if norm_leg > 1e-4 else np.array([0.0, -1.0], dtype=np.float32)
            p_shin_mid = p_throat + 1.20 * ref_len * u_leg
            v_shin_mid = min(v_thr, v_ach, max(v_hg, 0.4)) * 0.90

        # CASE B: Front / Top-down / Side views (toes visible)
        else:
            v_dorsal = p_throat - p_toe
            norm_dorsal = float(np.linalg.norm(v_dorsal))

            # Transmalleolar axis normal (leg rises perpendicular to malleoli line)
            v_malleoli = p_ml - p_mm
            perp = np.array([-v_malleoli[1], v_malleoli[0]], dtype=np.float32)
            if np.dot(perp, v_dorsal) < 0:
                perp = -perp
            norm_perp = float(np.linalg.norm(perp))
            u_perp = (perp / norm_perp) if norm_perp > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)

            u_dorsal = (v_dorsal / norm_dorsal) if norm_dorsal > 1e-4 else u_perp
            u_leg = 0.65 * u_dorsal + 0.35 * u_perp
            norm_u = float(np.linalg.norm(u_leg))
            u_leg = (u_leg / norm_u) if norm_u > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)

            p_shin_mid = p_throat + 1.10 * ref_len * u_leg
            v_shin_mid = min(v_thr, max(v_toe, 0.5)) * 0.92

        # Assemble 15 keypoints list
        out_15 = [
            {"index": 0, "name": KEYPOINT_NAMES_15KP[0], "x": k[0]["x"], "y": k[0]["y"], "visibility": k[0]["visibility"]},
            {"index": 1, "name": KEYPOINT_NAMES_15KP[1], "x": k[1]["x"], "y": k[1]["y"], "visibility": k[1]["visibility"]},
            {"index": 2, "name": KEYPOINT_NAMES_15KP[2], "x": k[2]["x"], "y": k[2]["y"], "visibility": k[2]["visibility"]},
            {"index": 3, "name": KEYPOINT_NAMES_15KP[3], "x": k[3]["x"], "y": k[3]["y"], "visibility": k[3]["visibility"]},
            {"index": 4, "name": KEYPOINT_NAMES_15KP[4], "x": k[4]["x"], "y": k[4]["y"], "visibility": k[4]["visibility"]},
            {"index": 5, "name": KEYPOINT_NAMES_15KP[5], "x": float(p_ball_top[0]), "y": float(p_ball_top[1]), "visibility": float(v_ball_top)},
            {"index": 6, "name": KEYPOINT_NAMES_15KP[6], "x": k[5]["x"], "y": k[5]["y"], "visibility": k[5]["visibility"]},
            {"index": 7, "name": KEYPOINT_NAMES_15KP[7], "x": k[6]["x"], "y": k[6]["y"], "visibility": k[6]["visibility"]},
            {"index": 8, "name": KEYPOINT_NAMES_15KP[8], "x": k[7]["x"], "y": k[7]["y"], "visibility": k[7]["visibility"]},
            {"index": 9, "name": KEYPOINT_NAMES_15KP[9], "x": k[8]["x"], "y": k[8]["y"], "visibility": k[8]["visibility"]},
            {"index": 10, "name": KEYPOINT_NAMES_15KP[10], "x": k[9]["x"], "y": k[9]["y"], "visibility": k[9]["visibility"]},
            {"index": 11, "name": KEYPOINT_NAMES_15KP[11], "x": k[10]["x"], "y": k[10]["y"], "visibility": k[10]["visibility"]},
            {"index": 12, "name": KEYPOINT_NAMES_15KP[12], "x": k[11]["x"], "y": k[11]["y"], "visibility": k[11]["visibility"]},
            {"index": 13, "name": KEYPOINT_NAMES_15KP[13], "x": k[12]["x"], "y": k[12]["y"], "visibility": k[12]["visibility"]},
            {"index": 14, "name": KEYPOINT_NAMES_15KP[14], "x": float(p_shin_mid[0]), "y": float(p_shin_mid[1]), "visibility": float(v_shin_mid)},
        ]
        return out_15

    else:
        # Numpy array shape (13, 2) or (13, 3)
        pts = np.asarray(keypoints_13, dtype=np.float32)
        has_vis = pts.shape[-1] >= 3

        p_bm = pts[3, :2]
        p_bl = pts[4, :2]
        p_instep = pts[5, :2]
        m_ball = 0.5 * (p_bm + p_bl)
        v_dorsal_ball = p_instep - m_ball
        p_ball_top = m_ball + 0.28 * v_dorsal_ball

        p_throat = pts[11, :2]
        p_ach = pts[12, :2]
        p_toe = pts[10, :2]
        p_hg = pts[2, :2]
        p_mm = pts[8, :2]
        p_ml = pts[9, :2]

        v_toe = pts[10, 2] if has_vis else 1.0
        v_hg = pts[2, 2] if has_vis else 1.0
        v_ach = pts[12, 2] if has_vis else 1.0
        v_thr = pts[11, 2] if has_vis else 1.0

        if bbox is not None:
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            ref_len = float(np.sqrt(bw * bw + bh * bh)) * 0.40
        else:
            bw = pts[:, 0].max() - pts[:, 0].min()
            bh = pts[:, 1].max() - pts[:, 1].min()
            ref_len = float(np.sqrt(bw * bw + bh * bh)) * 0.40

        # CASE A: Back view
        if v_toe < 0.30 and (v_hg > 0.40 or v_ach > 0.50):
            v_leg = p_ach - p_hg
            norm_leg = float(np.linalg.norm(v_leg))
            u_leg = (v_leg / norm_leg) if norm_leg > 1e-4 else np.array([0.0, -1.0], dtype=np.float32)
            p_shin_mid = p_throat + 1.20 * ref_len * u_leg
            v_shin_mid = min(v_thr, v_ach, max(v_hg, 0.4)) * 0.90

        # CASE B: Front / Top-down / Side
        else:
            v_dorsal = p_throat - p_toe
            norm_dorsal = float(np.linalg.norm(v_dorsal))

            v_malleoli = p_ml - p_mm
            perp = np.array([-v_malleoli[1], v_malleoli[0]], dtype=np.float32)
            if np.dot(perp, v_dorsal) < 0:
                perp = -perp
            norm_perp = float(np.linalg.norm(perp))
            u_perp = (perp / norm_perp) if norm_perp > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)

            u_dorsal = (v_dorsal / norm_dorsal) if norm_dorsal > 1e-4 else u_perp
            u_leg = 0.65 * u_dorsal + 0.35 * u_perp
            norm_u = float(np.linalg.norm(u_leg))
            u_leg = (u_leg / norm_u) if norm_u > 1e-4 else np.array([0.0, 1.0], dtype=np.float32)

            p_shin_mid = p_throat + 1.10 * ref_len * u_leg
            v_shin_mid = min(v_thr, max(v_toe, 0.5)) * 0.92

        if has_vis:
            v_ball_top = min(pts[3, 2], pts[4, 2], pts[5, 2])
            k_ball_top = np.array([p_ball_top[0], p_ball_top[1], v_ball_top], dtype=np.float32)
            k_shin_mid = np.array([p_shin_mid[0], p_shin_mid[1], v_shin_mid], dtype=np.float32)
        else:
            k_ball_top = p_ball_top
            k_shin_mid = p_shin_mid

        # Slice and construct 15 keypoints array
        out_pts = np.zeros((15, pts.shape[-1]), dtype=np.float32)
        out_pts[:5] = pts[:5]
        out_pts[5] = k_ball_top
        out_pts[6:14] = pts[5:13]
        out_pts[14] = k_shin_mid
        return out_pts



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

