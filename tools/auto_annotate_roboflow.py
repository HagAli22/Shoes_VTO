r"""
auto_annotate_roboflow.py
─────────────────────────
Automated Pre-Annotation / Pseudo-Labeling Tool for Roboflow Uploads.

Generates initial 16-keypoint YOLO-pose annotations for any folder of raw images
using the trained Shoe VTO model, formatted strictly to the Roboflow YOLOv8-pose standard.

Enhancements:
  1. Audited & Frozen 16-Keypoint Anatomical Mapping (Indices 0 to 15).
  2. Anatomical Geometric Chirality Disambiguation (Cross-product eliminates L/R flip).
  3. Physical Foot De-duplication (Cross-Class IoU filter).
  4. Precise 3-State Visibility flags: 0 (absent), 1 (occluded), 2 (visible).
  5. Dual Engine Support: Loads PyTorch (.pt via Ultralytics) or ONNX (.onnx via ONNXRuntime).
  6. High-Quality Inspection Overlays (Solid dots for visible, hollow rings for occluded).

Each image (e.g. `img_001.jpg`) gets a matching `img_001.txt` containing:
    <class_id> <cx> <cy> <w> <h> <kp0_x> <kp0_y> <kp0_v> ... <kp15_x> <kp15_y> <kp15_v>

Where in Roboflow:
    - class_id: 0 = Foot_Right, 1 = Foot_left
    - cx, cy, w, h: normalized box center and dimensions [0, 1]
    - kp_x, kp_y: normalized keypoint coordinates [0, 1]
    - kp_v: visibility flag (0 = absent, 1 = occluded, 2 = visible)

Usage:
    python tools/auto_annotate_roboflow.py --input_dir data/uploadtoroboflow/womantest
    python tools/auto_annotate_roboflow.py --input_dir path/to/images --weights deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx
"""

import os
import sys
import glob
import time
import argparse
import cv2
import numpy as np
from tqdm import tqdm

# ==============================================================================
# AUDITED & FROZEN 16-KEYPOINT MAPPING & SKELETON DEFINITION
# ==============================================================================
KEYPOINT_NAMES = [
    "toe_ground",        # Index 0
    "heel_back",         # Index 1
    "heel_ground",       # Index 2
    "ball_medial",       # Index 3
    "ball_lateral",      # Index 4
    "ball_top",          # Index 5
    "instep_top",        # Index 6
    "arch_medial",       # Index 7
    "midfoot_lateral",   # Index 8
    "malleolus_medial",  # Index 9
    "malleolus_lateral", # Index 10
    "toe_tip",           # Index 11
    "ankle_center",      # Index 12
    "throat",            # Index 13
    "achilles",          # Index 14
    "shin_mid"           # Index 15
]

# Anatomical skeleton connections based on frozen indices
SKELETON_PAIRS = [
    # Toe box & Forefoot
    (11, 0),   # toe_tip -> toe_ground
    (11, 3),   # toe_tip -> ball_medial
    (11, 4),   # toe_tip -> ball_lateral
    (0, 3),    # toe_ground -> ball_medial
    (0, 4),    # toe_ground -> ball_lateral
    (3, 5),    # ball_medial -> ball_top
    (4, 5),    # ball_lateral -> ball_top
    (5, 6),    # ball_top -> instep_top
    # Medial & Lateral borders to Calcaneus
    (3, 7),    # ball_medial -> arch_medial
    (4, 8),    # ball_lateral -> midfoot_lateral
    (7, 2),    # arch_medial -> heel_ground
    (8, 2),    # midfoot_lateral -> heel_ground
    # Heel & Achilles
    (2, 1),    # heel_ground -> heel_back
    (1, 14),   # heel_back -> achilles
    (14, 12),  # achilles -> ankle_center
    # Malleoli & Throat
    (9, 12),   # malleolus_medial -> ankle_center
    (10, 12),  # malleolus_lateral -> ankle_center
    (6, 13),   # instep_top -> throat
    (13, 12),  # throat -> ankle_center
    # Lower Leg
    (12, 15),  # ankle_center -> shin_mid
]

# Visual Palette
COLOR_LEFT = {"box": (70, 230, 0), "skel": (120, 255, 60), "name": "Left Foot (Roboflow Cls 1)"}
COLOR_RIGHT = {"box": (0, 165, 255), "skel": (60, 200, 255), "name": "Right Foot (Roboflow Cls 0)"}


def compute_iou(b1, b2):
    """Computes IoU between two bboxes [x1, y1, x2, y2]."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-7)


def determine_geometric_chirality(kpts_dict):
    """
    Deterministically computes anatomical foot chirality (Left vs Right):
      Vector 1 (Longitudinal axis): toe_tip (Index 11) - heel_back (Index 1)
      Vector 2 (Lateral vector):    ball_lateral (Index 4) - heel_back (Index 1)
      2D Cross-Product: V_axis.x * V_lat.y - V_axis.y * V_lat.x

    Returns:
      model_cls_id: 0 = Left Foot, 1 = Right Foot
    """
    p_toe = kpts_dict.get(11)
    p_heel = kpts_dict.get(1)
    p_ball_lat = kpts_dict.get(4)

    if not p_toe or not p_heel or not p_ball_lat:
        return None

    # Check minimum confidence for geometry anchor points
    if p_toe[2] < 0.15 or p_heel[2] < 0.15 or p_ball_lat[2] < 0.15:
        return None

    v_axis_x = p_toe[0] - p_heel[0]
    v_axis_y = p_toe[1] - p_heel[1]

    v_lat_x = p_ball_lat[0] - p_heel[0]
    v_lat_y = p_ball_lat[1] - p_heel[1]

    cross_prod = v_axis_x * v_lat_y - v_axis_y * v_lat_x

    # In standard image coordinates (Y pointing down):
    # Cross product >= 0 -> Right Foot (model class 1)
    # Cross product < 0  -> Left Foot (model class 0)
    return 1 if cross_prod >= 0 else 0


class UniversalFootDetector:
    """
    Unified detector wrapper supporting either PyTorch (.pt) or ONNX (.onnx) weights.
    """
    def __init__(self, weights_path: str, device: str = "0", imgsz: int = 320):
        self.weights_path = weights_path
        self.is_onnx = weights_path.lower().endswith(".onnx")
        self.imgsz = imgsz
        self.device = device

        if self.is_onnx:
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(weights_path, providers=providers)
            self.active_provider = self.session.get_providers()[0]
            self.input_name = self.session.get_inputs()[0].name
            self.is_fp16 = "float16" in self.session.get_inputs()[0].type
        else:
            from ultralytics import YOLO
            self.model = YOLO(weights_path)

    def letterbox(self, img, size=320):
        h, w = img.shape[:2]
        scale = min(size / w, size / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_x = (size - new_w) // 2
        pad_y = (size - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
        return canvas, scale, pad_x, pad_y

    def predict(self, img_bgr, conf_thresh=0.25, iou_thresh=0.45, dedup_iou=0.60, use_geom_chirality=True):
        orig_h, orig_w = img_bgr.shape[:2]
        raw_detections = []

        if self.is_onnx:
            lb_img, scale, pad_x, pad_y = self.letterbox(img_bgr, self.imgsz)
            lb_rgb = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            if self.is_fp16:
                lb_rgb = lb_rgb.astype(np.float16)
            tensor_in = np.transpose(lb_rgb, (2, 0, 1))[np.newaxis]

            raw_out = self.session.run(None, {self.input_name: tensor_in})[0]
            if raw_out.ndim == 3:
                raw_out = raw_out[0]

            num_anchors = raw_out.shape[1]
            candidates = []

            for i in range(num_anchors):
                score_left = float(raw_out[4, i])
                score_right = float(raw_out[5, i])
                max_score = max(score_left, score_right)

                if max_score < conf_thresh:
                    continue

                cx, cy, bw, bh = raw_out[0, i], raw_out[1, i], raw_out[2, i], raw_out[3, i]
                model_cls_id = 0 if score_left >= score_right else 1

                x1 = max(0.0, min(orig_w, ((cx - bw / 2.0) - pad_x) / scale))
                y1 = max(0.0, min(orig_h, ((cy - bh / 2.0) - pad_y) / scale))
                x2 = max(0.0, min(orig_w, ((cx + bw / 2.0) - pad_x) / scale))
                y2 = max(0.0, min(orig_h, ((cy + bh / 2.0) - pad_y) / scale))

                kpts = {}
                for k in range(16):
                    kx = float((raw_out[6 + k * 3, i] - pad_x) / scale)
                    ky = float((raw_out[7 + k * 3, i] - pad_y) / scale)
                    kv = float(raw_out[8 + k * 3, i])
                    kpts[k] = (kx, ky, kv)

                if use_geom_chirality:
                    geom_cls = determine_geometric_chirality(kpts)
                    if geom_cls is not None:
                        model_cls_id = geom_cls

                candidates.append({
                    "model_cls_id": model_cls_id,
                    "conf": max_score,
                    "bbox": [x1, y1, x2, y2],
                    "kpts": kpts
                })

            # Independent Per-Class NMS
            nmed = []
            for target_cls in [0, 1]:
                cls_dets = [d for d in candidates if d["model_cls_id"] == target_cls]
                cls_dets.sort(key=lambda d: d["conf"], reverse=True)
                kept = []
                for det in cls_dets:
                    suppress = False
                    for k in kept:
                        if compute_iou(det["bbox"], k["bbox"]) > iou_thresh:
                            suppress = True
                            break
                    if not suppress:
                        kept.append(det)
                nmed.extend(kept)

            # Physical foot deduplication
            nmed.sort(key=lambda d: d["conf"], reverse=True)
            for det in nmed:
                dup = False
                for k in raw_detections:
                    if compute_iou(det["bbox"], k["bbox"]) > dedup_iou:
                        dup = True
                        break
                if not dup:
                    raw_detections.append(det)

        else:
            # PyTorch YOLO inference
            results = self.model.predict(
                source=img_bgr,
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )[0]

            boxes = results.boxes
            keypoints = results.keypoints
            num_dets = len(boxes) if boxes is not None else 0

            candidates = []
            for i in range(num_dets):
                model_cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                kpts = {}
                if keypoints is not None:
                    k_xy = keypoints.xy[i].tolist()
                    k_conf = keypoints.conf[i].tolist() if keypoints.conf is not None else [1.0] * len(k_xy)
                    for k in range(min(16, len(k_xy))):
                        kx, ky = k_xy[k]
                        kc = float(k_conf[k])
                        kpts[k] = (kx, ky, kc)

                if use_geom_chirality:
                    geom_cls = determine_geometric_chirality(kpts)
                    if geom_cls is not None:
                        model_cls_id = geom_cls

                candidates.append({
                    "model_cls_id": model_cls_id,
                    "conf": conf,
                    "bbox": [x1, y1, x2, y2],
                    "kpts": kpts
                })

            # Physical foot deduplication across classes
            candidates.sort(key=lambda d: d["conf"], reverse=True)
            for det in candidates:
                dup = False
                for k in raw_detections:
                    if compute_iou(det["bbox"], k["bbox"]) > dedup_iou:
                        dup = True
                        break
                if not dup:
                    raw_detections.append(det)

        return raw_detections


def auto_annotate(
    input_dir: str,
    weights_path: str = "outputs/stage_a/run_shuffled_v3/weights/best.pt",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    dedup_iou: float = 0.60,
    kpt_thresh: float = 0.35,
    imgsz: int = 320,
    device: str = "0",
    use_geom_chirality: bool = True,
    save_vis: bool = True,
    generate_yaml: bool = True,
    overwrite: bool = True,
    max_vis: int = 100
) -> dict:
    """
    Runs automated pre-annotation with audited 16-KP mapping, geometric chirality & physical de-duplication.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Check weights fallback
    if not os.path.exists(weights_path):
        fallback_onnx = "deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx"
        if os.path.exists(fallback_onnx):
            print(f"[!] Primary weights '{weights_path}' not found, falling back to ONNX deliverable: {fallback_onnx}")
            weights_path = fallback_onnx
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")

    # Gather image files
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    image_files = []
    for ext in exts:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_files = sorted(list(set(image_files)))

    total_images = len(image_files)
    if total_images == 0:
        print(f"[!] No images found in {input_dir}")
        return {"total_images": 0, "annotated_images": 0}

    print("==================================================================")
    print("  ROBOFLOW AUTOMATED PRE-ANNOTATION PIPELINE (16-KP AUDITED)      ")
    print(f"  Target Folder : {input_dir}")
    print(f"  Total Images  : {total_images}")
    print(f"  Model Weights : {weights_path}")
    print(f"  Confidence    : {conf_thresh} | Keypoint Thresh: {kpt_thresh}")
    print(f"  Post-Process  : Geometric Chirality={'ON' if use_geom_chirality else 'OFF'} | De-dup IoU={dedup_iou}")
    print(f"  Resolution    : {imgsz}x{imgsz} | Device: {device}")
    print("==================================================================\n")

    # Initialize Detector
    print("[1] Initializing Universal Detector Engine...")
    detector = UniversalFootDetector(weights_path=weights_path, device=device, imgsz=imgsz)

    vis_dir = os.path.join(input_dir, "annotation_previews") if save_vis else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    stats = {
        "total_images": total_images,
        "annotated_images": 0,
        "total_boxes": 0,
        "left_feet_count": 0,
        "right_feet_count": 0,
        "empty_images": 0,
    }

    print("[2] Generating pre-annotations for all images...")
    t0 = time.time()

    for img_path in tqdm(image_files, desc="Pre-annotating"):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(input_dir, f"{base_name}.txt")

        if os.path.exists(txt_path) and not overwrite:
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
        h_img, w_img = img_bgr.shape[:2]

        detections = detector.predict(
            img_bgr,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
            dedup_iou=dedup_iou,
            use_geom_chirality=use_geom_chirality
        )

        lines = []
        vis_frame = img_bgr.copy() if save_vis else None

        if len(detections) > 0:
            stats["annotated_images"] += 1
            stats["total_boxes"] += len(detections)

            for det in detections:
                model_cls = det["model_cls_id"]
                conf = det["conf"]
                bx1, by1, bx2, by2 = det["bbox"]
                kpts = det["kpts"]

                # Roboflow Class Mapping:
                # In Roboflow dataset: Class 0 = Foot_Right, Class 1 = Foot_left
                # In our detector:    Class 0 = left_foot,  Class 1 = right_foot
                # So: Roboflow 0 = Model 1, Roboflow 1 = Model 0
                roboflow_cls_id = 1 if model_cls == 0 else 0

                if model_cls == 0:
                    stats["left_feet_count"] += 1
                else:
                    stats["right_feet_count"] += 1

                # Normalized Bounding Box center & dimensions
                cx = ((bx1 + bx2) / 2.0) / w_img
                cy = ((by1 + by2) / 2.0) / h_img
                bw = (bx2 - bx1) / w_img
                bh = (by2 - by1) / h_img

                line_parts = [
                    str(roboflow_cls_id),
                    f"{np.clip(cx, 0.0, 1.0):.6f}",
                    f"{np.clip(cy, 0.0, 1.0):.6f}",
                    f"{np.clip(bw, 0.0, 1.0):.6f}",
                    f"{np.clip(bh, 0.0, 1.0):.6f}"
                ]

                # Exactly 16 keypoints in audited order (0 to 15)
                pts_for_vis = []
                for k_idx in range(16):
                    if k_idx in kpts:
                        kx_px, ky_px, kc = kpts[k_idx]
                        kx_n = float(np.clip(kx_px / w_img, 0.0, 1.0))
                        ky_n = float(np.clip(ky_px / h_img, 0.0, 1.0))

                        # Visibility: 0 = absent, 1 = occluded, 2 = visible
                        if kx_px <= 1e-5 and ky_px <= 1e-5:
                            v = 0
                        elif kc < kpt_thresh:
                            v = 1  # Occluded
                        else:
                            v = 2  # Visible

                        pts_for_vis.append((int(kx_px), int(ky_px), kc, v))
                    else:
                        kx_n, ky_n, v = 0.0, 0.0, 0
                        pts_for_vis.append((0, 0, 0.0, 0))

                    line_parts.extend([f"{kx_n:.6f}", f"{ky_n:.6f}", str(v)])

                lines.append(" ".join(line_parts) + "\n")

                # Visual Overlay for inspection
                if save_vis and vis_frame is not None:
                    color_info = COLOR_LEFT if model_cls == 0 else COLOR_RIGHT
                    box_color = color_info["box"]
                    skel_color = color_info["skel"]
                    label_text = f"{'Left' if model_cls == 0 else 'Right'} {conf:.0%}"

                    # Draw Bounding Box
                    x1_i, y1_i, x2_i, y2_i = int(bx1), int(by1), int(bx2), int(by2)
                    cv2.rectangle(vis_frame, (x1_i, y1_i), (x2_i, y2_i), box_color, 2, cv2.LINE_AA)

                    # Header badge
                    font = cv2.FONT_HERSHEY_DUPLEX
                    (tw, th), _ = cv2.getTextSize(label_text, font, 0.55, 1)
                    cv2.rectangle(vis_frame, (x1_i, max(0, y1_i - th - 8)), (x1_i + tw + 10, y1_i), box_color, -1)
                    cv2.putText(vis_frame, label_text, (x1_i + 5, max(th + 2, y1_i - 4)), font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

                    # Draw Skeleton connections
                    for p1_i, p2_i in SKELETON_PAIRS:
                        if p1_i < len(pts_for_vis) and p2_i < len(pts_for_vis):
                            pt1 = pts_for_vis[p1_i]
                            pt2 = pts_for_vis[p2_i]
                            if pt1[3] > 0 and pt2[3] > 0:
                                cv2.line(vis_frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), skel_color, 1, cv2.LINE_AA)

                    # Draw Keypoints (Solid for visible v=2, Hollow ring for occluded v=1)
                    for k_idx, pt in enumerate(pts_for_vis):
                        px, py, kc, v = pt
                        if v == 2:
                            # Solid dot
                            cv2.circle(vis_frame, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(vis_frame, (px, py), 4, (0, 0, 0), 1, cv2.LINE_AA)
                        elif v == 1:
                            # Hollow ring for occluded
                            cv2.circle(vis_frame, (px, py), 3, (180, 180, 255), 1, cv2.LINE_AA)

        else:
            stats["empty_images"] += 1

        # Write txt annotation file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Save visual preview (up to max_vis images to conserve disk space)
        if save_vis and vis_frame is not None and stats["annotated_images"] <= max_vis:
            vis_path = os.path.join(vis_dir, f"vis_{base_name}.jpg")
            cv2.imwrite(vis_path, vis_frame)

    elapsed = max(time.time() - t0, 0.001)

    # Step 3: Create Roboflow-compatible data.yaml
    if generate_yaml:
        yaml_path = os.path.join(input_dir, "data.yaml")
        yaml_content = (
            "# Roboflow YOLOv8 Pose Dataset Specification\n"
            "# Audited & Frozen 16 Anatomical Keypoints\n\n"
            "nc: 2\n"
            "names:\n"
            "  0: Foot_Right\n"
            "  1: Foot_left\n\n"
            "kpt_shape: [16, 3]\n\n"
            "# Official Frozen 16 Anatomical Keypoints Order:\n"
            "#  0: toe_ground        (Inferior contact point under toe box / lateral toe base)\n"
            "#  1: heel_back         (Posterior-most prominence of calcaneus)\n"
            "#  2: heel_ground       (Inferior ground contact point of calcaneus)\n"
            "#  3: ball_medial       (1st metatarsal head prominence - medial ball)\n"
            "#  4: ball_lateral      (5th metatarsal head prominence - lateral ball)\n"
            "#  5: ball_top          (Superior dorsal point over metatarsal heads)\n"
            "#  6: instep_top        (Apex of dorsal instep arch)\n"
            "#  7: arch_medial       (Medial longitudinal arch apex)\n"
            "#  8: midfoot_lateral   (Lateral midfoot outer boundary)\n"
            "#  9: malleolus_medial  (Center of medial malleolus - inner ankle)\n"
            "# 10: malleolus_lateral (Center of lateral malleolus - outer ankle)\n"
            "# 11: toe_tip           (Anterior tip of distal phalanx of 1st big toe)\n"
            "# 12: ankle_center      (Geometric center of talocrural joint)\n"
            "# 13: throat            (Anterior shoe opening throat / lower instep flex)\n"
            "# 14: achilles          (Achilles tendon insertion onto calcaneus)\n"
            "# 15: shin_mid          (Anterior lower tibia midpoint for leg alignment)\n"
        )
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        print(f"\n[3] Created Roboflow dataset descriptor: {yaml_path}")

    # Summary
    print("\n" + "=" * 66)
    print("           PRE-ANNOTATION COMPLETED SUCCESSFULLY!                 ")
    print("=" * 66)
    print(f"  Total Images Processed : {stats['total_images']}")
    print(f"  Images With Detections : {stats['annotated_images']} ({stats['annotated_images']/stats['total_images']*100:.1f}%)")
    print(f"  Empty/Background Images: {stats['empty_images']}")
    print(f"  Total Feet Annotated   : {stats['total_boxes']}")
    print(f"    - Left Feet (Roboflow Cls 1): {stats['left_feet_count']}")
    print(f"    - Right Feet (Roboflow Cls 0): {stats['right_feet_count']}")
    print(f"  Processing Time        : {elapsed:.2f} s ({stats['total_images']/elapsed:.1f} img/s)")
    print(f"  Annotation Directory   : {input_dir}")
    if vis_dir:
        print(f"  Inspection Previews    : {vis_dir}")
    print("=" * 66 + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate raw foot images for Roboflow with audited 16 KPs.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to folder containing raw images")
    parser.add_argument("--weights", default="outputs/stage_a/run_shuffled_v3/weights/best.pt",
                        help="Path to trained YOLO (.pt) or ONNX (.onnx) model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="Per-Class NMS IoU threshold")
    parser.add_argument("--dedup_iou", type=float, default=0.60, help="Cross-class duplicate foot IoU threshold")
    parser.add_argument("--kpt_thresh", type=float, default=0.35, help="Keypoint visibility threshold (v=2 vs v=1)")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference resolution")
    parser.add_argument("--device", default="0", help="CUDA device ('0' or 'cpu')")
    parser.add_argument("--no_geom", action="store_true", help="Disable geometric chirality check")
    parser.add_argument("--no_vis", action="store_true", help="Disable saving visual previews")
    parser.add_argument("--no_yaml", action="store_true", help="Disable generating data.yaml")
    parser.add_argument("--max_vis", type=int, default=100, help="Maximum visual previews to save")
    args = parser.parse_args()

    auto_annotate(
        input_dir=args.input_dir,
        weights_path=args.weights,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        dedup_iou=args.dedup_iou,
        kpt_thresh=args.kpt_thresh,
        imgsz=args.imgsz,
        device=args.device,
        use_geom_chirality=not args.no_geom,
        save_vis=not args.no_vis,
        generate_yaml=not args.no_yaml,
        max_vis=args.max_vis
    )


if __name__ == "__main__":
    main()
