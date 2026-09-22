r"""
auto_annotate_roboflow.py
─────────────────────────
Automated Pre-Annotation / Pseudo-Labeling Tool for Roboflow Uploads.

Generates initial 16-keypoint YOLO-pose annotations for any folder of raw images
using the trained Shoe VTO model, formatted strictly to the Roboflow YOLOv8-pose standard.

Each image (e.g. `img_001.jpg`) gets a matching `img_001.txt` containing:
    <class_id> <cx> <cy> <w> <h> <kp0_x> <kp0_y> <kp0_v> ... <kp15_x> <kp15_y> <kp15_v>

Where:
    - class_id: 0 = Foot_Right, 1 = Foot_left
    - cx, cy, w, h: normalized box center and dimensions [0, 1]
    - kp_x, kp_y: normalized keypoint coordinates [0, 1]
    - kp_v: visibility flag (0 = absent, 1 = occluded, 2 = visible)

Usage:
    python tools/auto_annotate_roboflow.py --input_dir data/uploadtoroboflow/womantest
"""

import os
import sys
import glob
import time
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO


def auto_annotate(
    input_dir: str,
    weights_path: str = "outputs/stage_a/run_shuffled_v3/weights/best.pt",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    imgsz: int = 320,
    device: str = "0",
    save_vis: bool = True,
    generate_yaml: bool = True,
    overwrite: bool = True,
) -> dict:
    """
    Runs inference on all images in input_dir and generates Roboflow-compatible .txt files.
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found: {weights_path}")

    # Gather all image files
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
    print("  Roboflow Automated Pre-Annotation Pipeline                     ")
    print(f"  Target Folder : {input_dir}")
    print(f"  Total Images  : {total_images}")
    print(f"  Model Weights : {weights_path}")
    print(f"  Confidence    : {conf_thresh} | IOU Threshold: {iou_thresh}")
    print(f"  Resolution    : {imgsz}x{imgsz} | Device: {device}")
    print("==================================================================\n")

    # Load YOLO model
    print("[1] Loading trained YOLOv8-pose model...")
    model = YOLO(weights_path)

    # Optional: Visual preview folder
    vis_dir = os.path.join(input_dir, "annotation_previews") if save_vis else None
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    # Keypoint skeleton pairs for visualization
    skeleton = [
        (0, 1), (1, 3), (3, 2), (2, 14), (14, 15), (14, 12),
        (12, 10), (12, 11), (12, 13), (13, 7), (7, 6),
        (6, 4), (6, 5), (4, 8), (5, 9), (0, 4), (0, 5)
    ]
    color_left = (0, 230, 70)     # Green
    color_right = (0, 160, 255)   # Amber

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

        # Inference
        results = model.predict(
            source=img_path,
            conf=conf_thresh,
            iou=iou_thresh,
            imgsz=imgsz,
            device=device,
            verbose=False
        )[0]

        boxes = results.boxes
        keypoints = results.keypoints

        lines = []
        vis_frame = cv2.imread(img_path) if save_vis else None
        h_img, w_img = vis_frame.shape[:2] if save_vis else (0, 0)

        num_dets = len(boxes) if boxes is not None else 0

        if num_dets > 0:
            stats["annotated_images"] += 1
            stats["total_boxes"] += num_dets

            for i in range(num_dets):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                cx, cy, w, h = boxes.xywhn[i].tolist()

                if cls_id == 0:
                    stats["left_feet_count"] += 1
                else:
                    stats["right_feet_count"] += 1

                # Keypoints extraction
                kpts_xyn = keypoints.xyn[i].tolist() if keypoints is not None else []
                kpts_conf = (
                    keypoints.conf[i].tolist()
                    if keypoints is not None and keypoints.conf is not None
                    else [1.0] * len(kpts_xyn)
                )

                # Roboflow class mapping: 0 = Foot_Right, 1 = Foot_left
                # Model outputs: 0 = left_foot, 1 = right_foot
                roboflow_cls_id = 1 if cls_id == 0 else 0

                # Format Roboflow YOLO pose line
                line_parts = [
                    str(roboflow_cls_id),
                    f"{np.clip(cx, 0.0, 1.0):.6f}",
                    f"{np.clip(cy, 0.0, 1.0):.6f}",
                    f"{np.clip(w, 0.0, 1.0):.6f}",
                    f"{np.clip(h, 0.0, 1.0):.6f}"
                ]

                # Exactly 16 keypoints
                for k_idx in range(16):
                    if k_idx < len(kpts_xyn):
                        kx, ky = kpts_xyn[k_idx]
                        kc = kpts_conf[k_idx]

                        # Clamp
                        kx = float(np.clip(kx, 0.0, 1.0))
                        ky = float(np.clip(ky, 0.0, 1.0))

                        # Visibility: 0 = absent, 1 = occluded, 2 = visible
                        if kx <= 1e-5 and ky <= 1e-5:
                            v = 0
                        elif kc < 0.20:
                            v = 1
                        else:
                            v = 2
                    else:
                        kx, ky, v = 0.0, 0.0, 0

                    line_parts.extend([f"{kx:.6f}", f"{ky:.6f}", str(v)])

                lines.append(" ".join(line_parts) + "\n")

                # Visual overlay for preview
                if save_vis and vis_frame is not None:
                    col = color_left if cls_id == 0 else color_right
                    label_name = "Left Foot" if cls_id == 0 else "Right Foot"

                    bx1 = int((cx - w / 2.0) * w_img)
                    by1 = int((cy - h / 2.0) * h_img)
                    bx2 = int((cx + w / 2.0) * w_img)
                    by2 = int((cy + h / 2.0) * h_img)

                    cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), col, 2)
                    cv2.putText(
                        vis_frame,
                        f"{label_name} {conf*100:.0f}%",
                        (bx1, max(20, by1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        col,
                        2,
                        cv2.LINE_AA,
                    )

                    # Draw keypoints and skeleton
                    pts = []
                    for k_idx in range(min(16, len(kpts_xyn))):
                        kx, ky = kpts_xyn[k_idx]
                        px, py = int(kx * w_img), int(ky * h_img)
                        pts.append((px, py))
                        cv2.circle(vis_frame, (px, py), 4, (0, 255, 255), -1)
                        cv2.circle(vis_frame, (px, py), 5, (0, 0, 0), 1)

                    for p1_i, p2_i in skeleton:
                        if p1_i < len(pts) and p2_i < len(pts):
                            p1, p2 = pts[p1_i], pts[p2_i]
                            if p1 != (0, 0) and p2 != (0, 0):
                                cv2.line(vis_frame, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)
        else:
            stats["empty_images"] += 1

        # Write txt annotation file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

        # Save visual preview (first 50 images to conserve disk space)
        if save_vis and vis_frame is not None and stats["annotated_images"] <= 50:
            vis_path = os.path.join(vis_dir, f"vis_{base_name}.jpg")
            cv2.imwrite(vis_path, vis_frame)

    elapsed = time.time() - t0

    # Step 3: Create Roboflow-compatible data.yaml
    if generate_yaml:
        yaml_path = os.path.join(input_dir, "data.yaml")
        yaml_content = (
            "# Roboflow YOLOv8 Pose Dataset Specification\n"
            "nc: 2\n"
            "names:\n"
            "  0: Foot_Right\n"
            "  1: Foot_left\n\n"
            "kpt_shape: [16, 3]\n\n"
            "# 16 Frozen Anatomical Keypoints:\n"
            "# 0: toe_tip, 1: toe_ground, 2: heel_back, 3: heel_ground,\n"
            "# 4: ball_medial, 5: ball_lateral, 6: ball_top, 7: instep_top,\n"
            "# 8: arch_medial, 9: midfoot_lateral, 10: malleolus_medial, 11: malleolus_lateral,\n"
            "# 12: ankle_center, 13: throat, 14: achilles, 15: shin_mid\n"
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
    print(f"    - Left Feet (class 0): {stats['left_feet_count']}")
    print(f"    - Right Feet (class 1: {stats['right_feet_count']}")
    print(f"  Processing Time        : {elapsed:.2f} s ({stats['total_images']/elapsed:.1f} img/s)")
    print(f"  Annotation Directory   : {input_dir}")
    if vis_dir:
        print(f"  Inspection Previews    : {vis_dir}")
    print("=" * 66 + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate raw foot images for Roboflow.")
    parser.add_argument("--input_dir", default="data/uploadtoroboflow/7",
                        help="Path to folder containing raw images")
    parser.add_argument("--weights", default="outputs/stage_a/run_v2_1/weights/best.pt",
                        help="YOLO pose weights to use for auto-annotation")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--imgsz", type=int, default=320, help="Inference resolution")
    parser.add_argument("--device", default="0", help="CUDA device ('0' or 'cpu')")
    parser.add_argument("--no_vis", action="store_true", help="Disable saving visual previews")
    parser.add_argument("--no_yaml", action="store_true", help="Disable generating data.yaml")
    args = parser.parse_args()

    auto_annotate(
        input_dir=args.input_dir,
        weights_path=args.weights,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_vis=not args.no_vis,
        generate_yaml=not args.no_yaml,
    )


if __name__ == "__main__":
    main()

