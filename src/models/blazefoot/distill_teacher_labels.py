"""
distill_teacher_labels.py
─────────────────────────
Teacher Knowledge Distillation & High-Fidelity Pseudo-Labeling Engine.

Uses the frozen Stage A 16-KP Teacher Model (deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx)
to generate clean, sub-pixel accurate, consistent pseudo-labels across the dataset.
Eliminates human annotation noise, missing keypoint visibility (e.g. heel/toe ambiguity),
and inconsistent bounding boxes.

Usage:
  python -m src.models.blazefoot.distill_teacher_labels \
      --data "data/shuffled_v3" \
      --teacher "deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx" \
      --output "data/distilled_v3"
"""

import os
import sys
import glob
import shutil
import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# Force UTF-8 on Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

KEYPOINT_NAMES_16 = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial",
    "ball_lateral", "ball_top", "instep_top", "arch_medial",
    "midfoot_lateral", "malleolus_medial", "malleolus_lateral", "toe_tip",
    "ankle_center", "throat", "achilles", "shin_mid"
]


def letterbox(img: np.ndarray, size: int = 320):
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
    return canvas, scale, pad_x, pad_y


def compute_iou(b1, b2):
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    a1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    a2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, scores, classes, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    while indices:
        current = indices.pop(0)
        keep.append(current)
        indices = [
            i for i in indices
            if classes[i] != classes[current] or compute_iou(boxes[current], boxes[i]) < iou_thresh
        ]
    return keep


class TeacherInference:
    def __init__(self, onnx_path: str):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 4
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.img_size = self.input_shape[2] if len(self.input_shape) >= 3 and isinstance(self.input_shape[2], int) else 320

    def predict(self, bgr_img: np.ndarray, conf_thresh: float = 0.35, iou_thresh: float = 0.5):
        h_orig, w_orig = bgr_img.shape[:2]
        canvas, scale, pad_x, pad_y = letterbox(bgr_img, self.img_size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]

        raw_out = self.session.run(None, {self.input_name: blob})[0] # [1, 54, N]
        preds = raw_out[0].T # [N, 54]

        boxes = []
        scores = []
        classes = []
        kpts_list = []

        for p in preds:
            cls_scores = p[4:6]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = float(cls_scores[cls_id])
            if cls_conf < conf_thresh:
                continue

            cx, cy, w, h = p[0:4]
            # Convert letterbox px to original img normalized coords
            x1_px = (cx - w / 2.0 - pad_x) / scale
            y1_px = (cy - h / 2.0 - pad_y) / scale
            x2_px = (cx + w / 2.0 - pad_x) / scale
            y2_px = (cy + h / 2.0 - pad_y) / scale

            # Clamp
            x1_px = max(0.0, min(float(w_orig), x1_px))
            y1_px = max(0.0, min(float(h_orig), y1_px))
            x2_px = max(0.0, min(float(w_orig), x2_px))
            y2_px = max(0.0, min(float(h_orig), y2_px))

            # 16 Keypoints [x, y, conf]
            raw_kp = p[6:]
            kp = []
            for k in range(16):
                kx_px = (raw_kp[k * 3] - pad_x) / scale
                ky_px = (raw_kp[k * 3 + 1] - pad_y) / scale
                kconf = float(raw_kp[k * 3 + 2])
                
                kx_norm = max(0.0, min(1.0, kx_px / w_orig))
                ky_norm = max(0.0, min(1.0, ky_px / h_orig))
                vis = 2.0 if kconf >= 0.3 else (1.0 if kconf >= 0.15 else 0.0)
                kp.extend([kx_norm, ky_norm, vis])

            boxes.append([x1_px, y1_px, x2_px, y2_px])
            scores.append(cls_conf)
            classes.append(cls_id)
            kpts_list.append(kp)

        keep = nms_boxes(boxes, scores, classes, iou_thresh)
        results = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            bw = (x2 - x1) / w_orig
            bh = (y2 - y1) / h_orig
            cx = (x1 + x2) / (2.0 * w_orig)
            cy = (y1 + y2) / (2.0 * h_orig)
            results.append({
                "class_id": classes[i],
                "bbox": [cx, cy, bw, bh],
                "score": scores[i],
                "keypoints": kpts_list[i]
            })
        return results


def distill_dataset(data_dir: str, teacher_onnx: str, output_dir: str):
    teacher = TeacherInference(teacher_onnx)
    print("=" * 80)
    print("  [🚀] TEACHER KNOWLEDGE DISTILLATION & PSEUDO-LABEL GENERATION")
    print(f"  Teacher Model : {teacher_onnx}")
    print(f"  Input Dataset : {data_dir}")
    print(f"  Distilled Out : {output_dir}")
    print("=" * 80)

    splits = ["train", "valid", "val", "test"]
    found_splits = []
    for s in splits:
        img_d = os.path.join(data_dir, s, "images")
        if os.path.isdir(img_d):
            found_splits.append(s)

    if not found_splits:
        # Check if direct images folder
        if os.path.isdir(os.path.join(data_dir, "images")):
            found_splits = ["."]
        else:
            raise ValueError(f"No valid dataset splits found in {data_dir}")

    total_images_processed = 0
    total_pseudo_labels = 0
    total_fallback_labels = 0

    for split in found_splits:
        in_img_dir = os.path.join(data_dir, split, "images") if split != "." else os.path.join(data_dir, "images")
        in_lbl_dir = os.path.join(data_dir, split, "labels") if split != "." else os.path.join(data_dir, "labels")

        out_split = "val" if split == "valid" else split
        out_img_dir = os.path.join(output_dir, out_split, "images") if split != "." else os.path.join(output_dir, "images")
        out_lbl_dir = os.path.join(output_dir, out_split, "labels") if split != "." else os.path.join(output_dir, "labels")

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_lbl_dir, exist_ok=True)

        img_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
            img_files.extend(glob.glob(os.path.join(in_img_dir, ext)))

        print(f"\n📂 Processing split '{split}' ({len(img_files)} images)...")

        for img_path in tqdm(img_files, desc=f"Distilling {split}"):
            bname = os.path.basename(img_path)
            stem = os.path.splitext(bname)[0]
            out_img_path = os.path.join(out_img_dir, bname)
            out_lbl_path = os.path.join(out_lbl_dir, f"{stem}.txt")

            # Copy image if not exists
            if not os.path.exists(out_img_path):
                shutil.copy2(img_path, out_img_path)

            img = cv2.imread(img_path)
            if img is None:
                continue

            # 1. Run Teacher Inference
            detections = teacher.predict(img, conf_thresh=0.30, iou_thresh=0.50)

            # If teacher didn't find anything, check if original label exists as fallback
            orig_lbl_path = os.path.join(in_lbl_dir, f"{stem}.txt")
            if len(detections) == 0 and os.path.exists(orig_lbl_path):
                shutil.copy2(orig_lbl_path, out_lbl_path)
                total_fallback_labels += 1
                total_images_processed += 1
                continue

            # 2. Write Distilled High-Precision Labels
            with open(out_lbl_path, "w", encoding="utf-8") as f:
                for det in detections:
                    cls_id = det["class_id"]
                    cx, cy, bw, bh = det["bbox"]
                    kps = det["keypoints"]
                    
                    # Format: class cx cy w h kx0 ky0 v0 kx1 ky1 v1 ... (16 keypoints)
                    kp_str = " ".join([f"{v:.6f}" for v in kps])
                    line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kp_str}\n"
                    f.write(line)
                    total_pseudo_labels += 1

            total_images_processed += 1

    print("\n" + "=" * 80)
    print("  ✅ DISTILLATION COMPLETE!")
    print(f"  Total Images Processed : {total_images_processed}")
    print(f"  Total Teacher Targets  : {total_pseudo_labels}")
    print(f"  Fallback Manual Labels : {total_fallback_labels}")
    print(f"  Output Directory       : {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="BlazeFoot Teacher Knowledge Distillation")
    parser.add_argument("--data", type=str, default="data/shuffled_v3", help="Input dataset directory")
    parser.add_argument("--teacher", type=str, default="deliverables/stage_a_16kp/models/stage-a-320-16kp-fp32.onnx", help="Teacher ONNX model")
    parser.add_argument("--output", type=str, default="data/distilled_v3", help="Output distilled dataset directory")
    args = parser.parse_args()

    distill_dataset(args.data, args.teacher, args.output)


if __name__ == "__main__":
    main()
