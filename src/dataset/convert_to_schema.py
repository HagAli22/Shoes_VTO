"""
convert_to_schema.py
────────────────────
Converts public foot-keypoint datasets into our unified 18-KP COCO format.

Supported source datasets:
  --dataset cmu    : CMU Human Foot Keypoint Dataset  (3 KP/foot: big_toe, small_toe, heel)
  --dataset h3wb   : H3WB (Human3.6M 3D WholeBody)   (3 KP/foot extracted from 133 whole-body KPs)
  --dataset moof   : MOOF dataset                     (3 KP/foot)

Output schema (18 KP, COCO format):
  Indices 0–8  : left foot  (big_toe … upper_heel)
  Indices 9–17 : right foot (big_toe … upper_heel)

For source datasets with only 3 KP/foot, the remaining 6 KP per foot are set to
  (x=0, y=0, v=0)  — "not labeled".

Usage
-----
  python convert_to_schema.py --dataset cmu  --input /data/cmu  --output /data/cmu_18kp.json
  python convert_to_schema.py --dataset h3wb --input /data/h3wb --output /data/h3wb_18kp.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Unified schema metadata
# ──────────────────────────────────────────────────────────────────────────────

CATEGORY = {
    "id": 1,
    "name": "foot",
    "supercategory": "person",
    "keypoints": [
        "left_big_toe",        # 0
        "left_little_toe",     # 1
        "left_near_little_toe",# 2
        "left_near_big_toe",   # 3
        "left_far_big_toe",    # 4
        "left_far_little_toe", # 5
        "left_dorsum",         # 6
        "left_heel",           # 7
        "left_upper_heel",     # 8
        "right_big_toe",       # 9
        "right_little_toe",    # 10
        "right_near_little_toe",# 11
        "right_near_big_toe",  # 12
        "right_far_big_toe",   # 13
        "right_far_little_toe",# 14
        "right_dorsum",        # 15
        "right_heel",          # 16
        "right_upper_heel",    # 17
    ],
    "skeleton": [
        # Left foot connections
        [0, 1], [1, 2], [2, 5], [5, 4], [4, 3], [3, 0],
        [0, 7], [7, 8], [6, 7], [3, 6], [4, 6],
        # Right foot connections
        [9, 10], [10, 11], [11, 14], [14, 13], [13, 12], [12, 9],
        [9, 16], [16, 17], [15, 16], [12, 15], [13, 15],
    ],
}


def empty_keypoints() -> List[float]:
    """Return 18 × (0, 0, 0) — all keypoints unlabeled."""
    return [0.0] * (18 * 3)


def set_kp(kps: List[float], idx: int, x: float, y: float, v: int) -> None:
    """Set keypoint at *idx* in the flat list."""
    kps[idx * 3]     = x
    kps[idx * 3 + 1] = y
    kps[idx * 3 + 2] = v


# ──────────────────────────────────────────────────────────────────────────────
# CMU Foot Keypoint Dataset converter
# ──────────────────────────────────────────────────────────────────────────────
# CMU annotation format (assumed JSON with COCO-like structure):
#   keypoints per instance: [x_big_toe_l, y_big_toe_l, v, x_small_toe_l, ...]
#   KP order: left_big_toe(0), left_small_toe(1), left_heel(2),
#             right_big_toe(3), right_small_toe(4), right_heel(5)

CMU_TO_OURS = {
    # CMU index → our index
    0: 0,   # left_big_toe   → left_big_toe   (0)
    1: 1,   # left_small_toe → left_little_toe (1)
    2: 7,   # left_heel      → left_heel       (7)
    3: 9,   # right_big_toe  → right_big_toe   (9)
    4: 10,  # right_small_toe→ right_little_toe(10)
    5: 16,  # right_heel     → right_heel      (16)
}


def convert_cmu(input_dir: str) -> Tuple[List[dict], List[dict]]:
    """
    Parse CMU foot dataset and return (images, annotations) in our format.

    Expected directory structure:
      input_dir/
        annotations/
          train.json   (COCO-like, 6 KP/person)
          val.json
        images/
          *.jpg
    """
    images, annotations = [], []
    ann_id = 0

    for split_file in ["train.json", "val.json"]:
        fpath = os.path.join(input_dir, "annotations", split_file)
        if not os.path.exists(fpath):
            print(f"[WARN] CMU file not found: {fpath} — skipping.")
            continue

        with open(fpath) as f:
            coco_data = json.load(f)

        # Collect unique image ids already added
        seen_img_ids = {img["id"] for img in images}
        for img in coco_data["images"]:
            if img["id"] not in seen_img_ids:
                images.append(img)
                seen_img_ids.add(img["id"])

        for ann in tqdm(coco_data["annotations"], desc=f"CMU {split_file}"):
            raw_kps = ann.get("keypoints", [])
            if len(raw_kps) < 6 * 3:
                continue

            new_kps = empty_keypoints()
            for cmu_idx, our_idx in CMU_TO_OURS.items():
                x = raw_kps[cmu_idx * 3]
                y = raw_kps[cmu_idx * 3 + 1]
                v = int(raw_kps[cmu_idx * 3 + 2])
                set_kp(new_kps, our_idx, x, y, v)

            num_kps = sum(1 for i in range(18) if new_kps[i * 3 + 2] > 0)
            if num_kps == 0:
                continue

            annotations.append({
                "id":            ann_id,
                "image_id":      ann["image_id"],
                "category_id":   1,
                "keypoints":     new_kps,
                "num_keypoints": num_kps,
                "bbox":          ann.get("bbox", [0, 0, 1, 1]),
                "area":          ann.get("area", 1.0),
                "iscrowd":       0,
            })
            ann_id += 1

    return images, annotations


# ──────────────────────────────────────────────────────────────────────────────
# H3WB converter
# ──────────────────────────────────────────────────────────────────────────────
# H3WB has 133 whole-body KPs.  Foot KP indices (0-based within the 133):
#   left_big_toe=17, left_small_toe=20, left_heel=21
#   right_big_toe=18, right_small_toe=22, right_heel=23   (approx — verify with actual dataset)
# Reference: wholebody3d GitHub

H3WB_FOOT_INDICES = {
    # H3WB idx → our schema idx
    17: 0,   # left_big_toe
    20: 1,   # left_little_toe
    21: 7,   # left_heel
    18: 9,   # right_big_toe
    22: 10,  # right_little_toe
    23: 16,  # right_heel
}


def convert_h3wb(input_dir: str) -> Tuple[List[dict], List[dict]]:
    """
    Parse H3WB dataset and extract foot keypoints only.

    Expected directory structure:
      input_dir/
        annotations/
          h3wb_train.json   (COCO-like, 133 KP/person)
          h3wb_val.json
        images/
    """
    images, annotations = [], []
    ann_id = 0
    seen_img_ids: set = set()

    for split_file in ["h3wb_train.json", "h3wb_val.json"]:
        fpath = os.path.join(input_dir, "annotations", split_file)
        if not os.path.exists(fpath):
            print(f"[WARN] H3WB file not found: {fpath} — skipping.")
            continue

        with open(fpath) as f:
            coco_data = json.load(f)

        for img in coco_data["images"]:
            if img["id"] not in seen_img_ids:
                images.append(img)
                seen_img_ids.add(img["id"])

        for ann in tqdm(coco_data["annotations"], desc=f"H3WB {split_file}"):
            raw_kps = ann.get("keypoints", [])
            if len(raw_kps) < 133 * 3:
                continue

            new_kps = empty_keypoints()
            for h3wb_idx, our_idx in H3WB_FOOT_INDICES.items():
                x = raw_kps[h3wb_idx * 3]
                y = raw_kps[h3wb_idx * 3 + 1]
                v = int(raw_kps[h3wb_idx * 3 + 2])
                set_kp(new_kps, our_idx, x, y, v)

            num_kps = sum(1 for i in range(18) if new_kps[i * 3 + 2] > 0)
            if num_kps == 0:
                continue

            # Compute bbox from visible foot keypoints
            xs = [new_kps[i*3]   for i in range(18) if new_kps[i*3+2] > 0]
            ys = [new_kps[i*3+1] for i in range(18) if new_kps[i*3+2] > 0]
            if not xs:
                continue
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            padding = 20
            bbox = [max(0, x1 - padding), max(0, y1 - padding),
                    (x2 - x1) + 2 * padding, (y2 - y1) + 2 * padding]

            annotations.append({
                "id":            ann_id,
                "image_id":      ann["image_id"],
                "category_id":   1,
                "keypoints":     new_kps,
                "num_keypoints": num_kps,
                "bbox":          bbox,
                "area":          bbox[2] * bbox[3],
                "iscrowd":       0,
            })
            ann_id += 1

    return images, annotations


# ──────────────────────────────────────────────────────────────────────────────
# Save to COCO JSON
# ──────────────────────────────────────────────────────────────────────────────

def save_coco(images: List[dict], annotations: List[dict], output_path: str) -> None:
    coco_out = {
        "info":        {"description": "Foot Keypoint Dataset (18 KP unified schema)"},
        "licenses":    [],
        "images":      images,
        "annotations": annotations,
        "categories":  [CATEGORY],
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_out, f)
    print(f"Saved {len(images)} images, {len(annotations)} annotations → {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert public foot datasets to unified 18-KP COCO format."
    )
    parser.add_argument(
        "--dataset", required=True, choices=["cmu", "h3wb", "moof"],
        help="Source dataset name."
    )
    parser.add_argument("--input",  required=True, help="Root directory of source dataset.")
    parser.add_argument("--output", required=True, help="Output JSON file path.")
    args = parser.parse_args()

    converters = {
        "cmu":  convert_cmu,
        "h3wb": convert_h3wb,
        "moof": convert_cmu,   # MOOF has similar 3-KP structure — reuse CMU converter
    }

    convert_fn = converters[args.dataset]
    images, annotations = convert_fn(args.input)

    if not annotations:
        print("[ERROR] No annotations produced. Check input directory structure.")
        return

    save_coco(images, annotations, args.output)


if __name__ == "__main__":
    main()

