"""
filter_dataset_keypoints.py
───────────────────────────
Reusable script to remove specified keypoints from YOLO-pose datasets.
Transforms 16-keypoint annotations to 13-keypoint (or custom) schema,
generates the updated data.yaml, and produces keypoint_meta.json.

Default removed keypoints:
  - ball_top     (KP 5)
  - ankle_center (KP 12)
  - shin_mid     (KP 15)

Usage:
  python tools/filter_dataset_keypoints.py \
      --src_dir data/shuffled_v3 \
      --dst_dir data/shuffled_v4_13kp \
      --remove_indices 5,12,15
"""

import os
import sys
import shutil
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Official 16-keypoint reference schema
REFERENCE_16KP = [
    {"index": 0, "name": "toe_ground", "zone": "sole", "tier": 1},
    {"index": 1, "name": "heel_back", "zone": "sole", "tier": 1},
    {"index": 2, "name": "heel_ground", "zone": "sole", "tier": 1},
    {"index": 3, "name": "ball_medial", "zone": "ball", "tier": 1},
    {"index": 4, "name": "ball_lateral", "zone": "ball", "tier": 1},
    {"index": 5, "name": "ball_top", "zone": "ball", "tier": 1},
    {"index": 6, "name": "instep_top", "zone": "mid", "tier": 1},
    {"index": 7, "name": "arch_medial", "zone": "mid", "tier": 2},
    {"index": 8, "name": "midfoot_lateral", "zone": "mid", "tier": 2},
    {"index": 9, "name": "malleolus_medial", "zone": "ankle", "tier": 1},
    {"index": 10, "name": "malleolus_lateral", "zone": "ankle", "tier": 1},
    {"index": 11, "name": "toe_tip", "zone": "sole", "tier": 1},
    {"index": 12, "name": "ankle_center", "zone": "ankle", "tier": 1},
    {"index": 13, "name": "throat", "zone": "ankle", "tier": 1},
    {"index": 14, "name": "achilles", "zone": "ankle", "tier": 1},
    {"index": 15, "name": "shin_mid", "zone": "leg", "tier": 1},
]


def build_schema_mapping(remove_indices: List[int]) -> Tuple[List[Dict], List[int]]:
    """
    Computes surviving keypoints and old-to-new index mapping.
    """
    remove_set = set(remove_indices)
    surviving_schema = []
    old_to_new = []

    new_idx = 0
    for old_item in REFERENCE_16KP:
        old_idx = old_item["index"]
        if old_idx not in remove_set:
            new_item = dict(old_item)
            new_item["old_index"] = old_idx
            new_item["index"] = new_idx
            surviving_schema.append(new_item)
            old_to_new.append(old_idx)
            new_idx += 1

    return surviving_schema, old_to_new


def filter_label_line(line: str, surviving_old_indices: List[int], original_kp_count: int = 16) -> str:
    """
    Filters a single YOLO pose line:
    class cx cy w h (x0 y0 v0) (x1 y1 v1) ...
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return ""

    cls = parts[0]
    bbox = parts[1:5]
    raw_kpts = parts[5:]

    expected_raw_len = original_kp_count * 3
    if len(raw_kpts) < expected_raw_len:
        # Malformed or different keypoint format
        return ""

    # Parse keypoint triplets
    kpt_triplets = []
    for i in range(original_kp_count):
        x = raw_kpts[i * 3]
        y = raw_kpts[i * 3 + 1]
        v = raw_kpts[i * 3 + 2]
        kpt_triplets.append((x, y, v))

    # Keep only surviving keypoints
    new_kpts = []
    for old_idx in surviving_old_indices:
        if old_idx < len(kpt_triplets):
            x, y, v = kpt_triplets[old_idx]
            new_kpts.extend([x, y, v])
        else:
            new_kpts.extend(["0.000000", "0.000000", "0"])

    return " ".join([cls] + bbox + new_kpts)


def process_dataset(
    src_dir: Path,
    dst_dir: Path,
    remove_indices: List[int],
    copy_images: bool = True
) -> Dict:
    """
    Iterates through dataset splits, filters annotations, copies/links images,
    and creates data.yaml and keypoint_meta.json.
    """
    surviving_schema, surviving_old_indices = build_schema_mapping(remove_indices)
    num_surviving = len(surviving_schema)
    expected_line_tokens = 5 + num_surviving * 3

    print(f"[*] Original keypoints: 16")
    print(f"[*] Removing keypoints: {remove_indices}")
    print(f"[*] Surviving keypoints ({num_surviving}):")
    for s in surviving_schema:
        print(f"    New [{s['index']:2d}] <- Old [{s['old_index']:2d}]: {s['name']} ({s['zone']})")

    dst_dir.mkdir(parents=True, exist_ok=True)

    # Detect splits: look for train/valid/test or train/val/test
    splits = []
    for cand in ["train", "valid", "val", "test"]:
        if (src_dir / cand).exists():
            splits.append(cand)

    if not splits:
        # Check if direct labels/ and images/ exist
        if (src_dir / "labels").exists():
            splits = ["."]

    stats = {
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "total_boxes": 0
    }

    for split in splits:
        split_src = src_dir / split if split != "." else src_dir
        split_dst = dst_dir / ("valid" if split == "val" else split) if split != "." else dst_dir

        src_img_dir = split_src / "images" if (split_src / "images").exists() else split_src
        src_lbl_dir = split_src / "labels" if (split_src / "labels").exists() else split_src

        dst_img_dir = split_dst / "images"
        dst_lbl_dir = split_dst / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        split_images = 0
        split_labels = 0
        split_boxes = 0

        # Process each label file
        for lbl_file in src_lbl_dir.glob("*.txt"):
            lines = lbl_file.read_text(encoding="utf-8").splitlines()
            filtered_lines = []

            for line in lines:
                f_line = filter_label_line(line, surviving_old_indices, original_kp_count=16)
                if f_line:
                    tokens = f_line.split()
                    if len(tokens) == expected_line_tokens:
                        filtered_lines.append(f_line)
                        split_boxes += 1

            if not filtered_lines:
                continue

            # Write filtered label
            out_lbl_file = dst_lbl_dir / lbl_file.name
            out_lbl_file.write_text("\n".join(filtered_lines), encoding="utf-8")
            split_labels += 1

            # Match and copy corresponding image
            stem = lbl_file.stem
            found_img = None
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"]:
                cand_img = src_img_dir / f"{stem}{ext}"
                if cand_img.exists():
                    found_img = cand_img
                    break

            if found_img:
                out_img_file = dst_img_dir / found_img.name
                if not out_img_file.exists():
                    if copy_images:
                        shutil.copy2(found_img, out_img_file)
                    else:
                        try:
                            os.link(found_img, out_img_file)
                        except OSError:
                            shutil.copy2(found_img, out_img_file)
                split_images += 1

        stats["splits"][split] = {
            "images": split_images,
            "labels": split_labels,
            "boxes": split_boxes
        }
        stats["total_images"] += split_images
        stats["total_labels"] += split_labels
        stats["total_boxes"] += split_boxes
        print(f"  [+] Split '{split}': {split_images} images, {split_labels} label files, {split_boxes} foot instances")

    # Generate data.yaml
    # Standardize train/val/test paths
    train_split = "train" if (dst_dir / "train").exists() else "."
    val_split = "valid" if (dst_dir / "valid").exists() else ("val" if (dst_dir / "val").exists() else train_split)
    test_split = "test" if (dst_dir / "test").exists() else val_split

    flip_idx = list(range(num_surviving))

    yaml_text = f"""path: {dst_dir.resolve().as_posix()}
train: {train_split}/images
val: {val_split}/images
test: {test_split}/images

nc: 2
names:
  - left_foot
  - right_foot

kpt_shape: [{num_surviving}, 3]
flip_idx: {flip_idx}
"""
    (dst_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # Generate keypoint_meta.json
    meta = {
        "version": f"v4_{num_surviving}kp",
        "num_keypoints": num_surviving,
        "kpt_shape": [num_surviving, 3],
        "removed_original_indices": remove_indices,
        "keypoints": surviving_schema,
        "classes": {
            "0": "left_foot",
            "1": "right_foot"
        },
        "stats": stats
    }
    (dst_dir / "keypoint_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n[OK] Successfully generated {num_surviving}-KP dataset at: {dst_dir}")
    print(f"    - YAML Config: {dst_dir / 'data.yaml'}")
    print(f"    - Metadata:    {dst_dir / 'keypoint_meta.json'}")
    print(f"    - Total Foot Instances: {stats['total_boxes']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter out specified keypoints from YOLO-pose dataset.")
    parser.add_argument("--src_dir", type=str, default="data/shuffled_v3",
                        help="Source YOLO-pose dataset directory")
    parser.add_argument("--dst_dir", type=str, default="data/shuffled_v4_13kp",
                        help="Target filtered dataset directory")
    parser.add_argument("--remove_indices", type=str, default="5,12,15",
                        help="Comma-separated 0-indexed keypoints to remove (default: 5,12,15 for ball_top, ankle_center, shin_mid)")
    parser.add_argument("--no_copy_images", action="store_true",
                        help="Use hardlinks instead of file copying where supported")
    args = parser.parse_args()

    remove_indices = [int(x.strip()) for x in args.remove_indices.split(",") if x.strip()]
    src_path = Path(args.src_dir)
    dst_path = Path(args.dst_dir)

    if not src_path.exists():
        print(f"[!] Error: Source directory does not exist: {src_path}")
        sys.exit(1)

    process_dataset(
        src_dir=src_path,
        dst_dir=dst_path,
        remove_indices=remove_indices,
        copy_images=not args.no_copy_images
    )


if __name__ == "__main__":
    main()
