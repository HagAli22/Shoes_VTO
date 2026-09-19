"""
prepare_stage_a_data.py
───────────────────────
Prepares shoes_v1 dataset for Stage A training.

What this does:
  1. Fixes class label swap: Foot_Right(0)/Foot_left(1) → left_foot(0)/right_foot(1)
  2. Copies images and fixed labels into data/stage_a/
  3. Writes a clean data.yaml for YOLOv8n-pose training

Run once before training:
  python src/models/stage_a/prepare_stage_a_data.py
"""

import os
import shutil

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_ROOT = r"F:\Machine learning\Shoes_VTO\data\shoes_v2"
DST_ROOT = r"F:\Machine learning\Shoes_VTO\data\stage_a"


def fix_and_copy_labels(src_lbl_dir: str, dst_lbl_dir: str) -> dict:
    """
    Copy label files with class indices swapped:
        0 (Foot_Right) → 1 (right_foot)
        1 (Foot_left)  → 0 (left_foot)
    Keypoint coordinates and visibility are untouched.
    Returns counts of fixed instances per class.
    """
    os.makedirs(dst_lbl_dir, exist_ok=True)
    counts = {0: 0, 1: 0}

    for fname in os.listdir(src_lbl_dir):
        if not fname.endswith(".txt"):
            continue
        src_path = os.path.join(src_lbl_dir, fname)
        dst_path = os.path.join(dst_lbl_dir, fname)

        with open(src_path, "r") as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            old_cls = int(parts[0])
            # Swap: 0 ↔ 1
            new_cls = 1 if old_cls == 0 else 0
            parts[0] = str(new_cls)
            counts[new_cls] = counts.get(new_cls, 0) + 1
            fixed_lines.append(" ".join(parts) + "\n")

        with open(dst_path, "w") as f:
            f.writelines(fixed_lines)

    return counts


def copy_images(src_img_dir: str, dst_img_dir: str) -> int:
    os.makedirs(dst_img_dir, exist_ok=True)
    count = 0
    for fname in os.listdir(src_img_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            shutil.copy2(
                os.path.join(src_img_dir, fname),
                os.path.join(dst_img_dir, fname),
            )
            count += 1
    return count


def write_data_yaml(dst_root: str) -> None:
    """Write Stage A data.yaml with corrected class names and 16-KP schema."""

    # flip_idx: anatomical KPs — indices are identity (same physical point
    # when image is mirrored), only class label flips (§7 of requirements).
    flip_idx = list(range(16))

    yaml_content = f"""# Stage A — Foot Detector
# Class 0 = left_foot | Class 1 = right_foot
# 16 keypoints per foot (full schema for maximum supervision signal)
# flip_idx is identity: only class label flips on horizontal mirror (§7)

path: {dst_root.replace(chr(92), "/")}
train: train/images
val:   valid/images
test:  test/images

nc: 2
names:
  - left_foot
  - right_foot

kpt_shape: [16, 3]

flip_idx: {flip_idx}
"""
    with open(os.path.join(dst_root, "data.yaml"), "w") as f:
        f.write(yaml_content)


def main() -> None:
    print("=" * 60)
    print("  Stage A — Data Preparation")
    print("=" * 60)

    total_imgs = 0
    total_fixed = {0: 0, 1: 0}

    for split in ["train", "valid", "test"]:
        src_img = os.path.join(SRC_ROOT, split, "images")
        src_lbl = os.path.join(SRC_ROOT, split, "labels")
        dst_img = os.path.join(DST_ROOT, split, "images")
        dst_lbl = os.path.join(DST_ROOT, split, "labels")

        n_imgs = copy_images(src_img, dst_img)
        counts = fix_and_copy_labels(src_lbl, dst_lbl)
        total_imgs += n_imgs
        for k, v in counts.items():
            total_fixed[k] = total_fixed.get(k, 0) + v

        print(
            f"  {split:6}: {n_imgs} images  |  "
            f"left_foot(0)={counts.get(0,0)}  right_foot(1)={counts.get(1,0)}"
        )

    write_data_yaml(DST_ROOT)
    print()
    print(f"  Total images   : {total_imgs}")
    print(f"  Total left_foot (0) : {total_fixed.get(0,0)}")
    print(f"  Total right_foot(1) : {total_fixed.get(1,0)}")
    print(f"  data.yaml written to: {DST_ROOT}/data.yaml")
    print()
    print("  Class fix applied:")
    print("    OLD: 0=Foot_Right, 1=Foot_left")
    print("    NEW: 0=left_foot,  1=right_foot  (matches §5 frozen spec)")
    print("=" * 60)


if __name__ == "__main__":
    main()

