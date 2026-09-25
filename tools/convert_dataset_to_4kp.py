"""
convert_dataset_to_4kp.py
─────────────────────────
Converts 16-Keypoint YOLO Pose datasets (e.g. data/shoes_v2 and data/shuffled_v3)
into 4-Keypoint Native YOLO Pose datasets for Stage 1 coarse keypoint estimation.

4 Coarse Keypoints:
  KP 0: toe_tip       (extracted from 16-KP index 0, fallback index 1)
  KP 1: heel          (extracted from 16-KP index 2, fallback index 3/13/14)
  KP 2: ball_medial   (extracted from 16-KP index 4, fallback index 7)
  KP 3: ball_lateral  (extracted from 16-KP index 5, fallback index 8)

Classes:
  0: left_foot
  1: right_foot

Flip Index for Data Augmentation:
  [0, 1, 3, 2]  (toe_tip and heel stay, ball_medial <-> ball_lateral swap)

Usage:
  python tools/convert_dataset_to_4kp.py --all
  python tools/convert_dataset_to_4kp.py --src data/shoes_v2 --dst data/shoes_v2_4kp --remap
  python tools/convert_dataset_to_4kp.py --src data/shuffled_v3 --dst data/shuffled_v3_4kp
"""

import os
import shutil
import argparse
from pathlib import Path


def convert_label_line_16_to_4(line: str, remap_classes: bool = False) -> str:
    """
    Converts a single label line from 16-KP format (53 items) to 4-KP format (17 items).
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return ""
    
    try:
        cls_id = int(float(parts[0]))
        if remap_classes:
            # Roboflow raw convention: 0=Foot_Right, 1=Foot_left -> Remap to 0=left_foot, 1=right_foot
            cls_id = 0 if cls_id == 1 else 1
            
        cx = min(max(float(parts[1]), 0.0), 1.0)
        cy = min(max(float(parts[2]), 0.0), 1.0)
        w = min(max(float(parts[3]), 0.001), 1.0)
        h = min(max(float(parts[4]), 0.001), 1.0)
        
        # Extract 16 keypoints [(x, y, v), ...]
        kpts_16 = []
        if len(parts) >= 5 + 48:
            for i in range(16):
                kx = min(max(float(parts[5 + i*3]), 0.0), 1.0)
                ky = min(max(float(parts[6 + i*3]), 0.0), 1.0)
                kv = int(float(parts[7 + i*3]))
                kv = min(max(kv, 0), 2)
                kpts_16.append((kx, ky, kv))
        elif len(parts) >= 5 + 12:
            # Already 4 keypoints
            for i in range(4):
                kx = min(max(float(parts[5 + i*3]), 0.0), 1.0)
                ky = min(max(float(parts[6 + i*3]), 0.0), 1.0)
                kv = int(float(parts[7 + i*3]))
                kv = min(max(kv, 0), 2)
                kpts_16.append((kx, ky, kv))
            # Return directly
            repaired = [str(cls_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            for kx, ky, kv in kpts_16:
                repaired.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
            return " ".join(repaired)
        # Helper to pick keypoint with safe fallbacks
        def get_kp(primary_idx, fallbacks=()):
            if primary_idx < len(kpts_16) and kpts_16[primary_idx][2] > 0:
                return kpts_16[primary_idx]
            for fb in fallbacks:
                if fb < len(kpts_16) and kpts_16[fb][2] > 0:
                    return kpts_16[fb]
            if primary_idx < len(kpts_16):
                return kpts_16[primary_idx]
            return (0.0, 0.0, 0)

        # Exact 4 anatomical perimeter corners of the foot:
        # 0: Toe Tip      -> Index 0 (fallback Index 11)
        # 1: Heel         -> Index 2 (fallback Index 12, 13, 14)
        # 2: Ball Medial  -> Index 3 (fallback Index 7)
        # 3: Ball Lateral -> Index 4 (fallback Index 8, 9)
        kp0 = get_kp(0, (11, 1))          # toe_tip (Front)
        kp1 = get_kp(2, (12, 13, 14))     # heel (Back - heel_ground/heel_back/achilles)
        kp2 = get_kp(3, (7,))             # ball_medial (Inner ball)
        kp3 = get_kp(4, (8, 9))           # ball_lateral (Outer ball)

        four_kpts = [kp0, kp1, kp2, kp3]

        repaired = [str(cls_id), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
        for kx, ky, kv in four_kpts:
            repaired.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
            
        return " ".join(repaired)
    except Exception as e:
        return ""


def convert_dataset(src_dir: Path, dst_dir: Path, remap_classes: bool = False):
    """
    Copies images and converts label files from 16-KP to 4-KP.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    print(f"\n[CONVERT] Converting Dataset: {src_dir} -> {dst_dir}")
    print(f"   Class Remap (0<->1): {remap_classes}")

    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test", "val"]
    total_converted = 0

    for split in splits:
        src_split = src_dir / split
        if not src_split.exists():
            continue
        
        # Standardize 'val' to 'valid' or vice-versa
        dst_split_name = "valid" if split in ["valid", "val"] else split
        dst_img_dir = dst_dir / dst_split_name / "images"
        dst_lbl_dir = dst_dir / dst_split_name / "labels"
        
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        src_img_dir = src_split / "images" if (src_split / "images").exists() else src_split
        src_lbl_dir = src_split / "labels" if (src_split / "labels").exists() else src_split

        image_files = [f for f in src_img_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]]
        split_count = 0

        for img_path in image_files:
            lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                lbl_path = src_img_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            raw_lines = lbl_path.read_text(encoding="utf-8").splitlines()
            converted_lines = []
            for line in raw_lines:
                c_line = convert_label_line_16_to_4(line, remap_classes=remap_classes)
                if c_line:
                    converted_lines.append(c_line)

            if not converted_lines:
                continue

            # Copy image and write converted label
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            (dst_lbl_dir / f"{img_path.stem}.txt").write_text("\n".join(converted_lines), encoding="utf-8")
            split_count += 1

        print(f"   - Split '{dst_split_name}': {split_count} images converted")
        total_converted += split_count

    # Create dataset data.yaml
    yaml_content = f"""path: {dst_dir.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: left_foot
  1: right_foot

kpt_shape: [4, 3]
flip_idx: [0, 1, 3, 2]
"""
    (dst_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"[OK] Created {dst_dir / 'data.yaml'} ({total_converted} total samples)\n")
    return total_converted


def create_standalone_configs():
    """
    Creates standalone YAML config files in configs/ directory.
    """
    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    shoes_v2_yaml = """path: data/shoes_v2_4kp
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: left_foot
  1: right_foot

kpt_shape: [4, 3]
flip_idx: [0, 1, 3, 2]
"""
    shuffled_v3_yaml = """path: data/shuffled_v3_4kp
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: left_foot
  1: right_foot

kpt_shape: [4, 3]
flip_idx: [0, 1, 3, 2]
"""
    (configs_dir / "shoes_v2_4kp.yaml").write_text(shoes_v2_yaml, encoding="utf-8")
    (configs_dir / "shuffled_v3_4kp.yaml").write_text(shuffled_v3_yaml, encoding="utf-8")
    print("[OK] Created configs/shoes_v2_4kp.yaml and configs/shuffled_v3_4kp.yaml")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO pose dataset to 4 coarse keypoints format.")
    parser.add_argument("--src", type=str, default=None, help="Source dataset directory")
    parser.add_argument("--dst", type=str, default=None, help="Destination dataset directory")
    parser.add_argument("--remap", action="store_true", help="Remap classes (0<->1 for Roboflow raw)")
    parser.add_argument("--all", action="store_true", help="Convert both shoes_v2 and shuffled_v3 automatically")

    args = parser.parse_args()

    if args.all or (args.src is None and args.dst is None):
        print("=" * 70)
        print("  Converting all datasets to 4-KP Format (shoes_v2 & shuffled_v3)")
        print("=" * 70)
        
        # 1. shoes_v2 (needs remap)
        if Path("data/shoes_v2").exists():
            convert_dataset(Path("data/shoes_v2"), Path("data/shoes_v2_4kp"), remap_classes=True)
            
        # 2. shuffled_v3 (already standardized)
        if Path("data/shuffled_v3").exists():
            convert_dataset(Path("data/shuffled_v3"), Path("data/shuffled_v3_4kp"), remap_classes=False)

        create_standalone_configs()
    else:
        convert_dataset(Path(args.src), Path(args.dst), remap_classes=args.remap)
        create_standalone_configs()


if __name__ == "__main__":
    main()
