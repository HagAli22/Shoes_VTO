"""
prepare_shuffled_dataset.py
───────────────────────────
Pools images and labels from raw Roboflow downloads (or specified folders),
repairs any out-of-bounds / negative coordinates by clamping to [0, 1],
remaps classes (0<->1 for Roboflow raw convention),
and performs a stratified random shuffle into 80% train / 10% valid / 10% test.

Usage:
    # Prepare V3 dataset
    python tools/prepare_shuffled_dataset.py --source data/shoes_v3_raw --dest data/shuffled_v3 --remap

    # Prepare V2 dataset
    python tools/prepare_shuffled_dataset.py --source data/shoes_v2_raw --dest data/shoes_v2 --remap

    # Auto mode (pools from all default paths)
    python tools/prepare_shuffled_dataset.py
"""

import os
import glob
import random
import shutil
import argparse
from pathlib import Path

random.seed(42)

def clean_and_repair_label(line: str, remap_classes: bool = False) -> str:
    parts = line.strip().split()
    if len(parts) < 5:
        return ""
    try:
        cls = int(float(parts[0]))
        if remap_classes:
            # Roboflow raw convention: 0=Foot_Right, 1=Foot_left -> Remap to 0=left_foot, 1=right_foot
            cls = 1 if cls == 0 else 0

        cx = min(max(float(parts[1]), 0.0), 1.0)
        cy = min(max(float(parts[2]), 0.0), 1.0)
        w = min(max(float(parts[3]), 0.001), 1.0)
        h = min(max(float(parts[4]), 0.001), 1.0)
        
        repaired = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
        
        if len(parts) >= 5 + 48:
            for i in range(16):
                kx = min(max(float(parts[5 + i*3]), 0.0), 1.0)
                ky = min(max(float(parts[6 + i*3]), 0.0), 1.0)
                kv = int(float(parts[7 + i*3]))
                kv = min(max(kv, 0), 2)
                repaired.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
        elif len(parts) >= 5 + 12:
            for i in range(4):
                kx = min(max(float(parts[5 + i*3]), 0.0), 1.0)
                ky = min(max(float(parts[6 + i*3]), 0.0), 1.0)
                kv = int(float(parts[7 + i*3]))
                kv = min(max(kv, 0), 2)
                repaired.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
        return " ".join(repaired)
    except Exception:
        return ""


def process_dataset_folder(src_folder: str, dst_folder: str, remap: bool = True):
    dst_root = Path(dst_folder)
    if dst_root.exists():
        shutil.rmtree(dst_root)

    for split in ["train", "valid", "test"]:
        (dst_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / split / "labels").mkdir(parents=True, exist_ok=True)

    src_root = Path(src_folder)
    image_pairs = []
    seen_names = set()

    # Search in all subfolders of src_root
    for img_path in src_root.rglob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            continue
        
        # Try finding corresponding label in sibling labels folder or same folder
        lbl_path = None
        candidates = [
            img_path.parent.parent / "labels" / f"{img_path.stem}.txt",
            img_path.parent / "labels" / f"{img_path.stem}.txt",
            img_path.parent / f"{img_path.stem}.txt",
        ]
        for cand in candidates:
            if cand.exists():
                lbl_path = cand
                break
        
        if lbl_path is not None and img_path.name not in seen_names:
            seen_names.add(img_path.name)
            image_pairs.append((img_path, lbl_path, remap))

    print(f"[PREPARE] Found {len(image_pairs)} unique image-label pairs in {src_folder}")
    if len(image_pairs) == 0:
        print(f"[!] Warning: No images found in {src_folder}")
        return 0

    random.shuffle(image_pairs)
    n_total = len(image_pairs)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)
    
    splits_dict = {
        "train": image_pairs[:n_train],
        "valid": image_pairs[n_train:n_train + n_val],
        "test": image_pairs[n_train + n_val:]
    }

    total_saved = 0
    for split_name, pairs in splits_dict.items():
        dst_img_dir = dst_root / split_name / "images"
        dst_lbl_dir = dst_root / split_name / "labels"
        
        saved_count = 0
        for img_src, lbl_src, remap_flag in pairs:
            raw_lines = lbl_src.read_text(encoding="utf-8").splitlines()
            cleaned_lines = []
            for l in raw_lines:
                cl = clean_and_repair_label(l, remap_classes=remap_flag)
                if cl:
                    cleaned_lines.append(cl)
            
            if not cleaned_lines:
                continue

            dst_img = dst_img_dir / img_src.name
            dst_lbl = dst_lbl_dir / lbl_src.name
            
            try:
                shutil.copy2(img_src, dst_img)
                dst_lbl.write_text("\n".join(cleaned_lines), encoding="utf-8")
                saved_count += 1
            except Exception as e:
                print(f"Error copying {img_src}: {e}")
                continue
                
        print(f"  Split '{split_name}': {saved_count} images saved")
        total_saved += saved_count

    yaml_content = f"""path: {dst_root.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: left_foot
  1: right_foot

kpt_shape: [16, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
"""
    (dst_root / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"[OK] Shuffled dataset successfully created at: {dst_root}/data.yaml ({total_saved} samples)\n")
    return total_saved


def main():
    parser = argparse.ArgumentParser(description="Clean, repair, and shuffle Roboflow datasets.")
    parser.add_argument("--source", type=str, default=None, help="Source directory (e.g. data/shoes_v3_raw)")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory (e.g. data/shuffled_v3)")
    parser.add_argument("--remap", action="store_true", default=True, help="Remap Roboflow classes (0<->1)")
    args = parser.parse_args()

    if args.source and args.dest:
        process_dataset_folder(args.source, args.dest, remap=args.remap)
    else:
        # Default auto pool for V3
        if Path("data/shoes_v3_raw").exists():
            process_dataset_folder("data/shoes_v3_raw", "data/shuffled_v3", remap=True)
        elif Path("data/shoes_v3").exists():
            process_dataset_folder("data/shoes_v3", "data/shuffled_v3", remap=True)
        elif Path("data/stage_a").exists():
            process_dataset_folder("data/stage_a", "data/shuffled_v3", remap=False)
        else:
            print("[!] No default dataset sources found. Please specify --source and --dest.")


if __name__ == "__main__":
    main()
