"""
prepare_shuffled_dataset.py
---------------------------
Pools all images and labels from stage_a and shoes_v3_remapped,
repairs any out-of-bounds / negative coordinates by clamping to [0, 1],
and performs a stratified random shuffle into 80% train / 10% val / 10% test.
"""

import os
import glob
import random
import shutil
from pathlib import Path

random.seed(42)

def clean_and_repair_label(line: str) -> str:
    parts = line.strip().split()
    if len(parts) < 5:
        return ""
    try:
        cls = int(parts[0])
        cx = min(max(float(parts[1]), 0.0), 1.0)
        cy = min(max(float(parts[2]), 0.0), 1.0)
        w = min(max(float(parts[3]), 0.0), 1.0)
        h = min(max(float(parts[4]), 0.0), 1.0)
        
        repaired = [str(cls), f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
        
        if len(parts) >= 5 + 48:
            for i in range(16):
                kx = min(max(float(parts[5 + i*3]), 0.0), 1.0)
                ky = min(max(float(parts[6 + i*3]), 0.0), 1.0)
                kv = int(float(parts[7 + i*3]))
                kv = min(max(kv, 0), 2)
                repaired.extend([f"{kx:.6f}", f"{ky:.6f}", str(kv)])
        return " ".join(repaired)
    except Exception:
        return ""

def main():
    dst_root = Path("data/shuffled_v3")
    if dst_root.exists():
        shutil.rmtree(dst_root)

    for split in ["train", "valid", "test"]:
        (dst_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dst_root / split / "labels").mkdir(parents=True, exist_ok=True)

    sources = [
        "data/stage_a/train", "data/stage_a/valid", "data/stage_a/test",
        "data/shoes_v3_remapped/train", "data/shoes_v3_remapped/valid", "data/shoes_v3_remapped/test"
    ]

    image_pairs = []
    seen_names = set()

    for src in sources:
        img_dir = Path(src) / "images"
        lbl_dir = Path(src) / "labels"
        if not img_dir.exists():
            continue
        
        for img_path in img_dir.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue
            
            if img_path.name in seen_names:
                continue
            seen_names.add(img_path.name)
            image_pairs.append((img_path, lbl_path))

    print(f"Total unique image-label pairs found: {len(image_pairs)}")
    random.shuffle(image_pairs)

    n_total = len(image_pairs)
    n_train = int(n_total * 0.80)
    n_val = int(n_total * 0.10)
    
    splits_dict = {
        "train": image_pairs[:n_train],
        "valid": image_pairs[n_train:n_train + n_val],
        "test": image_pairs[n_train + n_val:]
    }

    for split_name, pairs in splits_dict.items():
        dst_img_dir = dst_root / split_name / "images"
        dst_lbl_dir = dst_root / split_name / "labels"
        
        saved_count = 0
        for img_src, lbl_src in pairs:
            raw_lines = lbl_src.read_text(encoding="utf-8").splitlines()
            cleaned_lines = []
            for l in raw_lines:
                cl = clean_and_repair_label(l)
                if cl:
                    cleaned_lines.append(cl)
            
            if not cleaned_lines:
                continue

            dst_img = dst_img_dir / img_src.name
            dst_lbl = dst_lbl_dir / lbl_src.name
            
            try:
                os.link(img_src, dst_img)
            except Exception:
                shutil.copy2(img_src, dst_img)
                
            dst_lbl.write_text("\n".join(cleaned_lines), encoding="utf-8")
            saved_count += 1
            
        print(f"  Split '{split_name}': {saved_count} images saved")

    yaml_content = f"""path: {dst_root.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  - left_foot
  - right_foot

kpt_shape: [16, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
"""
    (dst_root / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\n[OK] Shuffled dataset successfully created at: {dst_root}/data.yaml")

if __name__ == "__main__":
    main()

