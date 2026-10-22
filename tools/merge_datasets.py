"""
merge_datasets.py
─────────────────
General, reusable script to merge multiple YOLO-pose dataset versions
(e.g., v3 + v4 + future versions) into a single unified dataset.

Features:
  1. Multi-source pooling: Accepts arbitrary list of source directories (--src_dirs).
  2. Duplicate detection: Hashes images (SHA-256) to remove true duplicate frames.
  3. Collision resolution: Prefixes non-identical images that share filenames.
  4. Split modes:
       - 'pool' (default): Pools all samples and performs stratified 80/10/10 shuffle.
       - 'preserve': Merges train->train, valid->valid, test->test.
  5. Optional --filter_13kp flag to automatically excise keypoints [5, 12, 15] in one step.

Usage examples:
  # Basic merge of v3 and v4:
  python tools/merge_datasets.py \
      --src_dirs data/shuffled_v3 data/shuffled_v4 \
      --dst_dir data/merged_v3_v4_raw

  # Merge and directly filter to 13-KP in one step:
  python tools/merge_datasets.py \
      --src_dirs data/shuffled_v3 data/shuffled_v4 \
      --dst_dir data/merged_v3_v4_13kp \
      --filter_13kp
"""

import os
import sys
import shutil
import hashlib
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Set

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.filter_dataset_keypoints import process_dataset as filter_keypoints_dataset


def compute_file_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash of a file for exact deduplication."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_image_label_pairs(src_dir: Path) -> List[Tuple[Path, Path, str, str]]:
    """
    Finds all valid (image_path, label_path, split_name, src_name) in a dataset folder.
    Supports train/valid/test or flat directories.
    """
    pairs = []
    src_name = src_dir.name
    splits = ["train", "valid", "val", "test"]

    found_splits = [s for s in splits if (src_dir / s).exists()]
    if not found_splits:
        # Check flat images/ and labels/
        if (src_dir / "images").exists() and (src_dir / "labels").exists():
            found_splits = ["."]
        elif (src_dir / "labels").exists():
            found_splits = ["."]
        else:
            # Recursive search for any labels folder
            for lbl_dir in src_dir.rglob("labels"):
                parent_split = lbl_dir.parent.name
                found_splits.append(parent_split)

    for split in found_splits:
        split_path = src_dir / split if split != "." else src_dir
        img_dir = split_path / "images" if (split_path / "images").exists() else split_path
        lbl_dir = split_path / "labels" if (split_path / "labels").exists() else split_path

        if not lbl_dir.exists():
            continue

        for lbl_path in lbl_dir.glob("*.txt"):
            stem = lbl_path.stem
            found_img = None
            for ext in [".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"]:
                cand = img_dir / f"{stem}{ext}"
                if cand.exists():
                    found_img = cand
                    break

            if found_img:
                norm_split = "valid" if split == "val" else split
                pairs.append((found_img, lbl_path, norm_split, src_name))

    return pairs


def merge_datasets(
    src_dirs: List[Path],
    dst_dir: Path,
    split_mode: str = "pool",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
    copy_images: bool = True
) -> Dict:
    random.seed(seed)
    print("=" * 75)
    print("  YOLO DATASET MERGING PIPELINE")
    print(f"  Source Directories: {[str(d) for d in src_dirs]}")
    print(f"  Target Directory:   {dst_dir}")
    print(f"  Split Mode:         {split_mode}")
    print("=" * 75)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect all candidate pairs from all sources
    all_pairs = []
    for sdir in src_dirs:
        if not sdir.exists():
            print(f"[!] Warning: Source directory '{sdir}' does not exist. Skipping.")
            continue
        p = find_image_label_pairs(sdir)
        print(f"[*] Found {len(p)} image-label pairs in: {sdir}")
        all_pairs.extend(p)

    if not all_pairs:
        raise RuntimeError("No image-label pairs found in any of the provided source directories.")

    # 2. Deduplicate images by SHA-256 hash
    print("\n[*] Deduplicating images across datasets...")
    seen_hashes = {}
    unique_pairs = []
    duplicates_count = 0

    for img_path, lbl_path, split, src_name in all_pairs:
        try:
            h = compute_file_hash(img_path)
            if h in seen_hashes:
                duplicates_count += 1
                continue
            seen_hashes[h] = img_path
            unique_pairs.append((img_path, lbl_path, split, src_name))
        except Exception as e:
            print(f"[!] Error hashing {img_path}: {e}")

    print(f"[*] Total pairs collected: {len(all_pairs)}")
    print(f"[*] Duplicate images pruned: {duplicates_count}")
    print(f"[*] Unique images retained:  {len(unique_pairs)}")

    # 3. Partition into train / valid / test splits
    splits_dict = {"train": [], "valid": [], "test": []}

    if split_mode == "preserve":
        for item in unique_pairs:
            sp = item[2]
            if sp in splits_dict:
                splits_dict[sp].append(item)
            else:
                splits_dict["train"].append(item)
    else:  # 'pool' mode (stratified shuffle)
        random.shuffle(unique_pairs)
        n = len(unique_pairs)
        n_tr = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits_dict["train"] = unique_pairs[:n_tr]
        splits_dict["valid"] = unique_pairs[n_tr : n_tr + n_val]
        splits_dict["test"] = unique_pairs[n_tr + n_val :]

    # 4. Write to destination folders with collision-safe filenames
    used_filenames = set()
    stats = {}

    for split_name, items in splits_dict.items():
        dst_img_dir = dst_dir / split_name / "images"
        dst_lbl_dir = dst_dir / split_name / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for img_src, lbl_src, _, src_tag in items:
            out_img_name = img_src.name
            out_lbl_name = lbl_src.name

            # Handle filename collision
            if out_img_name in used_filenames:
                out_img_name = f"{src_tag}_{out_img_name}"
                out_lbl_name = f"{src_tag}_{out_lbl_name}"

            used_filenames.add(out_img_name)

            out_img = dst_img_dir / out_img_name
            out_lbl = dst_lbl_dir / out_lbl_name

            if copy_images:
                shutil.copy2(img_src, out_img)
            else:
                try:
                    os.link(img_src, out_img)
                except OSError:
                    shutil.copy2(img_src, out_img)

            shutil.copy2(lbl_src, out_lbl)
            saved += 1

        stats[split_name] = saved
        print(f"  [+] Split '{split_name}': {saved} image-label pairs written")

    # 5. Determine kpt_shape automatically from source data.yaml or label files
    detected_kps = 16
    for s_dir in src_dirs:
        yaml_cand = s_dir / "data.yaml"
        if yaml_cand.exists():
            try:
                import yaml
                with open(yaml_cand, "r", encoding="utf-8") as yf:
                    yd = yaml.safe_load(yf)
                    if "kpt_shape" in yd:
                        detected_kps = yd["kpt_shape"][0]
                        break
            except Exception:
                pass

    # If still 16, verify against non-empty labels in destination
    if detected_kps == 16:
        for lbl_file in (dst_dir / "train" / "labels").glob("*.txt"):
            if lbl_file.stat().st_size > 0:
                with open(lbl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cols = len(parts)
                            kps = (cols - 5) // 3
                            if kps in (13, 15, 16):
                                detected_kps = kps
                                break
                if detected_kps != 16:
                    break

    flip_indices = list(range(detected_kps))

    yaml_text = f"""path: {dst_dir.resolve().as_posix()}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  - left_foot
  - right_foot

kpt_shape: [{detected_kps}, 3]
flip_idx: {flip_indices}
"""
    (dst_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"\n[OK] Successfully merged dataset created at: {dst_dir}")
    print(f"     - Config: {dst_dir / 'data.yaml'} (kpt_shape: [{detected_kps}, 3])")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Merge multiple YOLO-pose dataset versions.")
    parser.add_argument("--src_dirs", nargs="+", required=True,
                        help="Paths to source dataset directories to merge")
    parser.add_argument("--dst_dir", type=str, default="data/merged_v3_v4_raw",
                        help="Target output directory for merged dataset")
    parser.add_argument("--split_mode", choices=["pool", "preserve"], default="pool",
                        help="'pool' shuffles all data into 80/10/10; 'preserve' keeps existing split folders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility")
    parser.add_argument("--filter_13kp", action="store_true",
                        help="Directly apply 13-keypoint filter after merging (excising KP 5, 12, 15)")
    args = parser.parse_args()

    src_paths = [Path(p) for p in args.src_dirs]
    dst_path = Path(args.dst_dir)

    merge_datasets(
        src_dirs=src_paths,
        dst_dir=dst_path,
        split_mode=args.split_mode,
        seed=args.seed
    )

    if args.filter_13kp:
        print("\n[*] Applying 13-keypoint filter to merged dataset...")
        intermediate_raw = dst_path
        final_13kp_dst = dst_path.parent / f"{dst_path.name}_13kp" if not dst_path.name.endswith("_13kp") else dst_path

        filter_keypoints_dataset(
            src_dir=intermediate_raw,
            dst_dir=final_13kp_dst,
            remove_indices=[5, 12, 15],
            copy_images=True
        )
        print(f"[OK] 13-KP merged dataset finalized at: {final_13kp_dst}")


if __name__ == "__main__":
    main()

