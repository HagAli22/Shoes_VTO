"""
finetune_stage_a.py
───────────────────
Fine-tune the best Stage A foot detector (run_v2_1/weights/best.pt)
on the new shoes_v3 dataset from Roboflow.

Key steps performed automatically:
  1. Validate input paths (model + dataset).
  2. Remap class IDs in shoes_v3 labels to match model convention:
       Roboflow 0 (Foot_Right) → model 1 (right_foot)
       Roboflow 1 (Foot_left)  → model 0 (left_foot)
     Saves the remapped copy to data/shoes_v3_remapped/ (idempotent).
  3. Fine-tune using YOLOv8 with a low LR and optional frozen backbone.
  4. Print a concise result table.

Usage:
    python -m src.models.detector.finetune_stage_a
    python -m src.models.detector.finetune_stage_a --epochs 50 --freeze 10

CLI options (all optional):
    --weights   Path to base model .pt  (default: outputs/stage_a/run_v2_1/weights/best.pt)
    --data      Source data.yaml        (default: data/shoes_v3/data.yaml)
    --epochs    Max training epochs     (default: 100)
    --batch     Batch size              (default: 16)
    --imgsz     Input resolution        (default: 320)
    --device    GPU id or 'cpu'         (default: '0')
    --lr0       Initial learning rate   (default: 0.0001)
    --patience  Early-stop patience     (default: 25)
    --freeze    Freeze first N layers   (default: 0  — no freezing)
    --project   Output project dir      (default: outputs/stage_a)
    --name      Run sub-directory name  (default: finetune_v3)
    --skip_remap  Skip class remapping if already done
"""

import argparse
import os
import shutil
import time
from pathlib import Path

from ultralytics import YOLO


# ─── Constants ───────────────────────────────────────────────────────────────

# Class remapping: Roboflow ID → model training ID
#   Roboflow: 0=Foot_Right, 1=Foot_left
#   Model:    0=left_foot,  1=right_foot
REMAP = {0: 1, 1: 0}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def remap_label_file(src_path: Path, dst_path: Path, remap: dict) -> bool:
    """
    Rewrite a YOLO-pose label file with remapped class IDs.

    Format per line: cls cx cy w h kp0x kp0y kp0v ... kp15x kp15y kp15v

    Returns True if any class ID was changed.
    """
    lines = src_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    changed = False
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        parts = line.split()
        original_cls = int(parts[0])
        new_cls = remap.get(original_cls, original_cls)
        if new_cls != original_cls:
            changed = True
        parts[0] = str(new_cls)
        new_lines.append(" ".join(parts))
    dst_path.write_text("\n".join(new_lines), encoding="utf-8")
    return changed


def build_remapped_dataset(src_root: Path, dst_root: Path, remap: dict) -> Path:
    """
    Copy shoes_v3 to shoes_v3_remapped with class IDs rewritten.
    Images are symlinked (or copied if symlinks fail) to save disk space.
    Returns the path to the generated data_finetune.yaml.
    """
    splits = ["train", "valid", "test"]
    total_files = 0
    total_changed = 0

    for split in splits:
        src_img_dir = src_root / split / "images"
        src_lbl_dir = src_root / split / "labels"
        dst_img_dir = dst_root / split / "images"
        dst_lbl_dir = dst_root / split / "labels"

        if not src_img_dir.exists():
            continue  # skip missing splits

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        # ── Labels: remap and write ───────────────────────────────────────
        if src_lbl_dir.exists():
            for lbl_file in src_lbl_dir.glob("*.txt"):
                dst_lbl = dst_lbl_dir / lbl_file.name
                changed = remap_label_file(lbl_file, dst_lbl, remap)
                total_files += 1
                if changed:
                    total_changed += 1

        # ── Images: copy (hard link or copy) ─────────────────────────────
        for img_file in src_img_dir.iterdir():
            dst_img = dst_img_dir / img_file.name
            if not dst_img.exists():
                try:
                    os.link(img_file, dst_img)  # hard-link (fast, same drive)
                except OSError:
                    shutil.copy2(img_file, dst_img)  # fallback: full copy

    print(f"[Remap] {total_changed}/{total_files} label files had class IDs changed.")

    # ── Write data_finetune.yaml into dst_root ────────────────────────────
    yaml_dst = dst_root / "data_finetune.yaml"
    yaml_content = (
        f"path: {dst_root.as_posix()}\n"
        "train: train/images\n"
        "val:   valid/images\n"
        "test:  test/images\n"
        "\n"
        "nc: 2\n"
        "names:\n"
        "  - left_foot    # class 0\n"
        "  - right_foot   # class 1\n"
        "\n"
        "kpt_shape: [16, 3]\n"
        "\n"
        "flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]\n"
    )
    yaml_dst.write_text(yaml_content, encoding="utf-8")
    print(f"[Remap] Data YAML written → {yaml_dst}")
    return yaml_dst


def finetune(
    weights_path: str = "outputs/stage_a/run_v2_1/weights/best.pt",
    data_yaml: str = "data/shoes_v3/data.yaml",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 320,
    device: str = "0",
    lr0: float = 0.0001,
    lrf: float = 0.01,
    patience: int = 25,
    freeze: int = 0,
    optimizer: str = "AdamW",
    project: str = "outputs/stage_a",
    name: str = "finetune_v3",
    skip_remap: bool = False,
) -> str:
    """
    Run the full fine-tuning pipeline and return path to best.pt.

    Parameters
    ----------
    weights_path : Base model checkpoint (best.pt from run_v2_1).
    data_yaml    : Original shoes_v3 data.yaml (Roboflow class convention).
    epochs       : Max training epochs.
    batch_size   : Batch size.
    imgsz        : Input resolution.
    device       : GPU id (e.g. '0') or 'cpu'.
    lr0          : Initial learning rate (kept low for fine-tuning).
    lrf          : Final LR as fraction of lr0.
    patience     : Early-stop patience epochs.
    freeze       : Freeze the first N layers (0 = no freeze).
    optimizer    : Optimizer name.
    project      : Output project directory.
    name         : Run sub-directory.
    skip_remap   : If True, skip dataset remapping (assumes already done).
    """
    weights_path = Path(weights_path)
    data_yaml_path = Path(data_yaml)

    # ── Validate inputs ───────────────────────────────────────────────────
    if not weights_path.exists():
        raise FileNotFoundError(f"Base weights not found: {weights_path}")
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

    # ── Step 1: Remap dataset ─────────────────────────────────────────────
    src_root = data_yaml_path.parent
    dst_root = src_root.parent / "shoes_v3_remapped"

    if skip_remap and dst_root.exists():
        yaml_dst = dst_root / "data_finetune.yaml"
        print(f"[Remap] Skipping (--skip_remap). Using existing: {yaml_dst}")
    else:
        print("\n[Step 1/2] Remapping class IDs in shoes_v3 dataset ...")
        print(f"  Roboflow 0 (Foot_Right) → model 1 (right_foot)")
        print(f"  Roboflow 1 (Foot_left)  → model 0 (left_foot)")
        yaml_dst = build_remapped_dataset(src_root, dst_root, REMAP)

    # ── Step 2: Fine-tune ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [Step 2/2] Fine-Tuning Stage A on shoes_v3                      ")
    print("=" * 70)
    print(f"  Base weights : {weights_path}")
    print(f"  Dataset      : {yaml_dst}")
    print(f"  Epochs       : {epochs}  (EarlyStop patience={patience})")
    print(f"  Image size   : {imgsz}×{imgsz}  |  Batch: {batch_size}")
    print(f"  LR           : {lr0} → {lr0*lrf:.2e}  |  Optimizer: {optimizer}")
    print(f"  Freeze layers: {freeze if freeze > 0 else 'None'}")
    print(f"  Output       : {os.path.join(project, name)}")
    print("=" * 70 + "\n")

    model = YOLO(str(weights_path))
    t0 = time.time()

    train_kwargs = dict(
        data=str(yaml_dst),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        optimizer=optimizer,
        lr0=lr0,
        lrf=lrf,
        patience=patience,
        save=True,
        verbose=True,
        plots=True,

        # ── LR Schedule ──────────────────────────────────────────────────
        warmup_epochs=2,          # Short warmup — dataset is small, ramp fast
        warmup_bias_lr=0.005,
        cos_lr=True,              # Cosine annealing to lrf*lr0

        # ── Regularization ───────────────────────────────────────────────
        weight_decay=0.0005,
        dropout=0.0,

        # ── Color / Photometric Augmentation ─────────────────────────────
        # Keep strong — helps generalize lighting/material without breaking geometry
        hsv_h=0.015,              # Hue jitter ±1.5%
        hsv_s=0.7,                # Saturation jitter ±70%
        hsv_v=0.4,                # Brightness jitter ±40%

        # ── Geometric Augmentation ────────────────────────────────────────
        degrees=10.0,             # Reduced from 15° — stable KP learning
        translate=0.1,            # Reduced from 0.15
        scale=0.5,                # Scale jitter
        shear=0.0,                # DISABLED — distorts KP positions too much
        perspective=0.0,          # DISABLED — not needed at this stage
        flipud=0.0,               # DISABLED — feet always point down
        fliplr=0.5,               # Horizontal flip (safe with flip_idx)

        # ── Advanced Augmentation ─────────────────────────────────────────
        # LESSON LEARNED: mosaic=1.0 + mixup caused validation collapse.
        # Fine-tuning on small datasets needs gentler augmentation.
        mosaic=0.5,               # 50% mosaic — enough diversity, not overwhelming
        mixup=0.0,                # DISABLED — interpolated KPs confuse pose head
        copy_paste=0.0,           # DISABLED — segmentation-only feature
        erasing=0.2,              # 20% random erasing (simulates partial occlusion)

        # ── Close Mosaic ──────────────────────────────────────────────────
        close_mosaic=15,          # Turn off mosaic for final 15 epochs → clean signal
    )

    if freeze > 0:
        train_kwargs["freeze"] = freeze

    results = model.train(**train_kwargs)

    elapsed_min = (time.time() - t0) / 60.0
    best_ckpt = Path(project) / name / "weights" / "best.pt"

    # ── Results Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Fine-Tuning Complete!")
    print("=" * 70)
    print(f"  Time elapsed   : {elapsed_min:.1f} minutes")
    print(f"  Best checkpoint: {best_ckpt}")
    if best_ckpt.exists():
        print(f"  File size      : {best_ckpt.stat().st_size / 1e6:.2f} MB")

    # Print key metrics from results object if available
    try:
        metrics = results.results_dict
        box_map50 = metrics.get("metrics/mAP50(B)", "N/A")
        pose_map50 = metrics.get("metrics/mAP50(P)", "N/A")
        print(f"  Box mAP@50     : {box_map50:.4f}" if isinstance(box_map50, float) else f"  Box mAP@50     : {box_map50}")
        print(f"  Pose mAP@50    : {pose_map50:.4f}" if isinstance(pose_map50, float) else f"  Pose mAP@50    : {pose_map50}")
    except Exception:
        pass

    print("=" * 70 + "\n")
    return str(best_ckpt)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stage A YOLOv8 pose detector on shoes_v3 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        default="outputs/stage_a/run_v2_1/weights/best.pt",
        help="Path to base model checkpoint (.pt).",
    )
    parser.add_argument(
        "--data",
        default="data/shoes_v3/data.yaml",
        help="Path to source shoes_v3 data.yaml (Roboflow convention).",
    )
    parser.add_argument("--epochs",   type=int,   default=100,    help="Max training epochs.")
    parser.add_argument("--batch",    type=int,   default=16,     help="Batch size.")
    parser.add_argument("--imgsz",    type=int,   default=320,    help="Input image resolution.")
    parser.add_argument("--device",   default="0",                help="GPU id or 'cpu'.")
    parser.add_argument("--lr0",      type=float, default=0.0001, help="Initial learning rate.")
    parser.add_argument("--lrf",      type=float, default=0.01,   help="Final LR fraction of lr0.")
    parser.add_argument("--patience", type=int,   default=25,     help="Early-stopping patience.")
    parser.add_argument("--freeze",   type=int,   default=0,      help="Freeze first N backbone layers.")
    parser.add_argument("--optimizer", default="AdamW",           help="Optimizer name.")
    parser.add_argument("--project",  default="outputs/stage_a",  help="Output project directory.")
    parser.add_argument("--name",     default="finetune_v3",       help="Run sub-directory name.")
    parser.add_argument("--skip_remap", action="store_true",
                        help="Skip class remapping (use if shoes_v3_remapped already exists).")
    args = parser.parse_args()

    finetune(
        weights_path=args.weights,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        lrf=args.lrf,
        patience=args.patience,
        freeze=args.freeze,
        optimizer=args.optimizer,
        project=args.project,
        name=args.name,
        skip_remap=args.skip_remap,
    )


if __name__ == "__main__":
    main()

'''
python -m src.models.detector.train_shuffled_v3 `
  --weights "outputs/stage_a/run_v2_1/weights/best.pt" `
  --data "data/shuffled_v3/data.yaml" `
  --epochs 150 `
  --batch 16 `
  --imgsz 320 `
  --device 0 `
  --lr0 0.001 `
  --patience 40 `
  --project "outputs/stage_a" `
  --name "run_shuffled_v3"
'''