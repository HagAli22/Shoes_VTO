"""
train_detector.py
─────────────────
YOLOv8n foot detector training script using Ultralytics.

Trains a foot bounding box detector on custom data.
The detector is the first stage in the AI Perception pipeline.

Dataset format: YOLO format
  datasets/foot_detection/
    images/
      train/  *.jpg
      val/    *.jpg
    labels/
      train/  *.txt   (class_id cx cy w h, normalised 0–1)
      val/    *.txt
    data.yaml

Usage
-----
  python src/models/detector/train_detector.py \\
      --data datasets/foot_detection/data.yaml \\
      --epochs 100 \\
      --batch 16 \\
      --imgsz 640 \\
      --work-dir outputs/detector/

After training, the best ONNX model is exported automatically.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def create_data_yaml(
    train_dir: str,
    val_dir: str,
    output_path: str = "datasets/foot_detection/data.yaml",
) -> str:
    """
    Create a YOLO data.yaml file for the foot detection dataset.

    Parameters
    ----------
    train_dir   : Path to training images directory.
    val_dir     : Path to validation images directory.
    output_path : Where to write the data.yaml file.

    Returns
    -------
    str : Path to the created data.yaml.
    """
    import yaml

    data = {
        "path":  str(Path(output_path).parent.resolve()),
        "train": str(Path(train_dir).resolve()),
        "val":   str(Path(val_dir).resolve()),
        "nc":    1,             # 1 class: "foot"
        "names": ["foot"],
    }
    os.makedirs(Path(output_path).parent, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"data.yaml written to: {output_path}")
    return output_path


def train_yolov8n(
    data_yaml: str,
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    work_dir: str = "outputs/detector/",
    device: str = "0",           # "0" = GPU 0, "cpu" = CPU
    pretrained: bool = True,
) -> None:
    """
    Train YOLOv8n on foot detection data.

    Parameters
    ----------
    data_yaml   : Path to YOLO data.yaml file.
    epochs      : Number of training epochs.
    batch       : Batch size.
    imgsz       : Input image size (square).
    work_dir    : Output directory for weights and logs.
    device      : "0" for GPU 0, "cpu" for CPU.
    pretrained  : Start from COCO-pretrained weights (recommended).
    """
    from ultralytics import YOLO

    # Load YOLOv8n (pretrained on COCO)
    weights = "yolov8n.pt" if pretrained else "yolov8n.yaml"
    model   = YOLO(weights)

    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=work_dir,
        name="foot_detector",
        save=True,
        save_period=10,
        val=True,
        workers=4,
        patience=20,            # early stopping
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        momentum=0.937,
        weight_decay=5e-4,
        warmup_epochs=3,
        augment=True,
        # Augmentation parameters
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )
    print(f"\nTraining complete.")
    print(f"Best mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")

    # ── Auto-export best model to ONNX ───────────────────────────────────────
    best_pt = Path(work_dir) / "foot_detector" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\nExporting best.pt to ONNX...")
        export_model = YOLO(str(best_pt))
        export_model.export(
            format="onnx",
            opset=17,
            simplify=True,
            dynamic=False,
            imgsz=imgsz,
        )
        onnx_src = best_pt.with_suffix(".onnx")
        onnx_dst = Path("outputs/exports/foot_detector.onnx")
        onnx_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx_src, onnx_dst)
        print(f"ONNX model exported to: {onnx_dst}")
    else:
        print(f"[WARN] best.pt not found at {best_pt}")


def validate_detector(
    model_path: str,
    data_yaml: str,
    imgsz: int = 640,
) -> None:
    """Run validation on the trained detector and print mAP metrics."""
    from ultralytics import YOLO
    model   = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz)
    print(f"mAP@50  : {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n foot detector.")
    parser.add_argument("--data",     required=True, help="Path to data.yaml")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--batch",    type=int, default=16)
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--work-dir", default="outputs/detector/")
    parser.add_argument("--device",   default="0", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train from scratch without COCO pretrained weights")
    parser.add_argument("--validate", default=None,
                        help="Path to .pt model to validate (skip training)")
    args = parser.parse_args()

    if args.validate:
        validate_detector(args.validate, args.data, args.imgsz)
    else:
        train_yolov8n(
            data_yaml=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            work_dir=args.work_dir,
            device=args.device,
            pretrained=not args.no_pretrained,
        )


if __name__ == "__main__":
    main()

