"""
train_and_package_compact_stage_a.py
───────────────────────────────────
Master One-Click Automated Pipeline for Stage A Compact Detector.

Orchestrates the entire lifecycle:
  Step 1: Knowledge Transfer (slices/transfers teacher weights into student architecture)
  Step 2: Model Training (Ultralytics YOLO with early stopping)
  Step 3: Path A ONNX Export & Channel Slicing ([1, 18, 2100])
  Step 4: True FP16 (WebGPU) & Dynamic INT8 (WASM) Quantization
  Step 5: Budget Verification (<= 5.0 MB FP32, <= 2.5 MB FP16, <= 1.5 MB INT8)
  Step 6: Contract Synchronization (model-contract.json)
  Step 7 (Optional): Video Benchmark on data/test_video.mp4

Whenever you update or augment the dataset in data/stage_a/, simply run:
    python tools/train_and_package_compact_stage_a.py
"""

import os
import sys
import time
import argparse

# Ensure project root is in PYTHONPATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.detector.transfer_compact_weights import transfer_weights
from src.models.detector.train_compact_detector import train_compact
from src.export.export_compact_stage_a import export_and_slice_compact


def run_full_compact_pipeline(
    teacher_path: str = "outputs/stage_a/run_v2_1/weights/best.pt",
    student_cfg: str = "configs/yolov8n-compact-16kp.yaml",
    data_yaml: str = "data/stage_a/data.yaml",
    epochs: int = 200,
    batch_size: int = 16,
    imgsz: int = 320,
    device: str = "0",
    patience: int = 40,
    lr0: float = 0.001,
    optimizer: str = "AdamW",
    run_name: str = "compact_automated_run",
    output_dir: str = "shoes-vto-ai-v2/models",
    contract_path: str = "shoes-vto-ai-v2/model-contract.json",
    eval_video: bool = False,
    video_path: str = "data/test_video.mp4",
):
    start_total = time.time()

    print("\n" + "=" * 76)
    print("      STAGE A COMPACT DETECTOR — FULL AUTOMATED RETRAINING PIPELINE      ")
    print("=" * 76)
    print(f"  Teacher Weights : {teacher_path}")
    print(f"  Student Config  : {student_cfg}")
    print(f"  Dataset YAML    : {data_yaml}")
    print(f"  Target Epochs   : {epochs} (Patience: {patience})")
    print(f"  Output Directory: {output_dir}")
    print("=" * 76 + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 1: Knowledge Transfer
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    init_weights = os.path.join("outputs", "stage_a", f"{run_name}_init.pt")
    print(">>> [STEP 1/5] Transferring Teacher Weights...")
    transfer_weights(
        teacher_path=teacher_path,
        student_cfg=student_cfg,
        output_path=init_weights
    )
    print(f">>> [STEP 1/5] Done in {time.time() - t0:.1f}s\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 2: Training
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print(">>> [STEP 2/5] Training Compact Detector to Convergence...")
    best_ckpt = train_compact(
        weights_path=init_weights,
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz,
        device=device,
        patience=patience,
        lr0=lr0,
        optimizer=optimizer,
        project="outputs/stage_a",
        name=run_name
    )
    print(f">>> [STEP 2/5] Done in {(time.time() - t0)/60.0:.2f} minutes\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Step 3, 4, 5, 6: Export, Path A Slicing, Quantization & Contract Sync
    # ──────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    print(">>> [STEP 3-6/5] Exporting ONNX, Slicing Path A [1, 18, 2100], and Quantizing...")
    manifest = export_and_slice_compact(
        weights_path=best_ckpt,
        output_dir=output_dir,
        contract_path=contract_path,
        mirror_dir="deliverables/models",
        imgsz=imgsz,
        opset=12
    )
    print(f">>> [STEP 3-6/5] Done in {time.time() - t0:.1f}s\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Optional Step 7: Video Benchmark Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    if eval_video and os.path.exists(video_path):
        t0 = time.time()
        print(">>> [OPTIONAL] Running Video Benchmark on test_video.mp4...")
        from tools.evaluation.eval_video_models import process_video_for_model
        vid_dir = "data/test_video_results"
        os.makedirs(vid_dir, exist_ok=True)

        for prec in ["fp32", "fp16", "int8"]:
            fn = f"stage-a-320-compact-{prec}.onnx"
            p = os.path.join(output_dir, fn)
            cfg = {
                "path": p,
                "display_name": "Stage A Compact 320",
                "precision": prec,
                "size_mb": manifest[fn]["size_mb"],
                "budget_ok": manifest[fn]["budget_compliant"],
                "conf_thresh": 0.25,
            }
            out_v = os.path.join(vid_dir, f"stage_a_320_compact_{prec}.mp4")
            process_video_for_model(cfg, video_path, out_v)
        print(f">>> [OPTIONAL] Video benchmark completed in {(time.time() - t0)/60.0:.2f} minutes\n")

    total_min = (time.time() - start_total) / 60.0
    print("=" * 76)
    print(f"      COMPACT PIPELINE FINISHED SUCCESSFULLY IN {total_min:.2f} MINUTES!      ")
    print("=" * 76)
    for model_name, info in manifest.items():
        print(f"  - {model_name:<32}: {info['size_mb']:>5.2f} MB | Budget: {'PASSED' if info['budget_compliant'] else 'FAILED'} | SHA-256: {info['sha256'][:16]}...")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Master automated training and packaging pipeline for Stage A Compact.")
    parser.add_argument("--teacher", default="outputs/stage_a/run_v2_1/weights/best.pt",
                        help="Path to trained teacher best.pt")
    parser.add_argument("--cfg", default="configs/yolov8n-compact-16kp.yaml",
                        help="Student architecture YAML")
    parser.add_argument("--data", default="data/stage_a/data.yaml",
                        help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size")
    parser.add_argument("--device", default="0", help="GPU device (e.g. '0' or 'cpu')")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument("--lr0", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--name", default=f"compact_run_{int(time.time())}", help="Run name")
    parser.add_argument("--output_dir", default="shoes-vto-ai-v2/models", help="Destination folder for ONNX")
    parser.add_argument("--eval_video", action="store_true", help="Also generate test videos")
    args = parser.parse_args()

    run_full_compact_pipeline(
        teacher_path=args.teacher,
        student_cfg=args.cfg,
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        run_name=args.name,
        output_dir=args.output_dir,
        eval_video=args.eval_video,
    )


if __name__ == "__main__":
    main()

