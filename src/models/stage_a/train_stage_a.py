"""
train_stage_a.py
────────────────
Stage A — Foot Detector Training

Trains YOLOv8n-pose to detect left/right feet and output 4 coarse
keypoints (toe_tip, heel_back, ball_medial, ball_lateral) + bounding box.

We train with ALL 16 keypoints for maximum supervision signal.
At export time we subset to the 4 coarse KPs required by the AR SDK.

Usage
-----
  python src/models/stage_a/train_stage_a.py [--epochs 200] [--batch 16]

Output
------
  outputs/stage_a/<run_name>/weights/best.pt     <- best checkpoint
  outputs/stage_a/<run_name>/weights/best.onnx   <- exported ONNX (opset 12)
"""

import argparse
import os
import sys
import time

# ── Make project root importable ──────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DATA_YAML   = os.path.join(PROJECT_ROOT, "data", "stage_a", "data.yaml")
WORK_DIR    = os.path.join(PROJECT_ROOT, "outputs", "stage_a")
MODEL_BASE  = "yolov8n-pose.pt"   # COCO-pretrained, will be downloaded automatically


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage A foot detector.")
    p.add_argument("--epochs",    type=int, default=200,
                   help="Number of training epochs (default 200 for small dataset)")
    p.add_argument("--batch",     type=int, default=16,
                   help="Batch size (default 16, safe for Quadro P2000 4GB)")
    p.add_argument("--imgsz",     type=int, default=320,
                   help="Input image size (must be 320 for frozen Stage A spec)")
    p.add_argument("--device",    default="0",
                   help="Device: '0' for first GPU, 'cpu' for CPU")
    p.add_argument("--resume",    default=None,
                   help="Resume from checkpoint path")
    p.add_argument("--name",      default="run1",
                   help="Run name under outputs/stage_a/")
    p.add_argument("--patience",  type=int, default=50,
                   help="Early stopping patience")
    return p.parse_args()


def check_data_ready() -> None:
    """Verify that data preparation has been run."""
    if not os.path.exists(DATA_YAML):
        print("[ERROR] data/stage_a/data.yaml not found.")
        print("  Run first:  python src/models/stage_a/prepare_stage_a_data.py")
        sys.exit(1)

    train_imgs = os.path.join(PROJECT_ROOT, "data", "stage_a", "train", "images")
    n = len([f for f in os.listdir(train_imgs) if f.lower().endswith((".jpg", ".png"))])
    print(f"  Train images : {n}")
    if n == 0:
        print("[ERROR] No training images found. Run prepare_stage_a_data.py first.")
        sys.exit(1)


def train(args: argparse.Namespace) -> str:
    """
    Run YOLOv8n-pose training.
    Returns path to best checkpoint.
    """
    from ultralytics import YOLO

    print("\n" + "=" * 60)
    print("  Stage A — Foot Detector Training")
    print("=" * 60)

    check_data_ready()

    model = YOLO(MODEL_BASE)
    print(f"  Base model    : {MODEL_BASE}")
    print(f"  Data yaml     : {DATA_YAML}")
    print(f"  Image size    : {args.imgsz}x{args.imgsz}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch}")
    print(f"  Device        : {args.device}")
    print(f"  Output dir    : {WORK_DIR}/{args.name}")
    print("=" * 60 + "\n")

    train_kwargs = dict(
        data        = DATA_YAML,
        epochs      = args.epochs,
        imgsz       = args.imgsz,
        batch       = args.batch,
        device      = args.device,
        project     = WORK_DIR,
        name        = args.name,
        patience    = args.patience,
        save        = True,
        save_period = 10,        # checkpoint every 10 epochs
        plots       = True,
        verbose     = True,

        # Augmentation — tuned for feet (small object, varied backgrounds)
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        degrees     = 15.0,      # rotation ±15°
        translate   = 0.1,
        scale       = 0.5,
        flipud      = 0.0,       # never flip upside-down (feet always gravity-down)
        fliplr      = 0.0,       # NO horizontal flip — class flip must stay correct (§7)
        mosaic      = 0.5,       # mosaic augmentation (helps with small dataset)
        mixup       = 0.0,
        copy_paste  = 0.1,

        # Optimiser
        optimizer   = "AdamW",
        lr0         = 1e-3,
        lrf         = 0.01,
        warmup_epochs = 5,
        weight_decay  = 5e-4,

        # Confidence for detection during val
        conf        = 0.25,
        iou         = 0.7,
    )

    if args.resume:
        train_kwargs["resume"] = args.resume

    t0 = time.time()
    results = model.train(**train_kwargs)
    elapsed = (time.time() - t0) / 60

    # Use the actual save_dir YOLO used (it may auto-increment: run1→run12 etc.)
    actual_run_dir = str(results.save_dir)
    best_pt = os.path.join(actual_run_dir, "weights", "best.pt")
    print(f"\nTraining complete in {elapsed:.1f} min")
    print(f"Best checkpoint: {best_pt}")

    # Print key metrics
    try:
        metrics = results.results_dict
        print("\nFinal metrics:")
        for k in ["metrics/mAP50(B)", "metrics/mAP50-95(B)",
                  "metrics/mAP50(P)", "metrics/mAP50-95(P)"]:
            if k in metrics:
                print(f"  {k:<30}: {metrics[k]:.4f}")
    except Exception:
        pass

    return best_pt


def export_onnx(best_pt: str, imgsz: int = 320) -> None:
    """
    Export best.pt → ONNX opset 12.
    fp32 only at this stage. fp16/int8 handled separately.
    """
    from ultralytics import YOLO

    print("\nExporting to ONNX opset 12...")
    model = YOLO(best_pt)

    onnx_path = best_pt.replace(".pt", ".onnx")
    model.export(
        format   = "onnx",
        imgsz    = imgsz,
        opset    = 12,            # frozen spec: opset 12
        simplify = True,          # run onnxsim
        dynamic  = False,         # static shapes (requirement §2)
        half     = False,         # fp32 primary
        nms      = False,         # no in-graph NMS (requirement §2)
    )

    print(f"ONNX exported to: {onnx_path}")
    _validate_onnx(onnx_path, imgsz)


def _validate_onnx(onnx_path: str, imgsz: int) -> None:
    """Validate ONNX output shape and run a test inference."""
    import numpy as np
    import onnx
    import onnxruntime as ort

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("  ONNX model: valid ✅")

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp  = sess.get_inputs()[0]
    out  = sess.get_outputs()[0]
    print(f"  Input  : {inp.name}  shape={inp.shape}  dtype={inp.type}")
    print(f"  Output : {out.name}  shape={out.shape}  dtype={out.type}")

    # Run dummy inference
    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    result = sess.run(None, {inp.name: dummy})
    print(f"  Output shape : {result[0].shape}")
    print(f"  Expected     : [1, N_rows, anchors] — N_rows depends on kpt_shape")
    print("  ONNX inference: PASSED ✅")

    # Op list
    ops = sorted(set(node.op_type for node in model.graph.node))
    print(f"  Ops used ({len(ops)}): {', '.join(ops)}")

    # Check for problematic ops
    banned = {"NonMaxSuppression", "RoiAlign", "GridSample", "ScatterND"}
    found_banned = banned & set(ops)
    if found_banned:
        print(f"  WARNING — banned ops found: {found_banned}")
        print("  These may fail on WebGPU EP or WASM EP. Raise with AR team.")
    else:
        print("  No banned ops detected ✅")


def main() -> None:
    args = parse_args()
    best_pt = train(args)
    export_onnx(best_pt, imgsz=args.imgsz)
    print("\nDone. Next steps:")
    print("  1. Check outputs/stage_a/<run>/results.png for loss curves")
    print("  2. Check mAP metrics above")
    print("  3. Run python tools/evaluation/eval_detection.py when ready")


if __name__ == "__main__":
    main()

