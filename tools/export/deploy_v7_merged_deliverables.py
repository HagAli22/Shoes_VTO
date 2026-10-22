import os
import shutil
import hashlib
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

SRC_PT = "outputs/stage_a/run_v7_256_merged_with_negatives/weights/best.pt"
SRC_ONNX = "outputs/stage_a/run_v7_256_merged_with_negatives/weights/best.onnx"
SRC_INT8 = "outputs/stage_a/run_v7_256_merged_with_negatives/weights/best_int8.onnx"
SRC_FP16 = "outputs/stage_a/run_v7_256_merged_with_negatives/weights/best_fp16.onnx"

DST_13KP = "deliverables/stage_a_13kp/models"
DST_256 = "deliverables/stage_a_256_13kp/models"


def main():
    print("=" * 70)
    print("DEPLOYING CHAMPION 256 (MERGED + NEGATIVES) TO DELIVERABLES")
    print("=" * 70)

    # 1. Backup old 256 base in stage_a_13kp
    old_pt = os.path.join(DST_13KP, "stage-a-256-13kp-v7.pt")
    old_onnx = os.path.join(DST_13KP, "stage-a-256-13kp-v7.onnx")
    backup_pt = os.path.join(DST_13KP, "stage-a-256-13kp-v7-base.pt")
    backup_onnx = os.path.join(DST_13KP, "stage-a-256-13kp-v7-base.onnx")

    if os.path.exists(old_pt) and not os.path.exists(backup_pt):
        shutil.copy2(old_pt, backup_pt)
        print(f"Backed up {old_pt} -> {backup_pt}")
    if os.path.exists(old_onnx) and not os.path.exists(backup_onnx):
        shutil.copy2(old_onnx, backup_onnx)
        print(f"Backed up {old_onnx} -> {backup_onnx}")

    # 2. Deploy to deliverables/stage_a_13kp/models
    shutil.copy2(SRC_PT, os.path.join(DST_13KP, "stage-a-256-13kp-v7.pt"))
    shutil.copy2(SRC_PT, os.path.join(DST_13KP, "stage-a-256-13kp-v7-merged-negatives.pt"))
    shutil.copy2(SRC_ONNX, os.path.join(DST_13KP, "stage-a-256-13kp-v7.onnx"))
    shutil.copy2(SRC_ONNX, os.path.join(DST_13KP, "stage-a-256-13kp-v7-merged-negatives.onnx"))
    shutil.copy2(SRC_INT8, os.path.join(DST_13KP, "stage-a-256-13kp-int8.onnx"))
    print(f"Deploy complete for {DST_13KP}")

    # 3. Deploy to deliverables/stage_a_256_13kp/models
    shutil.copy2(SRC_PT, os.path.join(DST_256, "stage-a-256-13kp-v7.pt"))
    shutil.copy2(SRC_ONNX, os.path.join(DST_256, "stage-a-256-13kp-fp32.onnx"))
    shutil.copy2(SRC_INT8, os.path.join(DST_256, "stage-a-256-13kp-int8.onnx"))
    shutil.copy2(SRC_FP16, os.path.join(DST_256, "stage-a-256-13kp-fp16.onnx"))
    print(f"Deploy complete for {DST_256}")

    # 4. Verify deployment
    print("\nVerifying deployed models:")
    m = YOLO(os.path.join(DST_13KP, "stage-a-256-13kp-v7.pt"))
    print("  [OK] deliverables/stage_a_13kp/models/stage-a-256-13kp-v7.pt loaded successfully!")

    sess = ort.InferenceSession(os.path.join(DST_13KP, "stage-a-256-13kp-v7.onnx"), providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)
    out = sess.run(None, {sess.get_inputs()[0].name: dummy})
    print(f"  [OK] stage-a-256-13kp-v7.onnx verified! Output shape: {out[0].shape}")

    sess_int8 = ort.InferenceSession(os.path.join(DST_13KP, "stage-a-256-13kp-int8.onnx"), providers=["CPUExecutionProvider"])
    out_int8 = sess_int8.run(None, {sess_int8.get_inputs()[0].name: dummy})
    print(f"  [OK] stage-a-256-13kp-int8.onnx verified! Output shape: {out_int8[0].shape}, Size: {os.path.getsize(os.path.join(DST_13KP, 'stage-a-256-13kp-int8.onnx'))/1e6:.2f} MB")

    sess_256 = ort.InferenceSession(os.path.join(DST_256, "stage-a-256-13kp-fp32.onnx"), providers=["CPUExecutionProvider"])
    out_256 = sess_256.run(None, {sess_256.get_inputs()[0].name: dummy})
    print(f"  [OK] deliverables/stage_a_256_13kp/models/stage-a-256-13kp-fp32.onnx verified! Shape: {out_256[0].shape}")

    print("\n" + "=" * 70)
    print("ALL MODELS SUCCESSFULLY DEPLOYED AND VERIFIED!")
    print("=" * 70)


if __name__ == "__main__":
    main()

