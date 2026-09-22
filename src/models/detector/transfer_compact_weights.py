"""
transfer_compact_weights.py
───────────────────────────
Knowledge Transfer / Weight Initialization Tool for Compact Stage A (YOLOv8-Pico).

Transfers learned feature representations from a full-capacity teacher detector
(e.g., standard YOLOv8n trained for 200 epochs) into the width-scaled compact student
architecture (width scale 0.16, <= 5 MB FP32 target).

Usage:
    python -m src.models.detector.transfer_compact_weights \\
        --teacher outputs/stage_a/run_v2_1/weights/best.pt \\
        --cfg configs/yolov8n-compact-16kp.yaml \\
        --output outputs/stage_a/compact_initialized.pt
"""

import os
import argparse
import torch
from ultralytics import YOLO


def transfer_weights(teacher_path: str, student_cfg: str, output_path: str) -> str:
    """
    Slices and transfers weights from teacher YOLOv8n to student YOLOv8-Pico.

    Parameters
    ----------
    teacher_path : str
        Path to best.pt checkpoint of the teacher model.
    student_cfg : str
        Path to the YAML architecture definition of the compact student model.
    output_path : str
        Destination path for the initialized PyTorch checkpoint.

    Returns
    -------
    output_path : str
    """
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher checkpoint not found at: {teacher_path}")
    if not os.path.exists(student_cfg):
        raise FileNotFoundError(f"Student YAML configuration not found at: {student_cfg}")

    print("==================================================================")
    print("  Initializing Compact Stage A with Teacher Knowledge Transfer    ")
    print(f"  Teacher Model : {teacher_path}")
    print(f"  Student Config: {student_cfg}")
    print("==================================================================")

    print("[1] Loading teacher model and initializing student architecture...")
    teacher = YOLO(teacher_path)
    student = YOLO(student_cfg)

    t_sd = teacher.model.state_dict()
    s_sd = student.model.state_dict()

    print(f"    Teacher tensors: {len(t_sd)}")
    print(f"    Student tensors: {len(s_sd)}")

    transferred = 0
    exact_matches = 0
    sliced_matches = 0

    print("[2] Slicing and transferring weights...")
    for k, s_v in s_sd.items():
        if k in t_sd:
            t_v = t_sd[k]
            if s_v.shape == t_v.shape:
                s_sd[k] = t_v.clone()
                exact_matches += 1
            else:
                slices = tuple(slice(0, min(s_dim, t_dim)) for s_dim, t_dim in zip(s_v.shape, t_v.shape))
                s_sd[k][slices] = t_v[slices].clone()
                sliced_matches += 1
            transferred += 1
        else:
            print(f"    [!] Warning: Layer {k} not found in teacher model")

    student.model.load_state_dict(s_sd)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    student.save(output_path)

    print(f"[3] Weight transfer complete:")
    print(f"    Total tensors transferred: {transferred}/{len(s_sd)} ({transferred/len(s_sd)*100:.1f}%)")
    print(f"    Exact shape matches       : {exact_matches}")
    print(f"    Channel-sliced matches    : {sliced_matches}")
    print(f"    Saved initialized weights : {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")
    print("==================================================================\n")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Transfer teacher weights into compact YOLOv8-Pico student.")
    parser.add_argument("--teacher", default="outputs/stage_a/run_v2_1/weights/best.pt",
                        help="Path to trained teacher best.pt")
    parser.add_argument("--cfg", default="configs/yolov8n-compact-16kp.yaml",
                        help="Path to student compact model YAML")
    parser.add_argument("--output", default="outputs/stage_a/compact_initialized.pt",
                        help="Output path for initialized checkpoint")
    args = parser.parse_args()

    transfer_weights(args.teacher, args.cfg, args.output)


if __name__ == "__main__":
    main()

