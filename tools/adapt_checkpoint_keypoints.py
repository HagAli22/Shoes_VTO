"""
adapt_checkpoint_keypoints.py
─────────────────────────────
Surgically adapts a pretrained 16-keypoint YOLO-pose checkpoint to a
13-keypoint architecture (or custom surviving keypoint subset).

Extracts and maps the weights and biases of the surviving keypoints directly
into the new pose head, preserving 100% of learned features in the backbone,
neck, bounding box head, class head, and the surviving keypoints.

Default removed keypoints:
  - ball_top     (original KP 5)
  - ankle_center (original KP 12)
  - shin_mid     (original KP 15)

Usage:
  python tools/adapt_checkpoint_keypoints.py \
      --src outputs/stage_a/run_shuffled_v3/weights/best.pt \
      --dst checkpoints/stage_a_13kp_adapted.pt \
      --remove_indices 5,12,15
"""

import argparse
from pathlib import Path
from typing import List
import torch
import torch.nn as nn
from ultralytics import YOLO


def adapt_pose_checkpoint(
    src_path: Path,
    dst_path: Path,
    remove_indices: List[int],
    original_kp_count: int = 16
) -> Path:
    print("=" * 70)
    print("  Adapting YOLO-Pose Checkpoint Keypoint Architecture")
    print(f"  Source Checkpoint: {src_path}")
    print(f"  Target Checkpoint: {dst_path}")
    print(f"  Removed Indices:   {remove_indices}")
    print("=" * 70)

    # 1. Load checkpoint safely
    ckpt = torch.load(str(src_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Checkpoint at {src_path} is missing 'model' key.")

    model = ckpt["model"].float()
    
    # Locate pose head (in YOLOv8 pose, it's typically layer 22)
    pose_head = None
    for m in model.model:
        if hasattr(m, "kpt_shape") and hasattr(m, "cv4"):
            pose_head = m
            break

    if pose_head is None:
        raise RuntimeError("Could not find Pose detection head in model.")

    old_kp_count = pose_head.kpt_shape[0] if pose_head.kpt_shape else original_kp_count
    remove_set = set(remove_indices)
    surviving_kps = [k for k in range(old_kp_count) if k not in remove_set]
    new_kp_count = len(surviving_kps)
    new_nk = new_kp_count * 3

    print(f"[*] Detected head keypoints: {old_kp_count}")
    print(f"[*] Surviving keypoints ({new_kp_count}): {surviving_kps}")
    print(f"[*] New output channels per anchor: {new_nk}")

    # Build channel slice indices (3 channels per keypoint: x, y, visibility)
    ch_indices = []
    for k in surviving_kps:
        ch_indices.extend([3 * k, 3 * k + 1, 3 * k + 2])
    ch_indices_t = torch.tensor(ch_indices, dtype=torch.long)

    # 2. Slice pose head conv layers
    for i in range(len(pose_head.cv4)):
        old_conv = pose_head.cv4[i][2]
        new_conv = nn.Conv2d(
            in_channels=old_conv.in_channels,
            out_channels=new_nk,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=True
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight[ch_indices_t])
            new_conv.bias.copy_(old_conv.bias[ch_indices_t])
        pose_head.cv4[i][2] = new_conv

    # 3. Update metadata on head and model
    pose_head.kpt_shape = [new_kp_count, 3]
    pose_head.nk = new_nk
    model.kpt_shape = [new_kp_count, 3]
    if hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model.yaml["kpt_shape"] = [new_kp_count, 3]

    # 4. Verify forward pass
    dummy = torch.zeros(1, 3, 320, 320)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    out_shape = out[0].shape if isinstance(out, tuple) else out.shape
    print(f"[*] Forward pass verified. Output shape: {out_shape}")

    # 5. Save adapted checkpoint
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt["model"] = model
    torch.save(ckpt, str(dst_path))
    print(f"[OK] Adapted checkpoint saved to: {dst_path}")

    # 6. Verify loading with Ultralytics YOLO
    yolo_model = YOLO(str(dst_path))
    print(f"[OK] Successfully verified model load with Ultralytics YOLO API.")
    return dst_path


def main():
    parser = argparse.ArgumentParser(description="Adapt YOLO-Pose checkpoint to new keypoint count.")
    parser.add_argument("--src", type=str, default="outputs/stage_a/run_shuffled_v3/weights/best.pt",
                        help="Source 16-keypoint checkpoint path")
    parser.add_argument("--dst", type=str, default="checkpoints/stage_a_13kp_adapted.pt",
                        help="Destination adapted checkpoint path")
    parser.add_argument("--remove_indices", type=str, default="5,12,15",
                        help="Comma-separated 0-indexed keypoints to remove (default: 5,12,15)")
    args = parser.parse_args()

    remove_indices = [int(x.strip()) for x in args.remove_indices.split(",") if x.strip()]
    adapt_pose_checkpoint(
        src_path=Path(args.src),
        dst_path=Path(args.dst),
        remove_indices=remove_indices
    )


if __name__ == "__main__":
    main()

