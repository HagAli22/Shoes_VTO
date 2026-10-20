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


def adapt_13kp_to_15kp(
    src_path: Path,
    dst_path: Path
) -> Path:
    """
    Adapts a modern 13-keypoint checkpoint (v7) into a 15-keypoint architecture
    restoring 'ball_top' (index 5) and 'shin_mid' (index 14), while excluding 'ankle_center' (Old 12).
    Preserves 100% of the backbone, neck, box head, class head, and 13 keypoint weights.
    """
    print("=" * 70)
    print("  Adapting Modern 13-KP Checkpoint to 15-KP Architecture")
    print(f"  Source (Modern 13-KP): {src_path}")
    print(f"  Target (New 15-KP):    {dst_path}")
    print("  Restored Keypoints:    ball_top (Index 5), shin_mid (Index 14) (Excluding ankle_center)")
    print("=" * 70)

    ckpt = torch.load(str(src_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"Checkpoint at {src_path} is missing 'model' key.")

    model = ckpt["model"].float()

    pose_head = None
    for m in model.model:
        if hasattr(m, "kpt_shape") and hasattr(m, "cv4"):
            pose_head = m
            break

    if pose_head is None:
        raise RuntimeError("Could not find Pose detection head in model.")

    new_nk = 15 * 3  # 45 channels
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
            # Copy 0..4 (toe_ground, heel_back, heel_ground, ball_medial, ball_lateral)
            new_conv.weight[:15].copy_(old_conv.weight[:15])
            new_conv.bias[:15].copy_(old_conv.bias[:15])

            # 5: ball_top -> average of ball_medial (channels 9..12) and ball_lateral (channels 12..15)
            new_conv.weight[15:18].copy_(0.5 * (old_conv.weight[9:12] + old_conv.weight[12:15]))
            new_conv.bias[15:18].copy_(0.5 * (old_conv.bias[9:12] + old_conv.bias[12:15]))

            # Copy 6..13 (instep_top, arch_medial, midfoot_lat, mall_medial, mall_lat, toe_tip, throat, achilles)
            new_conv.weight[18:42].copy_(old_conv.weight[15:39])
            new_conv.bias[18:42].copy_(old_conv.bias[15:39])

            # 14: shin_mid -> initialized from throat (channels 33..36 in 13-KP)
            new_conv.weight[42:45].copy_(old_conv.weight[33:36])
            new_conv.bias[42:45].copy_(old_conv.bias[33:36])

        pose_head.cv4[i][2] = new_conv

    pose_head.kpt_shape = [15, 3]
    pose_head.nk = new_nk
    model.kpt_shape = [15, 3]
    if hasattr(model, "yaml") and isinstance(model.yaml, dict):
        model.yaml["kpt_shape"] = [15, 3]

    dummy = torch.zeros(1, 3, 320, 320)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    out_shape = out[0].shape if isinstance(out, tuple) else out.shape
    print(f"[*] Forward pass verified. Output shape: {out_shape}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt["model"] = model
    torch.save(ckpt, str(dst_path))
    print(f"[OK] 15-KP adapted checkpoint saved to: {dst_path}")

    yolo_model = YOLO(str(dst_path))
    print(f"[OK] Successfully verified model load with Ultralytics YOLO API.")
    return dst_path


def main():
    parser = argparse.ArgumentParser(description="Adapt YOLO-Pose checkpoint to new keypoint count.")
    parser.add_argument("--src", type=str, default="deliverables/stage_a_13kp/models/stage-a-320-13kp-v7.pt",
                        help="Source checkpoint path")
    parser.add_argument("--dst", type=str, default="checkpoints/stage_a_15kp_adapted.pt",
                        help="Destination adapted checkpoint path")
    parser.add_argument("--target_kps", type=int, default=15, choices=[13, 15],
                        help="Target keypoints: 15 (restore ball_top & shin_mid from 13-KP) or 13 (slice from 16-KP)")
    parser.add_argument("--remove_indices", type=str, default="5,12,15",
                        help="Comma-separated 0-indexed keypoints to remove when slicing 16-KP down (default: 5,12,15)")
    args = parser.parse_args()

    src_p = Path(args.src)
    dst_p = Path(args.dst)

    if args.target_kps == 15:
        adapt_13kp_to_15kp(src_path=src_p, dst_path=dst_p)
    else:
        remove_indices = [int(x.strip()) for x in args.remove_indices.split(",") if x.strip()]
        adapt_pose_checkpoint(
            src_path=src_p,
            dst_path=dst_p,
            remove_indices=remove_indices
        )


if __name__ == "__main__":
    main()

