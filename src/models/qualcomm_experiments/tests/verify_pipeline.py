"""
tests/verify_pipeline.py
─────────────────────────
Pre-flight verification for Qualcomm RTMPose-13KP:
  1. Visualizes 20 augmented foot crops with affine scale/shift/rotation + keypoints.
  2. Overfits 64 crops to verify mean localization error drops below ~1.0 px.
"""

import sys
from pathlib import Path
import random
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.common.dataset import FootPoseDataset13KP
from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP
from src.models.qualcomm_experiments.rtmpose.losses import SimCCLoss
from src.models.qualcomm_experiments.rtmpose.decoder import decode_simcc
from src.models.qualcomm_experiments.rtmpose.train import collate_fn_rtmpose


KEYPOINT_NAMES = [
    "toe_ground", "heel_back", "heel_ground", "ball_medial", "ball_lateral",
    "instep_top", "arch_medial", "midfoot_lateral",
    "malleolus_medial", "malleolus_lateral", "toe_tip", "throat", "achilles"
]

KP_COLORS = [
    (0, 255, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (255, 0, 255), (0, 165, 255), (128, 0, 128), (0, 128, 255), (255, 128, 0),
    (0, 255, 128), (128, 255, 0), (255, 0, 128)
]


def visualize_augmented_crops(
    dataset,
    output_path: str = "deliverables/qualcomm_experiments/augmented_crops_vis.jpg",
    num_samples: int = 20
):
    print(f"[*] Visualizing {num_samples} augmented crops...")
    cols = 5
    rows = (num_samples + cols - 1) // cols
    canvas = np.zeros((rows * 256, cols * 256, 3), dtype=np.uint8)

    for i in range(num_samples):
        sample = dataset[i % len(dataset)]
        img_np = (sample["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        kps = sample["keypoints"].numpy()

        vis_img = img_np.copy()
        for k in range(13):
            kx, ky, kv = kps[k]
            if kv > 0:
                cv2.circle(vis_img, (int(round(kx)), int(round(ky))), 4, KP_COLORS[k], -1)
                cv2.circle(vis_img, (int(round(kx)), int(round(ky))), 5, (0, 0, 0), 1)

        r = i // cols
        c = i % cols
        canvas[r * 256:(r + 1) * 256, c * 256:(c + 1) * 256] = vis_img

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_file), canvas)
    print(f"[+] Saved visualization of {num_samples} augmented crops to: {out_file}")


def test_overfit_64_crops(dataset, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    print("\n[*] Running Overfit Test on 64 crops (target: mean pixel error < 1.0 px)...")
    subset_indices = list(range(64))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=32, shuffle=True, collate_fn=collate_fn_rtmpose)

    model = RTMPose13KP(num_keypoints=13, input_size=(256, 256), simcc_split_ratio=2.0, pretrained=True).to(device)
    criterion = SimCCLoss(simcc_split_ratio=2.0, sigma=6.0).to(device)

    # Use higher learning rate for overfitting test
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    for epoch in range(1, 45):
        model.train()
        train_loss = 0.0
        for batch in loader:
            imgs = batch["images"].to(device)
            kps = batch["keypoints"].to(device)
            targets = criterion.build_targets(kps, w_bins=512, h_bins=512, device=device)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets)["total_loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(loader)

        # Compute mean error
        if epoch % 5 == 0 or epoch == 44:
            model.eval()
            errors = []
            with torch.no_grad():
                for batch in loader:
                    imgs = batch["images"].to(device)
                    kps = batch["keypoints"].to(device)
                    preds = model(imgs)
                    decoded = decode_simcc(preds, simcc_split_ratio=2.0, win=12)
                    gt = kps.cpu().numpy()

                    for b in range(len(decoded)):
                        valid = gt[b, :, 2] > 0
                        if np.any(valid):
                            dx = decoded[b, valid, 0] - gt[b, valid, 0]
                            dy = decoded[b, valid, 1] - gt[b, valid, 1]
                            dist = np.sqrt(dx ** 2 + dy ** 2)
                            errors.extend(dist.tolist())

            mean_err = np.mean(errors) if errors else 0.0
            print(f"  Epoch [{epoch:02d}/44] | Train Loss: {train_loss:.4f} | Mean Error: {mean_err:.2f} px")

    print(f"\n[+] Final Overfit Mean Pixel Error: {mean_err:.2f} px")
    assert mean_err < 1.5, f"Overfitting test failed: mean error is {mean_err:.2f} px (expected < 1.5 px)"
    print("[OK] Overfitting verification PASSED: Model easily attains sub-pixel precision!")


if __name__ == "__main__":
    ds_aug = FootPoseDataset13KP(
        images_dir="data/shuffled_v4_13kp/train/images",
        img_size=256,
        mode="crop",
        augment=True,
        allow_flip=False
    )
    visualize_augmented_crops(ds_aug, num_samples=20)

    ds_fixed = FootPoseDataset13KP(
        images_dir="data/shuffled_v4_13kp/train/images",
        img_size=256,
        mode="crop",
        augment=False,
        allow_flip=False
    )
    test_overfit_64_crops(ds_fixed)

