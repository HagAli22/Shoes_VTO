"""
rtmpose/eval_val_metrics.py
───────────────────────────
Evaluates the trained RTMPose-13KP model on the validation dataset.
Computes:
  - Validation Loss (SimCC KL + Visibility BCE)
  - PCK@0.05 (% keypoints correct within 5% of foot bounding box diagonal)
  - PCK@0.10 (% keypoints correct within 10% of foot bounding box diagonal)
  - Mean NME (Normalized Mean Error)
  - Mean pixel error per keypoint
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def evaluate_rtmpose_metrics(
    weights_path: str = "checkpoints/rtmpose_13kp/best.pt",
    data_dir: str = "data/shuffled_v4_13kp",
    device_name: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    device = torch.device(device_name)
    val_ds = FootPoseDataset13KP(
        images_dir=str(Path(data_dir) / "valid" / "images"),
        img_size=256,
        mode="crop",
        augment=False
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn_rtmpose)

    model = RTMPose13KP(num_keypoints=13, input_size=(256, 256), simcc_split_ratio=2.0).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    criterion = SimCCLoss(simcc_split_ratio=2.0, sigma=6.0).to(device)

    total_loss = 0.0
    all_pck05 = []
    all_pck10 = []
    all_nme = []
    kp_errors = {i: [] for i in range(13)}

    ref_diag = np.sqrt(256.0**2 + 256.0**2)  # crop diagonal = 362.0 px

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["images"].to(device)
            kps_gt = batch["keypoints"].to(device)

            targets = criterion.build_targets(kps_gt, w_bins=512, h_bins=512, device=device)
            preds = model(imgs)
            loss = criterion(preds, targets)["total_loss"]
            total_loss += loss.item()

            pred_kps = decode_simcc(preds, simcc_split_ratio=2.0)  # [B, 13, 3]
            gt_np = kps_gt.cpu().numpy()

            B = pred_kps.shape[0]
            for b in range(B):
                inst_pck05 = 0
                inst_pck10 = 0
                inst_total = 0
                inst_errs = []

                for k in range(13):
                    if gt_np[b, k, 2] > 0:  # labeled / visible
                        dist = np.linalg.norm(pred_kps[b, k, :2] - gt_np[b, k, :2])
                        inst_errs.append(dist / ref_diag)
                        kp_errors[k].append(dist)

                        if dist <= 0.05 * ref_diag:
                            inst_pck05 += 1
                        if dist <= 0.10 * ref_diag:
                            inst_pck10 += 1
                        inst_total += 1

                if inst_total > 0:
                    all_pck05.append(inst_pck05 / inst_total)
                    all_pck10.append(inst_pck10 / inst_total)
                    all_nme.append(np.mean(inst_errs))

    mean_loss = total_loss / max(1, len(val_loader))
    pck05 = np.mean(all_pck05) * 100.0
    pck10 = np.mean(all_pck10) * 100.0
    nme = np.mean(all_nme)

    print("\n" + "=" * 65)
    print("  QUALCOMM RTMPOSE-13KP VALIDATION METRICS (186 FOOT CROPS)")
    print("=" * 65)
    print(f"  Best Val Loss:       {mean_loss:.4f}")
    print(f"  PCK@0.05 (Strict):   {pck05:.2f}%")
    print(f"  PCK@0.10 (Relaxed):  {pck10:.2f}%")
    print(f"  Mean NME:            {nme:.4f}")
    print("-" * 65)
    print("  Per-Keypoint Pixel Error (on 256x256 crop):")
    for k in range(13):
        err = np.mean(kp_errors[k]) if kp_errors[k] else 0.0
        print(f"    [{k:02d}] {KEYPOINT_NAMES[k]:<18}: {err:.2f} px")
    print("=" * 65)


if __name__ == "__main__":
    evaluate_rtmpose_metrics()

