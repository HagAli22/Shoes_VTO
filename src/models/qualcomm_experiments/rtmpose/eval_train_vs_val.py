"""
rtmpose/eval_train_vs_val.py
────────────────────────────
Evaluates RTMPose-13KP on both TRAIN and VALIDATION splits.
Diagnoses overfitting vs underfitting:
  - Loss (SimCC KL + Visibility BCE)
  - PCK@0.05 (Strict)
  - PCK@0.10 (Relaxed)
  - Mean NME (Normalized Mean Error)
  - Per-keypoint pixel errors
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


def evaluate_split(model, criterion, data_dir: str, split: str = "valid", device=torch.device("cuda")):
    ds = FootPoseDataset13KP(
        images_dir=str(Path(data_dir) / split / "images"),
        img_size=256,
        mode="crop",
        augment=False
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_fn_rtmpose)

    total_loss = 0.0
    all_pck05 = []
    all_pck10 = []
    all_nme = []
    kp_errors = {i: [] for i in range(13)}
    ref_diag = np.sqrt(256.0**2 + 256.0**2)  # 362.038 px

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            kps_gt = batch["keypoints"].to(device)

            targets = criterion.build_targets(kps_gt, w_bins=512, h_bins=512, device=device)
            preds = model(imgs)
            loss = criterion(preds, targets)["total_loss"]
            total_loss += loss.item()

            pred_kps = decode_simcc(preds, simcc_split_ratio=2.0)
            gt_np = kps_gt.cpu().numpy()

            B = pred_kps.shape[0]
            for b in range(B):
                inst_pck05 = 0
                inst_pck10 = 0
                inst_total = 0
                inst_errs = []

                for k in range(13):
                    if gt_np[b, k, 2] > 0:
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

    mean_loss = total_loss / max(1, len(loader))
    pck05 = np.mean(all_pck05) * 100.0 if all_pck05 else 0.0
    pck10 = np.mean(all_pck10) * 100.0 if all_pck10 else 0.0
    nme = np.mean(all_nme) if all_nme else 0.0
    mean_kp_err = {k: np.mean(kp_errors[k]) if kp_errors[k] else 0.0 for k in range(13)}

    return {
        "num_crops": len(ds),
        "loss": mean_loss,
        "pck05": pck05,
        "pck10": pck10,
        "nme": nme,
        "kp_errors": mean_kp_err
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="checkpoints/rtmpose_13kp_official_coco/best.pt")
    parser.add_argument("--data", type=str, default="data/shuffled_v4_13kp")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    weights_path = args.weights
    data_dir = args.data
    device = torch.device(args.device)

    print(f"[*] Loading model from {weights_path} on {device}...")
    ckpt = torch.load(weights_path, map_location=device)
    backbone_name = ckpt.get("backbone_name", "cspnext")
    model = RTMPose13KP(num_keypoints=13, input_size=(256, 256), simcc_split_ratio=2.0, backbone_name=backbone_name, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    criterion = SimCCLoss(simcc_split_ratio=2.0, sigma=6.0).to(device)

    print("[*] Evaluating on TRAIN split...")
    train_metrics = evaluate_split(model, criterion, data_dir, split="train", device=device)

    print("[*] Evaluating on VALIDATION split...")
    val_metrics = evaluate_split(model, criterion, data_dir, split="valid", device=device)

    print("\n" + "=" * 76)
    print("      QUALCOMM RTMPOSE-13KP: TRAIN vs. VALIDATION COMPARATIVE AUDIT")
    print("=" * 76)
    print(f"{'Metric':<26} | {'TRAIN Split (' + str(train_metrics['num_crops']) + ' crops)':<22} | {'VAL Split (' + str(val_metrics['num_crops']) + ' crops)':<22}")
    print("-" * 76)
    print(f"{'Loss (SimCC + Vis)':<26} | {train_metrics['loss']:<22.4f} | {val_metrics['loss']:<22.4f}")
    print(f"{'PCK@0.05 (Strict)':<26} | {train_metrics['pck05']:<21.2f}% | {val_metrics['pck05']:<21.2f}%")
    print(f"{'PCK@0.10 (Relaxed)':<26} | {train_metrics['pck10']:<21.2f}% | {val_metrics['pck10']:<21.2f}%")
    print(f"{'Mean NME (Normalized)':<26} | {train_metrics['nme']:<22.4f} | {val_metrics['nme']:<22.4f}")
    print("=" * 76)
    print("\nPer-Keypoint Pixel Error (256x256 patch):")
    print(f"{'Keypoint Name':<22} | {'Train Error (px)':<18} | {'Val Error (px)':<18} | {'Gap (Overfitting)':<18}")
    print("-" * 76)
    for k in range(13):
        t_err = train_metrics["kp_errors"][k]
        v_err = val_metrics["kp_errors"][k]
        gap = v_err - t_err
        print(f"[{k:02d}] {KEYPOINT_NAMES[k]:<18} | {t_err:<18.2f} | {v_err:<18.2f} | {gap:+18.2f}")
    print("=" * 76)


if __name__ == "__main__":
    main()

