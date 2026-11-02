"""
tools/evaluation/benchmark_rtmpose_compressed.py
────────────────────────────────────────────────
Comprehensive benchmark comparing Baseline RTMPose-13KP (improved5) against:
  - Option A: RTMPose-Slim (Stride 16, ~3.04 MB)
  - Option A+Lite: RTMPose-Slim-Stage4Lite (Stride 32, ~4.25 MB)
  - Option B: RTMPose-Pico (Width 0.5x, ~3.32 MB)

Evaluates:
  1. Parameter Counts & Memory Footprint (FP32, FP16, INT8)
  2. Latency Benchmarks (Batch-1 and Batch-2 on CPU / GPU)
  3. Foot Keypoint Accuracy (PCK@0.05, PCK@0.10, Mean Error)
  4. Handedness (Chirality) Classification Accuracy
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.common.dataset import FootPoseDataset13KP
from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP, load_improved5_weights
from src.models.qualcomm_experiments.rtmpose.decoder import decode_simcc
from src.models.qualcomm_experiments.rtmpose.train import collate_fn_rtmpose


def measure_latency(model: torch.nn.Module, device: torch.device, batch_size: int = 1, iterations: int = 50) -> Tuple[float, float]:
    dummy = torch.randn(batch_size, 3, 256, 256, device=device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)

    return float(np.mean(timings)), float(np.std(timings))


def evaluate_accuracy(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 25
) -> Dict[str, float]:
    model.eval()
    ref_diag = math.sqrt(256.0**2 + 256.0**2)
    all_pck05, all_pck10, errs = [], [], []
    hand_correct, hand_total = 0, 0

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            imgs = batch["images"].to(device)
            kps_gt = batch["keypoints"].to(device)
            classes = batch["classes"].to(device)

            preds = model(imgs)
            pred_kps = decode_simcc(preds, simcc_split_ratio=2.0)
            gt_np = kps_gt.cpu().numpy()

            for b in range(len(imgs)):
                p05, p10, tot = 0, 0, 0
                for k in range(13):
                    if gt_np[b, k, 2] > 0:
                        dist = float(np.linalg.norm(pred_kps[b, k, :2] - gt_np[b, k, :2]))
                        errs.append(dist)
                        if dist <= 0.05 * ref_diag:
                            p05 += 1
                        if dist <= 0.10 * ref_diag:
                            p10 += 1
                        tot += 1

                if tot > 0:
                    all_pck05.append(p05 / tot)
                    all_pck10.append(p10 / tot)

                if "hand" in preds:
                    pred_cls = 1 if preds["hand"][b].item() >= 0.5 else 0
                    if classes[b].item() >= 0:
                        hand_correct += int(pred_cls == classes[b].item())
                        hand_total += 1

    mean_pck05 = float(np.mean(all_pck05) * 100.0) if all_pck05 else 0.0
    mean_pck10 = float(np.mean(all_pck10) * 100.0) if all_pck10 else 0.0
    mean_err = float(np.mean(errs)) if errs else 999.0
    hand_acc = float(hand_correct / max(1, hand_total) * 100.0) if hand_total > 0 else 0.0

    return {
        "pck05": mean_pck05,
        "pck10": mean_pck10,
        "mean_err": mean_err,
        "hand_acc": hand_acc
    }


def run_benchmark(
    val_data_path: str = "data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid/images",
    teacher_ckpt: str = "checkpoints/rtmpose_13kp_improved5/best_with_hand.pt",
    output_json: str = "deliverables/qualcomm_experiments/rtmpose_compression_benchmark.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running benchmark on device: {device}")

    # Load validation dataset if present
    val_loader = None
    if Path(val_data_path).exists():
        val_ds = FootPoseDataset13KP(images_dir=val_data_path, img_size=256, mode="crop", augment=False)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn_rtmpose)
        print(f"[+] Validation dataset loaded: {len(val_ds)} crops.")
    else:
        print(f"[!] Validation dataset path {val_data_path} not found. Running synthetic latency & param tests.")

    # Model configurations to evaluate
    configs = [
        {
            "id": "baseline_improved5",
            "name": "RTMPose-13KP Baseline (improved5)",
            "backbone": "cspnext",
            "weights": teacher_ckpt,
            "hidden_dim": 256,
            "head_conv": "standard",
            "hand_head": "flatten"
        },
        {
            "id": "rtmpose_slim_optionA",
            "name": "Option A: RTMPose-Slim (Stride-16)",
            "backbone": "cspnext_slim",
            "weights": teacher_ckpt,
            "hidden_dim": 128,
            "head_conv": "separable",
            "hand_head": "pooling"
        },
        {
            "id": "rtmpose_slim_s4lite",
            "name": "Option A+Lite: RTMPose-Slim-Stage4Lite",
            "backbone": "cspnext_slim_s4lite",
            "weights": teacher_ckpt,
            "hidden_dim": 128,
            "head_conv": "separable",
            "hand_head": "pooling"
        },
        {
            "id": "rtmpose_pico_optionB",
            "name": "Option B: RTMPose-Pico (0.5x width)",
            "backbone": "cspnext_pico",
            "weights": teacher_ckpt,
            "hidden_dim": 128,
            "head_conv": "separable",
            "hand_head": "pooling"
        }
    ]

    results = []

    print("\n" + "=" * 105)
    print(f"{'Model Configuration':35s} | {'Params':>10s} | {'FP32 MB':>8s} | {'INT8 MB':>8s} | {'B1 CPU ms':>10s} | {'B2 CPU ms':>10s} | {'PCK@0.10':>9s} | {'Mean Err':>9s}")
    print("=" * 105)

    for cfg in configs:
        model = RTMPose13KP(
            backbone_name=cfg["backbone"],
            pretrained=False,
            hidden_dim=cfg["hidden_dim"],
            head_conv_type=cfg["head_conv"],
            hand_head_type=cfg["hand_head"]
        )

        if Path(cfg["weights"]).exists():
            if cfg["backbone"] == "cspnext":
                try:
                    ckpt = torch.load(cfg["weights"], map_location="cpu", weights_only=False)
                except TypeError:
                    ckpt = torch.load(cfg["weights"], map_location="cpu")
                model.load_state_dict(ckpt.get("model_state", ckpt), strict=True)
            else:
                load_improved5_weights(model, cfg["weights"])

        model.to(device)

        total_p = sum(p.numel() for p in model.parameters())
        fp32_mb = total_p * 4 / (1024 * 1024)
        int8_mb = total_p / (1024 * 1024)

        # Measure latency
        b1_mean, b1_std = measure_latency(model, device, batch_size=1, iterations=40)
        b2_mean, b2_std = measure_latency(model, device, batch_size=2, iterations=40)

        # Evaluate accuracy if validation data is available
        acc = {"pck05": 0.0, "pck10": 0.0, "mean_err": 0.0, "hand_acc": 0.0}
        if val_loader is not None:
            acc = evaluate_accuracy(model, val_loader, device, max_batches=15)

        entry = {
            "id": cfg["id"],
            "name": cfg["name"],
            "backbone": cfg["backbone"],
            "parameters": total_p,
            "size_fp32_mb": round(fp32_mb, 2),
            "size_int8_mb": round(int8_mb, 2),
            "latency_b1_ms": round(b1_mean, 2),
            "latency_b2_ms": round(b2_mean, 2),
            "pck05": round(acc["pck05"], 2),
            "pck10": round(acc["pck10"], 2),
            "mean_err_px": round(acc["mean_err"], 2),
            "hand_acc": round(acc["hand_acc"], 2)
        }
        results.append(entry)

        print(f"{cfg['name']:35s} | {total_p:10,d} | {fp32_mb:7.2f}M | {int8_mb:7.2f}M | {b1_mean:8.2f} ms | {b2_mean:8.2f} ms | {acc['pck10']:8.2f}% | {acc['mean_err']:6.2f} px")

    print("=" * 105)

    # Save JSON summary
    out_file = Path(output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Benchmark results saved to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid/images")
    parser.add_argument("--teacher", type=str, default="checkpoints/rtmpose_13kp_improved5/best_with_hand.pt")
    parser.add_argument("--output", type=str, default="deliverables/qualcomm_experiments/rtmpose_compression_benchmark.json")
    args = parser.parse_args()
    run_benchmark(val_data_path=args.data, teacher_ckpt=args.teacher, output_json=args.output)

