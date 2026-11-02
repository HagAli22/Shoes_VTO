import json
import numpy as np
from pathlib import Path

b_path = Path("deliverables/qualcomm_experiments/demo_videos/B1_twostage_nanodet_fp32_rtmpose_fp32.json")
old_s_path = Path("deliverables/qualcomm_experiments/stabilized_demo_videos/B1_twostage_nanodet_fp32_rtmpose_fp32.json")
new_s_path = Path("deliverables/qualcomm_experiments/parity_demo_videos/B1_twostage_nanodet_fp32_rtmpose_fp32.json")

with open(b_path, "r", encoding="utf-8") as f:
    b_data = json.load(f)
with open(old_s_path, "r", encoding="utf-8") as f:
    old_s_data = json.load(f)
with open(new_s_path, "r", encoding="utf-8") as f:
    new_s_data = json.load(f)

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / max(union, 1e-6)

def evaluate(exp_data, name):
    det_count = 0
    ious = []
    kpt_diffs = []
    jerks = []
    prev_kps = {"left": [], "right": []}
    tracking_frames = 0

    for i in range(min(len(b_data), len(exp_data))):
        bf = b_data[i]
        ef = exp_data[i]
        if ef.get("is_tracking_frame", False):
            tracking_frames += 1

        for side in ["left", "right"]:
            b_foot = bf["feet"].get(side)
            e_foot = ef["feet"].get(side)

            if e_foot is not None and e_foot.get("kps"):
                det_count += 1
                ek = np.array(e_foot["kps"])[:, :2]
                prev_kps[side].append(ek)
                if len(prev_kps[side]) >= 3:
                    jerk = np.mean(np.linalg.norm(prev_kps[side][-1] - 2 * prev_kps[side][-2] + prev_kps[side][-3], axis=-1))
                    jerks.append(jerk)
                    if len(prev_kps[side]) > 3:
                        prev_kps[side].pop(0)

            if b_foot is not None and e_foot is not None:
                if b_foot.get("kps") and e_foot.get("kps"):
                    bk = np.array(b_foot["kps"])[:, :2]
                    ek = np.array(e_foot["kps"])[:, :2]
                    diff = np.mean(np.linalg.norm(bk - ek, axis=-1))
                    kpt_diffs.append(diff)

                if b_foot.get("bbox") and e_foot.get("bbox"):
                    iou = compute_iou(b_foot["bbox"], e_foot["bbox"])
                    ious.append(iou)

    return {
        "name": name,
        "active_instances": det_count,
        "tracking_frames": tracking_frames,
        "skip_rate": tracking_frames / len(exp_data) * 100.0,
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "mean_kpt_diff": float(np.mean(kpt_diffs)),
        "median_kpt_diff": float(np.median(kpt_diffs)),
        "mean_jerk": float(np.mean(jerks))
    }

base_eval = evaluate(b_data, "Baseline Reference")
old_eval = evaluate(old_s_data, "Old Stabilized (Prior)")
new_eval = evaluate(new_s_data, "New Parity Stabilized (Current)")

old_skip_str = f"{old_eval['skip_rate']:.1f}%"
new_skip_str = f"{new_eval['skip_rate']:.1f}%"
old_iou_str = f"{old_eval['mean_iou']:.4f}"
new_iou_str = f"{new_eval['mean_iou']:.4f}"
old_miou_str = f"{old_eval['median_iou']:.4f}"
new_miou_str = f"{new_eval['median_iou']:.4f}"
old_kpt_str = f"{old_eval['mean_kpt_diff']:.2f} px"
new_kpt_str = f"{new_eval['mean_kpt_diff']:.2f} px"
old_mkpt_str = f"{old_eval['median_kpt_diff']:.2f} px"
new_mkpt_str = f"{new_eval['median_kpt_diff']:.2f} px"
base_jerk_str = f"{base_eval['mean_jerk']:.2f}"
old_jerk_str = f"{old_eval['mean_jerk']:.2f}"
new_jerk_str = f"{new_eval['mean_jerk']:.2f}"

print("=" * 95)
print(f"{'Metric':<32} | {'Baseline':<18} | {'Old Stabilized':<18} | {'New Parity':<18}")
print("=" * 95)
print(f"{'Total Active Instances':<32} | {base_eval['active_instances']:^18} | {old_eval['active_instances']:^18} | {new_eval['active_instances']:^18}")
print(f"{'NanoDet Skip Rate':<32} | {'0.0% (No Skips)':^18} | {old_skip_str:^18} | {new_skip_str:^18}")
print(f"{'Mean Bounding Box IoU':<32} | {'1.000 (Ref)':^18} | {old_iou_str:^18} | {new_iou_str:^18}")
print(f"{'Median Bounding Box IoU':<32} | {'1.000 (Ref)':^18} | {old_miou_str:^18} | {new_miou_str:^18}")
print(f"{'Mean Keypoint Distance':<32} | {'0.00 px (Ref)':^18} | {old_kpt_str:^18} | {new_kpt_str:^18}")
print(f"{'Median Keypoint Distance':<32} | {'0.00 px (Ref)':^18} | {old_mkpt_str:^18} | {new_mkpt_str:^18}")
print(f"{'Jitter Jerk (Lower = Better)':<32} | {base_jerk_str:^18} | {old_jerk_str:^18} | {new_jerk_str:^18}")
print("=" * 95)

# Save comparison results
report = {
    "baseline": base_eval,
    "old_stabilized": old_eval,
    "new_parity": new_eval
}
with open("deliverables/qualcomm_experiments/three_way_parity_comparison.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print("\n[+] Saved three-way comparison report to deliverables/qualcomm_experiments/three_way_parity_comparison.json")
