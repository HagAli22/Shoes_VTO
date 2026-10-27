import numpy as np
from pathlib import Path

labels_dir = Path("data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/valid/labels")
label_files = list(labels_dir.glob("*.txt"))

correct_geom = 0
total_geom = 0

for lf in label_files:
    with open(lf, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id not in (0, 1):
                continue
            kp_parts = parts[5:]
            if len(kp_parts) < 39:
                continue
            kps = np.array([float(x) for x in kp_parts]).reshape(-1, 3)  # [13, 3]

            # 2: heel_ground, 0: toe_ground, 10: toe_tip, 3: ball_medial, 4: ball_lateral, 6: arch_medial, 7: midfoot_lateral
            # Check visibility
            if kps[2, 2] == 0 or (kps[0, 2] == 0 and kps[10, 2] == 0):
                continue
            toe = kps[0, :2] if kps[0, 2] > 0 else kps[10, :2]
            heel = kps[2, :2]
            axis = toe - heel

            # 2D cross product: vx * my - vy * mx
            votes = []
            if kps[3, 2] > 0 and kps[4, 2] > 0:
                # Transverse vector from lateral to medial
                trans = kps[3, :2] - kps[4, :2]
                cross_t = axis[0] * trans[1] - axis[1] * trans[0]
                votes.append(cross_t > 0)
            if kps[6, 2] > 0 and kps[7, 2] > 0:
                trans_arch = kps[6, :2] - kps[7, :2]
                cross_a = axis[0] * trans_arch[1] - axis[1] * trans_arch[0]
                votes.append(cross_a > 0)

            if len(votes) == 0:
                continue

            # On image coordinates (y points down):
            pred_is_left = sum(votes) > len(votes) / 2
            pred_cls = 0 if pred_is_left else 1
            if pred_cls == cls_id:
                correct_geom += 1
            total_geom += 1

print(f"Total evaluated: {total_geom}")
print(f"Direct match: {correct_geom}/{total_geom} = {correct_geom/total_geom*100:.2f}%")
print(f"Inverted match: {total_geom - correct_geom}/{total_geom} = {(total_geom - correct_geom)/total_geom*100:.2f}%")

