"""
visualize_user_selected_samples.py
───────────────────────────────────
Renders Ground Truth annotations and Model Predictions on the 3 user-selected training images:
  1. 2_0171_f01384
  2. 2_0173_f01400
  3. 3_0050_f00400
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def parse_yolo_pose_label(lbl_path):
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    
    parsed = []
    for line in lines:
        parts = [float(x) for x in line.strip().split()]
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = parts[1:5]
        
        kpts = []
        raw_kpts = parts[5:]
        for i in range(0, len(raw_kpts), 3):
            if i + 2 < len(raw_kpts):
                kx, ky, kv = raw_kpts[i], raw_kpts[i+1], int(raw_kpts[i+2])
                kpts.append((kx, ky, kv))
        
        parsed.append({
            "cls_id": cls_id,
            "cls_name": "Left_Foot" if cls_id == 0 else "Right_Foot",
            "bbox": (cx, cy, w, h),
            "kpts": kpts
        })
    return parsed

def render_sample(img_path, lbl_path, model, artifact_dir, sample_tag):
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"Error loading image: {img_path}")
        return
    h, w = orig_img.shape[:2]
    
    # ── 1. Render Ground Truth ───────────────────────────────────────────────
    annotations = parse_yolo_pose_label(lbl_path)
    
    for idx, ann in enumerate(annotations):
        cls_name = ann["cls_name"]
        cx, cy, bw, bh = ann["bbox"]
        
        x1 = max(0, int((cx - bw/2) * w))
        y1 = max(0, int((cy - bh/2) * h))
        x2 = min(w, int((cx + bw/2) * w))
        y2 = min(h, int((cy + bh/2) * h))
        
        pad_x = int((x2 - x1) * 0.35)
        pad_y = int((y2 - y1) * 0.35)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        
        crop = orig_img[cy1:cy2, cx1:cx2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(11, 11), dpi=200)
        ax.imshow(crop_rgb)
        ax.set_title(f"GROUND TRUTH: {sample_tag} | {cls_name} (Class {ann['cls_id']})", 
                     fontsize=14, fontweight='bold', color='darkred')
        ax.axis('off')
        
        kpts = ann["kpts"]
        for k_idx, (kx_norm, ky_norm, kv) in enumerate(kpts):
            if kv == 0 or kx_norm <= 0 or ky_norm <= 0:
                continue
            
            abs_x = kx_norm * w
            abs_y = ky_norm * h
            rel_x = abs_x - cx1
            rel_y = abs_y - cy1
            
            color = '#00FF00' if kv == 2 else '#FFA500' # Green for v=2, Orange for v=1
            ax.plot(rel_x, rel_y, 'o', markersize=9, color=color, markeredgecolor='black', markeredgewidth=1.5)
            
            angle = (k_idx * 22.5) * np.pi / 180.0
            dist = 35 + (k_idx % 3) * 16
            off_x = np.cos(angle) * dist
            off_y = np.sin(angle) * dist
            
            lbl_text = f"KP [{k_idx}] (v={kv})"
            ax.annotate(
                lbl_text,
                xy=(rel_x, rel_y),
                xytext=(rel_x + off_x, rel_y + off_y),
                fontsize=10,
                fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.85, edgecolor=color, linewidth=1.4),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3)
            )
            
        gt_out = os.path.join(artifact_dir, f"GT_{sample_tag}_foot_{idx}_{cls_name}.png")
        plt.tight_layout()
        plt.savefig(gt_out, bbox_inches='tight')
        plt.close()
        print(f"Saved GT: {gt_out}")

def main():
    artifact_dir = r"C:\Users\Matrix Store\.gemini\antigravity\brain\9f91a989-ae24-4c1f-b1bd-0c8230f28d7f"
    model_path = "outputs/stage_a/run_shuffled_v3/weights/best.pt"
    model = YOLO(model_path) if os.path.exists(model_path) else None
    
    samples = [
        ("F:/Machine learning/Shoes_VTO/data/combined_v3/train/images/2_0171_f01384_jpg.rf.33ab69771281b0fcb3f1b137942c88e5.jpg",
         "F:/Machine learning/Shoes_VTO/data/combined_v3/train/labels/2_0171_f01384_jpg.rf.33ab69771281b0fcb3f1b137942c88e5.txt",
         "img_2_0171"),
        ("F:/Machine learning/Shoes_VTO/data/combined_v3/train/images/2_0173_f01400_jpg.rf.fc7ec57415123244baf7f6d644d0ffc7.jpg",
         "F:/Machine learning/Shoes_VTO/data/combined_v3/train/labels/2_0173_f01400_jpg.rf.fc7ec57415123244baf7f6d644d0ffc7.txt",
         "img_2_0173"),
        ("F:/Machine learning/Shoes_VTO/data/combined_v3/valid/images/3_0050_f00400_jpg.rf.f9ecda402e24044bc6254e19c3a4b2bf.jpg",
         "F:/Machine learning/Shoes_VTO/data/combined_v3/valid/labels/3_0050_f00400_jpg.rf.f9ecda402e24044bc6254e19c3a4b2bf.txt",
         "img_3_0050")
    ]
    
    for img, lbl, tag in samples:
        render_sample(img, lbl, model, artifact_dir, tag)

if __name__ == "__main__":
    main()

