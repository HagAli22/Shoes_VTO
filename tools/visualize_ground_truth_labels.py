"""
visualize_ground_truth_labels.py
─────────────────────────────────
Draws ground truth keypoints (indices 0..15) directly from the dataset .txt annotation files
onto the original dataset images.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def parse_yolo_pose_label(lbl_path):
    with open(lbl_path, "r") as f:
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

def render_gt(img_path, lbl_path, artifact_dir, sample_name):
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        return
    h, w = orig_img.shape[:2]
    
    annotations = parse_yolo_pose_label(lbl_path)
    
    for idx, ann in enumerate(annotations):
        cls_name = ann["cls_name"]
        cx, cy, bw, bh = ann["bbox"]
        
        # Pixel bbox
        x1 = max(0, int((cx - bw/2) * w))
        y1 = max(0, int((cy - bh/2) * h))
        x2 = min(w, int((cx + bw/2) * w))
        y2 = min(h, int((cy + bh/2) * h))
        
        # Crop with margin
        pad_x = int((x2 - x1) * 0.3)
        pad_y = int((y2 - y1) * 0.3)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        
        crop = orig_img[cy1:cy2, cx1:cx2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.imshow(crop_rgb)
        ax.set_title(f"GROUND TRUTH: {sample_name} | {cls_name} (Class {ann['cls_id']})", 
                     fontsize=14, fontweight='bold', color='darkred')
        ax.axis('off')
        
        kpts = ann["kpts"]
        for k_idx, (kx_norm, ky_norm, kv) in enumerate(kpts):
            if kv == 0 or kx_norm <= 0 or ky_norm <= 0:
                continue # not labeled
            
            abs_x = kx_norm * w
            abs_y = ky_norm * h
            
            rel_x = abs_x - cx1
            rel_y = abs_y - cy1
            
            # Draw point
            color = 'lime' if kv == 2 else 'orange'
            ax.plot(rel_x, rel_y, 'o', markersize=9, color=color, markeredgecolor='black', markeredgewidth=1.5)
            
            # Label with just the index number and visibility
            angle = (k_idx * 22.5) * np.pi / 180.0
            dist = 30 + (k_idx % 3) * 15
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
                bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.85, edgecolor=color, linewidth=1.2),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2)
            )
            
        out_path = os.path.join(artifact_dir, f"GT_{sample_name}_foot_{idx}_{cls_name}.png")
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved: {out_path}")

def main():
    artifact_dir = r"C:\Users\Matrix Store\.gemini\antigravity\brain\9f91a989-ae24-4c1f-b1bd-0c8230f28d7f"
    
    # Let's pick 2 different images and labels from train/test
    samples = [
        ("data/shuffled_v3/test/images/1_0013_f00104_jpg.rf.bdca68355d89800af24f6e253db00e50.jpg",
         "data/shuffled_v3/test/labels/1_0013_f00104_jpg.rf.bdca68355d89800af24f6e253db00e50.txt",
         "sample1"),
        ("data/shuffled_v3/test/images/1_0027_f00216_jpg.rf.7131ba9f0d84017407c3a03a7b638490.jpg",
         "data/shuffled_v3/test/labels/1_0027_f00216_jpg.rf.7131ba9f0d84017407c3a03a7b638490.txt",
         "sample2")
    ]
    
    for img, lbl, name in samples:
        if os.path.exists(img) and os.path.exists(lbl):
            render_gt(img, lbl, artifact_dir, name)

if __name__ == "__main__":
    main()

