"""
visualize_keypoint_crops.py
───────────────────────────
Crops each detected foot with a margin and draws keypoints with large, crystal-clear labels.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "0: toe_tip",
    "1: toe_ground",
    "2: heel_back",
    "3: heel_ground",
    "4: ball_medial",
    "5: ball_lateral",
    "6: ball_top",
    "7: instep_top",
    "8: arch_medial",
    "9: midfoot_lateral",
    "10: malleolus_medial",
    "11: malleolus_lateral",
    "12: ankle_center",
    "13: throat",
    "14: achilles",
    "15: shin_mid"
]

def annotate_crops(img_path, model, artifact_dir, prefix):
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        return
    h, w = orig_img.shape[:2]
    
    results = model.predict(img_path, imgsz=320, conf=0.25, verbose=False)[0]
    if results.keypoints is None or len(results.keypoints) == 0:
        return
    
    kpts_data = results.keypoints.data.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    
    for det_idx in range(len(boxes)):
        box = boxes[det_idx]
        cls_id = int(classes[det_idx])
        cls_name = "Left_Foot" if cls_id == 0 else "Right_Foot"
        conf = confs[det_idx]
        
        # Crop bbox with 20% margin
        bx1, by1, bx2, by2 = box
        bw, bh = bx2 - bx1, by2 - by1
        pad_x, pad_y = bw * 0.25, bh * 0.25
        cx1 = max(0, int(bx1 - pad_x))
        cy1 = max(0, int(by1 - pad_y))
        cx2 = min(w, int(bx2 + pad_x))
        cy2 = min(h, int(by2 + pad_y))
        
        crop = orig_img[cy1:cy2, cx1:cx2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=200)
        ax.imshow(crop_rgb)
        ax.set_title(f"{prefix} - {cls_name} (conf={conf:.2f})", fontsize=14, fontweight='bold', color='navy')
        ax.axis('off')
        
        kpts = kpts_data[det_idx]
        for k in range(min(16, len(kpts))):
            kx, ky = kpts[k][0], kpts[k][1]
            kv = kpts[k][2] if len(kpts[k]) > 2 else 1.0
            
            if kv < 0.2 or kx <= 0 or ky <= 0:
                continue
            
            # Adjust to crop coordinates
            rel_x = kx - cx1
            rel_y = ky - cy1
            
            ax.plot(rel_x, rel_y, 'o', markersize=8, color='red', markeredgecolor='white', markeredgewidth=1.5)
            
            # Staggered offsets
            angle = (k * 22.5) * np.pi / 180.0
            dist = 35 + (k % 4) * 12
            off_x = np.cos(angle) * dist
            off_y = np.sin(angle) * dist
            
            ax.annotate(
                KEYPOINT_NAMES[k],
                xy=(rel_x, rel_y),
                xytext=(rel_x + off_x, rel_y + off_y),
                fontsize=9,
                fontweight='bold',
                color='yellow',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.85, edgecolor='yellow', linewidth=0.8),
                arrowprops=dict(arrowstyle='->', color='yellow', lw=1.0)
            )
            
        out_crop_path = os.path.join(artifact_dir, f"{prefix}_foot_{det_idx}_{cls_name}.png")
        plt.tight_layout()
        plt.savefig(out_crop_path, bbox_inches='tight')
        plt.close()
        print(f"Crop saved: {out_crop_path}")

def main():
    model_path = "outputs/stage_a/run_shuffled_v3/weights/best.pt"
    model = YOLO(model_path)
    artifact_dir = r"C:\Users\Matrix Store\.gemini\antigravity\brain\9f91a989-ae24-4c1f-b1bd-0c8230f28d7f"
    
    test_img_dir = "data/shuffled_v3/test/images"
    test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))][:3]
    
    for i, img_path in enumerate(test_images):
        annotate_crops(img_path, model, artifact_dir, f"sample_{i+1}")

if __name__ == "__main__":
    main()

