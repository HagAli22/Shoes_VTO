"""
visualize_keypoint_order.py
───────────────────────────
Runs inference on test images and draws every keypoint with its Index + Anatomical Name
so the user can visually verify if any two keypoints are swapped by Roboflow.
Outputs high-res annotated images directly to the artifacts directory.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# 16-KP Frozen Anatomical Names
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

# Color palette for distinct keypoint groups
COLORS = [
    (255, 0, 0),     # 0: toe_tip (Blue/Red in RGB)
    (255, 128, 0),   # 1: toe_ground
    (0, 0, 255),     # 2: heel_back
    (0, 128, 255),   # 3: heel_ground
    (0, 255, 0),     # 4: ball_medial
    (0, 200, 200),   # 5: ball_lateral
    (255, 255, 0),   # 6: ball_top
    (255, 0, 255),   # 7: instep_top
    (128, 0, 255),   # 8: arch_medial
    (255, 128, 128), # 9: midfoot_lateral
    (0, 255, 128),   # 10: malleolus_medial
    (128, 255, 0),   # 11: malleolus_lateral
    (255, 255, 255), # 12: ankle_center
    (200, 100, 50),  # 13: throat
    (50, 100, 200),  # 14: achilles
    (100, 255, 100)  # 15: shin_mid
]

def annotate_image(img_path, model, output_path):
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        return
    h, w = orig_img.shape[:2]
    
    results = model.predict(img_path, imgsz=320, conf=0.25, verbose=False)[0]
    
    # Create matplotlib figure for clear, legible text rendering
    fig, ax = plt.subplots(figsize=(14, 10), dpi=200)
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    ax.imshow(img_rgb)
    ax.set_title(f"Keypoint Index Verification: {os.path.basename(img_path)}", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    if results.keypoints is not None and len(results.keypoints) > 0:
        kpts_data = results.keypoints.data.cpu().numpy() # [N, 16, 3] or [N, 16, 2]
        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        
        for det_idx in range(len(boxes)):
            box = boxes[det_idx]
            cls_id = int(classes[det_idx])
            cls_name = "Left Foot" if cls_id == 0 else "Right Foot"
            conf = confs[det_idx]
            
            # Draw bbox
            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                 fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
            ax.add_patch(rect)
            ax.text(box[0], box[1] - 8, f"{cls_name} ({conf:.2f})", 
                    color='cyan', fontsize=12, fontweight='bold', 
                    bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))
            
            kpts = kpts_data[det_idx]
            
            for k in range(min(16, len(kpts))):
                kx, ky = kpts[k][0], kpts[k][1]
                kv = kpts[k][2] if len(kpts[k]) > 2 else 1.0
                
                if kv < 0.2 or kx <= 0 or ky <= 0:
                    continue
                
                # Plot keypoint dot
                ax.plot(kx, ky, 'o', markersize=7, color='red', markeredgecolor='white', markeredgewidth=1.2)
                
                # Plot label text with arrow / offset
                offset_x = 10 if (k % 2 == 0) else -15
                offset_y = -8 if (k % 3 == 0) else (8 if k % 3 == 1 else 0)
                
                label_text = KEYPOINT_NAMES[k]
                ax.annotate(
                    label_text,
                    xy=(kx, ky),
                    xytext=(kx + offset_x * 2.5, ky + offset_y * 2.5),
                    fontsize=8,
                    fontweight='bold',
                    color='yellow',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.75, edgecolor='yellow', linewidth=0.5),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=0.8)
                )
                
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    model_path = "outputs/stage_a/run_shuffled_v3/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    model = YOLO(model_path)
    
    artifact_dir = r"C:\Users\Matrix Store\.gemini\antigravity\brain\9f91a989-ae24-4c1f-b1bd-0c8230f28d7f"
    os.makedirs(artifact_dir, exist_ok=True)
    
    # Pick 4 diverse test images
    test_img_dir = "data/shuffled_v3/test/images"
    test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))][:4]
    
    for i, img_path in enumerate(test_images):
        out_name = f"verify_kpt_sample_{i+1}.png"
        out_path = os.path.join(artifact_dir, out_name)
        annotate_image(img_path, model, out_path)

if __name__ == "__main__":
    main()

