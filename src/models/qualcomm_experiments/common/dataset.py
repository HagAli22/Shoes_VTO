"""
common/dataset.py
─────────────────
Dataset loader for 13-Keypoint Foot Pose Estimation supporting both
CenterNet (full-frame keypoint heatmaps & bounding boxes) and RTMPose
(cropped foot bounding boxes with SimCC 1D coordinate labels).

Strictly adheres to the 13-KP Anatomical Schema:
  0: toe_ground, 1: heel_back, 2: heel_ground, 3: ball_medial, 4: ball_lateral,
  5: instep_top, 6: arch_medial, 7: midfoot_lateral, 8: malleolus_medial,
  9: malleolus_lateral, 10: toe_tip, 11: throat, 12: achilles.

Chirality Swap Rule on Horizontal Mirroring:
  - Class: 0 (left_foot) <-> 1 (right_foot)
  - Medial <-> Lateral keypoint pairs:
      3 (ball_medial) <-> 4 (ball_lateral)
      6 (arch_medial) <-> 7 (midfoot_lateral)
      8 (malleolus_medial) <-> 9 (malleolus_lateral)
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CHIRALITY_FLIP_MAP = {
    0: 0,   # toe_ground
    1: 1,   # heel_back
    2: 2,   # heel_ground
    3: 4,   # ball_medial <-> ball_lateral
    4: 3,   # ball_lateral <-> ball_medial
    5: 5,   # instep_top
    6: 7,   # arch_medial <-> midfoot_lateral
    7: 6,   # midfoot_lateral <-> arch_medial
    8: 9,   # malleolus_medial <-> malleolus_lateral
    9: 8,   # malleolus_lateral <-> malleolus_medial
    10: 10, # toe_tip
    11: 11, # throat
    12: 12  # achilles
}


class FootPoseDataset13KP(Dataset):
    """
    Standard PyTorch dataset parsing YOLO 13-keypoint annotated directories.
    Can return either:
      1) 'full': Full-frame image with multi-object boxes and keypoints (for CenterNet)
      2) 'crop': Cropped single-foot patches (for RTMPose top-down)
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: Optional[str] = None,
        img_size: int = 256,
        mode: str = "full",  # "full" or "crop"
        augment: bool = False,
        allow_flip: bool = False,
        crop_padding: float = 0.25,
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        if labels_dir is not None:
            self.labels_dir = Path(labels_dir)
        else:
            # Standard YOLO layout: images/ -> labels/
            self.labels_dir = self.images_dir.parent / "labels"

        self.img_size = img_size
        self.mode = mode
        self.augment = augment
        self.allow_flip = allow_flip
        self.crop_padding = crop_padding

        # Find all valid image files
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_files: List[Path] = [
            p for p in sorted(self.images_dir.iterdir())
            if p.suffix.lower() in valid_exts
        ]

        # For "crop" mode, build an index of (img_idx, box_idx)
        self.crops_index: List[Tuple[int, int]] = []
        if self.mode == "crop":
            self._index_crops()

    def _index_crops(self):
        for img_idx, img_path in enumerate(self.image_files):
            label_path = self.labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            for box_idx in range(len(lines)):
                self.crops_index.append((img_idx, box_idx))

    def __len__(self) -> int:
        if self.mode == "crop":
            return len(self.crops_index)
        return len(self.image_files)

    def _parse_label_file(self, label_path: Path) -> List[Dict[str, Any]]:
        instances = []
        if not label_path.exists():
            return instances

        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                # Parse keypoints (13 keypoints x 3: x, y, v)
                kps = []
                kp_parts = parts[5:]
                num_kps = len(kp_parts) // 3
                for i in range(num_kps):
                    kx = float(kp_parts[i * 3])
                    ky = float(kp_parts[i * 3 + 1])
                    kv = float(kp_parts[i * 3 + 2])
                    kps.append([kx, ky, kv])

                # Pad to 13 keypoints if necessary
                while len(kps) < 13:
                    kps.append([0.0, 0.0, 0.0])

                instances.append({
                    "class_id": cls_id,
                    "bbox": [cx, cy, w, h],  # normalized cx, cy, w, h
                    "keypoints": np.array(kps[:13], dtype=np.float32)  # [13, 3]
                })
        return instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.mode == "crop":
            img_idx, box_idx = self.crops_index[index]
            img_path = self.image_files[img_idx]
            instances = self._parse_label_file(self.labels_dir / f"{img_path.stem}.txt")
            inst = instances[box_idx]
            return self._get_crop_item(img_path, inst)
        else:
            img_path = self.image_files[index]
            instances = self._parse_label_file(self.labels_dir / f"{img_path.stem}.txt")
            return self._get_full_item(img_path, instances)

    def _get_full_item(self, img_path: Path, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h_orig, w_orig = img.shape[:2]

        # Convert instances to absolute pixel coordinates
        boxes_abs = []
        kps_abs = []
        classes = []
        for inst in instances:
            classes.append(inst["class_id"])
            cx, cy, bw, bh = inst["bbox"]
            x1 = (cx - bw / 2.0) * w_orig
            y1 = (cy - bh / 2.0) * h_orig
            x2 = (cx + bw / 2.0) * w_orig
            y2 = (cy + bh / 2.0) * h_orig
            boxes_abs.append([x1, y1, x2, y2])

            kp = inst["keypoints"].copy()
            kp[:, 0] *= w_orig
            kp[:, 1] *= h_orig
            kps_abs.append(kp)

        # Resize / letterbox to (img_size, img_size)
        scale = min(self.img_size / w_orig, self.img_size / h_orig)
        nw = int(w_orig * scale)
        nh = int(h_orig * scale)
        resized_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        pad_x = (self.img_size - nw) // 2
        pad_y = (self.img_size - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized_img

        # Adjust boxes and keypoints to canvas coordinates
        transformed_boxes = []
        for b in boxes_abs:
            tx1 = b[0] * scale + pad_x
            ty1 = b[1] * scale + pad_y
            tx2 = b[2] * scale + pad_x
            ty2 = b[3] * scale + pad_y
            transformed_boxes.append([tx1, ty1, tx2, ty2])

        transformed_kps = []
        for kp in kps_abs:
            tkp = kp.copy()
            tkp[:, 0] = kp[:, 0] * scale + pad_x
            tkp[:, 1] = kp[:, 1] * scale + pad_y
            transformed_kps.append(tkp)

        # Horizontal flip with Chirality swap
        if self.augment and self.allow_flip and random.random() < 0.5:
            canvas = cv2.flip(canvas, 1)
            flipped_boxes = []
            for b in transformed_boxes:
                fx1 = self.img_size - 1 - b[2]
                fx2 = self.img_size - 1 - b[0]
                flipped_boxes.append([fx1, b[1], fx2, b[3]])
            transformed_boxes = flipped_boxes

            flipped_classes = [1 - c for c in classes]
            classes = flipped_classes

            flipped_kps = []
            for kp in transformed_kps:
                fkp = kp.copy()
                fkp[:, 0] = self.img_size - 1 - kp[:, 0]
                # Swap medial <-> lateral
                swapped = fkp.copy()
                for old_idx, new_idx in CHIRALITY_FLIP_MAP.items():
                    swapped[old_idx] = fkp[new_idx]
                flipped_kps.append(swapped)
            transformed_kps = flipped_kps

        # Normalize image tensor
        img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0

        return {
            "image": img_tensor,
            "classes": torch.tensor(classes, dtype=torch.long) if classes else torch.empty((0,), dtype=torch.long),
            "boxes": torch.tensor(transformed_boxes, dtype=torch.float32) if transformed_boxes else torch.empty((0, 4), dtype=torch.float32),
            "keypoints": torch.tensor(np.array(transformed_kps), dtype=torch.float32) if transformed_kps else torch.empty((0, 13, 3), dtype=torch.float32),
            "img_path": str(img_path)
        }

    def _get_crop_item(self, img_path: Path, inst: Dict[str, Any]) -> Dict[str, Any]:
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h0, w0 = img.shape[:2]

        cx = inst["bbox"][0] * w0
        cy = inst["bbox"][1] * h0
        bw = inst["bbox"][2] * w0
        bh = inst["bbox"][3] * h0

        ar = max(bw / max(1.0, bh), bh / max(1.0, bw))
        size = max(bw, bh) * (1.0 + self.crop_padding + (0.15 if ar > 1.3 else 0.05))
        rot = 0.0

        # Jitter augmentation (scale, shift, rotation) for robust detection tolerance
        if self.augment:
            size *= random.uniform(0.85, 1.25)
            cx += random.uniform(-0.1, 0.1) * size
            cy += random.uniform(-0.1, 0.1) * size
            rot = random.uniform(-30, 30)

        # Affine warp matrix centered at (cx, cy)
        M = cv2.getRotationMatrix2D((cx, cy), rot, self.img_size / size)
        M[0, 2] += self.img_size / 2.0 - cx
        M[1, 2] += self.img_size / 2.0 - cy

        # Constant border prevents mirror ghost reflections
        crop = cv2.warpAffine(
            img, M, (self.img_size, self.img_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114)
        )

        # Transform keypoints via the exact affine transformation
        kps = inst["keypoints"].copy()
        kps[:, :2] = np.stack([kps[:, 0] * w0, kps[:, 1] * h0, np.ones(len(kps))], axis=1) @ M.T

        # Invalidate keypoints transformed outside crop boundaries
        out = (kps[:, 0] < 0) | (kps[:, 0] >= self.img_size) | (kps[:, 1] < 0) | (kps[:, 1] >= self.img_size)
        kps[out, 2] = 0.0

        img_tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0

        return {
            "image": img_tensor,
            "class_id": torch.tensor(inst["class_id"], dtype=torch.long),
            "keypoints": torch.tensor(kps, dtype=torch.float32),  # [13, 3]
            "img_path": str(img_path)
        }

