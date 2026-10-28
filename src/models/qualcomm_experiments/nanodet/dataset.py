"""
nanodet/dataset.py
──────────────────
Dataset loader for NanoDet-Plus Foot Detector (Apache-2.0).
Directly parses standard YOLO-format text bounding box annotations without
needing to modify or convert the existing dataset on disk.
Supports negative background images (empty label files) to suppress false positives.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.qualcomm_experiments.common.dataset import safe_imread, safe_path_str


class NanoDetFootDataset(Dataset):
    """
    YOLO Bounding Box Dataset for NanoDet-Plus.
    Classes:
      num_classes = 1: 0 (foot)
      num_classes = 2: 0 (left_foot), 1 (right_foot)
    Negative images: Empty text files or no objects (produces empty boxes [0, 4]).
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: Optional[str] = None,
        img_size: int = 256,
        num_classes: int = 1,
        augment: bool = False,
        allow_flip: bool = True,  # Safe for single-class 'foot' detector
        include_negatives: bool = True
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        if labels_dir is not None:
            self.labels_dir = Path(labels_dir)
        else:
            self.labels_dir = self.images_dir.parent / "labels"

        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
        self.allow_flip = allow_flip
        self.include_negatives = include_negatives

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_files = [
            p for p in sorted(self.images_dir.iterdir())
            if p.suffix.lower() in valid_exts
        ]

        if not self.include_negatives:
            filtered = []
            for p in self.image_files:
                lbl_path = self.labels_dir / f"{p.stem}.txt"
                p_str = safe_path_str(lbl_path)
                if Path(p_str).exists() and Path(p_str).stat().st_size > 0:
                    filtered.append(p)
            self.image_files = filtered

    def __len__(self) -> int:
        return len(self.image_files)

    def _parse_yolo_labels(self, label_path: Path, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
        boxes = []
        labels = []
        p_str = safe_path_str(label_path)
        if not Path(p_str).exists():
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        with open(p_str, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if self.num_classes == 1:
                    cls_id = 0  # Map both left and right to unified 'foot' class

                cx, cy, bw, bh = map(float, parts[1:5])

                x1 = (cx - bw / 2.0) * img_w
                y1 = (cy - bh / 2.0) * img_h
                x2 = (cx + bw / 2.0) * img_w
                y2 = (cy + bh / 2.0) * img_h

                x1 = max(0.0, min(float(img_w), x1))
                y1 = max(0.0, min(float(img_h), y1))
                x2 = max(0.0, min(float(img_w), x2))
                y2 = max(0.0, min(float(img_h), y2))

                if (x2 - x1) >= 2.0 and (y2 - y1) >= 2.0:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls_id)

        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        img = safe_imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        h0, w0 = img.shape[:2]
        boxes, labels = self._parse_yolo_labels(self.labels_dir / f"{img_path.stem}.txt", w0, h0)

        base_scale = min(self.img_size / w0, self.img_size / h0)

        if self.augment:
            # Scale jitter: 0.85x to 1.15x
            scale_jitter = random.uniform(0.85, 1.15)
            scale = base_scale * scale_jitter
            nw = max(16, int(round(w0 * scale)))
            nh = max(16, int(round(h0 * scale)))

            # Translation jitter: +/- 8% of canvas size
            max_shift = int(round(0.08 * self.img_size))
            tx = random.randint(-max_shift, max_shift)
            ty = random.randint(-max_shift, max_shift)
            pad_x = (self.img_size - nw) // 2 + tx
            pad_y = (self.img_size - nh) // 2 + ty
        else:
            scale = base_scale
            nw = int(round(w0 * scale))
            nh = int(round(h0 * scale))
            pad_x = (self.img_size - nw) // 2
            pad_y = (self.img_size - nh) // 2

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)

        # Robust bounded blit
        dst_x1 = max(0, pad_x)
        dst_y1 = max(0, pad_y)
        dst_x2 = min(self.img_size, pad_x + nw)
        dst_y2 = min(self.img_size, pad_y + nh)

        src_x1 = max(0, -pad_x)
        src_y1 = max(0, -pad_y)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1 and src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y

            # Clamp boxes to canvas boundary
            boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(self.img_size))
            boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(self.img_size))
            boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(self.img_size))
            boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(self.img_size))

            # Discard any box that was shifted completely outside or degenerated
            valid = (boxes[:, 2] - boxes[:, 0] >= 4.0) & (boxes[:, 3] - boxes[:, 1] >= 4.0)
            boxes = boxes[valid]
            labels = labels[valid]

        if self.augment:
            # HSV color jitter (50% prob)
            if random.random() < 0.50:
                hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
                h_delta = random.uniform(-15.0, 15.0)
                hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180.0
                s_scale = random.uniform(0.65, 1.35)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0.0, 255.0)
                v_scale = random.uniform(0.70, 1.30)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0.0, 255.0)
                canvas = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            elif random.random() < 0.50:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-20, 20)
                canvas = cv2.convertScaleAbs(canvas, alpha=alpha, beta=beta)

            # Random Cutout (20% prob)
            if random.random() < 0.20:
                pw = random.randint(int(0.08 * self.img_size), int(0.20 * self.img_size))
                ph = random.randint(int(0.08 * self.img_size), int(0.20 * self.img_size))
                px = random.randint(0, max(0, self.img_size - pw))
                py = random.randint(0, max(0, self.img_size - ph))
                canvas[py:py + ph, px:px + pw] = 114

            # Horizontal flip
            if self.allow_flip and random.random() < 0.5:
                canvas = cv2.flip(canvas, 1)
                if len(boxes) > 0:
                    x1 = self.img_size - 1.0 - boxes[:, 2]
                    x2 = self.img_size - 1.0 - boxes[:, 0]
                    boxes[:, 0] = x1
                    boxes[:, 2] = x2
                    if self.num_classes > 1:
                        labels = 1 - labels

        img_tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0

        return {
            "image": img_tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "img_path": str(img_path)
        }


def collate_fn_nanodet(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    paths = [b["img_path"] for b in batch]
    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "paths": paths
    }
