"""
nanodet/dataset.py
──────────────────
Dataset loader for NanoDet-Plus Foot Detector (Apache-2.0).
Directly parses standard YOLO-format text bounding box annotations without
needing to modify or convert the existing dataset on disk.
Includes Mosaic 4-Image Augmentation, multi-scale jitter, and robust negative
background handling (empty label files) to permanently suppress false positives.
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
    YOLO Bounding Box Dataset for NanoDet-Plus with Mosaic 4-Image Augmentation.
    Classes:
      num_classes = 1: 0 (foot)
      num_classes = 2: 0 (left_foot), 1 (right_foot)
    Negative images: Empty text files or no objects (produces empty boxes [0, 4]).
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: Optional[str] = None,
        img_size: int = 320,
        num_classes: int = 2,
        augment: bool = False,
        allow_flip: bool = True,
        include_negatives: bool = True,
        mosaic_prob: float = 0.50
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
        self.mosaic_prob = mosaic_prob if augment else 0.0

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

    def _load_single_image(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
        img_path = self.image_files[idx]
        img = safe_imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h0, w0 = img.shape[:2]
        boxes, labels = self._parse_yolo_labels(self.labels_dir / f"{img_path.stem}.txt", w0, h0)
        return img, boxes, labels, img_path

    def _load_mosaic(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads 4 images and collates into 1 mosaic canvas."""
        indices = [idx] + [random.randint(0, len(self.image_files) - 1) for _ in range(3)]
        s = self.img_size
        xc = int(random.uniform(s * 0.35, s * 0.65))
        yc = int(random.uniform(s * 0.35, s * 0.65))

        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        mosaic_boxes = []
        mosaic_labels = []

        for i, img_idx in enumerate(indices):
            img, boxes, labels, _ = self._load_single_image(img_idx)
            h, w = img.shape[:2]

            # Quadrants: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
            if i == 0:
                x1a, y1a, x2a, y2a = 0, 0, xc, yc
                scale = min(xc / w, yc / h)
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, 0, s, yc
                scale = min((s - xc) / w, yc / h)
            elif i == 2:
                x1a, y1a, x2a, y2a = 0, yc, xc, s
                scale = min(xc / w, (s - yc) / h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, s, s
                scale = min((s - xc) / w, (s - yc) / h)

            nw = int(round(w * scale))
            nh = int(round(h * scale))
            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

            # Center resized in quadrant
            qw, qh = x2a - x1a, y2a - y1a
            pad_x = x1a + max(0, (qw - nw) // 2)
            pad_y = y1a + max(0, (qh - nh) // 2)
            nw = min(nw, s - pad_x)
            nh = min(nh, s - pad_y)

            if nw > 0 and nh > 0:
                mosaic_img[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized[:nh, :nw]

            if len(boxes) > 0:
                b_scaled = boxes.copy()
                b_scaled[:, [0, 2]] = np.clip(b_scaled[:, [0, 2]] * scale + pad_x, 0, s - 1)
                b_scaled[:, [1, 3]] = np.clip(b_scaled[:, [1, 3]] * scale + pad_y, 0, s - 1)

                valid = ((b_scaled[:, 2] - b_scaled[:, 0]) >= 4.0) & ((b_scaled[:, 3] - b_scaled[:, 1]) >= 4.0)
                if valid.any():
                    mosaic_boxes.append(b_scaled[valid])
                    mosaic_labels.append(labels[valid])

        if mosaic_boxes:
            all_boxes = np.concatenate(mosaic_boxes, axis=0)
            all_labels = np.concatenate(mosaic_labels, axis=0)
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_labels = np.zeros((0,), dtype=np.int64)

        return mosaic_img, all_boxes, all_labels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.augment and random.random() < self.mosaic_prob:
            canvas, boxes, labels = self._load_mosaic(idx)
            img_path = self.image_files[idx]
        else:
            img, boxes, labels, img_path = self._load_single_image(idx)
            h0, w0 = img.shape[:2]

            scale = min(self.img_size / w0, self.img_size / h0)
            nw = int(round(w0 * scale))
            nh = int(round(h0 * scale))

            resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
            pad_x = (self.img_size - nw) // 2
            pad_y = (self.img_size - nh) // 2
            canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

            if len(boxes) > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y

        if self.augment:
            # Photometric color jitter
            if random.random() < 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-20, 20)
                canvas = cv2.convertScaleAbs(canvas, alpha=alpha, beta=beta)

            # Random horizontal flip with chirality label swap
            if self.allow_flip and random.random() < 0.5:
                canvas = cv2.flip(canvas, 1)
                if len(boxes) > 0:
                    x1 = self.img_size - 1 - boxes[:, 2]
                    x2 = self.img_size - 1 - boxes[:, 0]
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
