"""
single_stage_rtmpose/dataset.py
───────────────────────────────
Full-Image YOLO-Pose Dataset Loader for Single-Stage RTMPose (Apache-2.0).
Parses 2-class chirality (Left/Right foot), bounding boxes, and 13 keypoints.
Includes negative background images (empty label files) to suppress false positives.
Enforces allow_flip = False to prevent Left/Right chirality collisions.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.models.qualcomm_experiments.common.dataset import safe_imread, safe_path_str


class SingleStageFootDataset(Dataset):
    """
    YOLO-Pose Dataset Loader for Single-Stage RTMPose.
    Line format: class_id cx cy w h x1 y1 v1 x2 y2 v2 ... x13 y13 v13
      class_id: 0 (left_foot), 1 (right_foot)
      v_i: 0 (unlabeled/invisible), 1 (labeled/occluded), 2 (labeled/visible)
    """
    def __init__(
        self,
        images_dir: str,
        labels_dir: Optional[str] = None,
        img_size: int = 320,
        num_classes: int = 2,
        num_keypoints: int = 13,
        augment: bool = False,
        allow_flip: bool = False,  # Strict False for chiral Left/Right foot detection
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
        self.num_keypoints = num_keypoints
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

    def _parse_yolo_pose(
        self,
        label_path: Path,
        img_w: int,
        img_h: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes = []
        labels = []
        keypoints = []

        p_str = safe_path_str(label_path)
        if not Path(p_str).exists():
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, self.num_keypoints, 3), dtype=np.float32)
            )

        expected_fields = 5 + self.num_keypoints * 3

        with open(p_str, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                if self.num_classes == 1:
                    cls_id = 0

                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2.0) * img_w
                y1 = (cy - bh / 2.0) * img_h
                x2 = (cx + bw / 2.0) * img_w
                y2 = (cy + bh / 2.0) * img_h

                x1 = max(0.0, min(float(img_w), x1))
                y1 = max(0.0, min(float(img_h), y1))
                x2 = max(0.0, min(float(img_w), x2))
                y2 = max(0.0, min(float(img_h), y2))

                if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
                    continue

                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id)

                kpt_list = []
                if len(parts) >= expected_fields:
                    kpt_vals = list(map(float, parts[5:expected_fields]))
                    for i in range(self.num_keypoints):
                        kx = kpt_vals[i * 3] * img_w
                        ky = kpt_vals[i * 3 + 1] * img_h
                        kv = kpt_vals[i * 3 + 2]
                        kpt_list.append([kx, ky, kv])
                else:
                    # Dummy keypoints if missing
                    for _ in range(self.num_keypoints):
                        kpt_list.append([0.0, 0.0, 0.0])

                keypoints.append(kpt_list)

        if not boxes:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0, self.num_keypoints, 3), dtype=np.float32)
            )

        return (
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int64),
            np.array(keypoints, dtype=np.float32)
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        img = safe_imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        h0, w0 = img.shape[:2]
        boxes, labels, kpts = self._parse_yolo_pose(
            self.labels_dir / f"{img_path.stem}.txt",
            w0,
            h0
        )

        # Aspect-ratio preserving letterbox
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

            # Transform keypoint coordinates
            for b_idx in range(len(kpts)):
                vis = kpts[b_idx, :, 2] > 0
                kpts[b_idx, vis, 0] = kpts[b_idx, vis, 0] * scale + pad_x
                kpts[b_idx, vis, 1] = kpts[b_idx, vis, 1] * scale + pad_y

        if self.augment:
            # Photometric color jitter
            if random.random() < 0.5:
                alpha = random.uniform(0.8, 1.2)
                beta = random.uniform(-20, 20)
                canvas = cv2.convertScaleAbs(canvas, alpha=alpha, beta=beta)

            # Strict: allow_flip is False by default for chiral 2-class foot detection
            if self.allow_flip and random.random() < 0.5:
                canvas = cv2.flip(canvas, 1)
                if len(boxes) > 0:
                    x1 = self.img_size - 1 - boxes[:, 2]
                    x2 = self.img_size - 1 - boxes[:, 0]
                    boxes[:, 0] = x1
                    boxes[:, 2] = x2
                    if self.num_classes > 1:
                        labels = 1 - labels
                    for b_idx in range(len(kpts)):
                        vis = kpts[b_idx, :, 2] > 0
                        kpts[b_idx, vis, 0] = self.img_size - 1 - kpts[b_idx, vis, 0]

        # Convert to RGB normalized tensor in [0, 1]
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(canvas_rgb).permute(2, 0, 1).float() / 255.0

        return {
            "image": img_tensor,
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "keypoints": torch.from_numpy(kpts),
            "img_path": str(img_path)
        }


def collate_fn_single_stage(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    boxes = [b["boxes"] for b in batch]
    labels = [b["labels"] for b in batch]
    keypoints = [b["keypoints"] for b in batch]
    paths = [b["img_path"] for b in batch]
    return {
        "images": images,
        "boxes": boxes,
        "labels": labels,
        "keypoints": keypoints,
        "paths": paths
    }

