"""
YOLOv8-Pose Structured Channel Pruning Module
─────────────────────────────────────────────
Implements dependency-aware structured channel pruning for YOLOv8-pose models,
supporting both 16-keypoint and 4-coarse-keypoint architectures.
"""

from .depgraph_yolo import get_yolo_pruning_layers, get_ignored_head_layers
from .importance import get_importance_criterion
from .pruner import prune_yolo_model, measure_model_size_and_params
from .distiller import FeaturePoseDistillationLoss

__all__ = [
    "get_yolo_pruning_layers",
    "get_ignored_head_layers",
    "get_importance_criterion",
    "prune_yolo_model",
    "measure_model_size_and_params",
    "FeaturePoseDistillationLoss",
]
