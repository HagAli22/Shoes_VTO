"""
depgraph_yolo.py
────────────────
Dependency Graph Builder & Layer Protection for YOLOv8-Pose architectures.

Identifies:
  1. Prunable Convolutional layers in Backbone & Neck (C2f bottleneck convolutions).
  2. Protected layers: Pose & Detect prediction heads (cv2, cv3, cv4, dfl) and stem layers.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Set


def get_ignored_head_layers(model: nn.Module) -> List[nn.Module]:
    """
    Returns a list of all layers that must be protected from channel pruning
    to preserve 1:1 output tensor semantics, keypoint count, and bounding box decoding.
    """
    ignored_layers = []
    
    # Unwrap down to the internal layers sequence
    net = model
    while hasattr(net, "model"):
        net = net.model
        
    # Now net is either the nn.Sequential or ModuleList
    if isinstance(net, (nn.Sequential, nn.ModuleList)) or hasattr(net, "__getitem__"):
        head = net[-1]
        for name, m in head.named_modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                ignored_layers.append(m)
                
        stem = net[0]
        for name, m in stem.named_modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                ignored_layers.append(m)
    else:
        for name, m in model.named_modules():
            if "cv2" in name or "cv3" in name or "cv4" in name or "dfl" in name:
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                    ignored_layers.append(m)
                
    return ignored_layers


def get_yolo_pruning_layers(model: nn.Module) -> Tuple[List[nn.Module], List[nn.Module]]:
    """
    Returns (prunable_layers, ignored_layers) for YOLOv8-Pose.
    """
    net = model.model if hasattr(model, "model") else model
    ignored_layers = get_ignored_head_layers(model)
    ignored_set = set(ignored_layers)
    
    prunable_layers = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) and m not in ignored_set:
            prunable_layers.append(m)
            
    return prunable_layers, ignored_layers
