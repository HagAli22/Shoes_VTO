"""
importance.py
─────────────
Channel Importance Scoring Methods for Structured Channel Pruning:
  1. Magnitude-based Importance (L1 / L2 norm of Conv filter weights)
  2. First-Order Taylor Importance (Weight x Gradient magnitude from task loss)
  3. BN Scaling / Group Norm Importance (BatchNorm scaling parameter gamma magnitude)
"""

import torch
import torch.nn as nn
import torch_pruning as tp
from typing import Optional


def get_importance_criterion(
    criterion_name: str = "taylor",
    p: int = 1,
    group_reduction: str = "mean",
    normalizer: str = "mean"
):
    """
    Factory function to instantiate the specified channel importance criterion.
    
    Args:
        criterion_name: 'magnitude', 'l1', 'l2', 'taylor', 'first_order_taylor', 'bn_scale', 'group_norm'
        p: Norm degree (1 for L1, 2 for L2)
        group_reduction: Reduction across coupled channels in DepGraph ('mean', 'sum', 'max')
        normalizer: Normalization method across layers ('mean', 'max', 'none')
    """
    name = criterion_name.lower().strip()
    
    if name in ["magnitude", "l1", "l2"]:
        norm_p = 2 if name == "l2" else p
        return tp.importance.MagnitudeImportance(
            p=norm_p,
            group_reduction=group_reduction,
            normalizer=normalizer
        )
    elif name in ["taylor", "first_order_taylor", "gradient"]:
        return tp.importance.TaylorImportance(
            group_reduction=group_reduction,
            normalizer=normalizer
        )
    elif name in ["bn_scale", "group_norm", "bn"]:
        return tp.importance.BNScaleImportance(
            group_reduction=group_reduction,
            normalizer=normalizer
        )
    elif name in ["hessian", "second_order_taylor"]:
        return tp.importance.HessianImportance(
            group_reduction=group_reduction,
            normalizer=normalizer
        )
    else:
        raise ValueError(f"Unknown importance criterion '{criterion_name}'. Choose from: magnitude, l1, l2, taylor, bn_scale.")
