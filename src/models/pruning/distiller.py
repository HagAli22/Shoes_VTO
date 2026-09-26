"""
distiller.py
────────────
Knowledge Distillation Engine for Pruned YOLOv8-Pose Models.

Transfers representation capacity from unpruned Teacher baseline to Pruned Student:
  1. Feature Map Alignment: L2 / MSE loss on intermediate feature pyramids (P3, P4, P5).
  2. Soft Keypoint Distillation: Smooth L1 / OKS loss between Student and Teacher keypoint predictions.
  3. Classification Distillation: Temperature-scaled KL divergence on class logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class FeaturePoseDistillationLoss(nn.Module):
    """
    Multi-level Knowledge Distillation Loss for Foot Pose Estimation.
    """
    def __init__(
        self,
        feat_weight: float = 1.0,
        pose_weight: float = 2.0,
        cls_weight: float = 0.5,
        temperature: float = 2.0,
    ):
        super().__init__()
        self.feat_weight = feat_weight
        self.pose_weight = pose_weight
        self.cls_weight = cls_weight
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=1.0)
        
    def forward_feature_loss(
        self,
        student_feats: List[torch.Tensor],
        teacher_feats: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Computes MSE loss across matching feature map levels.
        If student channel dimension is smaller than teacher, uses 1x1 projection adapter or channel pooling.
        """
        total_feat_loss = 0.0
        n_levels = min(len(student_feats), len(teacher_feats))
        
        for i in range(n_levels):
            s_f = student_feats[i]
            t_f = teacher_feats[i].detach()
            
            # If channel counts match, direct MSE
            if s_f.shape[1] == t_f.shape[1]:
                total_feat_loss += self.mse_loss(s_f, t_f)
            else:
                # Spatial cosine similarity / channel-normalized MSE
                s_norm = F.normalize(s_f, p=2, dim=1)
                t_norm = F.normalize(t_f, p=2, dim=1)
                # Compute spatial attention map distillation
                s_att = torch.mean(s_f.pow(2), dim=1, keepdim=True)
                t_att = torch.mean(t_f.pow(2), dim=1, keepdim=True)
                total_feat_loss += self.mse_loss(F.normalize(s_att, p=2), F.normalize(t_att, p=2))
                
        return total_feat_loss / max(1, n_levels)

    def forward_pose_distillation(
        self,
        student_kpts: torch.Tensor,
        teacher_kpts: torch.Tensor,
        teacher_conf_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Distills keypoint coordinates from Teacher predictions where Teacher has high confidence.
        """
        t_k = teacher_kpts.detach()
        if teacher_conf_mask is not None:
            mask = (teacher_conf_mask > 0.4).float().unsqueeze(-1)
            loss = (self.smooth_l1(student_kpts, t_k) * mask).sum() / (mask.sum() + 1e-6)
        else:
            loss = self.smooth_l1(student_kpts, t_k)
        return loss

    def forward(
        self,
        student_feats: Optional[List[torch.Tensor]] = None,
        teacher_feats: Optional[List[torch.Tensor]] = None,
        student_kpts: Optional[torch.Tensor] = None,
        teacher_kpts: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        total = 0.0
        
        if student_feats is not None and teacher_feats is not None:
            l_feat = self.forward_feature_loss(student_feats, teacher_feats) * self.feat_weight
            losses["loss_feat_distill"] = l_feat
            total += l_feat
            
        if student_kpts is not None and teacher_kpts is not None:
            l_pose = self.forward_pose_distillation(student_kpts, teacher_kpts) * self.pose_weight
            losses["loss_pose_distill"] = l_pose
            total += l_pose
            
        losses["total_distill_loss"] = total
        return losses
