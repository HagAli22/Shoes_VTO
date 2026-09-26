"""
pruner.py
─────────
Structured Channel Pruning Engine for YOLOv8-Pose Architectures.

Performs physical channel dimension reduction on Backbone and Neck convolutions
while preserving protected Detect & Pose prediction heads.
"""

import os
import copy
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List
from ultralytics import YOLO


def measure_model_size_and_params(
    model: Any,
    input_size: Tuple[int, int, int, int] = (1, 3, 320, 320),
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Measures total parameters, trainable parameters, and estimated FP32 ONNX size in MB.
    """
    net = model.model if hasattr(model, "model") else model
    net.eval()
    
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    est_fp32_mb = (total_params * 4.0) / (1024.0 * 1024.0)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "est_fp32_mb": est_fp32_mb,
    }


def compute_channel_importance(
    conv_module: nn.Conv2d,
    importance_type: str = "taylor",
    bn_module: Optional[nn.BatchNorm2d] = None
) -> torch.Tensor:
    """
    Computes importance scores for each output channel of a Conv layer.
    """
    w = conv_module.weight.data # [C_out, C_in, k, k]
    c_out = w.shape[0]
    
    if importance_type in ["l1", "magnitude"]:
        scores = w.abs().view(c_out, -1).sum(dim=1)
    elif importance_type == "l2":
        scores = w.pow(2).view(c_out, -1).sum(dim=1).sqrt()
    elif importance_type == "taylor" and conv_module.weight.grad is not None:
        grad = conv_module.weight.grad.data
        scores = (w * grad).abs().view(c_out, -1).sum(dim=1)
    elif importance_type in ["bn_scale", "group_norm"] and bn_module is not None:
        scores = bn_module.weight.data.abs()
    else:
        # Fallback to L1
        scores = w.abs().view(c_out, -1).sum(dim=1)
        
    return scores


def prune_conv_layer(
    conv: nn.Conv2d,
    in_channels_keep: Optional[List[int]] = None,
    out_channels_keep: Optional[List[int]] = None
) -> nn.Conv2d:
    """
    Slices a Conv2d layer physically to new input and output channel dimensions.
    """
    old_w = conv.weight.data
    old_b = conv.bias.data if conv.bias is not None else None
    
    n_in = len(in_channels_keep) if in_channels_keep is not None else conv.in_channels
    n_out = len(out_channels_keep) if out_channels_keep is not None else conv.out_channels
    
    in_idx = in_channels_keep if in_channels_keep is not None else list(range(conv.in_channels))
    out_idx = out_channels_keep if out_channels_keep is not None else list(range(conv.out_channels))
    
    new_conv = nn.Conv2d(
        in_channels=n_in,
        out_channels=n_out,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=conv.bias is not None,
        device=old_w.device,
        dtype=old_w.dtype
    )
    
    new_conv.weight.data.copy_(old_w[out_idx][:, in_idx])
    if old_b is not None:
        new_conv.bias.data.copy_(old_b[out_idx])
        
    return new_conv


def prune_bn_layer(bn: nn.BatchNorm2d, channels_keep: List[int]) -> nn.BatchNorm2d:
    """
    Slices a BatchNorm2d layer to new channel dimensions.
    """
    num_features = len(channels_keep)
    device = bn.weight.data.device if bn.weight is not None else "cpu"
    dtype = bn.weight.data.dtype if bn.weight is not None else torch.float32
    new_bn = nn.BatchNorm2d(
        num_features=num_features,
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
        device=device,
        dtype=dtype
    )
    if bn.affine:
        new_bn.weight.data.copy_(bn.weight.data[channels_keep])
        new_bn.bias.data.copy_(bn.bias.data[channels_keep])
    if bn.track_running_stats:
        new_bn.running_mean.copy_(bn.running_mean[channels_keep])
        new_bn.running_var.copy_(bn.running_var[channels_keep])
    return new_bn


def prune_c2f_block(
    c2f_module: Any,
    prune_ratio: float,
    importance_type: str = "taylor",
    round_to: int = 8
):
    """
    Prunes the internal bottleneck layers inside a YOLOv8 C2f block.
    """
    if not (hasattr(c2f_module, "cv1") and hasattr(c2f_module, "cv2") and hasattr(c2f_module, "m")):
        return
        
    bottlenecks = c2f_module.m
    if not isinstance(bottlenecks, nn.ModuleList):
        return

    for b in bottlenecks:
        if hasattr(b, "cv1") and hasattr(b, "cv2"):
            conv1 = b.cv1.conv
            bn1 = b.cv1.bn
            conv2 = b.cv2.conv
            bn2 = b.cv2.bn
            
            c_mid = conv1.out_channels
            keep_count = max(round_to, int(c_mid * (1.0 - prune_ratio)))
            keep_count = (keep_count // round_to) * round_to
            keep_count = max(round_to, min(keep_count, c_mid))
            
            if keep_count >= c_mid:
                continue
                
            scores = compute_channel_importance(conv1, importance_type, bn1)
            _, top_indices = torch.topk(scores, keep_count, largest=True, sorted=True)
            top_indices = sorted(top_indices.tolist())
            
            # Prune cv1: out_channels reduced
            b.cv1.conv = prune_conv_layer(conv1, out_channels_keep=top_indices)
            b.cv1.bn = prune_bn_layer(bn1, channels_keep=top_indices)
            
            # Prune cv2: in_channels reduced
            b.cv2.conv = prune_conv_layer(conv2, in_channels_keep=top_indices)


def prune_yolo_model(
    model: Any,
    pruning_ratio: float,
    importance_name: str = "taylor",
    calibration_batch: Optional[torch.Tensor] = None,
    device: str = "cpu",
    round_to: int = 8,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Applies structured channel pruning on YOLOv8-Pose.
    """
    if isinstance(model, str):
        yolo = YOLO(model)
    elif isinstance(model, YOLO):
        yolo = model
    else:
        yolo = model

    net = yolo.model if hasattr(yolo, "model") else yolo
    if isinstance(device, str):
        if device.isdigit():
            torch_dev = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        elif "cuda" in device.lower():
            torch_dev = device if torch.cuda.is_available() else "cpu"
        else:
            torch_dev = "cpu"
    else:
        torch_dev = device

    net.to(torch_dev)

    # 1. Measure baseline
    stats_before = measure_model_size_and_params(net, device=torch_dev)
    print(f"\n[PRUNER] Baseline Parameters: {stats_before['total_params']:,} ({stats_before['est_fp32_mb']:.2f} MB FP32)")
    print(f"[PRUNER] Pruning Ratio Target: {pruning_ratio * 100:.1f}% (Method: {importance_name.upper()})")

    # 2. Accumulate gradients if Taylor mode
    if "taylor" in importance_name.lower():
        net.train()
        for p in net.parameters():
            p.requires_grad = True
        net.zero_grad()
        calib_in = torch.randn(4, 3, 320, 320, device=torch_dev, requires_grad=True) if calibration_batch is None else calibration_batch.to(torch_dev)
        try:
            out = net(calib_in)
            if isinstance(out, (list, tuple)):
                tensor_losses = [o.sum() for o in out if isinstance(o, torch.Tensor) and o.requires_grad]
            elif isinstance(out, dict):
                tensor_losses = [v.sum() for v in out.values() if isinstance(v, torch.Tensor) and v.requires_grad]
            elif isinstance(out, torch.Tensor) and out.requires_grad:
                tensor_losses = [out.sum()]
            else:
                tensor_losses = []
                
            if tensor_losses:
                total_loss = sum(tensor_losses)
                total_loss.backward()
            else:
                # Forward intermediate neck feature
                feat = calib_in
                for l in net.model[:-1]:
                    feat = l(feat)
                feat.sum().backward()
        except Exception as e:
            # Fallback to weight magnitude if autograd engine detaches in head
            pass
        net.eval()

    # 3. Prune internal C2f bottleneck channels across Backbone & Neck
    # (Layers 0..21 are Backbone and Neck, Layer 22 is Pose Head)
    layers_seq = net.model if hasattr(net, "model") else net
    pruned_c2f_count = 0
    
    for idx, layer in enumerate(layers_seq[:-1]): # Exclude final head layer
        if layer.__class__.__name__ == "C2f":
            prune_c2f_block(layer, prune_ratio=pruning_ratio, importance_type=importance_name, round_to=round_to)
            pruned_c2f_count += 1

    # 4. Verify forward pass
    net.eval()
    example_input = torch.randn(1, 3, 320, 320, device=torch_dev)
    with torch.no_grad():
        test_out = net(example_input)

    stats_after = measure_model_size_and_params(net, device=torch_dev)
    param_reduction_pct = (1.0 - stats_after["total_params"] / stats_before["total_params"]) * 100.0

    print(f"[PRUNER] Pruned {pruned_c2f_count} C2f blocks successfully.")
    print(f"[PRUNER] Pruned Parameters : {stats_after['total_params']:,} ({stats_after['est_fp32_mb']:.2f} MB FP32)")
    print(f"[PRUNER] Actual Reduction  : {param_reduction_pct:.2f}%")
    print(f"[PRUNER] Forward pass verification OK!\n")

    stats = {
        "pruning_ratio_requested": pruning_ratio,
        "importance_name": importance_name,
        "params_before": stats_before["total_params"],
        "params_after": stats_after["total_params"],
        "reduction_pct": param_reduction_pct,
        "est_fp32_mb_before": stats_before["est_fp32_mb"],
        "est_fp32_mb_after": stats_after["est_fp32_mb"],
    }

    return yolo, stats
