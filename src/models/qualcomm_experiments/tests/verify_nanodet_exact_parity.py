"""
verify_nanodet_exact_parity.py
───────────────────────────────
The Decisive Pre-Training Numerical Parity Verification Test:
Builds both:
1. Official NanoDet-Plus (from cloned official repository)
2. Our Self-Contained NanoDet-Plus (num_classes=80)
Loads the official checkpoint checkpoints/nanodet_plus_m_320_checkpoint.ckpt into both.
Passes an identical deterministic random input tensor [1, 3, 320, 320] through:
  - Backbone (ShuffleNetV2 with LeakyReLU)
  - Neck (Ghost-PAN with verified top-down / bottom-up wiring & extra level)
  - Head (NanoDetPlusHead with DFL)
Computes and reports the maximum absolute difference across every stage.
Target: max difference < 1e-4.
"""

import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_NANODET = REPO_ROOT / "src" / "models" / "qualcomm_experiments" / "official_nanodet"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OFFICIAL_NANODET) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_NANODET))

from src.models.qualcomm_experiments.nanodet.model import (
    NanoDetPlus,
    load_official_nanodet_weights
)

# Official components
from nanodet.model.backbone.shufflenetv2 import ShuffleNetV2 as OfficialShuffleNetV2
from nanodet.model.fpn.ghost_pan import GhostPAN as OfficialGhostPAN
from nanodet.model.head.nanodet_plus_head import NanoDetPlusHead as OfficialNanoDetPlusHead


def test_numerical_parity():
    print("=" * 75)
    print("  DECISIVE NUMERICAL PARITY TEST: OFFICIAL vs OUR NANODET-PLUS")
    print("=" * 75)

    device = torch.device("cpu")
    torch.manual_seed(42)
    x = torch.randn(1, 3, 320, 320, device=device)

    # 1. Build our model with num_classes=80
    our_model = NanoDetPlus(
        num_classes=80,
        input_size=(320, 320),
        out_channels=96,
        stacked_convs=2,
        reg_max=7,
        pretrained=False
    ).to(device)

    # 2. Build official models
    official_backbone = OfficialShuffleNetV2(
        model_size="1.0x",
        out_stages=(2, 3, 4),
        activation="LeakyReLU"
    ).to(device)

    official_fpn = OfficialGhostPAN(
        in_channels=[116, 232, 464],
        out_channels=96,
        kernel_size=5,
        num_extra_level=1,
        use_depthwise=True,
        activation="LeakyReLU"
    ).to(device)

    loss_cfg = type("LossCfg", (), {
        "loss_qfl": type("QFL", (), {"beta": 2.0, "loss_weight": 1.0})(),
        "loss_dfl": type("DFL", (), {"loss_weight": 0.25})(),
        "loss_bbox": type("Bbox", (), {"loss_weight": 2.0})()
    })()

    official_head = OfficialNanoDetPlusHead(
        num_classes=80,
        loss=loss_cfg,
        input_channel=96,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32, 64],
        conv_type="DWConv",
        reg_max=7,
        activation="LeakyReLU"
    ).to(device)

    # 3. Load official checkpoint
    ckpt_path = "checkpoints/nanodet_plus_m_320_checkpoint.ckpt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    # Strip prefixes to load into official components (using avg_model weights)
    prefix = "avg_model." if any(k.startswith("avg_model.") for k in state.keys()) else "model."
    bb_state = {}
    fpn_state = {}
    head_state = {}

    for k, v in state.items():
        if k.startswith(f"{prefix}backbone."):
            bb_state[k.replace(f"{prefix}backbone.", "")] = v
        elif k.startswith(f"{prefix}fpn."):
            fpn_state[k.replace(f"{prefix}fpn.", "")] = v
        elif k.startswith(f"{prefix}head."):
            head_state[k.replace(f"{prefix}head.", "")] = v

    # Load into official components
    m_bb, u_bb = official_backbone.load_state_dict(bb_state, strict=False)
    m_fpn, u_fpn = official_fpn.load_state_dict(fpn_state, strict=False)
    m_head, u_head = official_head.load_state_dict(head_state, strict=False)

    print(f"[*] Official Backbone loaded: missing={len(m_bb)}, unexpected={len(u_bb)}")
    print(f"[*] Official GhostPAN loaded: missing={len(m_fpn)}, unexpected={len(u_fpn)}")
    print(f"[*] Official Head loaded: missing={len(m_head)}, unexpected={len(u_head)}")

    # Load into our self-contained model
    load_official_nanodet_weights(our_model, ckpt_path=ckpt_path)

    # Set both to eval mode
    official_backbone.eval()
    official_fpn.eval()
    official_head.eval()
    our_model.eval()

    # 4. Forward comparison
    with torch.no_grad():
        # A. Backbone (both receive identical input tensor x)
        off_feats = official_backbone(x)
        our_feats = our_model.backbone(x)

        print("\n--- 1. BACKBONE OUTPUT DIFFERENCE ---")
        for lvl, (off_f, our_f) in enumerate(zip(off_feats, our_feats)):
            diff = (off_f - our_f).abs().max().item()
            print(f"  Stage {lvl+2} (shape {list(off_f.shape)}): max diff = {diff:.6e}")
            assert diff < 1e-4, f"Backbone stage {lvl+2} mismatch: {diff}"

        # B. Neck (Ghost-PAN)
        off_neck = official_fpn(off_feats)
        our_neck = our_model.neck(our_feats)

        print("\n--- 2. GHOST-PAN NECK OUTPUT DIFFERENCE ---")
        for lvl, (off_n, our_n) in enumerate(zip(off_neck, our_neck)):
            diff = (off_n - our_n).abs().max().item()
            print(f"  Level {lvl} (shape {list(off_n.shape)}): max diff = {diff:.6e}")
            assert diff < 1e-4, f"GhostPAN level {lvl} mismatch: {diff}"

        # C. Head comparison
        off_head_out = official_head(off_neck)  # [B, total_anchors, 112]

        our_flat_outputs = []
        for feat, convs, head in zip(our_neck, our_model.head.cls_convs, our_model.head.gfl_cls):
            for c in convs:
                feat = c(feat)
            out = head(feat)
            our_flat_outputs.append(out.flatten(start_dim=2))
        our_head_flat = torch.cat(our_flat_outputs, dim=2).permute(0, 2, 1)

        print("\n--- 3. DETECTION HEAD OUTPUT DIFFERENCE ---")
        diff_total = (off_head_out - our_head_flat).abs().max().item()
        print(f"  Complete Head Tensor (shape {list(off_head_out.shape)}): max diff = {diff_total:.6e}")
        assert diff_total < 1e-4, f"Head output mismatch: {diff_total}"

    print("\n" + "=" * 75)
    print("  [SUCCESS] EXACT NUMERICAL PARITY CONFIRMED! (All max diffs < 1e-4)")
    print("=" * 75)


if __name__ == "__main__":
    test_numerical_parity()
