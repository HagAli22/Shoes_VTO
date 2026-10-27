import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[4]
OFFICIAL_NANODET = REPO_ROOT / "src" / "models" / "qualcomm_experiments" / "official_nanodet"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(OFFICIAL_NANODET) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_NANODET))

import torch
from src.models.qualcomm_experiments.nanodet.model import NanoDetPlus, load_official_nanodet_weights
from nanodet.model.fpn.ghost_pan import GhostPAN as OfficialGhostPAN

ckpt_path = "checkpoints/nanodet_plus_m_320_checkpoint.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt["state_dict"]

official_fpn = OfficialGhostPAN(
    in_channels=[116, 232, 464],
    out_channels=96,
    kernel_size=5,
    num_extra_level=1,
    use_depthwise=True,
    activation="LeakyReLU"
)
prefix = "avg_model." if any(k.startswith("avg_model.") for k in state.keys()) else "model."
fpn_state = {k.replace(f"{prefix}fpn.", ""): v for k, v in state.items() if k.startswith(f"{prefix}fpn.")}
official_fpn.load_state_dict(fpn_state)
official_fpn.eval()

our_model = NanoDetPlus(num_classes=80, input_size=(320, 320), pretrained=False)
load_official_nanodet_weights(our_model, ckpt_path=ckpt_path)
our_fpn = our_model.neck
our_fpn.eval()

# Dummy backbone features (reproducible)
torch.manual_seed(42)
c2 = torch.randn(1, 116, 40, 40)
c3 = torch.randn(1, 232, 20, 20)
c4 = torch.randn(1, 464, 10, 10)
feats = [c2, c3, c4]

with torch.no_grad():
    # 1. Reduce layers
    for i in range(3):
        off_r = official_fpn.reduce_layers[i](feats[i])
        our_r = our_fpn.reduce_layers[i](feats[i])
        print(f"reduce_layers[{i}] max diff:", (off_r - our_r).abs().max().item())

    # 2. Top-down block 0 (p4_up + c3)
    p4 = official_fpn.reduce_layers[2](c4)
    p4_up = torch.nn.functional.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)
    p3_in = torch.cat([p4_up, official_fpn.reduce_layers[1](c3)], 1)
    off_td0 = official_fpn.top_down_blocks[0](p3_in)
    our_td0 = our_fpn.top_down_blocks[0](p3_in)
    print("top_down_blocks[0] max diff:", (off_td0 - our_td0).abs().max().item())

