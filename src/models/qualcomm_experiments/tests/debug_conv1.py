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
from nanodet.model.backbone.shufflenetv2 import ShuffleNetV2 as OfficialShuffleNetV2

ckpt_path = "checkpoints/nanodet_plus_m_320_checkpoint.ckpt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt["state_dict"]

official_bb = OfficialShuffleNetV2(model_size="1.0x", out_stages=(2, 3, 4), activation="LeakyReLU")
bb_state = {k.replace("avg_model.backbone.", ""): v for k, v in state.items() if k.startswith("avg_model.backbone.")}
official_bb.load_state_dict(bb_state)

our_model = NanoDetPlus(num_classes=80, input_size=(320, 320), pretrained=False)
load_official_nanodet_weights(our_model, ckpt_path=ckpt_path)
our_bb = our_model.backbone

print("conv1[0].weight max diff:", (official_bb.conv1[0].weight - our_bb.conv1[0].weight).abs().max().item())
print("conv1[1].weight max diff:", (official_bb.conv1[1].weight - our_bb.conv1[1].weight).abs().max().item())
print("conv1[1].bias max diff:", (official_bb.conv1[1].bias - our_bb.conv1[1].bias).abs().max().item())
print("conv1[1].running_mean max diff:", (official_bb.conv1[1].running_mean - our_bb.conv1[1].running_mean).abs().max().item())
print("conv1[1].running_var max diff:", (official_bb.conv1[1].running_var - our_bb.conv1[1].running_var).abs().max().item())
print("official conv1[2]:", official_bb.conv1[2])
print("our conv1[2]:", our_bb.conv1[2])
