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
bb_state = {k.replace("model.backbone.", ""): v for k, v in state.items() if k.startswith("model.backbone.")}
official_bb.load_state_dict(bb_state)
official_bb.eval()

our_model = NanoDetPlus(num_classes=80, input_size=(320, 320), pretrained=False)
load_official_nanodet_weights(our_model, ckpt_path=ckpt_path)
our_bb = our_model.backbone
our_bb.eval()

x = torch.randn(1, 3, 320, 320)

with torch.no_grad():
    # 1. conv1
    off_c1 = official_bb.conv1(x)
    our_c1 = our_bb.conv1(x)
    print("conv1 max diff:", (off_c1 - our_c1).abs().max().item())

    # 2. maxpool
    off_mp = official_bb.maxpool(off_c1)
    our_mp = our_bb.maxpool(our_c1)
    print("maxpool max diff:", (off_mp - our_mp).abs().max().item())

    # 3. stage2 block 0
    off_s2_0 = official_bb.stage2[0](off_mp)
    our_s2_0 = our_bb.stage2[0](our_mp)
    print("stage2[0] max diff:", (off_s2_0 - our_s2_0).abs().max().item())

