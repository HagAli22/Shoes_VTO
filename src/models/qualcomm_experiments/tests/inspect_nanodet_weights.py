import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from src.models.qualcomm_experiments.nanodet.model import _setup_lightning_mock

_setup_lightning_mock()
ckpt = torch.load("checkpoints/nanodet_plus_m_320_checkpoint.ckpt", map_location="cpu", weights_only=False)
state = ckpt["state_dict"]

print("--- FP Network Layers in official checkpoint ---")
fpn_keys = [k for k in state.keys() if "fpn." in k]
for k in fpn_keys[:25]:
    print(k, state[k].shape)

print("\n--- Extra Level Layers in official checkpoint ---")
for k in [k for k in state.keys() if "extra" in k]:
    print(k, state[k].shape)

