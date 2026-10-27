"""
nanodet
───────
Self-contained, ultra-lightweight anchor-free foot detection model (Apache-2.0).
Zero dependencies on Ultralytics or GPL/AGPL copyleft libraries.
"""

from src.models.qualcomm_experiments.nanodet.model import NanoDetPlus, nms_postprocess
from src.models.qualcomm_experiments.nanodet.shufflenetv2 import ShuffleNetV2Backbone
from src.models.qualcomm_experiments.nanodet.ghost_pan import GhostPAN
from src.models.qualcomm_experiments.nanodet.head import NanoDetPlusHead
from src.models.qualcomm_experiments.nanodet.loss import NanoDetPlusLoss
from src.models.qualcomm_experiments.nanodet.dataset import NanoDetFootDataset

