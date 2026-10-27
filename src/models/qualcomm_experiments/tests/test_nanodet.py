"""
tests/test_nanodet.py
─────────────────────
Automated unit tests for Official NanoDet-Plus Foot Detector (Apache-2.0):
  1. ShuffleNetV2Backbone multi-scale feature shapes
  2. Official Ghost-PAN pyramid aggregation (4 levels: strides 8, 16, 32, 64)
  3. NanoDetPlusHead DFL and Anchor-Free classification
  4. 100% Official COCO Pretrained Checkpoint Transfer (0 missing keys)
  5. Composite loss calculation and gradient backprop
  6. Decoded box outputs and NMS post-processing
"""

import sys
from pathlib import Path
import unittest
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.qualcomm_experiments.nanodet.shufflenetv2 import ShuffleNetV2Backbone
from src.models.qualcomm_experiments.nanodet.ghost_pan import GhostPAN
from src.models.qualcomm_experiments.nanodet.head import NanoDetPlusHead, Integral
from src.models.qualcomm_experiments.nanodet.model import NanoDetPlus, nms_postprocess, load_official_nanodet_weights, DEFAULT_CKPT_PATH
from src.models.qualcomm_experiments.nanodet.loss import NanoDetPlusLoss


class TestNanoDetSuite(unittest.TestCase):

    def test_backbone_and_neck(self):
        B = 2
        img = torch.randn(B, 3, 256, 256)
        backbone = ShuffleNetV2Backbone()
        feats = backbone(img)

        # Stride 8 (32x32), 16 (16x16), 32 (8x8)
        self.assertEqual(feats[0].shape, (B, 116, 32, 32))
        self.assertEqual(feats[1].shape, (B, 232, 16, 16))
        self.assertEqual(feats[2].shape, (B, 464, 8, 8))

        neck = GhostPAN(in_channels=[116, 232, 464], out_channels=96)
        pan_feats = neck(feats)

        # 4 levels: strides 8 (32x32), 16 (16x16), 32 (8x8), 64 (4x4)
        self.assertEqual(len(pan_feats), 4)
        self.assertEqual(pan_feats[0].shape, (B, 96, 32, 32))
        self.assertEqual(pan_feats[1].shape, (B, 96, 16, 16))
        self.assertEqual(pan_feats[2].shape, (B, 96, 8, 8))
        self.assertEqual(pan_feats[3].shape, (B, 96, 4, 4))

    def test_head_and_integral(self):
        B = 2
        head = NanoDetPlusHead(num_classes=1, in_channels=96, reg_max=7, strides=[8, 16, 32, 64])
        dummy_feats = [
            torch.randn(B, 96, 32, 32),
            torch.randn(B, 96, 16, 16),
            torch.randn(B, 96, 8, 8),
            torch.randn(B, 96, 4, 4)
        ]
        cls_scores, reg_dists = head(dummy_feats)

        # Class logits: [B, 1, H, W]
        self.assertEqual(len(cls_scores), 4)
        self.assertEqual(cls_scores[0].shape, (B, 1, 32, 32))
        self.assertEqual(cls_scores[1].shape, (B, 1, 16, 16))
        self.assertEqual(cls_scores[2].shape, (B, 1, 8, 8))
        self.assertEqual(cls_scores[3].shape, (B, 1, 4, 4))

        # Reg distribution: 4 * (7 + 1) = 32 channels
        self.assertEqual(len(reg_dists), 4)
        self.assertEqual(reg_dists[0].shape, (B, 32, 32, 32))
        self.assertEqual(reg_dists[1].shape, (B, 32, 16, 16))
        self.assertEqual(reg_dists[2].shape, (B, 32, 8, 8))
        self.assertEqual(reg_dists[3].shape, (B, 32, 4, 4))

    def test_official_coco_weights_loading(self):
        """Verifies 100% loading of official COCO checkpoint with zero missing keys."""
        if Path(DEFAULT_CKPT_PATH).exists():
            model = NanoDetPlus(num_classes=1, input_size=(256, 256), pretrained=False)
            res = load_official_nanodet_weights(model, ckpt_path=DEFAULT_CKPT_PATH)

            self.assertEqual(res["missing_keys"], 0)
            self.assertEqual(res["loaded_keys"], 637)
            self.assertEqual(res["adapted_keys"], 8)
            # Total model tensors: 637 matching + 8 adapted + 2 norm buffers (mean, std) = 647
            self.assertEqual(len(model.state_dict()), 647)

    def test_model_forward_train_and_eval(self):
        B = 2
        model = NanoDetPlus(num_classes=1, input_size=(256, 256), pretrained=False)
        img = torch.randn(B, 3, 256, 256)

        model.train()
        train_out = model(img)
        self.assertIn("cls_preds", train_out)
        self.assertIn("reg_preds", train_out)

        model.eval()
        with torch.no_grad():
            eval_out = model(img)
        self.assertIn("boxes", eval_out)
        self.assertIn("scores", eval_out)

        # 4 levels: 32*32 + 16*16 + 8*8 + 4*4 = 1024 + 256 + 64 + 16 = 1360
        self.assertEqual(eval_out["boxes"].shape, (B, 1360, 4))
        self.assertEqual(eval_out["scores"].shape, (B, 1360, 1))

        dets = nms_postprocess(eval_out["boxes"][0], eval_out["scores"][0], conf_thresh=0.01)
        self.assertIsInstance(dets, list)

    def test_composite_loss_and_gradients(self):
        B = 2
        model = NanoDetPlus(num_classes=1, input_size=(256, 256), pretrained=False)
        criterion = NanoDetPlusLoss(num_classes=1, reg_max=7, strides=[8, 16, 32, 64])

        img = torch.randn(B, 3, 256, 256)
        gt_boxes = [
            torch.tensor([[30.0, 40.0, 120.0, 180.0], [130.0, 50.0, 220.0, 200.0]], dtype=torch.float32),
            torch.tensor([[50.0, 60.0, 150.0, 210.0]], dtype=torch.float32)
        ]
        gt_labels = [
            torch.tensor([0, 0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long)
        ]

        model.train()
        preds = model(img)
        losses = criterion(preds, gt_boxes, gt_labels, img_size=(256, 256))

        self.assertIn("total_loss", losses)
        self.assertIn("loss_cls", losses)
        self.assertIn("loss_giou", losses)
        self.assertIn("loss_dfl", losses)
        self.assertGreater(losses["total_loss"].item(), 0.0)

        losses["total_loss"].backward()
        self.assertIsNotNone(model.head.gfl_cls[0].weight.grad)
        self.assertGreater(model.head.gfl_cls[0].weight.grad.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
