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
from src.models.qualcomm_experiments.nanodet.ghost_pan import GhostPAN, RepPAN
from src.models.qualcomm_experiments.nanodet.head import NanoDetPlusHead, Integral
from src.models.qualcomm_experiments.nanodet.model import NanoDetPlus, nms_postprocess, load_official_nanodet_weights, DEFAULT_CKPT_PATH
from src.models.qualcomm_experiments.nanodet.loss import NanoDetPlusLoss


class TestNanoDetSuite(unittest.TestCase):

    def test_backbone_and_neck(self):
        B = 2
        img = torch.randn(B, 3, 256, 256)
        backbone = ShuffleNetV2Backbone(activation="relu")
        feats = backbone(img)

        # Stride 8 (32x32), 16 (16x16), 32 (8x8)
        self.assertEqual(feats[0].shape, (B, 116, 32, 32))
        self.assertEqual(feats[1].shape, (B, 232, 16, 16))
        self.assertEqual(feats[2].shape, (B, 464, 8, 8))

        # Test improved RepPAN (3 levels: strides 8, 16, 32)
        rep_neck = RepPAN(in_channels=[116, 232, 464], out_channels=96, num_levels=3, act_type="relu")
        pan_feats = rep_neck(feats)
        self.assertEqual(len(pan_feats), 3)
        self.assertEqual(pan_feats[0].shape, (B, 96, 32, 32))
        self.assertEqual(pan_feats[1].shape, (B, 96, 16, 16))
        self.assertEqual(pan_feats[2].shape, (B, 96, 8, 8))

        # Test legacy GhostPAN (4 levels: strides 8, 16, 32, 64)
        ghost_neck = GhostPAN(in_channels=[116, 232, 464], out_channels=96)
        ghost_feats = ghost_neck(feats)
        self.assertEqual(len(ghost_feats), 4)

    def test_head_and_integral(self):
        B = 2
        head = NanoDetPlusHead(num_classes=1, in_channels=96, reg_max=16, strides=[8, 16, 32], act_type="relu", kernel_size=3)
        dummy_feats = [
            torch.randn(B, 96, 32, 32),
            torch.randn(B, 96, 16, 16),
            torch.randn(B, 96, 8, 8)
        ]
        cls_scores, reg_dists = head(dummy_feats)

        # Class logits: [B, 1, H, W]
        self.assertEqual(len(cls_scores), 3)
        self.assertEqual(cls_scores[0].shape, (B, 1, 32, 32))
        self.assertEqual(cls_scores[1].shape, (B, 1, 16, 16))
        self.assertEqual(cls_scores[2].shape, (B, 1, 8, 8))

        # Reg distribution: 4 * (16 + 1) = 68 channels
        self.assertEqual(len(reg_dists), 3)
        self.assertEqual(reg_dists[0].shape, (B, 68, 32, 32))
        self.assertEqual(reg_dists[1].shape, (B, 68, 16, 16))
        self.assertEqual(reg_dists[2].shape, (B, 68, 8, 8))

    def test_official_coco_weights_loading(self):
        """Verifies 100% loading of official COCO checkpoint with zero missing keys in legacy config."""
        if Path(DEFAULT_CKPT_PATH).exists():
            model = NanoDetPlus(
                num_classes=1,
                input_size=(256, 256),
                reg_max=7,
                strides=[8, 16, 32, 64],
                neck_type="ghost_pan",
                act_type="leaky",
                kernel_size=5,
                pretrained=False
            )
            res = load_official_nanodet_weights(model, ckpt_path=DEFAULT_CKPT_PATH)

            self.assertEqual(res["missing_keys"], 0)
            self.assertEqual(res["loaded_keys"], 637)
            self.assertEqual(res["adapted_keys"], 8)
            self.assertEqual(len(model.state_dict()), 647)


    def test_model_forward_train_and_eval(self):
        B = 2
        # Hardware-optimized model: strides 8, 16, 32 -> 1024 + 256 + 64 = 1344 anchors
        model = NanoDetPlus(
            num_classes=1,
            input_size=(256, 256),
            reg_max=16,
            strides=[8, 16, 32],
            neck_type="rep_pan",
            act_type="relu",
            pretrained=False
        )
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

        # Exactly 1344 anchors matching Stage A champion contract
        self.assertEqual(eval_out["boxes"].shape, (B, 1344, 4))
        self.assertEqual(eval_out["scores"].shape, (B, 1344, 1))

        dets = nms_postprocess(eval_out["boxes"][0], eval_out["scores"][0], conf_thresh=0.01)
        self.assertIsInstance(dets, list)

    def test_composite_loss_and_gradients(self):
        B = 2
        model = NanoDetPlus(
            num_classes=1,
            input_size=(256, 256),
            reg_max=16,
            strides=[8, 16, 32],
            neck_type="rep_pan",
            act_type="relu",
            pretrained=False
        )
        criterion = NanoDetPlusLoss(
            num_classes=1,
            reg_max=16,
            strides=[8, 16, 32],
            loss_bbox_type="ciou",
            assigner="tal",
            neg_penalty_weight=1.5
        )

        img = torch.randn(B, 3, 256, 256)
        # 1 image with feet, 1 empty background image (negative sample)
        gt_boxes = [
            torch.tensor([[30.0, 40.0, 120.0, 180.0]], dtype=torch.float32),
            torch.zeros((0, 4), dtype=torch.float32)
        ]
        gt_labels = [
            torch.tensor([0], dtype=torch.long),
            torch.zeros((0,), dtype=torch.long)
        ]

        model.train()
        preds = model(img)
        losses = criterion(preds, gt_boxes, gt_labels, img_size=(256, 256))

        self.assertIn("total_loss", losses)
        self.assertIn("loss_cls", losses)
        self.assertIn("loss_bbox", losses)
        self.assertIn("loss_dfl", losses)
        self.assertGreater(losses["total_loss"].item(), 0.0)

        losses["total_loss"].backward()
        self.assertIsNotNone(model.head.gfl_cls[0].weight.grad)
        self.assertGreater(model.head.gfl_cls[0].weight.grad.norm().item(), 0.0)

    def test_finetune_checkpoint_loading(self):
        ckpt_path = Path("checkpoints/nanodet_stage_a_improved2/best.pt")
        if not ckpt_path.exists():
            return

        model = NanoDetPlus(
            num_classes=1,
            input_size=(256, 256),
            reg_max=16,
            strides=[8, 16, 32],
            neck_type="rep_pan",
            act_type="relu",
            pretrained=False
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = ckpt.get("ema_state", ckpt.get("model_state", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=True)
        self.assertEqual(len(missing), 0)
        self.assertEqual(len(unexpected), 0)

        # Forward pass verification
        model.eval()
        inp = torch.randn(1, 3, 256, 256)
        out = model(inp)
        self.assertEqual(out["boxes"].shape, (1, 1344, 4))
        self.assertEqual(out["scores"].shape, (1, 1344, 1))



if __name__ == "__main__":
    unittest.main()
