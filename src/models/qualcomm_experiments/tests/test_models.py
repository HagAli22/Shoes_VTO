"""
tests/test_models.py
────────────────────
Automated unit tests for Qualcomm RTMPose-13KP (CSPNeXt + RTMCCHead).
Tests forward passes, GAU token mixing, vectorized SimCC losses, and sub-pixel peak decoding.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
import torch
import numpy as np

from src.models.qualcomm_experiments.rtmpose.backbone import CSPNeXtTiny, load_official_coco_weights
from src.models.qualcomm_experiments.rtmpose.simcc_head import RTMPose13KP, RTMCCHead13KP
from src.models.qualcomm_experiments.rtmpose.losses import SimCCLoss
from src.models.qualcomm_experiments.rtmpose.decoder import decode_simcc


class TestRTMPoseSuite(unittest.TestCase):

    def test_cspnext_backbone_and_weights(self):
        B = 2
        img = torch.randn(B, 3, 256, 256)
        backbone = CSPNeXtTiny()
        load_official_coco_weights(backbone)
        backbone.eval()

        with torch.no_grad():
            feat = backbone(img)

        # Output should be stride 32 (8x8 for 256x256), 384 channels
        self.assertEqual(feat.shape, (B, 384, 8, 8))

    def test_rtmpose_forward_and_shapes(self):
        B = 2
        img = torch.randn(B, 3, 256, 256)
        model = RTMPose13KP(num_keypoints=13, input_size=(256, 256), simcc_split_ratio=2.0, pretrained=False)
        model.eval()

        with torch.no_grad():
            preds = model(img)

        # 256 * 2.0 = 512 discrete bins
        self.assertEqual(preds["simcc_x"].shape, (B, 13, 512))
        self.assertEqual(preds["simcc_y"].shape, (B, 13, 512))
        self.assertEqual(preds["vis_logits"].shape, (B, 13, 1))
        self.assertEqual(preds["vis"].shape, (B, 13, 1))
        # In-model handedness classification
        self.assertIn("hand_logits", preds)
        self.assertEqual(preds["hand_logits"].shape, (B,))
        self.assertIn("hand", preds)
        self.assertEqual(preds["hand"].shape, (B,))

    def test_vectorized_simcc_loss_and_peak_decoder(self):
        B = 2
        img = torch.randn(B, 3, 256, 256)
        model = RTMPose13KP(num_keypoints=13, input_size=(256, 256), simcc_split_ratio=2.0, pretrained=False)
        criterion = SimCCLoss(simcc_split_ratio=2.0, sigma=6.0)

        # Mock ground truth keypoints
        kps = torch.ones((B, 13, 3)) * 128.0
        kps[:, :, 2] = 2.0  # visible

        targets = criterion.build_targets(kps, w_bins=512, h_bins=512)
        targets["classes"] = torch.tensor([0, 1], dtype=torch.long)
        self.assertEqual(targets["target_x"].shape, (B, 13, 512))
        self.assertEqual(targets["target_y"].shape, (B, 13, 512))
        self.assertEqual(targets["mask"].shape, (B, 13))

        preds = model(img)
        losses = criterion(preds, targets)

        self.assertIn("total_loss", losses)
        self.assertIn("loss_hand", losses)
        self.assertGreater(losses["total_loss"].item(), 0.0)
        self.assertGreater(losses["loss_hand"].item(), 0.0)

        # Ensure gradient flows back into handedness linear layer
        losses["total_loss"].backward()
        self.assertIsNotNone(model.head.hand.weight.grad)
        self.assertGreater(model.head.hand.weight.grad.norm().item(), 0.0)

        # Test peak-windowed decoder
        decoded_kps = decode_simcc(preds, simcc_split_ratio=2.0, win=6)
        self.assertEqual(decoded_kps.shape, (B, 13, 3))
        # Ensure confidence is well-scaled between [0, 1]
        self.assertTrue(np.all(decoded_kps[:, :, 2] >= 0.0))
        self.assertTrue(np.all(decoded_kps[:, :, 2] <= 1.0))


if __name__ == "__main__":
    unittest.main()
