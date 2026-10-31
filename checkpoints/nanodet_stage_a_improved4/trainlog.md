======================================================================
  EXPERIMENT: NanoDet-Plus Stage A Foot Detector (Fine-Tuning Continuation)
  Dataset:         /content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3 (6620 train, 827 val)
  Backbone:        ShuffleNetV2 1.0x (Pretrained: False, Act: RELU)
  Neck:            REP_PAN | Strides: [8, 16, 32] | BBox Loss: CIOU
  Assigner:        TAL (Warmup: 1 ep) | Loss Weights (Box: 3.0, DFL: 0.15)
  Device:          cuda (AMP: True)
  Epochs:          40 | Batch: 16 | LR: 0.000300 | Classes: 1
  Resolution:      256x256
  Fine-Tuning From:/checkpoints/nanodet_stage_a_improved3/best.pt
======================================================================
[*] Loading fine-tuning checkpoint: /checkpoints/nanodet_stage_a_improved3/best.pt
[+] Loaded weights successfully (missing=0, unexpected=0)
[*] Continuation baseline best F1@0.75: 67.77%
Epoch [01/40] (62.3s) | Loss: 1.5415 (Cls: 0.291, GIoU: 0.342, DFL: 1.491)
Epoch [02/40] (53.9s) | Loss: 0.8207 (Cls: 0.170, GIoU: 0.173, DFL: 0.877) | F1@0.50: 87.2% | F1@0.75: 66.6% (c35: 71.0%) | mIoU: 81.5% | MaxConf: 99.5% | NegPrec: 73.8%
  [*] New best detector checkpoint saved! (F1@0.75: 66.57%, c35: 71.00%, mIoU: 81.51%)
Epoch [03/40] (54.5s) | Loss: 0.7003 (Cls: 0.051, GIoU: 0.172, DFL: 0.887)
Epoch [04/40] (54.0s) | Loss: 0.6979 (Cls: 0.049, GIoU: 0.172, DFL: 0.890) | F1@0.50: 88.1% | F1@0.75: 68.5% (c35: 72.7%) | mIoU: 81.6% | MaxConf: 99.8% | NegPrec: 78.6%
  [*] New best detector checkpoint saved! (F1@0.75: 68.45%, c35: 72.67%, mIoU: 81.59%)
Epoch [05/40] (54.2s) | Loss: 0.6951 (Cls: 0.048, GIoU: 0.171, DFL: 0.887)
Epoch [06/40] (54.0s) | Loss: 0.6915 (Cls: 0.048, GIoU: 0.170, DFL: 0.887) | F1@0.50: 87.0% | F1@0.75: 65.9% (c35: 70.8%) | mIoU: 81.2% | MaxConf: 99.5% | NegPrec: 77.4%
Epoch [07/40] (54.4s) | Loss: 0.6878 (Cls: 0.047, GIoU: 0.170, DFL: 0.878)
Epoch [08/40] (54.4s) | Loss: 0.6838 (Cls: 0.046, GIoU: 0.169, DFL: 0.875) | F1@0.50: 87.6% | F1@0.75: 67.1% (c35: 71.9%) | mIoU: 81.5% | MaxConf: 99.7% | NegPrec: 79.8%
Epoch [09/40] (54.4s) | Loss: 0.6835 (Cls: 0.046, GIoU: 0.169, DFL: 0.878)
Epoch [10/40] (54.4s) | Loss: 0.6787 (Cls: 0.046, GIoU: 0.167, DFL: 0.874) | F1@0.50: 87.5% | F1@0.75: 68.5% (c35: 72.9%) | mIoU: 81.6% | MaxConf: 99.7% | NegPrec: 78.6%
  [*] New best detector checkpoint saved! (F1@0.75: 68.46%, c35: 72.93%, mIoU: 81.56%)
Epoch [11/40] (54.4s) | Loss: 0.6804 (Cls: 0.045, GIoU: 0.168, DFL: 0.875)
Epoch [12/40] (54.1s) | Loss: 0.6721 (Cls: 0.045, GIoU: 0.166, DFL: 0.869) | F1@0.50: 85.9% | F1@0.75: 67.4% (c35: 72.9%) | mIoU: 81.7% | MaxConf: 99.7% | NegPrec: 82.1%
Epoch [13/40] (54.3s) | Loss: 0.6734 (Cls: 0.044, GIoU: 0.166, DFL: 0.868)
Epoch [14/40] (54.4s) | Loss: 0.6685 (Cls: 0.044, GIoU: 0.165, DFL: 0.862) | F1@0.50: 88.1% | F1@0.75: 69.8% (c35: 74.2%) | mIoU: 81.8% | MaxConf: 99.5% | NegPrec: 77.4%
  [*] New best detector checkpoint saved! (F1@0.75: 69.75%, c35: 74.20%, mIoU: 81.83%)
Epoch [15/40] (54.1s) | Loss: 0.6628 (Cls: 0.043, GIoU: 0.164, DFL: 0.860)
Epoch [16/40] (54.0s) | Loss: 0.6597 (Cls: 0.042, GIoU: 0.163, DFL: 0.858) | F1@0.50: 88.3% | F1@0.75: 69.5% (c35: 73.4%) | mIoU: 82.0% | MaxConf: 99.8% | NegPrec: 79.8%
Epoch [17/40] (54.0s) | Loss: 0.6613 (Cls: 0.043, GIoU: 0.163, DFL: 0.861)
Epoch [18/40] (54.1s) | Loss: 0.6591 (Cls: 0.043, GIoU: 0.162, DFL: 0.863) | F1@0.50: 88.1% | F1@0.75: 69.7% (c35: 73.8%) | mIoU: 82.1% | MaxConf: 99.7% | NegPrec: 81.0%
Epoch [19/40] (54.3s) | Loss: 0.6559 (Cls: 0.042, GIoU: 0.162, DFL: 0.857)
Epoch [20/40] (54.4s) | Loss: 0.6537 (Cls: 0.042, GIoU: 0.161, DFL: 0.854) | F1@0.50: 87.6% | F1@0.75: 68.8% (c35: 73.4%) | mIoU: 82.1% | MaxConf: 99.7% | NegPrec: 81.0%
Epoch [21/40] (54.3s) | Loss: 0.6513 (Cls: 0.041, GIoU: 0.161, DFL: 0.853)
Epoch [22/40] (54.2s) | Loss: 0.6490 (Cls: 0.041, GIoU: 0.160, DFL: 0.852) | F1@0.50: 87.7% | F1@0.75: 70.3% (c35: 75.1%) | mIoU: 82.2% | MaxConf: 99.7% | NegPrec: 79.8%
  [*] New best detector checkpoint saved! (F1@0.75: 70.34%, c35: 75.13%, mIoU: 82.20%)
Epoch [23/40] (54.0s) | Loss: 0.6447 (Cls: 0.040, GIoU: 0.159, DFL: 0.848)
Epoch [24/40] (54.2s) | Loss: 0.6426 (Cls: 0.040, GIoU: 0.158, DFL: 0.849) | F1@0.50: 88.3% | F1@0.75: 69.3% (c35: 73.4%) | mIoU: 82.2% | MaxConf: 99.7% | NegPrec: 81.0%
Epoch [25/40] (54.7s) | Loss: 0.6403 (Cls: 0.039, GIoU: 0.158, DFL: 0.847)
Epoch [26/40] (54.4s) | Loss: 0.6394 (Cls: 0.040, GIoU: 0.158, DFL: 0.845) | F1@0.50: 88.6% | F1@0.75: 69.4% (c35: 73.4%) | mIoU: 82.2% | MaxConf: 99.8% | NegPrec: 83.3%
Epoch [27/40] (54.3s) | Loss: 0.6366 (Cls: 0.039, GIoU: 0.157, DFL: 0.845)
Epoch [28/40] (54.3s) | Loss: 0.6362 (Cls: 0.039, GIoU: 0.157, DFL: 0.842) | F1@0.50: 88.5% | F1@0.75: 69.7% (c35: 74.0%) | mIoU: 82.2% | MaxConf: 99.7% | NegPrec: 81.0%
Epoch [29/40] (54.4s) | Loss: 0.6328 (Cls: 0.040, GIoU: 0.156, DFL: 0.835)
Epoch [30/40] (53.9s) | Loss: 0.6341 (Cls: 0.040, GIoU: 0.156, DFL: 0.838) | F1@0.50: 88.6% | F1@0.75: 70.3% (c35: 74.3%) | mIoU: 82.2% | MaxConf: 99.7% | NegPrec: 81.0%
Epoch [31/40] (53.9s) | Loss: 0.6308 (Cls: 0.039, GIoU: 0.155, DFL: 0.837)
Epoch [32/40] (53.9s) | Loss: 0.6285 (Cls: 0.037, GIoU: 0.155, DFL: 0.839) | F1@0.50: 88.3% | F1@0.75: 69.5% (c35: 73.6%) | mIoU: 82.3% | MaxConf: 99.7% | NegPrec: 81.0%
Epoch [33/40] (54.7s) | Loss: 0.6278 (Cls: 0.038, GIoU: 0.155, DFL: 0.839)
Epoch [34/40] (54.6s) | Loss: 0.6288 (Cls: 0.038, GIoU: 0.155, DFL: 0.837) | F1@0.50: 88.3% | F1@0.75: 69.4% (c35: 73.6%) | mIoU: 82.3% | MaxConf: 99.7% | NegPrec: 79.8%
Epoch [35/40] (54.3s) | Loss: 0.6275 (Cls: 0.038, GIoU: 0.155, DFL: 0.836)
Epoch [36/40] (54.1s) | Loss: 0.6263 (Cls: 0.038, GIoU: 0.154, DFL: 0.833) | F1@0.50: 88.3% | F1@0.75: 69.8% (c35: 73.8%) | mIoU: 82.4% | MaxConf: 99.7% | NegPrec: 81.0%
Epoch [37/40] (54.4s) | Loss: 0.6246 (Cls: 0.037, GIoU: 0.154, DFL: 0.833)
Epoch [38/40] (54.3s) | Loss: 0.6243 (Cls: 0.038, GIoU: 0.154, DFL: 0.835) | F1@0.50: 88.4% | F1@0.75: 69.6% (c35: 73.5%) | mIoU: 82.4% | MaxConf: 99.7% | NegPrec: 81.0%
Epoch [39/40] (54.5s) | Loss: 0.6258 (Cls: 0.038, GIoU: 0.154, DFL: 0.832)
Epoch [40/40] (54.4s) | Loss: 0.6233 (Cls: 0.038, GIoU: 0.153, DFL: 0.833) | F1@0.50: 88.3% | F1@0.75: 69.5% (c35: 73.4%) | mIoU: 82.4% | MaxConf: 99.7% | NegPrec: 81.0%

[OK] NanoDet-Plus training finished.
Best Checkpoint: /checkpoints/nanodet_stage_a_improved4/best.pt (Best F1@0.75: 70.34%)
