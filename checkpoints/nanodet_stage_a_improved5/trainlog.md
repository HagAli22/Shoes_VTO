======================================================================
  EXPERIMENT: NanoDet-Plus Stage A Foot Detector (Fine-Tuning Continuation)
  Dataset:         /content/Shoes_VTO/data/merged_v6_with_negatives (9120 train, 1138 val)
  Backbone:        ShuffleNetV2 1.0x (Pretrained: False, Act: RELU)
  Neck:            REP_PAN | Strides: [8, 16, 32] | BBox Loss: CIOU
  Assigner:        TAL (Warmup: 1 ep) | Loss Weights (Box: 3.0, DFL: 0.15)
  Device:          cuda (AMP: True)
  Epochs:          40 | Batch: 16 | LR: 0.000300 | Classes: 1
  Resolution:      256x256
  Fine-Tuning From:/content/Shoes_VTO/checkpoints/nanodet_stage_a_improved4/best.pt
======================================================================
[*] Loading fine-tuning checkpoint: /content/Shoes_VTO/checkpoints/nanodet_stage_a_improved4/best.pt
[+] Loaded weights successfully (missing=0, unexpected=0)
[*] Continuation baseline best F1@0.75: 70.34%
Epoch [01/40] (90.0s) | Loss: 1.4284 (Cls: 0.222, GIoU: 0.330, DFL: 1.444)
Epoch [02/40] (72.9s) | Loss: 0.8438 (Cls: 0.136, GIoU: 0.190, DFL: 0.920) | F1@0.50: 85.6% | F1@0.75: 64.8% (c35: 69.8%) | mIoU: 80.9% | MaxConf: 99.5% | NegPrec: 71.6%
  [*] New best detector checkpoint saved! (F1@0.75: 64.81%, c35: 69.77%, mIoU: 80.87%)
Epoch [03/40] (73.0s) | Loss: 0.7629 (Cls: 0.062, GIoU: 0.188, DFL: 0.920)
Epoch [04/40] (73.4s) | Loss: 0.7546 (Cls: 0.058, GIoU: 0.186, DFL: 0.917) | F1@0.50: 81.6% | F1@0.75: 62.0% (c35: 69.5%) | mIoU: 80.9% | MaxConf: 99.7% | NegPrec: 74.7%
Epoch [05/40] (74.5s) | Loss: 0.7464 (Cls: 0.057, GIoU: 0.184, DFL: 0.910)
Epoch [06/40] (74.5s) | Loss: 0.7413 (Cls: 0.055, GIoU: 0.184, DFL: 0.902) | F1@0.50: 83.5% | F1@0.75: 63.8% (c35: 70.2%) | mIoU: 81.0% | MaxConf: 99.6% | NegPrec: 75.3%
Epoch [07/40] (73.8s) | Loss: 0.7328 (Cls: 0.053, GIoU: 0.182, DFL: 0.898)
Epoch [08/40] (73.7s) | Loss: 0.7331 (Cls: 0.053, GIoU: 0.182, DFL: 0.898) | F1@0.50: 84.6% | F1@0.75: 63.9% (c35: 69.9%) | mIoU: 80.8% | MaxConf: 99.4% | NegPrec: 84.6%
Epoch [09/40] (73.5s) | Loss: 0.7253 (Cls: 0.052, GIoU: 0.180, DFL: 0.892)
Epoch [10/40] (74.6s) | Loss: 0.7252 (Cls: 0.051, GIoU: 0.180, DFL: 0.894) | F1@0.50: 84.5% | F1@0.75: 65.4% (c35: 72.1%) | mIoU: 81.3% | MaxConf: 99.8% | NegPrec: 80.9%
  [*] New best detector checkpoint saved! (F1@0.75: 65.40%, c35: 72.06%, mIoU: 81.34%)
Epoch [11/40] (73.4s) | Loss: 0.7220 (Cls: 0.050, GIoU: 0.179, DFL: 0.891)
Epoch [12/40] (74.0s) | Loss: 0.7169 (Cls: 0.050, GIoU: 0.178, DFL: 0.888) | F1@0.50: 83.6% | F1@0.75: 64.5% (c35: 71.3%) | mIoU: 81.4% | MaxConf: 99.8% | NegPrec: 80.2%
======================================================================
  EXPERIMENT: NanoDet-Plus Stage A Foot Detector (Fine-Tuning Continuation)
  Dataset:         /content/Shoes_VTO/data/merged_v6_with_negatives (9120 train, 1138 val)
  Backbone:        ShuffleNetV2 1.0x (Pretrained: False, Act: RELU)
  Neck:            REP_PAN | Strides: [8, 16, 32] | BBox Loss: CIOU
  Assigner:        TAL (Warmup: 1 ep) | Loss Weights (Box: 3.0, DFL: 0.15)
  Device:          cuda (AMP: True)
  Epochs:          100 | Batch: 16 | LR: 0.000300 | Classes: 1
  Resolution:      256x256
  Fine-Tuning From:/content/Shoes_VTO/checkpoints/nanodet_stage_a_improved4/best.pt
======================================================================
[*] Loading fine-tuning checkpoint: /content/Shoes_VTO/checkpoints/nanodet_stage_a_improved4/best.pt
[+] Loaded weights successfully (missing=0, unexpected=0)
[*] Continuation baseline best F1@0.75: 70.34%
Epoch [01/100] (78.1s) | Loss: 1.4279 (Cls: 0.222, GIoU: 0.330, DFL: 1.444)
Epoch [02/100] (72.7s) | Loss: 0.8441 (Cls: 0.136, GIoU: 0.190, DFL: 0.920) | F1@0.50: 85.3% | F1@0.75: 63.7% (c35: 68.9%) | mIoU: 80.8% | MaxConf: 99.4% | NegPrec: 71.6%
  [*] New best detector checkpoint saved! (F1@0.75: 63.70%, c35: 68.89%, mIoU: 80.80%)
Epoch [03/100] (72.7s) | Loss: 0.7619 (Cls: 0.062, GIoU: 0.187, DFL: 0.918)
Epoch [04/100] (73.0s) | Loss: 0.7544 (Cls: 0.058, GIoU: 0.186, DFL: 0.915) | F1@0.50: 82.9% | F1@0.75: 63.3% (c35: 70.3%) | mIoU: 81.1% | MaxConf: 99.7% | NegPrec: 78.4%
Epoch [05/100] (73.3s) | Loss: 0.7459 (Cls: 0.056, GIoU: 0.184, DFL: 0.915)
Epoch [06/100] (72.6s) | Loss: 0.7412 (Cls: 0.054, GIoU: 0.184, DFL: 0.906) | F1@0.50: 83.1% | F1@0.75: 63.7% (c35: 70.6%) | mIoU: 81.0% | MaxConf: 99.7% | NegPrec: 72.8%
Epoch [07/100] (73.4s) | Loss: 0.7332 (Cls: 0.053, GIoU: 0.182, DFL: 0.901)
Epoch [08/100] (73.1s) | Loss: 0.7327 (Cls: 0.053, GIoU: 0.182, DFL: 0.896) | F1@0.50: 84.7% | F1@0.75: 63.6% (c35: 69.8%) | mIoU: 81.0% | MaxConf: 99.6% | NegPrec: 79.6%
Epoch [09/100] (73.1s) | Loss: 0.7263 (Cls: 0.052, GIoU: 0.180, DFL: 0.896)
Epoch [10/100] (72.8s) | Loss: 0.7275 (Cls: 0.051, GIoU: 0.181, DFL: 0.896) | F1@0.50: 83.9% | F1@0.75: 64.8% (c35: 72.0%) | mIoU: 81.3% | MaxConf: 99.7% | NegPrec: 80.2%
  [*] New best detector checkpoint saved! (F1@0.75: 64.77%, c35: 71.98%, mIoU: 81.25%)
Epoch [11/100] (72.2s) | Loss: 0.7236 (Cls: 0.050, GIoU: 0.180, DFL: 0.891)
Epoch [12/100] (72.9s) | Loss: 0.7186 (Cls: 0.051, GIoU: 0.178, DFL: 0.890) | F1@0.50: 83.6% | F1@0.75: 64.3% (c35: 71.0%) | mIoU: 81.3% | MaxConf: 99.6% | NegPrec: 76.5%
Epoch [13/100] (73.5s) | Loss: 0.7134 (Cls: 0.049, GIoU: 0.177, DFL: 0.884)
Epoch [14/100] (72.5s) | Loss: 0.7104 (Cls: 0.048, GIoU: 0.177, DFL: 0.882) | F1@0.50: 86.7% | F1@0.75: 66.4% (c35: 71.7%) | mIoU: 81.2% | MaxConf: 99.8% | NegPrec: 81.5%
  [*] New best detector checkpoint saved! (F1@0.75: 66.35%, c35: 71.68%, mIoU: 81.24%)
Epoch [15/100] (72.5s) | Loss: 0.7091 (Cls: 0.048, GIoU: 0.176, DFL: 0.883)
Epoch [16/100] (72.9s) | Loss: 0.7033 (Cls: 0.047, GIoU: 0.175, DFL: 0.875) | F1@0.50: 85.1% | F1@0.75: 65.4% (c35: 71.4%) | mIoU: 81.4% | MaxConf: 99.8% | NegPrec: 82.7%
Epoch [17/100] (73.7s) | Loss: 0.7029 (Cls: 0.047, GIoU: 0.174, DFL: 0.881)
Epoch [18/100] (74.4s) | Loss: 0.6983 (Cls: 0.046, GIoU: 0.174, DFL: 0.873) | F1@0.50: 86.1% | F1@0.75: 65.8% (c35: 71.5%) | mIoU: 81.3% | MaxConf: 99.7% | NegPrec: 82.1%
Epoch [19/100] (73.0s) | Loss: 0.6986 (Cls: 0.046, GIoU: 0.174, DFL: 0.875)
Epoch [20/100] (73.1s) | Loss: 0.6967 (Cls: 0.045, GIoU: 0.173, DFL: 0.873) | F1@0.50: 86.4% | F1@0.75: 67.3% (c35: 72.4%) | mIoU: 81.5% | MaxConf: 99.8% | NegPrec: 84.0%
  [*] New best detector checkpoint saved! (F1@0.75: 67.26%, c35: 72.45%, mIoU: 81.52%)
Epoch [21/100] (73.0s) | Loss: 0.6914 (Cls: 0.045, GIoU: 0.172, DFL: 0.872)
Epoch [22/100] (73.7s) | Loss: 0.6882 (Cls: 0.044, GIoU: 0.171, DFL: 0.872) | F1@0.50: 86.3% | F1@0.75: 66.3% (c35: 71.3%) | mIoU: 81.5% | MaxConf: 99.8% | NegPrec: 84.0%
Epoch [23/100] (73.3s) | Loss: 0.6870 (Cls: 0.043, GIoU: 0.171, DFL: 0.867)
Epoch [24/100] (72.9s) | Loss: 0.6885 (Cls: 0.044, GIoU: 0.171, DFL: 0.871) | F1@0.50: 86.6% | F1@0.75: 67.1% (c35: 72.0%) | mIoU: 81.7% | MaxConf: 99.8% | NegPrec: 84.6%
Epoch [25/100] (73.6s) | Loss: 0.6847 (Cls: 0.043, GIoU: 0.170, DFL: 0.868)
Epoch [26/100] (73.3s) | Loss: 0.6789 (Cls: 0.043, GIoU: 0.169, DFL: 0.863) | F1@0.50: 86.5% | F1@0.75: 67.2% (c35: 72.2%) | mIoU: 81.7% | MaxConf: 99.8% | NegPrec: 84.0%
Epoch [27/100] (73.5s) | Loss: 0.6791 (Cls: 0.042, GIoU: 0.169, DFL: 0.863)
Epoch [28/100] (73.4s) | Loss: 0.6788 (Cls: 0.043, GIoU: 0.169, DFL: 0.862) | F1@0.50: 86.9% | F1@0.75: 67.5% (c35: 72.7%) | mIoU: 81.6% | MaxConf: 99.8% | NegPrec: 83.3%
  [*] New best detector checkpoint saved! (F1@0.75: 67.47%, c35: 72.68%, mIoU: 81.61%)
Epoch [29/100] (73.3s) | Loss: 0.6751 (Cls: 0.042, GIoU: 0.168, DFL: 0.858)
Epoch [30/100] (73.7s) | Loss: 0.6755 (Cls: 0.043, GIoU: 0.168, DFL: 0.862) | F1@0.50: 87.1% | F1@0.75: 67.4% (c35: 72.4%) | mIoU: 81.6% | MaxConf: 99.8% | NegPrec: 83.3%
Epoch [31/100] (73.2s) | Loss: 0.6682 (Cls: 0.041, GIoU: 0.166, DFL: 0.856)
Epoch [32/100] (73.5s) | Loss: 0.6668 (Cls: 0.040, GIoU: 0.166, DFL: 0.856) | F1@0.50: 86.9% | F1@0.75: 67.5% (c35: 72.2%) | mIoU: 81.6% | MaxConf: 99.8% | NegPrec: 82.1%
  [*] New best detector checkpoint saved! (F1@0.75: 67.53%, c35: 72.22%, mIoU: 81.61%)
Epoch [33/100] (73.2s) | Loss: 0.6681 (Cls: 0.041, GIoU: 0.166, DFL: 0.857)
Epoch [34/100] (73.1s) | Loss: 0.6628 (Cls: 0.040, GIoU: 0.165, DFL: 0.856) | F1@0.50: 87.1% | F1@0.75: 67.7% (c35: 72.3%) | mIoU: 81.7% | MaxConf: 99.8% | NegPrec: 84.6%
  [*] New best detector checkpoint saved! (F1@0.75: 67.71%, c35: 72.31%, mIoU: 81.65%)
Epoch [35/100] (73.5s) | Loss: 0.6631 (Cls: 0.040, GIoU: 0.165, DFL: 0.856)
Epoch [36/100] (73.9s) | Loss: 0.6600 (Cls: 0.040, GIoU: 0.164, DFL: 0.851) | F1@0.50: 86.9% | F1@0.75: 67.5% (c35: 72.4%) | mIoU: 81.7% | MaxConf: 99.9% | NegPrec: 85.2%
Epoch [37/100] (73.1s) | Loss: 0.6579 (Cls: 0.039, GIoU: 0.164, DFL: 0.852)
Epoch [38/100] (72.5s) | Loss: 0.6577 (Cls: 0.039, GIoU: 0.164, DFL: 0.846) | F1@0.50: 86.7% | F1@0.75: 67.2% (c35: 72.4%) | mIoU: 81.7% | MaxConf: 99.9% | NegPrec: 84.6%
Epoch [39/100] (73.8s) | Loss: 0.6500 (Cls: 0.038, GIoU: 0.162, DFL: 0.841)
Epoch [40/100] (72.7s) | Loss: 0.6495 (Cls: 0.038, GIoU: 0.162, DFL: 0.843) | F1@0.50: 87.0% | F1@0.75: 67.8% (c35: 72.9%) | mIoU: 81.8% | MaxConf: 99.9% | NegPrec: 84.6%
  [*] New best detector checkpoint saved! (F1@0.75: 67.81%, c35: 72.86%, mIoU: 81.80%)
Epoch [41/100] (72.9s) | Loss: 0.6470 (Cls: 0.037, GIoU: 0.161, DFL: 0.841)
Epoch [42/100] (72.3s) | Loss: 0.6503 (Cls: 0.038, GIoU: 0.162, DFL: 0.844) | F1@0.50: 86.9% | F1@0.75: 68.1% (c35: 73.1%) | mIoU: 81.8% | MaxConf: 99.9% | NegPrec: 84.6%
  [*] New best detector checkpoint saved! (F1@0.75: 68.10%, c35: 73.07%, mIoU: 81.85%)
Epoch [43/100] (73.5s) | Loss: 0.6466 (Cls: 0.037, GIoU: 0.161, DFL: 0.839)
Epoch [44/100] (72.6s) | Loss: 0.6438 (Cls: 0.037, GIoU: 0.160, DFL: 0.839) | F1@0.50: 87.2% | F1@0.75: 67.9% (c35: 72.8%) | mIoU: 81.8% | MaxConf: 99.9% | NegPrec: 85.2%
Epoch [45/100] (74.0s) | Loss: 0.6442 (Cls: 0.037, GIoU: 0.160, DFL: 0.841)
Epoch [46/100] (73.6s) | Loss: 0.6422 (Cls: 0.036, GIoU: 0.160, DFL: 0.838) | F1@0.50: 87.2% | F1@0.75: 67.9% (c35: 72.7%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 85.2%
Epoch [47/100] (73.2s) | Loss: 0.6408 (Cls: 0.036, GIoU: 0.160, DFL: 0.836)
Epoch [48/100] (73.0s) | Loss: 0.6405 (Cls: 0.037, GIoU: 0.159, DFL: 0.833) | F1@0.50: 87.4% | F1@0.75: 68.1% (c35: 72.7%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 86.4%
  [*] New best detector checkpoint saved! (F1@0.75: 68.10%, c35: 72.74%, mIoU: 81.90%)
Epoch [49/100] (73.5s) | Loss: 0.6397 (Cls: 0.036, GIoU: 0.159, DFL: 0.836)
Epoch [50/100] (72.8s) | Loss: 0.6359 (Cls: 0.035, GIoU: 0.159, DFL: 0.832) | F1@0.50: 87.5% | F1@0.75: 68.3% (c35: 73.0%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 85.8%
  [*] New best detector checkpoint saved! (F1@0.75: 68.34%, c35: 72.99%, mIoU: 81.90%)
Epoch [51/100] (73.0s) | Loss: 0.6314 (Cls: 0.035, GIoU: 0.157, DFL: 0.830)
Epoch [52/100] (73.2s) | Loss: 0.6325 (Cls: 0.036, GIoU: 0.158, DFL: 0.827) | F1@0.50: 87.5% | F1@0.75: 68.6% (c35: 73.1%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 86.4%
  [*] New best detector checkpoint saved! (F1@0.75: 68.57%, c35: 73.13%, mIoU: 81.90%)
Epoch [53/100] (72.8s) | Loss: 0.6340 (Cls: 0.036, GIoU: 0.158, DFL: 0.829)
Epoch [54/100] (72.6s) | Loss: 0.6273 (Cls: 0.034, GIoU: 0.156, DFL: 0.826) | F1@0.50: 87.5% | F1@0.75: 68.5% (c35: 73.0%) | mIoU: 81.9% | MaxConf: 99.8% | NegPrec: 85.2%
Epoch [55/100] (73.4s) | Loss: 0.6307 (Cls: 0.035, GIoU: 0.157, DFL: 0.830)
Epoch [56/100] (73.5s) | Loss: 0.6299 (Cls: 0.035, GIoU: 0.157, DFL: 0.827) | F1@0.50: 87.8% | F1@0.75: 68.7% (c35: 73.1%) | mIoU: 81.9% | MaxConf: 99.8% | NegPrec: 85.2%
  [*] New best detector checkpoint saved! (F1@0.75: 68.66%, c35: 73.10%, mIoU: 81.85%)
Epoch [57/100] (73.7s) | Loss: 0.6273 (Cls: 0.035, GIoU: 0.156, DFL: 0.828)
Epoch [58/100] (73.5s) | Loss: 0.6238 (Cls: 0.034, GIoU: 0.155, DFL: 0.825) | F1@0.50: 88.0% | F1@0.75: 68.9% (c35: 73.4%) | mIoU: 81.9% | MaxConf: 99.8% | NegPrec: 85.8%
  [*] New best detector checkpoint saved! (F1@0.75: 68.90%, c35: 73.36%, mIoU: 81.88%)
Epoch [59/100] (72.8s) | Loss: 0.6259 (Cls: 0.034, GIoU: 0.156, DFL: 0.826)
Epoch [60/100] (73.0s) | Loss: 0.6230 (Cls: 0.034, GIoU: 0.155, DFL: 0.822) | F1@0.50: 88.1% | F1@0.75: 69.1% (c35: 73.5%) | mIoU: 81.9% | MaxConf: 99.8% | NegPrec: 85.8%
  [*] New best detector checkpoint saved! (F1@0.75: 69.15%, c35: 73.49%, mIoU: 81.89%)
Epoch [61/100] (73.4s) | Loss: 0.6201 (Cls: 0.034, GIoU: 0.154, DFL: 0.822)
Epoch [62/100] (73.5s) | Loss: 0.6207 (Cls: 0.034, GIoU: 0.155, DFL: 0.821) | F1@0.50: 88.3% | F1@0.75: 69.1% (c35: 73.5%) | mIoU: 81.9% | MaxConf: 99.8% | NegPrec: 86.4%
Epoch [63/100] (73.9s) | Loss: 0.6197 (Cls: 0.033, GIoU: 0.154, DFL: 0.820)
Epoch [64/100] (73.9s) | Loss: 0.6154 (Cls: 0.033, GIoU: 0.153, DFL: 0.817) | F1@0.50: 88.4% | F1@0.75: 69.1% (c35: 73.3%) | mIoU: 81.9% | MaxConf: 99.8% | NegPrec: 86.4%
Epoch [65/100] (74.2s) | Loss: 0.6159 (Cls: 0.033, GIoU: 0.154, DFL: 0.815)
Epoch [66/100] (73.8s) | Loss: 0.6161 (Cls: 0.033, GIoU: 0.153, DFL: 0.817) | F1@0.50: 88.4% | F1@0.75: 69.2% (c35: 73.4%) | mIoU: 81.9% | MaxConf: 99.8% | NegPrec: 86.4%
  [*] New best detector checkpoint saved! (F1@0.75: 69.19%, c35: 73.39%, mIoU: 81.89%)
Epoch [67/100] (73.7s) | Loss: 0.6164 (Cls: 0.033, GIoU: 0.154, DFL: 0.818)
Epoch [68/100] (73.4s) | Loss: 0.6116 (Cls: 0.033, GIoU: 0.152, DFL: 0.815) | F1@0.50: 88.4% | F1@0.75: 69.1% (c35: 73.3%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 86.4%
Epoch [69/100] (73.4s) | Loss: 0.6127 (Cls: 0.033, GIoU: 0.153, DFL: 0.814)
Epoch [70/100] (73.3s) | Loss: 0.6129 (Cls: 0.033, GIoU: 0.152, DFL: 0.818) | F1@0.50: 88.3% | F1@0.75: 69.2% (c35: 73.3%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [71/100] (72.5s) | Loss: 0.6081 (Cls: 0.032, GIoU: 0.151, DFL: 0.814)
Epoch [72/100] (73.2s) | Loss: 0.6133 (Cls: 0.033, GIoU: 0.153, DFL: 0.815) | F1@0.50: 88.1% | F1@0.75: 69.0% (c35: 73.1%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 87.0%
Epoch [73/100] (72.9s) | Loss: 0.6094 (Cls: 0.033, GIoU: 0.152, DFL: 0.813)
Epoch [74/100] (73.3s) | Loss: 0.6102 (Cls: 0.033, GIoU: 0.152, DFL: 0.814) | F1@0.50: 88.2% | F1@0.75: 69.2% (c35: 73.3%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 86.4%
  [*] New best detector checkpoint saved! (F1@0.75: 69.21%, c35: 73.31%, mIoU: 81.94%)
Epoch [75/100] (72.8s) | Loss: 0.6079 (Cls: 0.032, GIoU: 0.151, DFL: 0.813)
Epoch [76/100] (73.8s) | Loss: 0.6060 (Cls: 0.032, GIoU: 0.151, DFL: 0.812) | F1@0.50: 88.2% | F1@0.75: 69.2% (c35: 73.2%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 86.4%
Epoch [77/100] (73.0s) | Loss: 0.6066 (Cls: 0.032, GIoU: 0.151, DFL: 0.810)
Epoch [78/100] (73.1s) | Loss: 0.6047 (Cls: 0.031, GIoU: 0.151, DFL: 0.810) | F1@0.50: 88.1% | F1@0.75: 69.1% (c35: 73.2%) | mIoU: 82.0% | MaxConf: 99.9% | NegPrec: 86.4%
Epoch [79/100] (73.5s) | Loss: 0.6065 (Cls: 0.032, GIoU: 0.151, DFL: 0.809)
Epoch [80/100] (74.2s) | Loss: 0.6043 (Cls: 0.031, GIoU: 0.150, DFL: 0.809) | F1@0.50: 88.2% | F1@0.75: 69.2% (c35: 73.3%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 86.4%
  [*] New best detector checkpoint saved! (F1@0.75: 69.23%, c35: 73.26%, mIoU: 81.94%)
Epoch [81/100] (73.8s) | Loss: 0.6061 (Cls: 0.032, GIoU: 0.151, DFL: 0.811)
Epoch [82/100] (73.5s) | Loss: 0.6055 (Cls: 0.032, GIoU: 0.151, DFL: 0.810) | F1@0.50: 88.2% | F1@0.75: 69.1% (c35: 73.2%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [83/100] (73.1s) | Loss: 0.6021 (Cls: 0.032, GIoU: 0.150, DFL: 0.809)
Epoch [84/100] (73.1s) | Loss: 0.6030 (Cls: 0.031, GIoU: 0.150, DFL: 0.809) | F1@0.50: 88.3% | F1@0.75: 69.2% (c35: 73.3%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [85/100] (73.4s) | Loss: 0.6038 (Cls: 0.031, GIoU: 0.151, DFL: 0.809)
Epoch [86/100] (72.8s) | Loss: 0.6039 (Cls: 0.031, GIoU: 0.150, DFL: 0.810) | F1@0.50: 88.2% | F1@0.75: 69.1% (c35: 73.3%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [87/100] (73.3s) | Loss: 0.6047 (Cls: 0.032, GIoU: 0.150, DFL: 0.810)
Epoch [88/100] (73.7s) | Loss: 0.6004 (Cls: 0.031, GIoU: 0.149, DFL: 0.806) | F1@0.50: 88.2% | F1@0.75: 69.1% (c35: 73.2%) | mIoU: 81.9% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [89/100] (73.6s) | Loss: 0.6003 (Cls: 0.032, GIoU: 0.149, DFL: 0.806)
Epoch [90/100] (73.7s) | Loss: 0.6031 (Cls: 0.032, GIoU: 0.150, DFL: 0.806) | F1@0.50: 88.1% | F1@0.75: 69.2% (c35: 73.4%) | mIoU: 82.0% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [91/100] (73.4s) | Loss: 0.5983 (Cls: 0.032, GIoU: 0.149, DFL: 0.804)
Epoch [92/100] (73.4s) | Loss: 0.5991 (Cls: 0.032, GIoU: 0.149, DFL: 0.806) | F1@0.50: 88.0% | F1@0.75: 69.1% (c35: 73.3%) | mIoU: 82.0% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [93/100] (73.6s) | Loss: 0.5968 (Cls: 0.031, GIoU: 0.149, DFL: 0.803)
Epoch [94/100] (73.1s) | Loss: 0.5984 (Cls: 0.031, GIoU: 0.149, DFL: 0.804) | F1@0.50: 88.1% | F1@0.75: 69.1% (c35: 73.3%) | mIoU: 82.0% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [95/100] (73.5s) | Loss: 0.6011 (Cls: 0.032, GIoU: 0.149, DFL: 0.806)
Epoch [96/100] (73.2s) | Loss: 0.5980 (Cls: 0.031, GIoU: 0.149, DFL: 0.802) | F1@0.50: 88.1% | F1@0.75: 69.1% (c35: 73.3%) | mIoU: 82.0% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [97/100] (74.4s) | Loss: 0.5976 (Cls: 0.031, GIoU: 0.149, DFL: 0.804)
Epoch [98/100] (73.6s) | Loss: 0.5984 (Cls: 0.031, GIoU: 0.149, DFL: 0.805) | F1@0.50: 88.0% | F1@0.75: 69.1% (c35: 73.3%) | mIoU: 82.0% | MaxConf: 99.9% | NegPrec: 85.8%
Epoch [99/100] (73.8s) | Loss: 0.5995 (Cls: 0.031, GIoU: 0.149, DFL: 0.805)
Epoch [100/100] (73.6s) | Loss: 0.6002 (Cls: 0.031, GIoU: 0.150, DFL: 0.806) | F1@0.50: 87.9% | F1@0.75: 69.1% (c35: 73.3%) | mIoU: 82.0% | MaxConf: 99.9% | NegPrec: 86.4%

[OK] NanoDet-Plus training finished.
Best Checkpoint: /content/Shoes_VTO/checkpoints/nanodet_stage_a_improved5/best.pt (Best F1@0.75: 69.23%)
