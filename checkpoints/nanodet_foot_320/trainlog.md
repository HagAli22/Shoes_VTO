======================================================================
  EXPERIMENT: NanoDet-Plus Foot Detector (Apache-2.0) Training
  Dataset:         /kaggle/working/Shoes_VTO/data (6620 train, 827 val)
  Backbone:        ShuffleNetV2 1.0x (Pretrained COCO: True)
  Device:          cuda (AMP: True)
  Epochs:          180 | Batch: 64 | LR: 0.001000 | Classes: 1
  Resolution:      320x320
======================================================================
Downloading: "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth" to /root/.cache/torch/hub/checkpoints/shufflenetv2_x1-5666bf0f80.pth
100%|██████████████████████████████████████| 8.79M/8.79M [00:00<00:00, 77.9MB/s]
[+] Successfully loaded official NanoDet-Plus-m COCO checkpoint:
    - Exact matching tensors loaded: 637 / 647
    - Adapted head tensors (regression transferred + cls init): 8
    - Missing / Unloaded tensors: 0
Epoch [01/180] (94.2s) | Loss: 1.6113 (Cls: 0.626, GIoU: 0.346, DFL: 1.171)
Epoch [02/180] (41.9s) | Loss: 0.8717 (Cls: 0.145, GIoU: 0.250, DFL: 0.908) | F1@0.50: 58.3% | F1@0.75: 38.4% | mIoU: 78.0% | MaxConf: 89.6% | NegPrec: 41.7%
  [*] New best detector checkpoint saved! (F1@0.75: 38.36%, mIoU: 77.97%)
Epoch [03/180] (41.2s) | Loss: 0.7731 (Cls: 0.116, GIoU: 0.223, DFL: 0.848)
Epoch [04/180] (42.8s) | Loss: 0.7139 (Cls: 0.101, GIoU: 0.205, DFL: 0.811) | F1@0.50: 70.3% | F1@0.75: 51.4% | mIoU: 79.9% | MaxConf: 94.5% | NegPrec: 57.1%
  [*] New best detector checkpoint saved! (F1@0.75: 51.39%, mIoU: 79.93%)
Epoch [05/180] (42.1s) | Loss: 0.6702 (Cls: 0.092, GIoU: 0.191, DFL: 0.783)
Epoch [06/180] (42.7s) | Loss: 0.6384 (Cls: 0.084, GIoU: 0.182, DFL: 0.764) | F1@0.50: 78.2% | F1@0.75: 58.5% | mIoU: 80.4% | MaxConf: 97.3% | NegPrec: 64.3%
  [*] New best detector checkpoint saved! (F1@0.75: 58.52%, mIoU: 80.37%)
Epoch [07/180] (42.9s) | Loss: 0.6124 (Cls: 0.079, GIoU: 0.173, DFL: 0.748)
Epoch [08/180] (42.8s) | Loss: 0.5890 (Cls: 0.073, GIoU: 0.166, DFL: 0.732) | F1@0.50: 80.9% | F1@0.75: 61.7% | mIoU: 81.0% | MaxConf: 96.7% | NegPrec: 72.6%
  [*] New best detector checkpoint saved! (F1@0.75: 61.71%, mIoU: 81.01%)
Epoch [09/180] (42.3s) | Loss: 0.5722 (Cls: 0.069, GIoU: 0.161, DFL: 0.722)
Epoch [10/180] (42.2s) | Loss: 0.5561 (Cls: 0.066, GIoU: 0.156, DFL: 0.712) | F1@0.50: 78.8% | F1@0.75: 60.9% | mIoU: 81.4% | MaxConf: 98.2% | NegPrec: 70.2%
Epoch [11/180] (42.2s) | Loss: 0.5395 (Cls: 0.063, GIoU: 0.151, DFL: 0.701)
Epoch [12/180] (41.9s) | Loss: 0.5242 (Cls: 0.060, GIoU: 0.146, DFL: 0.691) | F1@0.50: 83.7% | F1@0.75: 65.2% | mIoU: 81.5% | MaxConf: 98.3% | NegPrec: 78.6%
  [*] New best detector checkpoint saved! (F1@0.75: 65.23%, mIoU: 81.49%)
Epoch [13/180] (42.1s) | Loss: 0.5085 (Cls: 0.057, GIoU: 0.141, DFL: 0.681)
Epoch [14/180] (42.2s) | Loss: 0.5008 (Cls: 0.055, GIoU: 0.139, DFL: 0.676) | F1@0.50: 82.6% | F1@0.75: 64.5% | mIoU: 81.5% | MaxConf: 98.8% | NegPrec: 72.6%
Epoch [15/180] (42.0s) | Loss: 0.4860 (Cls: 0.052, GIoU: 0.134, DFL: 0.667)
Epoch [16/180] (42.4s) | Loss: 0.4789 (Cls: 0.050, GIoU: 0.131, DFL: 0.663) | F1@0.50: 83.3% | F1@0.75: 65.0% | mIoU: 81.7% | MaxConf: 99.7% | NegPrec: 70.2%
Epoch [17/180] (42.7s) | Loss: 0.4698 (Cls: 0.048, GIoU: 0.129, DFL: 0.657)
Epoch [18/180] (42.5s) | Loss: 0.4625 (Cls: 0.047, GIoU: 0.126, DFL: 0.652) | F1@0.50: 84.1% | F1@0.75: 66.6% | mIoU: 82.0% | MaxConf: 98.6% | NegPrec: 73.8%
  [*] New best detector checkpoint saved! (F1@0.75: 66.57%, mIoU: 81.99%)
Epoch [19/180] (42.1s) | Loss: 0.4558 (Cls: 0.046, GIoU: 0.124, DFL: 0.648)
Epoch [20/180] (42.5s) | Loss: 0.4502 (Cls: 0.044, GIoU: 0.123, DFL: 0.645) | F1@0.50: 85.5% | F1@0.75: 67.3% | mIoU: 82.1% | MaxConf: 99.4% | NegPrec: 81.0%
  [*] New best detector checkpoint saved! (F1@0.75: 67.29%, mIoU: 82.09%)
Epoch [21/180] (42.2s) | Loss: 0.4382 (Cls: 0.042, GIoU: 0.118, DFL: 0.638)
Epoch [22/180] (42.1s) | Loss: 0.4310 (Cls: 0.040, GIoU: 0.116, DFL: 0.634) | F1@0.50: 85.5% | F1@0.75: 67.7% | mIoU: 82.1% | MaxConf: 99.5% | NegPrec: 78.6%
  [*] New best detector checkpoint saved! (F1@0.75: 67.73%, mIoU: 82.12%)
Epoch [23/180] (42.5s) | Loss: 0.4260 (Cls: 0.039, GIoU: 0.115, DFL: 0.631)
Epoch [24/180] (41.9s) | Loss: 0.4187 (Cls: 0.037, GIoU: 0.112, DFL: 0.626) | F1@0.50: 86.6% | F1@0.75: 67.1% | mIoU: 81.9% | MaxConf: 99.8% | NegPrec: 73.8%
Epoch [25/180] (41.9s) | Loss: 0.4143 (Cls: 0.037, GIoU: 0.111, DFL: 0.624)
Epoch [26/180] (41.8s) | Loss: 0.4095 (Cls: 0.036, GIoU: 0.109, DFL: 0.620) | F1@0.50: 86.3% | F1@0.75: 68.1% | mIoU: 82.0% | MaxConf: 99.8% | NegPrec: 81.0%
  [*] New best detector checkpoint saved! (F1@0.75: 68.12%, mIoU: 82.02%)
Epoch [27/180] (41.8s) | Loss: 0.4014 (Cls: 0.034, GIoU: 0.107, DFL: 0.616)
Epoch [28/180] (42.0s) | Loss: 0.3967 (Cls: 0.033, GIoU: 0.105, DFL: 0.613) | F1@0.50: 86.5% | F1@0.75: 68.2% | mIoU: 82.0% | MaxConf: 99.8% | NegPrec: 75.0%
  [*] New best detector checkpoint saved! (F1@0.75: 68.21%, mIoU: 82.00%)
Epoch [29/180] (42.2s) | Loss: 0.3925 (Cls: 0.032, GIoU: 0.104, DFL: 0.610)
Epoch [30/180] (42.5s) | Loss: 0.3872 (Cls: 0.031, GIoU: 0.102, DFL: 0.608) | F1@0.50: 86.9% | F1@0.75: 68.5% | mIoU: 82.0% | MaxConf: 99.9% | NegPrec: 78.6%
  [*] New best detector checkpoint saved! (F1@0.75: 68.49%, mIoU: 81.97%)
Epoch [31/180] (42.2s) | Loss: 0.3862 (Cls: 0.031, GIoU: 0.102, DFL: 0.607)
Epoch [32/180] (42.6s) | Loss: 0.3792 (Cls: 0.029, GIoU: 0.100, DFL: 0.603) | F1@0.50: 87.5% | F1@0.75: 68.8% | mIoU: 82.2% | MaxConf: 99.8% | NegPrec: 76.2%
  [*] New best detector checkpoint saved! (F1@0.75: 68.81%, mIoU: 82.19%)
Epoch [33/180] (42.6s) | Loss: 0.3743 (Cls: 0.029, GIoU: 0.098, DFL: 0.600)
Epoch [34/180] (42.6s) | Loss: 0.3730 (Cls: 0.028, GIoU: 0.097, DFL: 0.599) | F1@0.50: 88.1% | F1@0.75: 69.5% | mIoU: 82.2% | MaxConf: 99.9% | NegPrec: 77.4%
  [*] New best detector checkpoint saved! (F1@0.75: 69.46%, mIoU: 82.17%)
Epoch [35/180] (42.2s) | Loss: 0.3681 (Cls: 0.027, GIoU: 0.096, DFL: 0.597)
Epoch [36/180] (42.4s) | Loss: 0.3631 (Cls: 0.026, GIoU: 0.094, DFL: 0.594) | F1@0.50: 87.2% | F1@0.75: 69.2% | mIoU: 82.4% | MaxConf: 99.9% | NegPrec: 73.8%
Epoch [37/180] (41.9s) | Loss: 0.3606 (Cls: 0.026, GIoU: 0.093, DFL: 0.593)
Epoch [38/180] (42.0s) | Loss: 0.3571 (Cls: 0.025, GIoU: 0.092, DFL: 0.591) | F1@0.50: 87.0% | F1@0.75: 68.6% | mIoU: 82.2% | MaxConf: 99.9% | NegPrec: 82.1%
Epoch [39/180] (42.2s) | Loss: 0.3550 (Cls: 0.025, GIoU: 0.092, DFL: 0.589)
Epoch [40/180] (42.2s) | Loss: 0.3499 (Cls: 0.023, GIoU: 0.090, DFL: 0.587) | F1@0.50: 87.0% | F1@0.75: 70.3% | mIoU: 82.7% | MaxConf: 99.9% | NegPrec: 77.4%
  [*] New best detector checkpoint saved! (F1@0.75: 70.32%, mIoU: 82.69%)
Epoch [41/180] (42.2s) | Loss: 0.3474 (Cls: 0.023, GIoU: 0.089, DFL: 0.586)
Epoch [42/180] (41.6s) | Loss: 0.3424 (Cls: 0.022, GIoU: 0.087, DFL: 0.583) | F1@0.50: 87.7% | F1@0.75: 69.9% | mIoU: 82.5% | MaxConf: 99.9% | NegPrec: 83.3%
Epoch [43/180] (42.4s) | Loss: 0.3405 (Cls: 0.021, GIoU: 0.087, DFL: 0.582)
Epoch [44/180] (42.3s) | Loss: 0.3379 (Cls: 0.021, GIoU: 0.086, DFL: 0.580) | F1@0.50: 88.9% | F1@0.75: 71.0% | mIoU: 82.3% | MaxConf: 99.9% | NegPrec: 82.1%
  [*] New best detector checkpoint saved! (F1@0.75: 70.98%, mIoU: 82.30%)
Epoch [45/180] (42.1s) | Loss: 0.3368 (Cls: 0.021, GIoU: 0.085, DFL: 0.580)
Epoch [46/180] (42.3s) | Loss: 0.3318 (Cls: 0.020, GIoU: 0.084, DFL: 0.578) | F1@0.50: 88.8% | F1@0.75: 70.7% | mIoU: 82.6% | MaxConf: 99.9% | NegPrec: 79.8%
Epoch [47/180] (42.1s) | Loss: 0.3300 (Cls: 0.020, GIoU: 0.083, DFL: 0.576)
Epoch [48/180] (42.6s) | Loss: 0.3276 (Cls: 0.019, GIoU: 0.083, DFL: 0.575) | F1@0.50: 88.5% | F1@0.75: 69.8% | mIoU: 82.4% | MaxConf: 99.9% | NegPrec: 78.6%
Epoch [49/180] (42.0s) | Loss: 0.3241 (Cls: 0.018, GIoU: 0.081, DFL: 0.573)
Epoch [50/180] (42.5s) | Loss: 0.3218 (Cls: 0.018, GIoU: 0.080, DFL: 0.573) | F1@0.50: 87.3% | F1@0.75: 69.9% | mIoU: 82.6% | MaxConf: 100.0% | NegPrec: 76.2%
Epoch [51/180] (42.3s) | Loss: 0.3180 (Cls: 0.017, GIoU: 0.079, DFL: 0.571)
Epoch [52/180] (42.6s) | Loss: 0.3169 (Cls: 0.017, GIoU: 0.079, DFL: 0.570) | F1@0.50: 88.6% | F1@0.75: 70.2% | mIoU: 82.5% | MaxConf: 99.9% | NegPrec: 83.3%
Epoch [53/180] (42.7s) | Loss: 0.3169 (Cls: 0.017, GIoU: 0.079, DFL: 0.570)
Epoch [54/180] (42.4s) | Loss: 0.3134 (Cls: 0.016, GIoU: 0.078, DFL: 0.569) | F1@0.50: 88.9% | F1@0.75: 69.9% | mIoU: 82.5% | MaxConf: 100.0% | NegPrec: 78.6%
Epoch [55/180] (42.2s) | Loss: 0.3110 (Cls: 0.016, GIoU: 0.077, DFL: 0.567)
Epoch [56/180] (42.1s) | Loss: 0.3077 (Cls: 0.015, GIoU: 0.075, DFL: 0.566) | F1@0.50: 89.1% | F1@0.75: 71.1% | mIoU: 82.6% | MaxConf: 99.9% | NegPrec: 84.5%
  [*] New best detector checkpoint saved! (F1@0.75: 71.07%, mIoU: 82.59%)
Epoch [57/180] (42.8s) | Loss: 0.3052 (Cls: 0.015, GIoU: 0.075, DFL: 0.565)
Epoch [58/180] (42.4s) | Loss: 0.3053 (Cls: 0.015, GIoU: 0.075, DFL: 0.564) | F1@0.50: 89.1% | F1@0.75: 70.8% | mIoU: 82.7% | MaxConf: 99.9% | NegPrec: 79.8%
Epoch [59/180] (42.0s) | Loss: 0.3046 (Cls: 0.015, GIoU: 0.074, DFL: 0.564)
Epoch [60/180] (41.6s) | Loss: 0.3018 (Cls: 0.014, GIoU: 0.074, DFL: 0.562) | F1@0.50: 89.4% | F1@0.75: 70.8% | mIoU: 82.6% | MaxConf: 99.9% | NegPrec: 82.1%
Epoch [61/180] (42.6s) | Loss: 0.2996 (Cls: 0.014, GIoU: 0.073, DFL: 0.562)
Epoch [62/180] (41.9s) | Loss: 0.2950 (Cls: 0.013, GIoU: 0.071, DFL: 0.560) | F1@0.50: 89.0% | F1@0.75: 70.4% | mIoU: 82.6% | MaxConf: 99.9% | NegPrec: 82.1%
Epoch [63/180] (43.7s) | Loss: 0.2953 (Cls: 0.013, GIoU: 0.071, DFL: 0.560)
Epoch [64/180] (43.8s) | Loss: 0.2940 (Cls: 0.013, GIoU: 0.071, DFL: 0.559) | F1@0.50: 89.3% | F1@0.75: 70.5% | mIoU: 82.6% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [65/180] (44.0s) | Loss: 0.2932 (Cls: 0.012, GIoU: 0.071, DFL: 0.558)
Epoch [66/180] (44.5s) | Loss: 0.2903 (Cls: 0.012, GIoU: 0.069, DFL: 0.557) | F1@0.50: 89.1% | F1@0.75: 70.7% | mIoU: 82.7% | MaxConf: 99.9% | NegPrec: 81.0%
Epoch [67/180] (43.6s) | Loss: 0.2890 (Cls: 0.012, GIoU: 0.069, DFL: 0.556)
Epoch [68/180] (42.9s) | Loss: 0.2866 (Cls: 0.012, GIoU: 0.068, DFL: 0.555) | F1@0.50: 89.4% | F1@0.75: 71.7% | mIoU: 82.6% | MaxConf: 99.9% | NegPrec: 79.8%
  [*] New best detector checkpoint saved! (F1@0.75: 71.67%, mIoU: 82.56%)
Epoch [69/180] (43.0s) | Loss: 0.2854 (Cls: 0.011, GIoU: 0.068, DFL: 0.554)
Epoch [70/180] (41.4s) | Loss: 0.2847 (Cls: 0.011, GIoU: 0.067, DFL: 0.555) | F1@0.50: 89.4% | F1@0.75: 72.1% | mIoU: 82.8% | MaxConf: 99.9% | NegPrec: 81.0%
  [*] New best detector checkpoint saved! (F1@0.75: 72.07%, mIoU: 82.80%)
Epoch [71/180] (42.0s) | Loss: 0.2809 (Cls: 0.011, GIoU: 0.066, DFL: 0.552)
Epoch [72/180] (41.5s) | Loss: 0.2798 (Cls: 0.011, GIoU: 0.066, DFL: 0.552) | F1@0.50: 89.7% | F1@0.75: 71.7% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 83.3%
Epoch [73/180] (41.8s) | Loss: 0.2799 (Cls: 0.011, GIoU: 0.066, DFL: 0.552)
Epoch [74/180] (42.4s) | Loss: 0.2773 (Cls: 0.010, GIoU: 0.065, DFL: 0.551) | F1@0.50: 89.5% | F1@0.75: 72.3% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 82.1%
  [*] New best detector checkpoint saved! (F1@0.75: 72.35%, mIoU: 82.77%)
Epoch [75/180] (42.1s) | Loss: 0.2762 (Cls: 0.010, GIoU: 0.064, DFL: 0.550)
Epoch [76/180] (42.4s) | Loss: 0.2749 (Cls: 0.010, GIoU: 0.064, DFL: 0.550) | F1@0.50: 89.4% | F1@0.75: 71.0% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 83.3%
Epoch [77/180] (41.8s) | Loss: 0.2730 (Cls: 0.009, GIoU: 0.063, DFL: 0.549)
Epoch [78/180] (42.1s) | Loss: 0.2713 (Cls: 0.009, GIoU: 0.062, DFL: 0.549) | F1@0.50: 89.6% | F1@0.75: 71.9% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 82.1%
Epoch [79/180] (42.2s) | Loss: 0.2711 (Cls: 0.009, GIoU: 0.062, DFL: 0.548)
Epoch [80/180] (41.9s) | Loss: 0.2684 (Cls: 0.009, GIoU: 0.061, DFL: 0.547) | F1@0.50: 89.8% | F1@0.75: 71.8% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [81/180] (41.6s) | Loss: 0.2673 (Cls: 0.009, GIoU: 0.061, DFL: 0.547)
Epoch [82/180] (42.1s) | Loss: 0.2653 (Cls: 0.008, GIoU: 0.060, DFL: 0.545) | F1@0.50: 89.6% | F1@0.75: 71.8% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 82.1%
Epoch [83/180] (42.0s) | Loss: 0.2650 (Cls: 0.009, GIoU: 0.060, DFL: 0.545)
Epoch [84/180] (42.5s) | Loss: 0.2636 (Cls: 0.008, GIoU: 0.060, DFL: 0.545) | F1@0.50: 90.4% | F1@0.75: 71.9% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 84.5%
Epoch [85/180] (42.4s) | Loss: 0.2633 (Cls: 0.008, GIoU: 0.059, DFL: 0.544)
Epoch [86/180] (42.1s) | Loss: 0.2618 (Cls: 0.008, GIoU: 0.059, DFL: 0.544) | F1@0.50: 90.2% | F1@0.75: 71.6% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 82.1%
Epoch [87/180] (41.9s) | Loss: 0.2605 (Cls: 0.008, GIoU: 0.058, DFL: 0.544)
Epoch [88/180] (41.4s) | Loss: 0.2589 (Cls: 0.007, GIoU: 0.058, DFL: 0.543) | F1@0.50: 90.1% | F1@0.75: 70.9% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 83.3%
Epoch [89/180] (41.4s) | Loss: 0.2575 (Cls: 0.007, GIoU: 0.057, DFL: 0.542)
Epoch [90/180] (41.5s) | Loss: 0.2567 (Cls: 0.007, GIoU: 0.057, DFL: 0.542) | F1@0.50: 90.1% | F1@0.75: 71.4% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 83.3%
Epoch [91/180] (41.9s) | Loss: 0.2564 (Cls: 0.007, GIoU: 0.057, DFL: 0.541)
Epoch [92/180] (40.9s) | Loss: 0.2544 (Cls: 0.007, GIoU: 0.056, DFL: 0.541) | F1@0.50: 90.4% | F1@0.75: 72.0% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 83.3%
Epoch [93/180] (41.7s) | Loss: 0.2521 (Cls: 0.007, GIoU: 0.055, DFL: 0.540)
Epoch [94/180] (41.4s) | Loss: 0.2516 (Cls: 0.007, GIoU: 0.055, DFL: 0.540) | F1@0.50: 89.9% | F1@0.75: 72.6% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 83.3%
  [*] New best detector checkpoint saved! (F1@0.75: 72.56%, mIoU: 82.89%)
Epoch [95/180] (41.9s) | Loss: 0.2518 (Cls: 0.007, GIoU: 0.055, DFL: 0.540)
Epoch [96/180] (41.4s) | Loss: 0.2506 (Cls: 0.006, GIoU: 0.055, DFL: 0.539) | F1@0.50: 90.1% | F1@0.75: 72.3% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 82.1%
Epoch [97/180] (41.3s) | Loss: 0.2488 (Cls: 0.006, GIoU: 0.054, DFL: 0.539)
Epoch [98/180] (41.7s) | Loss: 0.2476 (Cls: 0.006, GIoU: 0.054, DFL: 0.538) | F1@0.50: 89.8% | F1@0.75: 72.7% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 79.8%
  [*] New best detector checkpoint saved! (F1@0.75: 72.75%, mIoU: 82.87%)
Epoch [99/180] (41.1s) | Loss: 0.2462 (Cls: 0.006, GIoU: 0.053, DFL: 0.537)
Epoch [100/180] (41.5s) | Loss: 0.2448 (Cls: 0.005, GIoU: 0.053, DFL: 0.537) | F1@0.50: 90.1% | F1@0.75: 72.2% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [101/180] (41.6s) | Loss: 0.2449 (Cls: 0.006, GIoU: 0.053, DFL: 0.537)
Epoch [102/180] (41.5s) | Loss: 0.2442 (Cls: 0.005, GIoU: 0.052, DFL: 0.537) | F1@0.50: 90.4% | F1@0.75: 72.0% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [103/180] (41.5s) | Loss: 0.2433 (Cls: 0.005, GIoU: 0.052, DFL: 0.536)
Epoch [104/180] (41.3s) | Loss: 0.2422 (Cls: 0.005, GIoU: 0.051, DFL: 0.536) | F1@0.50: 90.6% | F1@0.75: 72.5% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [105/180] (41.6s) | Loss: 0.2407 (Cls: 0.005, GIoU: 0.051, DFL: 0.536)
Epoch [106/180] (41.5s) | Loss: 0.2413 (Cls: 0.005, GIoU: 0.051, DFL: 0.536) | F1@0.50: 91.0% | F1@0.75: 72.8% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 82.1%
  [*] New best detector checkpoint saved! (F1@0.75: 72.79%, mIoU: 82.66%)
Epoch [107/180] (41.5s) | Loss: 0.2387 (Cls: 0.005, GIoU: 0.050, DFL: 0.535)
Epoch [108/180] (41.0s) | Loss: 0.2380 (Cls: 0.005, GIoU: 0.050, DFL: 0.534) | F1@0.50: 90.5% | F1@0.75: 72.4% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [109/180] (41.3s) | Loss: 0.2363 (Cls: 0.005, GIoU: 0.049, DFL: 0.534)
Epoch [110/180] (41.9s) | Loss: 0.2359 (Cls: 0.005, GIoU: 0.049, DFL: 0.534) | F1@0.50: 90.5% | F1@0.75: 72.5% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [111/180] (41.7s) | Loss: 0.2359 (Cls: 0.005, GIoU: 0.049, DFL: 0.533)
Epoch [112/180] (41.6s) | Loss: 0.2344 (Cls: 0.004, GIoU: 0.048, DFL: 0.533) | F1@0.50: 90.3% | F1@0.75: 72.5% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 78.6%
Epoch [113/180] (41.3s) | Loss: 0.2333 (Cls: 0.004, GIoU: 0.048, DFL: 0.533)
Epoch [114/180] (41.5s) | Loss: 0.2329 (Cls: 0.004, GIoU: 0.048, DFL: 0.532) | F1@0.50: 90.7% | F1@0.75: 72.6% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [115/180] (41.3s) | Loss: 0.2314 (Cls: 0.004, GIoU: 0.047, DFL: 0.532)
Epoch [116/180] (41.1s) | Loss: 0.2312 (Cls: 0.004, GIoU: 0.047, DFL: 0.532) | F1@0.50: 90.4% | F1@0.75: 72.3% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [117/180] (41.6s) | Loss: 0.2311 (Cls: 0.004, GIoU: 0.047, DFL: 0.532)
Epoch [118/180] (41.5s) | Loss: 0.2301 (Cls: 0.004, GIoU: 0.047, DFL: 0.531) | F1@0.50: 90.2% | F1@0.75: 72.4% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [119/180] (41.4s) | Loss: 0.2291 (Cls: 0.004, GIoU: 0.046, DFL: 0.531)
Epoch [120/180] (41.4s) | Loss: 0.2290 (Cls: 0.004, GIoU: 0.046, DFL: 0.531) | F1@0.50: 90.4% | F1@0.75: 72.2% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [121/180] (41.5s) | Loss: 0.2287 (Cls: 0.004, GIoU: 0.046, DFL: 0.531)
Epoch [122/180] (41.4s) | Loss: 0.2280 (Cls: 0.004, GIoU: 0.046, DFL: 0.531) | F1@0.50: 90.4% | F1@0.75: 72.3% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [123/180] (42.0s) | Loss: 0.2275 (Cls: 0.004, GIoU: 0.046, DFL: 0.530)
Epoch [124/180] (42.0s) | Loss: 0.2265 (Cls: 0.004, GIoU: 0.045, DFL: 0.530) | F1@0.50: 90.4% | F1@0.75: 72.8% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 78.6%
Epoch [125/180] (41.1s) | Loss: 0.2258 (Cls: 0.004, GIoU: 0.045, DFL: 0.530)
Epoch [126/180] (41.5s) | Loss: 0.2243 (Cls: 0.003, GIoU: 0.044, DFL: 0.530) | F1@0.50: 90.5% | F1@0.75: 73.2% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 81.0%
  [*] New best detector checkpoint saved! (F1@0.75: 73.15%, mIoU: 82.93%)
Epoch [127/180] (41.3s) | Loss: 0.2250 (Cls: 0.003, GIoU: 0.045, DFL: 0.529)
Epoch [128/180] (41.3s) | Loss: 0.2235 (Cls: 0.003, GIoU: 0.044, DFL: 0.529) | F1@0.50: 90.2% | F1@0.75: 72.9% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [129/180] (41.4s) | Loss: 0.2225 (Cls: 0.003, GIoU: 0.044, DFL: 0.529)
Epoch [130/180] (41.3s) | Loss: 0.2221 (Cls: 0.003, GIoU: 0.043, DFL: 0.528) | F1@0.50: 90.2% | F1@0.75: 72.6% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [131/180] (40.8s) | Loss: 0.2219 (Cls: 0.003, GIoU: 0.043, DFL: 0.529)
Epoch [132/180] (41.2s) | Loss: 0.2211 (Cls: 0.003, GIoU: 0.043, DFL: 0.528) | F1@0.50: 90.1% | F1@0.75: 72.6% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [133/180] (41.3s) | Loss: 0.2197 (Cls: 0.003, GIoU: 0.043, DFL: 0.528)
Epoch [134/180] (41.0s) | Loss: 0.2205 (Cls: 0.003, GIoU: 0.043, DFL: 0.528) | F1@0.50: 90.3% | F1@0.75: 72.7% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [135/180] (41.0s) | Loss: 0.2191 (Cls: 0.003, GIoU: 0.042, DFL: 0.527)
Epoch [136/180] (41.1s) | Loss: 0.2187 (Cls: 0.003, GIoU: 0.042, DFL: 0.527) | F1@0.50: 90.3% | F1@0.75: 72.6% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [137/180] (41.0s) | Loss: 0.2193 (Cls: 0.003, GIoU: 0.042, DFL: 0.528)
Epoch [138/180] (41.1s) | Loss: 0.2196 (Cls: 0.003, GIoU: 0.042, DFL: 0.528) | F1@0.50: 90.1% | F1@0.75: 72.5% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 79.8%
Epoch [139/180] (40.9s) | Loss: 0.2185 (Cls: 0.003, GIoU: 0.042, DFL: 0.527)
Epoch [140/180] (41.0s) | Loss: 0.2174 (Cls: 0.003, GIoU: 0.042, DFL: 0.527) | F1@0.50: 90.2% | F1@0.75: 72.5% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [141/180] (40.7s) | Loss: 0.2178 (Cls: 0.003, GIoU: 0.042, DFL: 0.527)
Epoch [142/180] (40.8s) | Loss: 0.2169 (Cls: 0.003, GIoU: 0.041, DFL: 0.527) | F1@0.50: 90.2% | F1@0.75: 72.4% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [143/180] (41.0s) | Loss: 0.2167 (Cls: 0.003, GIoU: 0.041, DFL: 0.526)
Epoch [144/180] (41.0s) | Loss: 0.2164 (Cls: 0.003, GIoU: 0.041, DFL: 0.527) | F1@0.50: 90.5% | F1@0.75: 72.4% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [145/180] (41.2s) | Loss: 0.2153 (Cls: 0.002, GIoU: 0.041, DFL: 0.526)
Epoch [146/180] (41.1s) | Loss: 0.2155 (Cls: 0.002, GIoU: 0.041, DFL: 0.526) | F1@0.50: 90.3% | F1@0.75: 72.1% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [147/180] (41.5s) | Loss: 0.2151 (Cls: 0.002, GIoU: 0.041, DFL: 0.526)
Epoch [148/180] (41.1s) | Loss: 0.2154 (Cls: 0.003, GIoU: 0.041, DFL: 0.526) | F1@0.50: 90.4% | F1@0.75: 71.9% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [149/180] (40.7s) | Loss: 0.2143 (Cls: 0.003, GIoU: 0.040, DFL: 0.526)
Epoch [150/180] (41.1s) | Loss: 0.2141 (Cls: 0.002, GIoU: 0.040, DFL: 0.526) | F1@0.50: 90.3% | F1@0.75: 72.1% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [151/180] (41.3s) | Loss: 0.2136 (Cls: 0.002, GIoU: 0.040, DFL: 0.526)
Epoch [152/180] (41.2s) | Loss: 0.2134 (Cls: 0.002, GIoU: 0.040, DFL: 0.526) | F1@0.50: 90.3% | F1@0.75: 72.2% | mIoU: 82.9% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [153/180] (40.5s) | Loss: 0.2126 (Cls: 0.002, GIoU: 0.040, DFL: 0.525)
Epoch [154/180] (40.8s) | Loss: 0.2122 (Cls: 0.002, GIoU: 0.039, DFL: 0.525) | F1@0.50: 90.2% | F1@0.75: 72.2% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [155/180] (40.6s) | Loss: 0.2129 (Cls: 0.002, GIoU: 0.040, DFL: 0.525)
Epoch [156/180] (40.6s) | Loss: 0.2121 (Cls: 0.002, GIoU: 0.039, DFL: 0.525) | F1@0.50: 90.1% | F1@0.75: 72.1% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [157/180] (40.3s) | Loss: 0.2136 (Cls: 0.002, GIoU: 0.040, DFL: 0.526)
Epoch [158/180] (41.2s) | Loss: 0.2124 (Cls: 0.002, GIoU: 0.039, DFL: 0.525) | F1@0.50: 90.1% | F1@0.75: 72.3% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [159/180] (41.4s) | Loss: 0.2120 (Cls: 0.002, GIoU: 0.039, DFL: 0.525)
Epoch [160/180] (41.3s) | Loss: 0.2113 (Cls: 0.002, GIoU: 0.039, DFL: 0.525) | F1@0.50: 90.1% | F1@0.75: 72.3% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [161/180] (41.5s) | Loss: 0.2117 (Cls: 0.002, GIoU: 0.039, DFL: 0.525)
Epoch [162/180] (41.1s) | Loss: 0.2119 (Cls: 0.002, GIoU: 0.039, DFL: 0.525) | F1@0.50: 90.2% | F1@0.75: 72.1% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [163/180] (41.5s) | Loss: 0.2111 (Cls: 0.002, GIoU: 0.039, DFL: 0.525)
Epoch [164/180] (41.4s) | Loss: 0.2111 (Cls: 0.002, GIoU: 0.039, DFL: 0.525) | F1@0.50: 90.1% | F1@0.75: 72.1% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [165/180] (41.5s) | Loss: 0.2106 (Cls: 0.002, GIoU: 0.039, DFL: 0.525)
Epoch [166/180] (41.5s) | Loss: 0.2105 (Cls: 0.002, GIoU: 0.039, DFL: 0.525) | F1@0.50: 90.1% | F1@0.75: 72.2% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [167/180] (41.5s) | Loss: 0.2114 (Cls: 0.002, GIoU: 0.039, DFL: 0.525)
Epoch [168/180] (41.4s) | Loss: 0.2096 (Cls: 0.002, GIoU: 0.038, DFL: 0.525) | F1@0.50: 90.1% | F1@0.75: 72.0% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [169/180] (41.4s) | Loss: 0.2109 (Cls: 0.002, GIoU: 0.039, DFL: 0.525)
Epoch [170/180] (41.4s) | Loss: 0.2103 (Cls: 0.002, GIoU: 0.039, DFL: 0.525) | F1@0.50: 90.2% | F1@0.75: 72.0% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [171/180] (41.8s) | Loss: 0.2095 (Cls: 0.002, GIoU: 0.038, DFL: 0.525)
Epoch [172/180] (41.7s) | Loss: 0.2099 (Cls: 0.002, GIoU: 0.038, DFL: 0.524) | F1@0.50: 90.3% | F1@0.75: 72.2% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [173/180] (41.9s) | Loss: 0.2105 (Cls: 0.002, GIoU: 0.039, DFL: 0.525)
Epoch [174/180] (42.0s) | Loss: 0.2102 (Cls: 0.002, GIoU: 0.038, DFL: 0.525) | F1@0.50: 90.3% | F1@0.75: 72.2% | mIoU: 82.7% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [175/180] (41.9s) | Loss: 0.2102 (Cls: 0.002, GIoU: 0.038, DFL: 0.525)
Epoch [176/180] (41.3s) | Loss: 0.2094 (Cls: 0.002, GIoU: 0.038, DFL: 0.524) | F1@0.50: 90.2% | F1@0.75: 72.1% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [177/180] (41.6s) | Loss: 0.2097 (Cls: 0.002, GIoU: 0.038, DFL: 0.525)
Epoch [178/180] (41.8s) | Loss: 0.2100 (Cls: 0.002, GIoU: 0.038, DFL: 0.525) | F1@0.50: 90.1% | F1@0.75: 72.1% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%
Epoch [179/180] (41.0s) | Loss: 0.2093 (Cls: 0.002, GIoU: 0.038, DFL: 0.524)
Epoch [180/180] (41.1s) | Loss: 0.2100 (Cls: 0.002, GIoU: 0.038, DFL: 0.525) | F1@0.50: 90.2% | F1@0.75: 72.2% | mIoU: 82.8% | MaxConf: 100.0% | NegPrec: 81.0%

[OK] NanoDet-Plus training finished.
Best Checkpoint: checkpoints/nanodet_foot_320/best.pt (Best F1@0.75: 73.15%)