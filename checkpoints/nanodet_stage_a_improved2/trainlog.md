======================================================================
  EXPERIMENT: NanoDet-Plus Stage A Foot Detector (Hardware-Optimized)
  Dataset:         /content/Shoes_VTO/data/shuffled_v4_1_v4_2_13kp (1286 train, 160 val)
  Backbone:        ShuffleNetV2 1.0x (Pretrained: True, Act: RELU)
  Neck:            REP_PAN | Strides: [8, 16, 32] | BBox Loss: CIOU
  Assigner:        TAL (Warmup: 8 ep) | Loss Weights (Box: 2.5, DFL: 0.05)
  Device:          cuda (AMP: True)
  Epochs:          100 | Batch: 16 | LR: 0.001500 | Classes: 1
  Resolution:      256x256
======================================================================
[+] Loaded pretrained foot detector weights from checkpoints/nanodet_foot_320/best.pt:
    - Exact matching tensors: 414 / 501
    - Adapted head tensors:   6
    - Unloaded tensors:       79
Epoch [01/100] (28.3s) | Loss: 3.4339 (Cls: 1.284, GIoU: 0.790, DFL: 3.508)
Epoch [02/100] (8.8s) | Loss: 2.0835 (Cls: 0.211, GIoU: 0.695, DFL: 2.676) | F1@0.50: 11.2% | F1@0.75: 0.8% | mIoU: 60.2% | MaxConf: 98.1% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 0.77%, mIoU: 60.16%)
Epoch [03/100] (8.8s) | Loss: 1.6607 (Cls: 0.127, GIoU: 0.565, DFL: 2.407)
Epoch [04/100] (8.8s) | Loss: 1.4018 (Cls: 0.106, GIoU: 0.476, DFL: 2.098) | F1@0.50: 22.4% | F1@0.75: 3.4% | mIoU: 65.1% | MaxConf: 98.8% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 3.44%, mIoU: 65.08%)
Epoch [05/100] (8.9s) | Loss: 1.2518 (Cls: 0.098, GIoU: 0.424, DFL: 1.886)
Epoch [06/100] (9.1s) | Loss: 1.1327 (Cls: 0.086, GIoU: 0.383, DFL: 1.769) | F1@0.50: 36.0% | F1@0.75: 12.2% | mIoU: 69.6% | MaxConf: 98.8% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 12.20%, mIoU: 69.57%)
Epoch [07/100] (9.1s) | Loss: 1.0586 (Cls: 0.082, GIoU: 0.357, DFL: 1.680)
Epoch [08/100] (8.7s) | Loss: 1.0056 (Cls: 0.075, GIoU: 0.340, DFL: 1.627) | F1@0.50: 35.7% | F1@0.75: 14.5% | mIoU: 71.8% | MaxConf: 99.6% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 14.49%, mIoU: 71.80%)
Epoch [09/100] (11.2s) | Loss: 1.1993 (Cls: 0.451, GIoU: 0.273, DFL: 1.331)
Epoch [10/100] (11.5s) | Loss: 0.8302 (Cls: 0.124, GIoU: 0.258, DFL: 1.242) | F1@0.50: 76.0% | F1@0.75: 38.7% | mIoU: 74.7% | MaxConf: 95.9% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 38.74%, mIoU: 74.66%)
Epoch [11/100] (11.4s) | Loss: 0.7611 (Cls: 0.103, GIoU: 0.239, DFL: 1.197)
Epoch [12/100] (11.4s) | Loss: 0.7274 (Cls: 0.091, GIoU: 0.231, DFL: 1.162) | F1@0.50: 83.1% | F1@0.75: 46.0% | mIoU: 75.3% | MaxConf: 96.8% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 46.01%, mIoU: 75.28%)
Epoch [13/100] (11.3s) | Loss: 0.6893 (Cls: 0.080, GIoU: 0.221, DFL: 1.116)
Epoch [14/100] (11.4s) | Loss: 0.6638 (Cls: 0.076, GIoU: 0.213, DFL: 1.091) | F1@0.50: 85.3% | F1@0.75: 57.1% | mIoU: 77.0% | MaxConf: 98.6% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 57.05%, mIoU: 77.01%)
Epoch [15/100] (11.5s) | Loss: 0.6303 (Cls: 0.070, GIoU: 0.203, DFL: 1.056)
Epoch [16/100] (11.3s) | Loss: 0.6162 (Cls: 0.067, GIoU: 0.199, DFL: 1.047) | F1@0.50: 79.2% | F1@0.75: 48.4% | mIoU: 76.9% | MaxConf: 99.2% | NegPrec: 0.0%
Epoch [17/100] (11.5s) | Loss: 0.5915 (Cls: 0.060, GIoU: 0.192, DFL: 1.028)
Epoch [18/100] (11.5s) | Loss: 0.5705 (Cls: 0.056, GIoU: 0.186, DFL: 1.004) | F1@0.50: 81.7% | F1@0.75: 53.2% | mIoU: 77.5% | MaxConf: 99.6% | NegPrec: 0.0%
Epoch [19/100] (11.3s) | Loss: 0.5600 (Cls: 0.057, GIoU: 0.181, DFL: 0.992)
Epoch [20/100] (11.3s) | Loss: 0.5334 (Cls: 0.051, GIoU: 0.174, DFL: 0.964) | F1@0.50: 89.0% | F1@0.75: 56.7% | mIoU: 77.6% | MaxConf: 99.8% | NegPrec: 0.0%
Epoch [21/100] (11.3s) | Loss: 0.5245 (Cls: 0.049, GIoU: 0.171, DFL: 0.948)
Epoch [22/100] (11.4s) | Loss: 0.5256 (Cls: 0.050, GIoU: 0.171, DFL: 0.943) | F1@0.50: 79.9% | F1@0.75: 52.1% | mIoU: 77.7% | MaxConf: 99.8% | NegPrec: 0.0%
Epoch [23/100] (11.4s) | Loss: 0.5050 (Cls: 0.046, GIoU: 0.165, DFL: 0.926)
Epoch [24/100] (11.4s) | Loss: 0.4878 (Cls: 0.043, GIoU: 0.160, DFL: 0.912) | F1@0.50: 89.1% | F1@0.75: 60.8% | mIoU: 78.7% | MaxConf: 99.8% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 60.77%, mIoU: 78.75%)
Epoch [25/100] (11.4s) | Loss: 0.4848 (Cls: 0.044, GIoU: 0.158, DFL: 0.900)
Epoch [26/100] (11.1s) | Loss: 0.4660 (Cls: 0.041, GIoU: 0.153, DFL: 0.881) | F1@0.50: 86.7% | F1@0.75: 57.5% | mIoU: 78.0% | MaxConf: 99.8% | NegPrec: 0.0%
Epoch [27/100] (11.2s) | Loss: 0.4568 (Cls: 0.040, GIoU: 0.149, DFL: 0.876)
Epoch [28/100] (11.2s) | Loss: 0.4584 (Cls: 0.040, GIoU: 0.150, DFL: 0.874) | F1@0.50: 89.0% | F1@0.75: 58.5% | mIoU: 78.1% | MaxConf: 99.9% | NegPrec: 0.0%
Epoch [29/100] (11.1s) | Loss: 0.4417 (Cls: 0.037, GIoU: 0.144, DFL: 0.864)
Epoch [30/100] (11.2s) | Loss: 0.4312 (Cls: 0.036, GIoU: 0.141, DFL: 0.853) | F1@0.50: 87.9% | F1@0.75: 59.8% | mIoU: 78.5% | MaxConf: 99.9% | NegPrec: 0.0%
Epoch [31/100] (11.4s) | Loss: 0.4277 (Cls: 0.036, GIoU: 0.140, DFL: 0.840)
Epoch [32/100] (11.2s) | Loss: 0.4175 (Cls: 0.035, GIoU: 0.136, DFL: 0.832) | F1@0.50: 86.9% | F1@0.75: 58.0% | mIoU: 78.8% | MaxConf: 99.9% | NegPrec: 0.0%
Epoch [33/100] (11.2s) | Loss: 0.4051 (Cls: 0.032, GIoU: 0.133, DFL: 0.824)
Epoch [34/100] (11.2s) | Loss: 0.4065 (Cls: 0.033, GIoU: 0.133, DFL: 0.819) | F1@0.50: 85.7% | F1@0.75: 56.2% | mIoU: 78.8% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [35/100] (11.2s) | Loss: 0.3975 (Cls: 0.033, GIoU: 0.130, DFL: 0.814)
Epoch [36/100] (11.6s) | Loss: 0.3899 (Cls: 0.031, GIoU: 0.127, DFL: 0.812) | F1@0.50: 90.0% | F1@0.75: 61.2% | mIoU: 78.8% | MaxConf: 100.0% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 61.20%, mIoU: 78.84%)
Epoch [37/100] (11.4s) | Loss: 0.3733 (Cls: 0.029, GIoU: 0.122, DFL: 0.798)
Epoch [38/100] (11.5s) | Loss: 0.3746 (Cls: 0.029, GIoU: 0.122, DFL: 0.794) | F1@0.50: 88.4% | F1@0.75: 57.0% | mIoU: 78.2% | MaxConf: 99.9% | NegPrec: 0.0%
Epoch [39/100] (11.3s) | Loss: 0.3772 (Cls: 0.029, GIoU: 0.123, DFL: 0.796)
Epoch [40/100] (11.4s) | Loss: 0.3674 (Cls: 0.028, GIoU: 0.120, DFL: 0.792) | F1@0.50: 88.3% | F1@0.75: 62.7% | mIoU: 79.5% | MaxConf: 99.9% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 62.67%, mIoU: 79.49%)
Epoch [41/100] (11.4s) | Loss: 0.3578 (Cls: 0.027, GIoU: 0.117, DFL: 0.780)
Epoch [42/100] (11.3s) | Loss: 0.3496 (Cls: 0.026, GIoU: 0.114, DFL: 0.772) | F1@0.50: 90.5% | F1@0.75: 61.7% | mIoU: 79.3% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [43/100] (11.2s) | Loss: 0.3475 (Cls: 0.025, GIoU: 0.113, DFL: 0.768)
Epoch [44/100] (11.3s) | Loss: 0.3410 (Cls: 0.025, GIoU: 0.111, DFL: 0.761) | F1@0.50: 90.4% | F1@0.75: 62.6% | mIoU: 79.5% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [45/100] (11.3s) | Loss: 0.3379 (Cls: 0.025, GIoU: 0.110, DFL: 0.757)
Epoch [46/100] (11.2s) | Loss: 0.3391 (Cls: 0.025, GIoU: 0.111, DFL: 0.762) | F1@0.50: 89.1% | F1@0.75: 62.6% | mIoU: 79.5% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [47/100] (11.3s) | Loss: 0.3236 (Cls: 0.024, GIoU: 0.105, DFL: 0.749)
Epoch [48/100] (11.3s) | Loss: 0.3227 (Cls: 0.023, GIoU: 0.105, DFL: 0.749) | F1@0.50: 91.1% | F1@0.75: 61.8% | mIoU: 79.3% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [49/100] (11.4s) | Loss: 0.3145 (Cls: 0.022, GIoU: 0.102, DFL: 0.744)
Epoch [50/100] (11.3s) | Loss: 0.3150 (Cls: 0.022, GIoU: 0.102, DFL: 0.747) | F1@0.50: 88.1% | F1@0.75: 62.6% | mIoU: 79.8% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [51/100] (11.4s) | Loss: 0.3086 (Cls: 0.021, GIoU: 0.100, DFL: 0.738)
Epoch [52/100] (11.2s) | Loss: 0.3081 (Cls: 0.023, GIoU: 0.099, DFL: 0.737) | F1@0.50: 89.9% | F1@0.75: 62.1% | mIoU: 79.6% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [53/100] (11.2s) | Loss: 0.3031 (Cls: 0.021, GIoU: 0.098, DFL: 0.731)
Epoch [54/100] (11.2s) | Loss: 0.2935 (Cls: 0.020, GIoU: 0.095, DFL: 0.727) | F1@0.50: 87.9% | F1@0.75: 60.7% | mIoU: 79.8% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [55/100] (11.5s) | Loss: 0.2919 (Cls: 0.020, GIoU: 0.094, DFL: 0.722)
Epoch [56/100] (11.3s) | Loss: 0.2868 (Cls: 0.019, GIoU: 0.093, DFL: 0.719) | F1@0.50: 90.6% | F1@0.75: 65.7% | mIoU: 79.7% | MaxConf: 100.0% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 65.66%, mIoU: 79.73%)
Epoch [57/100] (11.2s) | Loss: 0.2811 (Cls: 0.020, GIoU: 0.090, DFL: 0.716)
Epoch [58/100] (11.2s) | Loss: 0.2791 (Cls: 0.019, GIoU: 0.090, DFL: 0.714) | F1@0.50: 90.0% | F1@0.75: 65.2% | mIoU: 79.9% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [59/100] (11.1s) | Loss: 0.2782 (Cls: 0.019, GIoU: 0.090, DFL: 0.711)
Epoch [60/100] (11.3s) | Loss: 0.2727 (Cls: 0.019, GIoU: 0.087, DFL: 0.707) | F1@0.50: 90.4% | F1@0.75: 64.8% | mIoU: 79.8% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [61/100] (11.5s) | Loss: 0.2720 (Cls: 0.018, GIoU: 0.088, DFL: 0.706)
Epoch [62/100] (11.3s) | Loss: 0.2663 (Cls: 0.017, GIoU: 0.086, DFL: 0.703) | F1@0.50: 90.8% | F1@0.75: 64.6% | mIoU: 79.9% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [63/100] (11.4s) | Loss: 0.2592 (Cls: 0.017, GIoU: 0.083, DFL: 0.700)
Epoch [64/100] (11.3s) | Loss: 0.2585 (Cls: 0.017, GIoU: 0.083, DFL: 0.701) | F1@0.50: 91.4% | F1@0.75: 67.8% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 67.81%, mIoU: 80.12%)
Epoch [65/100] (11.4s) | Loss: 0.2566 (Cls: 0.017, GIoU: 0.082, DFL: 0.699)
Epoch [66/100] (11.3s) | Loss: 0.2552 (Cls: 0.017, GIoU: 0.081, DFL: 0.697) | F1@0.50: 90.7% | F1@0.75: 66.2% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [67/100] (11.1s) | Loss: 0.2492 (Cls: 0.016, GIoU: 0.079, DFL: 0.695)
Epoch [68/100] (11.1s) | Loss: 0.2439 (Cls: 0.016, GIoU: 0.077, DFL: 0.694) | F1@0.50: 90.8% | F1@0.75: 67.2% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [69/100] (11.6s) | Loss: 0.2419 (Cls: 0.015, GIoU: 0.077, DFL: 0.692)
Epoch [70/100] (11.1s) | Loss: 0.2403 (Cls: 0.015, GIoU: 0.076, DFL: 0.690) | F1@0.50: 90.8% | F1@0.75: 67.5% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [71/100] (11.3s) | Loss: 0.2402 (Cls: 0.015, GIoU: 0.076, DFL: 0.690)
Epoch [72/100] (11.2s) | Loss: 0.2346 (Cls: 0.015, GIoU: 0.074, DFL: 0.686) | F1@0.50: 90.5% | F1@0.75: 65.8% | mIoU: 79.9% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [73/100] (11.3s) | Loss: 0.2340 (Cls: 0.015, GIoU: 0.074, DFL: 0.686)
Epoch [74/100] (11.1s) | Loss: 0.2301 (Cls: 0.014, GIoU: 0.073, DFL: 0.683) | F1@0.50: 90.0% | F1@0.75: 64.2% | mIoU: 80.2% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [75/100] (11.4s) | Loss: 0.2276 (Cls: 0.014, GIoU: 0.072, DFL: 0.681)
Epoch [76/100] (11.3s) | Loss: 0.2286 (Cls: 0.014, GIoU: 0.072, DFL: 0.682) | F1@0.50: 90.9% | F1@0.75: 65.5% | mIoU: 80.2% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [77/100] (11.5s) | Loss: 0.2270 (Cls: 0.014, GIoU: 0.071, DFL: 0.680)
Epoch [78/100] (11.1s) | Loss: 0.2234 (Cls: 0.014, GIoU: 0.070, DFL: 0.679) | F1@0.50: 90.5% | F1@0.75: 65.3% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [79/100] (11.3s) | Loss: 0.2208 (Cls: 0.014, GIoU: 0.069, DFL: 0.676)
Epoch [80/100] (11.2s) | Loss: 0.2187 (Cls: 0.013, GIoU: 0.069, DFL: 0.675) | F1@0.50: 90.8% | F1@0.75: 65.5% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [81/100] (11.3s) | Loss: 0.2188 (Cls: 0.013, GIoU: 0.069, DFL: 0.674)
Epoch [82/100] (11.3s) | Loss: 0.2171 (Cls: 0.014, GIoU: 0.068, DFL: 0.675) | F1@0.50: 91.6% | F1@0.75: 66.1% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [83/100] (11.4s) | Loss: 0.2139 (Cls: 0.013, GIoU: 0.067, DFL: 0.673)
Epoch [84/100] (11.2s) | Loss: 0.2109 (Cls: 0.013, GIoU: 0.066, DFL: 0.671) | F1@0.50: 91.3% | F1@0.75: 64.8% | mIoU: 80.0% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [85/100] (11.2s) | Loss: 0.2151 (Cls: 0.013, GIoU: 0.067, DFL: 0.673)
Epoch [86/100] (11.4s) | Loss: 0.2086 (Cls: 0.013, GIoU: 0.065, DFL: 0.668) | F1@0.50: 91.4% | F1@0.75: 66.3% | mIoU: 80.2% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [87/100] (11.3s) | Loss: 0.2086 (Cls: 0.013, GIoU: 0.065, DFL: 0.670)
Epoch [88/100] (11.3s) | Loss: 0.2100 (Cls: 0.012, GIoU: 0.066, DFL: 0.669) | F1@0.50: 90.6% | F1@0.75: 66.8% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [89/100] (11.2s) | Loss: 0.2105 (Cls: 0.013, GIoU: 0.066, DFL: 0.670)
Epoch [90/100] (11.2s) | Loss: 0.2061 (Cls: 0.012, GIoU: 0.064, DFL: 0.668) | F1@0.50: 90.8% | F1@0.75: 65.0% | mIoU: 79.9% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [91/100] (11.4s) | Loss: 0.2070 (Cls: 0.012, GIoU: 0.065, DFL: 0.669)
Epoch [92/100] (11.3s) | Loss: 0.2066 (Cls: 0.013, GIoU: 0.064, DFL: 0.668) | F1@0.50: 90.0% | F1@0.75: 64.9% | mIoU: 80.2% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [93/100] (11.2s) | Loss: 0.2054 (Cls: 0.012, GIoU: 0.064, DFL: 0.669)
Epoch [94/100] (11.2s) | Loss: 0.2018 (Cls: 0.012, GIoU: 0.063, DFL: 0.665) | F1@0.50: 91.1% | F1@0.75: 66.8% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [95/100] (11.3s) | Loss: 0.1999 (Cls: 0.012, GIoU: 0.062, DFL: 0.666)
Epoch [96/100] (11.2s) | Loss: 0.2029 (Cls: 0.012, GIoU: 0.063, DFL: 0.666) | F1@0.50: 90.0% | F1@0.75: 66.9% | mIoU: 80.2% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [97/100] (11.3s) | Loss: 0.2032 (Cls: 0.012, GIoU: 0.063, DFL: 0.666)
Epoch [98/100] (11.3s) | Loss: 0.1991 (Cls: 0.012, GIoU: 0.062, DFL: 0.665) | F1@0.50: 90.6% | F1@0.75: 66.0% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%
Epoch [99/100] (11.3s) | Loss: 0.2023 (Cls: 0.012, GIoU: 0.063, DFL: 0.666)
Epoch [100/100] (11.3s) | Loss: 0.2022 (Cls: 0.012, GIoU: 0.063, DFL: 0.668) | F1@0.50: 90.4% | F1@0.75: 66.6% | mIoU: 80.1% | MaxConf: 100.0% | NegPrec: 0.0%

[OK] NanoDet-Plus training finished.
Best Checkpoint: checkpoints/nanodet_stage_a_improved2/best.pt (Best F1@0.75: 67.81%)
