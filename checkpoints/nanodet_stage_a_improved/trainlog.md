======================================================================
  EXPERIMENT: NanoDet-Plus Stage A Foot Detector (Hardware-Optimized)
  Dataset:         /content/Shoes_VTO/data/shuffled_v4_1_v4_2_13kp (1286 train, 160 val)
  Backbone:        ShuffleNetV2 1.0x (Pretrained: True, Act: RELU)
  Neck:            REP_PAN | Strides: [8, 16, 32] | BBox Loss: CIOU
  Assigner:        TAL | Neg Penalty: 1.5x | reg_max: 16
  Device:          cuda (AMP: True)
  Epochs:          60 | Batch: 32 | LR: 0.001000 | Classes: 1
  Resolution:      256x256
======================================================================
Downloading: "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth" to /root/.cache/torch/hub/checkpoints/shufflenetv2_x1-5666bf0f80.pth
100% 8.79M/8.79M [00:00<00:00, 198MB/s]
Epoch [01/60] (34.0s) | Loss: 2.6962 (Cls: 0.905, GIoU: 0.582, DFL: 2.511)
Epoch [02/60] (8.3s) | Loss: 1.8410 (Cls: 0.416, GIoU: 0.459, DFL: 2.028) | F1@0.50: 29.8% | F1@0.75: 4.2% | mIoU: 63.2% | MaxConf: 93.4% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 4.20%, mIoU: 63.19%)
Epoch [03/60] (8.1s) | Loss: 1.4683 (Cls: 0.267, GIoU: 0.381, DFL: 1.761)
Epoch [04/60] (8.2s) | Loss: 1.2640 (Cls: 0.252, GIoU: 0.313, DFL: 1.547) | F1@0.50: 48.1% | F1@0.75: 13.6% | mIoU: 68.8% | MaxConf: 84.5% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 13.62%, mIoU: 68.82%)
Epoch [05/60] (8.3s) | Loss: 1.1513 (Cls: 0.241, GIoU: 0.279, DFL: 1.410)
Epoch [06/60] (8.2s) | Loss: 1.0709 (Cls: 0.236, GIoU: 0.253, DFL: 1.318) | F1@0.50: 60.1% | F1@0.75: 24.7% | mIoU: 71.4% | MaxConf: 82.2% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 24.69%, mIoU: 71.39%)
Epoch [07/60] (8.2s) | Loss: 1.0132 (Cls: 0.230, GIoU: 0.236, DFL: 1.249)
Epoch [08/60] (8.2s) | Loss: 0.9671 (Cls: 0.224, GIoU: 0.222, DFL: 1.192) | F1@0.50: 67.5% | F1@0.75: 31.2% | mIoU: 72.6% | MaxConf: 86.1% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 31.22%, mIoU: 72.57%)
Epoch [09/60] (8.2s) | Loss: 0.9257 (Cls: 0.220, GIoU: 0.210, DFL: 1.144)
Epoch [10/60] (8.3s) | Loss: 0.8981 (Cls: 0.216, GIoU: 0.203, DFL: 1.107) | F1@0.50: 69.8% | F1@0.75: 34.2% | mIoU: 73.8% | MaxConf: 85.9% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 34.22%, mIoU: 73.78%)
Epoch [11/60] (8.3s) | Loss: 0.8601 (Cls: 0.208, GIoU: 0.193, DFL: 1.068)
Epoch [12/60] (8.2s) | Loss: 0.8324 (Cls: 0.203, GIoU: 0.185, DFL: 1.039) | F1@0.50: 73.0% | F1@0.75: 35.0% | mIoU: 73.8% | MaxConf: 87.7% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 34.97%, mIoU: 73.85%)
Epoch [13/60] (8.3s) | Loss: 0.8046 (Cls: 0.196, GIoU: 0.178, DFL: 1.008)
Epoch [14/60] (8.2s) | Loss: 0.7730 (Cls: 0.191, GIoU: 0.169, DFL: 0.978) | F1@0.50: 77.0% | F1@0.75: 42.1% | mIoU: 75.3% | MaxConf: 84.7% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 42.07%, mIoU: 75.32%)
Epoch [15/60] (8.3s) | Loss: 0.7583 (Cls: 0.188, GIoU: 0.165, DFL: 0.961)
Epoch [16/60] (8.3s) | Loss: 0.7429 (Cls: 0.183, GIoU: 0.162, DFL: 0.945) | F1@0.50: 78.3% | F1@0.75: 43.1% | mIoU: 75.3% | MaxConf: 93.5% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 43.12%, mIoU: 75.27%)
Epoch [17/60] (8.3s) | Loss: 0.7184 (Cls: 0.176, GIoU: 0.156, DFL: 0.923)
Epoch [18/60] (8.2s) | Loss: 0.7012 (Cls: 0.172, GIoU: 0.152, DFL: 0.903) | F1@0.50: 75.1% | F1@0.75: 45.8% | mIoU: 75.7% | MaxConf: 93.4% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 45.76%, mIoU: 75.73%)
Epoch [19/60] (8.2s) | Loss: 0.6949 (Cls: 0.172, GIoU: 0.149, DFL: 0.896)
Epoch [20/60] (8.2s) | Loss: 0.6661 (Cls: 0.163, GIoU: 0.143, DFL: 0.872) | F1@0.50: 78.5% | F1@0.75: 45.5% | mIoU: 75.8% | MaxConf: 87.8% | NegPrec: 0.0%
Epoch [21/60] (8.1s) | Loss: 0.6609 (Cls: 0.161, GIoU: 0.142, DFL: 0.865)
Epoch [22/60] (8.1s) | Loss: 0.6529 (Cls: 0.159, GIoU: 0.139, DFL: 0.858) | F1@0.50: 81.3% | F1@0.75: 46.4% | mIoU: 76.5% | MaxConf: 91.6% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 46.42%, mIoU: 76.46%)
Epoch [23/60] (8.3s) | Loss: 0.6366 (Cls: 0.153, GIoU: 0.136, DFL: 0.844)
Epoch [24/60] (8.2s) | Loss: 0.6259 (Cls: 0.147, GIoU: 0.135, DFL: 0.834) | F1@0.50: 80.9% | F1@0.75: 50.3% | mIoU: 76.8% | MaxConf: 93.3% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 50.31%, mIoU: 76.82%)
Epoch [25/60] (8.1s) | Loss: 0.6209 (Cls: 0.148, GIoU: 0.133, DFL: 0.830)
Epoch [26/60] (8.2s) | Loss: 0.6065 (Cls: 0.144, GIoU: 0.129, DFL: 0.816) | F1@0.50: 81.3% | F1@0.75: 49.8% | mIoU: 76.7% | MaxConf: 94.6% | NegPrec: 0.0%
Epoch [27/60] (8.3s) | Loss: 0.6026 (Cls: 0.141, GIoU: 0.129, DFL: 0.814)
Epoch [28/60] (8.3s) | Loss: 0.5964 (Cls: 0.139, GIoU: 0.128, DFL: 0.809) | F1@0.50: 82.1% | F1@0.75: 49.1% | mIoU: 76.7% | MaxConf: 92.0% | NegPrec: 0.0%
Epoch [29/60] (8.2s) | Loss: 0.5779 (Cls: 0.133, GIoU: 0.123, DFL: 0.793)
Epoch [30/60] (8.2s) | Loss: 0.5666 (Cls: 0.130, GIoU: 0.120, DFL: 0.784) | F1@0.50: 80.5% | F1@0.75: 50.9% | mIoU: 77.5% | MaxConf: 93.6% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 50.94%, mIoU: 77.52%)
Epoch [31/60] (8.2s) | Loss: 0.5683 (Cls: 0.130, GIoU: 0.121, DFL: 0.785)
Epoch [32/60] (8.1s) | Loss: 0.5519 (Cls: 0.125, GIoU: 0.117, DFL: 0.774) | F1@0.50: 83.4% | F1@0.75: 52.8% | mIoU: 77.3% | MaxConf: 90.8% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 52.82%, mIoU: 77.34%)
Epoch [33/60] (8.4s) | Loss: 0.5523 (Cls: 0.124, GIoU: 0.117, DFL: 0.775)
Epoch [34/60] (8.1s) | Loss: 0.5368 (Cls: 0.120, GIoU: 0.114, DFL: 0.760) | F1@0.50: 81.0% | F1@0.75: 50.5% | mIoU: 77.3% | MaxConf: 94.6% | NegPrec: 0.0%
Epoch [35/60] (8.1s) | Loss: 0.5301 (Cls: 0.116, GIoU: 0.113, DFL: 0.757)
Epoch [36/60] (8.3s) | Loss: 0.5278 (Cls: 0.115, GIoU: 0.112, DFL: 0.754) | F1@0.50: 83.5% | F1@0.75: 51.0% | mIoU: 77.0% | MaxConf: 95.9% | NegPrec: 0.0%
Epoch [37/60] (8.3s) | Loss: 0.5185 (Cls: 0.113, GIoU: 0.109, DFL: 0.748)
Epoch [38/60] (8.3s) | Loss: 0.5203 (Cls: 0.112, GIoU: 0.111, DFL: 0.749) | F1@0.50: 81.3% | F1@0.75: 51.1% | mIoU: 77.1% | MaxConf: 96.2% | NegPrec: 0.0%
Epoch [39/60] (8.1s) | Loss: 0.5183 (Cls: 0.113, GIoU: 0.109, DFL: 0.746)
Epoch [40/60] (8.1s) | Loss: 0.5093 (Cls: 0.108, GIoU: 0.108, DFL: 0.741) | F1@0.50: 82.9% | F1@0.75: 53.3% | mIoU: 77.2% | MaxConf: 95.4% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 53.33%, mIoU: 77.20%)
Epoch [41/60] (8.3s) | Loss: 0.5047 (Cls: 0.106, GIoU: 0.107, DFL: 0.738)
Epoch [42/60] (8.2s) | Loss: 0.4945 (Cls: 0.103, GIoU: 0.104, DFL: 0.731) | F1@0.50: 82.3% | F1@0.75: 52.9% | mIoU: 77.5% | MaxConf: 96.4% | NegPrec: 0.0%
Epoch [43/60] (8.3s) | Loss: 0.4965 (Cls: 0.103, GIoU: 0.105, DFL: 0.734)
Epoch [44/60] (8.2s) | Loss: 0.4936 (Cls: 0.102, GIoU: 0.104, DFL: 0.732) | F1@0.50: 81.9% | F1@0.75: 52.6% | mIoU: 77.7% | MaxConf: 97.0% | NegPrec: 0.0%
Epoch [45/60] (8.1s) | Loss: 0.4877 (Cls: 0.101, GIoU: 0.103, DFL: 0.726)
Epoch [46/60] (8.3s) | Loss: 0.4860 (Cls: 0.100, GIoU: 0.102, DFL: 0.723) | F1@0.50: 81.9% | F1@0.75: 52.8% | mIoU: 77.7% | MaxConf: 97.4% | NegPrec: 0.0%
Epoch [47/60] (8.2s) | Loss: 0.4855 (Cls: 0.100, GIoU: 0.102, DFL: 0.723)
Epoch [48/60] (8.2s) | Loss: 0.4852 (Cls: 0.099, GIoU: 0.103, DFL: 0.723) | F1@0.50: 82.7% | F1@0.75: 53.5% | mIoU: 77.6% | MaxConf: 97.2% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 53.48%, mIoU: 77.57%)
Epoch [49/60] (8.4s) | Loss: 0.4723 (Cls: 0.096, GIoU: 0.099, DFL: 0.716)
Epoch [50/60] (8.4s) | Loss: 0.4716 (Cls: 0.096, GIoU: 0.099, DFL: 0.713) | F1@0.50: 82.9% | F1@0.75: 55.1% | mIoU: 78.1% | MaxConf: 97.6% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 55.14%, mIoU: 78.09%)
Epoch [51/60] (8.0s) | Loss: 0.4749 (Cls: 0.097, GIoU: 0.099, DFL: 0.718)
Epoch [52/60] (8.2s) | Loss: 0.4766 (Cls: 0.097, GIoU: 0.100, DFL: 0.718) | F1@0.50: 82.7% | F1@0.75: 53.4% | mIoU: 77.9% | MaxConf: 97.7% | NegPrec: 0.0%
Epoch [53/60] (8.3s) | Loss: 0.4750 (Cls: 0.097, GIoU: 0.099, DFL: 0.717)
Epoch [54/60] (8.3s) | Loss: 0.4684 (Cls: 0.095, GIoU: 0.097, DFL: 0.713) | F1@0.50: 82.4% | F1@0.75: 53.8% | mIoU: 77.9% | MaxConf: 97.2% | NegPrec: 0.0%
Epoch [55/60] (8.3s) | Loss: 0.4677 (Cls: 0.094, GIoU: 0.098, DFL: 0.712)
Epoch [56/60] (8.2s) | Loss: 0.4666 (Cls: 0.094, GIoU: 0.097, DFL: 0.711) | F1@0.50: 83.6% | F1@0.75: 55.6% | mIoU: 78.0% | MaxConf: 97.5% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 55.61%, mIoU: 77.97%)
Epoch [57/60] (8.2s) | Loss: 0.4703 (Cls: 0.094, GIoU: 0.098, DFL: 0.717)
Epoch [58/60] (8.2s) | Loss: 0.4654 (Cls: 0.094, GIoU: 0.097, DFL: 0.711) | F1@0.50: 82.8% | F1@0.75: 53.4% | mIoU: 77.9% | MaxConf: 97.3% | NegPrec: 0.0%
Epoch [59/60] (8.4s) | Loss: 0.4633 (Cls: 0.092, GIoU: 0.097, DFL: 0.711)
Epoch [60/60] (8.3s) | Loss: 0.4674 (Cls: 0.094, GIoU: 0.098, DFL: 0.713) | F1@0.50: 83.1% | F1@0.75: 53.9% | mIoU: 77.7% | MaxConf: 97.7% | NegPrec: 0.0%

[OK] NanoDet-Plus training finished.
Best Checkpoint: checkpoints/nanodet_stage_a_improved/best.pt (Best F1@0.75: 55.61%)
