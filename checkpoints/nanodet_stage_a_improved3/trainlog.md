======================================================================
  EXPERIMENT: NanoDet-Plus Stage A Foot Detector (Initial Training)
  Dataset:         /content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3 (6620 train, 827 val)
  Backbone:        ShuffleNetV2 1.0x (Pretrained: True, Act: RELU)
  Neck:            REP_PAN | Strides: [8, 16, 32] | BBox Loss: CIOU
  Assigner:        TAL (Warmup: 8 ep) | Loss Weights (Box: 3.0, DFL: 0.15)
  Device:          cuda (AMP: True)
  Epochs:          60 | Batch: 24 | LR: 0.000600 | Classes: 1
  Resolution:      256x256
======================================================================
Epoch [01/60] (47.0s) | Loss: 3.6574 (Cls: 0.917, GIoU: 0.757, DFL: 3.141)
Epoch [02/60] (32.0s) | Loss: 2.3190 (Cls: 0.168, GIoU: 0.601, DFL: 2.329) | F1@0.50: 15.8% | F1@0.75: 2.0% (c35: 3.5%) | mIoU: 62.9% | MaxConf: 98.6% | NegPrec: 0.0%
  [*] New best detector checkpoint saved! (F1@0.75: 2.01%, c35: 3.50%, mIoU: 62.87%)
Epoch [03/60] (32.7s) | Loss: 1.7731 (Cls: 0.125, GIoU: 0.452, DFL: 1.949)
Epoch [04/60] (32.5s) | Loss: 1.5392 (Cls: 0.111, GIoU: 0.388, DFL: 1.760) | F1@0.50: 28.2% | F1@0.75: 11.3% (c35: 18.8%) | mIoU: 70.5% | MaxConf: 97.9% | NegPrec: 10.7%
  [*] New best detector checkpoint saved! (F1@0.75: 11.30%, c35: 18.83%, mIoU: 70.47%)
Epoch [05/60] (32.3s) | Loss: 1.4045 (Cls: 0.102, GIoU: 0.352, DFL: 1.639)
Epoch [06/60] (32.1s) | Loss: 1.3227 (Cls: 0.095, GIoU: 0.331, DFL: 1.556) | F1@0.50: 33.4% | F1@0.75: 16.2% (c35: 24.2%) | mIoU: 72.5% | MaxConf: 98.2% | NegPrec: 17.9%
  [*] New best detector checkpoint saved! (F1@0.75: 16.18%, c35: 24.21%, mIoU: 72.45%)
Epoch [07/60] (32.5s) | Loss: 1.2539 (Cls: 0.090, GIoU: 0.313, DFL: 1.492)
Epoch [08/60] (32.1s) | Loss: 1.2096 (Cls: 0.086, GIoU: 0.302, DFL: 1.449) | F1@0.50: 37.6% | F1@0.75: 18.4% (c35: 25.9%) | mIoU: 72.9% | MaxConf: 99.2% | NegPrec: 22.6%
  [*] New best detector checkpoint saved! (F1@0.75: 18.40%, c35: 25.90%, mIoU: 72.90%)
Epoch [09/60] (42.6s) | Loss: 1.1212 (Cls: 0.214, GIoU: 0.247, DFL: 1.096)
Epoch [10/60] (42.9s) | Loss: 0.9775 (Cls: 0.116, GIoU: 0.234, DFL: 1.059) | F1@0.50: 80.4% | F1@0.75: 51.4% (c35: 57.9%) | mIoU: 78.1% | MaxConf: 97.0% | NegPrec: 52.4%
  [*] New best detector checkpoint saved! (F1@0.75: 51.38%, c35: 57.93%, mIoU: 78.06%)
Epoch [11/60] (43.3s) | Loss: 0.9459 (Cls: 0.105, GIoU: 0.228, DFL: 1.042)
Epoch [12/60] (42.5s) | Loss: 0.9132 (Cls: 0.097, GIoU: 0.221, DFL: 1.021) | F1@0.50: 78.6% | F1@0.75: 53.5% (c35: 61.5%) | mIoU: 79.1% | MaxConf: 98.6% | NegPrec: 54.8%
  [*] New best detector checkpoint saved! (F1@0.75: 53.51%, c35: 61.49%, mIoU: 79.11%)
Epoch [13/60] (43.0s) | Loss: 0.8939 (Cls: 0.092, GIoU: 0.217, DFL: 1.019)
Epoch [14/60] (42.8s) | Loss: 0.8804 (Cls: 0.089, GIoU: 0.213, DFL: 1.014) | F1@0.50: 80.2% | F1@0.75: 57.2% (c35: 64.5%) | mIoU: 79.6% | MaxConf: 99.0% | NegPrec: 58.3%
  [*] New best detector checkpoint saved! (F1@0.75: 57.22%, c35: 64.49%, mIoU: 79.62%)
Epoch [15/60] (43.2s) | Loss: 0.8602 (Cls: 0.083, GIoU: 0.209, DFL: 1.006)
Epoch [16/60] (43.0s) | Loss: 0.8475 (Cls: 0.080, GIoU: 0.206, DFL: 0.991) | F1@0.50: 83.7% | F1@0.75: 59.3% (c35: 65.5%) | mIoU: 79.8% | MaxConf: 99.2% | NegPrec: 60.7%
  [*] New best detector checkpoint saved! (F1@0.75: 59.33%, c35: 65.45%, mIoU: 79.84%)
Epoch [17/60] (42.6s) | Loss: 0.8406 (Cls: 0.078, GIoU: 0.205, DFL: 0.989)
Epoch [18/60] (43.1s) | Loss: 0.8272 (Cls: 0.077, GIoU: 0.201, DFL: 0.977) | F1@0.50: 81.9% | F1@0.75: 59.8% (c35: 67.2%) | mIoU: 80.1% | MaxConf: 99.0% | NegPrec: 60.7%
  [*] New best detector checkpoint saved! (F1@0.75: 59.78%, c35: 67.24%, mIoU: 80.08%)
Epoch [19/60] (43.1s) | Loss: 0.8166 (Cls: 0.073, GIoU: 0.199, DFL: 0.971)
Epoch [20/60] (43.2s) | Loss: 0.8082 (Cls: 0.071, GIoU: 0.198, DFL: 0.962) | F1@0.50: 84.3% | F1@0.75: 63.7% (c35: 70.3%) | mIoU: 80.7% | MaxConf: 99.1% | NegPrec: 66.7%
  [*] New best detector checkpoint saved! (F1@0.75: 63.73%, c35: 70.32%, mIoU: 80.68%)
Epoch [21/60] (43.2s) | Loss: 0.8029 (Cls: 0.070, GIoU: 0.196, DFL: 0.963)
Epoch [22/60] (43.5s) | Loss: 0.7918 (Cls: 0.066, GIoU: 0.194, DFL: 0.957) | F1@0.50: 84.0% | F1@0.75: 64.3% (c35: 70.7%) | mIoU: 80.8% | MaxConf: 99.2% | NegPrec: 65.5%
  [*] New best detector checkpoint saved! (F1@0.75: 64.35%, c35: 70.72%, mIoU: 80.77%)
Epoch [23/60] (43.0s) | Loss: 0.7815 (Cls: 0.064, GIoU: 0.192, DFL: 0.944)
Epoch [24/60] (43.2s) | Loss: 0.7757 (Cls: 0.063, GIoU: 0.191, DFL: 0.941) | F1@0.50: 85.2% | F1@0.75: 64.8% (c35: 70.8%) | mIoU: 80.8% | MaxConf: 99.0% | NegPrec: 71.4%
  [*] New best detector checkpoint saved! (F1@0.75: 64.85%, c35: 70.76%, mIoU: 80.81%)
Epoch [25/60] (43.3s) | Loss: 0.7725 (Cls: 0.062, GIoU: 0.190, DFL: 0.945)
Epoch [26/60] (43.3s) | Loss: 0.7684 (Cls: 0.061, GIoU: 0.189, DFL: 0.943) | F1@0.50: 85.1% | F1@0.75: 63.4% (c35: 69.1%) | mIoU: 80.9% | MaxConf: 99.2% | NegPrec: 71.4%
Epoch [27/60] (42.7s) | Loss: 0.7596 (Cls: 0.060, GIoU: 0.187, DFL: 0.932)
Epoch [28/60] (43.0s) | Loss: 0.7555 (Cls: 0.060, GIoU: 0.185, DFL: 0.930) | F1@0.50: 85.1% | F1@0.75: 65.2% (c35: 71.2%) | mIoU: 81.2% | MaxConf: 99.4% | NegPrec: 71.4%
  [*] New best detector checkpoint saved! (F1@0.75: 65.17%, c35: 71.23%, mIoU: 81.22%)
Epoch [29/60] (43.0s) | Loss: 0.7460 (Cls: 0.057, GIoU: 0.183, DFL: 0.925)
Epoch [30/60] (42.2s) | Loss: 0.7391 (Cls: 0.055, GIoU: 0.182, DFL: 0.921) | F1@0.50: 85.7% | F1@0.75: 66.5% (c35: 71.9%) | mIoU: 81.3% | MaxConf: 99.5% | NegPrec: 69.0%
  [*] New best detector checkpoint saved! (F1@0.75: 66.53%, c35: 71.93%, mIoU: 81.30%)
Epoch [31/60] (43.4s) | Loss: 0.7357 (Cls: 0.056, GIoU: 0.181, DFL: 0.913)
Epoch [32/60] (43.4s) | Loss: 0.7322 (Cls: 0.055, GIoU: 0.180, DFL: 0.912) | F1@0.50: 85.7% | F1@0.75: 67.0% (c35: 72.8%) | mIoU: 81.4% | MaxConf: 99.5% | NegPrec: 71.4%
  [*] New best detector checkpoint saved! (F1@0.75: 67.04%, c35: 72.84%, mIoU: 81.43%)
Epoch [33/60] (43.6s) | Loss: 0.7245 (Cls: 0.053, GIoU: 0.179, DFL: 0.905)
Epoch [34/60] (42.7s) | Loss: 0.7225 (Cls: 0.054, GIoU: 0.178, DFL: 0.906) | F1@0.50: 85.8% | F1@0.75: 66.5% (c35: 72.0%) | mIoU: 81.4% | MaxConf: 99.4% | NegPrec: 75.0%
Epoch [35/60] (43.6s) | Loss: 0.7158 (Cls: 0.052, GIoU: 0.176, DFL: 0.903)
Epoch [36/60] (43.1s) | Loss: 0.7154 (Cls: 0.053, GIoU: 0.176, DFL: 0.902) | F1@0.50: 85.8% | F1@0.75: 66.2% (c35: 71.9%) | mIoU: 81.4% | MaxConf: 99.5% | NegPrec: 73.8%
Epoch [37/60] (43.3s) | Loss: 0.7082 (Cls: 0.051, GIoU: 0.174, DFL: 0.892)
Epoch [38/60] (43.7s) | Loss: 0.7044 (Cls: 0.050, GIoU: 0.174, DFL: 0.891) | F1@0.50: 85.8% | F1@0.75: 66.8% (c35: 72.1%) | mIoU: 81.6% | MaxConf: 99.5% | NegPrec: 75.0%
Epoch [39/60] (43.0s) | Loss: 0.7002 (Cls: 0.050, GIoU: 0.173, DFL: 0.887)
Epoch [40/60] (42.8s) | Loss: 0.6955 (Cls: 0.048, GIoU: 0.172, DFL: 0.885) | F1@0.50: 86.4% | F1@0.75: 67.2% (c35: 72.3%) | mIoU: 81.6% | MaxConf: 99.5% | NegPrec: 77.4%
  [*] New best detector checkpoint saved! (F1@0.75: 67.23%, c35: 72.31%, mIoU: 81.62%)
Epoch [41/60] (42.9s) | Loss: 0.6961 (Cls: 0.048, GIoU: 0.172, DFL: 0.888)
Epoch [42/60] (42.9s) | Loss: 0.6910 (Cls: 0.047, GIoU: 0.170, DFL: 0.887) | F1@0.50: 86.4% | F1@0.75: 67.3% (c35: 72.4%) | mIoU: 81.7% | MaxConf: 99.5% | NegPrec: 77.4%
  [*] New best detector checkpoint saved! (F1@0.75: 67.34%, c35: 72.35%, mIoU: 81.70%)
Epoch [43/60] (43.2s) | Loss: 0.6907 (Cls: 0.047, GIoU: 0.170, DFL: 0.884)
Epoch [44/60] (43.1s) | Loss: 0.6897 (Cls: 0.047, GIoU: 0.170, DFL: 0.882) | F1@0.50: 85.9% | F1@0.75: 66.9% (c35: 72.1%) | mIoU: 81.8% | MaxConf: 99.6% | NegPrec: 77.4%
Epoch [45/60] (42.9s) | Loss: 0.6886 (Cls: 0.048, GIoU: 0.169, DFL: 0.882)
Epoch [46/60] (43.0s) | Loss: 0.6837 (Cls: 0.047, GIoU: 0.168, DFL: 0.877) | F1@0.50: 85.9% | F1@0.75: 67.6% (c35: 72.7%) | mIoU: 81.9% | MaxConf: 99.5% | NegPrec: 78.6%
  [*] New best detector checkpoint saved! (F1@0.75: 67.60%, c35: 72.73%, mIoU: 81.87%)
Epoch [47/60] (42.8s) | Loss: 0.6833 (Cls: 0.047, GIoU: 0.168, DFL: 0.875)
Epoch [48/60] (42.8s) | Loss: 0.6813 (Cls: 0.046, GIoU: 0.168, DFL: 0.874) | F1@0.50: 85.9% | F1@0.75: 67.5% (c35: 72.7%) | mIoU: 81.8% | MaxConf: 99.5% | NegPrec: 77.4%
Epoch [49/60] (43.4s) | Loss: 0.6739 (Cls: 0.045, GIoU: 0.166, DFL: 0.870)
Epoch [50/60] (42.9s) | Loss: 0.6775 (Cls: 0.046, GIoU: 0.167, DFL: 0.869) | F1@0.50: 86.0% | F1@0.75: 67.6% (c35: 72.8%) | mIoU: 81.9% | MaxConf: 99.6% | NegPrec: 78.6%
  [*] New best detector checkpoint saved! (F1@0.75: 67.60%, c35: 72.81%, mIoU: 81.89%)
Epoch [51/60] (42.5s) | Loss: 0.6715 (Cls: 0.046, GIoU: 0.165, DFL: 0.866)
Epoch [52/60] (43.1s) | Loss: 0.6720 (Cls: 0.045, GIoU: 0.166, DFL: 0.869) | F1@0.50: 86.2% | F1@0.75: 67.5% (c35: 72.6%) | mIoU: 81.8% | MaxConf: 99.6% | NegPrec: 78.6%
Epoch [53/60] (43.5s) | Loss: 0.6723 (Cls: 0.044, GIoU: 0.166, DFL: 0.867)
Epoch [54/60] (43.9s) | Loss: 0.6722 (Cls: 0.045, GIoU: 0.166, DFL: 0.870) | F1@0.50: 86.2% | F1@0.75: 67.4% (c35: 72.7%) | mIoU: 81.9% | MaxConf: 99.6% | NegPrec: 78.6%
Epoch [55/60] (42.7s) | Loss: 0.6692 (Cls: 0.045, GIoU: 0.165, DFL: 0.863)
Epoch [56/60] (42.8s) | Loss: 0.6669 (Cls: 0.045, GIoU: 0.164, DFL: 0.863) | F1@0.50: 86.2% | F1@0.75: 67.4% (c35: 72.5%) | mIoU: 81.8% | MaxConf: 99.6% | NegPrec: 78.6%
Epoch [57/60] (42.9s) | Loss: 0.6705 (Cls: 0.045, GIoU: 0.165, DFL: 0.867)
Epoch [58/60] (43.2s) | Loss: 0.6713 (Cls: 0.045, GIoU: 0.165, DFL: 0.870) | F1@0.50: 86.3% | F1@0.75: 67.7% (c35: 72.6%) | mIoU: 81.9% | MaxConf: 99.6% | NegPrec: 78.6%
  [*] New best detector checkpoint saved! (F1@0.75: 67.65%, c35: 72.63%, mIoU: 81.88%)
Epoch [59/60] (42.8s) | Loss: 0.6655 (Cls: 0.044, GIoU: 0.164, DFL: 0.863)
Epoch [60/60] (42.9s) | Loss: 0.6641 (Cls: 0.044, GIoU: 0.164, DFL: 0.861) | F1@0.50: 86.4% | F1@0.75: 67.8% (c35: 72.8%) | mIoU: 81.9% | MaxConf: 99.6% | NegPrec: 78.6%
  [*] New best detector checkpoint saved! (F1@0.75: 67.77%, c35: 72.80%, mIoU: 81.90%)

[OK] NanoDet-Plus training finished.
Best Checkpoint: /checkpoints/nanodet_stage_a_improved3/best.pt (Best F1@0.75: 67.77%)
