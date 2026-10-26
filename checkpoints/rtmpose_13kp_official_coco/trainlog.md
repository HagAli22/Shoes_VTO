===========================================================================
  EXPERIMENT 2: Qualcomm RTMPose-13KP Foot Pose (SimCC + GAU) Training
  Dataset:         /content/Shoes_VTO/data/shuffled_v4_13kp
  Backbone:        cspnext (Pretrained: True)
  Device:          cuda
  Epochs:          60 (Warmup: 4) | Batch: 32
  Backbone LR:     0.000100 | Head LR: 0.000500
  Resolution:      256x256 (SimCC Bins: 512x512, Dec Win: 12)
  Data Workers:    4 (Pin Memory: True)
===========================================================================
/usr/local/lib/python3.13/dist-packages/torch/utils/data/dataloader.py:424: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
[+] Successfully transferred all 206 tensors from official RTMPose-tiny COCO checkpoint!
/usr/local/lib/python3.13/dist-packages/torch/utils/data/dataloader.py:432: UserWarning: This DataLoader will create 4 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  self.check_worker_number_rationality()
Epoch [01/60] (50.6s) | Train: 5.7235 | Val: 5.2986 | Val PCK@0.05: 3.76% | PCK@0.10: 14.27% | Mean Err: 48.95px
  [*] New best checkpoint saved! (Mean Err: 48.95px, PCK@0.05: 3.76%)
Epoch [02/60] (41.7s) | Train: 4.9812 | Val: 4.3134 | Val PCK@0.05: 10.73% | PCK@0.10: 33.87% | Mean Err: 30.58px
  [*] New best checkpoint saved! (Mean Err: 30.58px, PCK@0.05: 10.73%)
Epoch [03/60] (41.8s) | Train: 4.3667 | Val: 3.8436 | Val PCK@0.05: 18.19% | PCK@0.10: 49.06% | Mean Err: 24.75px
  [*] New best checkpoint saved! (Mean Err: 24.75px, PCK@0.05: 18.19%)
Epoch [04/60] (41.9s) | Train: 3.9783 | Val: 3.5167 | Val PCK@0.05: 24.35% | PCK@0.10: 57.04% | Mean Err: 22.08px
  [*] New best checkpoint saved! (Mean Err: 22.08px, PCK@0.05: 24.35%)
Epoch [05/60] (41.0s) | Train: 3.6597 | Val: 3.2104 | Val PCK@0.05: 30.56% | PCK@0.10: 65.02% | Mean Err: 19.34px
  [*] New best checkpoint saved! (Mean Err: 19.34px, PCK@0.05: 30.56%)
Epoch [06/60] (40.8s) | Train: 3.4347 | Val: 3.0780 | Val PCK@0.05: 33.65% | PCK@0.10: 67.09% | Mean Err: 18.72px
  [*] New best checkpoint saved! (Mean Err: 18.72px, PCK@0.05: 33.65%)
Epoch [07/60] (41.3s) | Train: 3.2627 | Val: 2.9564 | Val PCK@0.05: 35.75% | PCK@0.10: 69.58% | Mean Err: 17.72px
  [*] New best checkpoint saved! (Mean Err: 17.72px, PCK@0.05: 35.75%)
Epoch [08/60] (41.2s) | Train: 3.1262 | Val: 2.8235 | Val PCK@0.05: 39.82% | PCK@0.10: 72.32% | Mean Err: 16.62px
  [*] New best checkpoint saved! (Mean Err: 16.62px, PCK@0.05: 39.82%)
Epoch [09/60] (41.1s) | Train: 3.0407 | Val: 2.7543 | Val PCK@0.05: 41.48% | PCK@0.10: 74.09% | Mean Err: 15.94px
  [*] New best checkpoint saved! (Mean Err: 15.94px, PCK@0.05: 41.48%)
Epoch [10/60] (41.4s) | Train: 2.9585 | Val: 2.6928 | Val PCK@0.05: 43.18% | PCK@0.10: 75.45% | Mean Err: 15.58px
  [*] New best checkpoint saved! (Mean Err: 15.58px, PCK@0.05: 43.18%)
Epoch [11/60] (41.1s) | Train: 2.8836 | Val: 2.6434 | Val PCK@0.05: 44.87% | PCK@0.10: 76.62% | Mean Err: 14.95px
  [*] New best checkpoint saved! (Mean Err: 14.95px, PCK@0.05: 44.87%)
Epoch [12/60] (40.4s) | Train: 2.8287 | Val: 2.5751 | Val PCK@0.05: 45.65% | PCK@0.10: 77.59% | Mean Err: 14.75px
  [*] New best checkpoint saved! (Mean Err: 14.75px, PCK@0.05: 45.65%)
Epoch [13/60] (40.7s) | Train: 2.7795 | Val: 2.5276 | Val PCK@0.05: 47.70% | PCK@0.10: 78.91% | Mean Err: 14.13px
  [*] New best checkpoint saved! (Mean Err: 14.13px, PCK@0.05: 47.70%)
Epoch [14/60] (40.5s) | Train: 2.7205 | Val: 2.4967 | Val PCK@0.05: 48.36% | PCK@0.10: 79.13% | Mean Err: 13.94px
  [*] New best checkpoint saved! (Mean Err: 13.94px, PCK@0.05: 48.36%)
Epoch [15/60] (40.0s) | Train: 2.6588 | Val: 2.4711 | Val PCK@0.05: 48.56% | PCK@0.10: 79.79% | Mean Err: 13.59px
  [*] New best checkpoint saved! (Mean Err: 13.59px, PCK@0.05: 48.56%)
Epoch [16/60] (40.2s) | Train: 2.6224 | Val: 2.4551 | Val PCK@0.05: 49.57% | PCK@0.10: 80.10% | Mean Err: 13.56px
  [*] New best checkpoint saved! (Mean Err: 13.56px, PCK@0.05: 49.57%)
Epoch [17/60] (40.3s) | Train: 2.5819 | Val: 2.4065 | Val PCK@0.05: 50.40% | PCK@0.10: 80.72% | Mean Err: 13.39px
  [*] New best checkpoint saved! (Mean Err: 13.39px, PCK@0.05: 50.40%)
Epoch [18/60] (40.3s) | Train: 2.5627 | Val: 2.4014 | Val PCK@0.05: 51.17% | PCK@0.10: 81.29% | Mean Err: 13.34px
  [*] New best checkpoint saved! (Mean Err: 13.34px, PCK@0.05: 51.17%)
Epoch [19/60] (39.9s) | Train: 2.5326 | Val: 2.3489 | Val PCK@0.05: 52.53% | PCK@0.10: 81.63% | Mean Err: 12.91px
  [*] New best checkpoint saved! (Mean Err: 12.91px, PCK@0.05: 52.53%)
Epoch [20/60] (40.2s) | Train: 2.4943 | Val: 2.3295 | Val PCK@0.05: 52.86% | PCK@0.10: 82.36% | Mean Err: 12.66px
  [*] New best checkpoint saved! (Mean Err: 12.66px, PCK@0.05: 52.86%)
Epoch [21/60] (41.1s) | Train: 2.4711 | Val: 2.3238 | Val PCK@0.05: 52.88% | PCK@0.10: 82.44% | Mean Err: 12.74px
Epoch [22/60] (39.6s) | Train: 2.4435 | Val: 2.3025 | Val PCK@0.05: 52.94% | PCK@0.10: 83.19% | Mean Err: 12.41px
  [*] New best checkpoint saved! (Mean Err: 12.41px, PCK@0.05: 52.94%)
Epoch [23/60] (40.2s) | Train: 2.4179 | Val: 2.2857 | Val PCK@0.05: 53.81% | PCK@0.10: 83.24% | Mean Err: 12.47px
Epoch [24/60] (40.5s) | Train: 2.3930 | Val: 2.2599 | Val PCK@0.05: 54.40% | PCK@0.10: 83.56% | Mean Err: 12.32px
  [*] New best checkpoint saved! (Mean Err: 12.32px, PCK@0.05: 54.40%)
Epoch [25/60] (40.7s) | Train: 2.3765 | Val: 2.2437 | Val PCK@0.05: 54.71% | PCK@0.10: 84.30% | Mean Err: 12.12px
  [*] New best checkpoint saved! (Mean Err: 12.12px, PCK@0.05: 54.71%)
Epoch [26/60] (40.5s) | Train: 2.3676 | Val: 2.2369 | Val PCK@0.05: 56.01% | PCK@0.10: 84.22% | Mean Err: 12.02px
  [*] New best checkpoint saved! (Mean Err: 12.02px, PCK@0.05: 56.01%)
Epoch [27/60] (41.6s) | Train: 2.3449 | Val: 2.2247 | Val PCK@0.05: 55.81% | PCK@0.10: 84.41% | Mean Err: 11.98px
  [*] New best checkpoint saved! (Mean Err: 11.98px, PCK@0.05: 55.81%)
Epoch [28/60] (42.2s) | Train: 2.3290 | Val: 2.2092 | Val PCK@0.05: 56.41% | PCK@0.10: 84.65% | Mean Err: 11.89px
  [*] New best checkpoint saved! (Mean Err: 11.89px, PCK@0.05: 56.41%)
Epoch [29/60] (42.0s) | Train: 2.3052 | Val: 2.1895 | Val PCK@0.05: 56.98% | PCK@0.10: 84.95% | Mean Err: 11.78px
  [*] New best checkpoint saved! (Mean Err: 11.78px, PCK@0.05: 56.98%)
Epoch [30/60] (41.1s) | Train: 2.3022 | Val: 2.1959 | Val PCK@0.05: 57.06% | PCK@0.10: 85.18% | Mean Err: 11.61px
  [*] New best checkpoint saved! (Mean Err: 11.61px, PCK@0.05: 57.06%)
Epoch [31/60] (41.5s) | Train: 2.2852 | Val: 2.1959 | Val PCK@0.05: 56.89% | PCK@0.10: 85.17% | Mean Err: 11.70px
Epoch [32/60] (42.2s) | Train: 2.2650 | Val: 2.1827 | Val PCK@0.05: 57.31% | PCK@0.10: 85.58% | Mean Err: 11.60px
  [*] New best checkpoint saved! (Mean Err: 11.60px, PCK@0.05: 57.31%)
Epoch [33/60] (41.1s) | Train: 2.2497 | Val: 2.1707 | Val PCK@0.05: 57.35% | PCK@0.10: 85.72% | Mean Err: 11.56px
  [*] New best checkpoint saved! (Mean Err: 11.56px, PCK@0.05: 57.35%)
Epoch [34/60] (40.7s) | Train: 2.2521 | Val: 2.1597 | Val PCK@0.05: 57.22% | PCK@0.10: 85.86% | Mean Err: 11.53px
  [*] New best checkpoint saved! (Mean Err: 11.53px, PCK@0.05: 57.22%)
Epoch [35/60] (41.0s) | Train: 2.2215 | Val: 2.1428 | Val PCK@0.05: 58.90% | PCK@0.10: 86.02% | Mean Err: 11.30px
  [*] New best checkpoint saved! (Mean Err: 11.30px, PCK@0.05: 58.90%)
Epoch [36/60] (40.5s) | Train: 2.2312 | Val: 2.1395 | Val PCK@0.05: 58.45% | PCK@0.10: 86.09% | Mean Err: 11.31px
Epoch [37/60] (40.8s) | Train: 2.1978 | Val: 2.1385 | Val PCK@0.05: 58.34% | PCK@0.10: 86.09% | Mean Err: 11.37px
Epoch [38/60] (40.9s) | Train: 2.1977 | Val: 2.1128 | Val PCK@0.05: 59.62% | PCK@0.10: 86.60% | Mean Err: 11.17px
  [*] New best checkpoint saved! (Mean Err: 11.17px, PCK@0.05: 59.62%)
Epoch [39/60] (40.9s) | Train: 2.1817 | Val: 2.1243 | Val PCK@0.05: 59.30% | PCK@0.10: 86.54% | Mean Err: 11.23px
Epoch [40/60] (41.1s) | Train: 2.1751 | Val: 2.1116 | Val PCK@0.05: 59.47% | PCK@0.10: 86.41% | Mean Err: 11.17px
Epoch [41/60] (40.9s) | Train: 2.1776 | Val: 2.1131 | Val PCK@0.05: 59.47% | PCK@0.10: 86.32% | Mean Err: 11.28px
Epoch [42/60] (41.0s) | Train: 2.1697 | Val: 2.1119 | Val PCK@0.05: 59.40% | PCK@0.10: 86.76% | Mean Err: 11.26px
Epoch [43/60] (40.8s) | Train: 2.1491 | Val: 2.1007 | Val PCK@0.05: 60.02% | PCK@0.10: 86.67% | Mean Err: 11.16px
  [*] New best checkpoint saved! (Mean Err: 11.16px, PCK@0.05: 60.02%)
Epoch [44/60] (41.0s) | Train: 2.1432 | Val: 2.1000 | Val PCK@0.05: 59.95% | PCK@0.10: 86.70% | Mean Err: 11.10px
  [*] New best checkpoint saved! (Mean Err: 11.10px, PCK@0.05: 59.95%)
Epoch [45/60] (40.9s) | Train: 2.1382 | Val: 2.0917 | Val PCK@0.05: 60.62% | PCK@0.10: 86.92% | Mean Err: 11.08px
  [*] New best checkpoint saved! (Mean Err: 11.08px, PCK@0.05: 60.62%)
Epoch [46/60] (41.1s) | Train: 2.1344 | Val: 2.0900 | Val PCK@0.05: 60.14% | PCK@0.10: 86.98% | Mean Err: 11.07px
  [*] New best checkpoint saved! (Mean Err: 11.07px, PCK@0.05: 60.14%)
Epoch [47/60] (41.2s) | Train: 2.1193 | Val: 2.0961 | Val PCK@0.05: 60.13% | PCK@0.10: 86.71% | Mean Err: 11.08px
Epoch [48/60] (41.1s) | Train: 2.1350 | Val: 2.0848 | Val PCK@0.05: 60.48% | PCK@0.10: 86.94% | Mean Err: 11.01px
  [*] New best checkpoint saved! (Mean Err: 11.01px, PCK@0.05: 60.48%)
Epoch [49/60] (40.7s) | Train: 2.1270 | Val: 2.0847 | Val PCK@0.05: 60.06% | PCK@0.10: 86.98% | Mean Err: 10.96px
  [*] New best checkpoint saved! (Mean Err: 10.96px, PCK@0.05: 60.06%)
Epoch [50/60] (41.3s) | Train: 2.1249 | Val: 2.0787 | Val PCK@0.05: 60.61% | PCK@0.10: 86.98% | Mean Err: 10.94px
  [*] New best checkpoint saved! (Mean Err: 10.94px, PCK@0.05: 60.61%)
Epoch [51/60] (40.7s) | Train: 2.1090 | Val: 2.0782 | Val PCK@0.05: 61.06% | PCK@0.10: 87.05% | Mean Err: 10.92px
  [*] New best checkpoint saved! (Mean Err: 10.92px, PCK@0.05: 61.06%)
Epoch [52/60] (40.9s) | Train: 2.1124 | Val: 2.0753 | Val PCK@0.05: 60.76% | PCK@0.10: 87.25% | Mean Err: 10.92px
Epoch [53/60] (41.2s) | Train: 2.1047 | Val: 2.0749 | Val PCK@0.05: 60.83% | PCK@0.10: 87.18% | Mean Err: 10.87px
  [*] New best checkpoint saved! (Mean Err: 10.87px, PCK@0.05: 60.83%)
Epoch [54/60] (40.5s) | Train: 2.0962 | Val: 2.0748 | Val PCK@0.05: 60.99% | PCK@0.10: 87.19% | Mean Err: 10.86px
  [*] New best checkpoint saved! (Mean Err: 10.86px, PCK@0.05: 60.99%)
Epoch [55/60] (40.7s) | Train: 2.1121 | Val: 2.0766 | Val PCK@0.05: 60.68% | PCK@0.10: 86.89% | Mean Err: 10.96px
Epoch [56/60] (40.6s) | Train: 2.1073 | Val: 2.0740 | Val PCK@0.05: 60.86% | PCK@0.10: 87.08% | Mean Err: 10.88px
Epoch [57/60] (40.7s) | Train: 2.0998 | Val: 2.0751 | Val PCK@0.05: 60.99% | PCK@0.10: 87.12% | Mean Err: 10.85px
  [*] New best checkpoint saved! (Mean Err: 10.85px, PCK@0.05: 60.99%)
Epoch [58/60] (40.4s) | Train: 2.1018 | Val: 2.0703 | Val PCK@0.05: 61.02% | PCK@0.10: 87.22% | Mean Err: 10.85px
Epoch [59/60] (41.1s) | Train: 2.0987 | Val: 2.0704 | Val PCK@0.05: 60.92% | PCK@0.10: 87.22% | Mean Err: 10.83px
  [*] New best checkpoint saved! (Mean Err: 10.83px, PCK@0.05: 60.92%)
Epoch [60/60] (42.2s) | Train: 2.0941 | Val: 2.0731 | Val PCK@0.05: 61.19% | PCK@0.10: 87.08% | Mean Err: 10.90px

[OK] Training completed. Best checkpoint: checkpoints/rtmpose_13kp_official_coco/best.pt (Mean Err: 10.83px, PCK@0.05: 60.92%)
