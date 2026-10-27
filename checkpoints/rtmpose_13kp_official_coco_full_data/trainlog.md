===========================================================================
  EXPERIMENT 2: Qualcomm RTMPose-13KP Foot Pose (SimCC + GAU) Training
  Dataset:         /kaggle/working/Shoes_VTO/data
  Backbone:        cspnext (Pretrained: True)
  Device:          cuda
  Epochs:          150 (Warmup: 2) | Batch: 32
  Backbone LR:     0.000050 | Head LR: 0.000250
  Resolution:      256x256 (SimCC Bins: 512x512, Dec Win: 12)
  Data Workers:    4 (Pin Memory: True, Negatives: True)
  Warm-Start Init: checkpoints/rtmpose_13kp_official_coco/best.pt
===========================================================================
[*] Warm-starting model weights from: checkpoints/rtmpose_13kp_official_coco/best.pt
[+] Model parameters successfully warm-started from checkpoint.
Epoch [01/150] (57.1s) | Train: 2.3141 | Val: 2.0416 | Val PCK@0.05: 59.64% | PCK@0.10: 86.45% | Mean Err: 11.13px
  [*] New best checkpoint saved! (Mean Err: 11.13px, PCK@0.05: 59.64%)
Epoch [02/150] (51.2s) | Train: 2.3057 | Val: 2.0521 | Val PCK@0.05: 59.99% | PCK@0.10: 86.23% | Mean Err: 11.13px
  [*] New best checkpoint saved! (Mean Err: 11.13px, PCK@0.05: 59.99%)
Epoch [03/150] (54.6s) | Train: 2.2823 | Val: 2.0274 | Val PCK@0.05: 60.11% | PCK@0.10: 86.38% | Mean Err: 11.18px
Epoch [04/150] (51.6s) | Train: 2.2520 | Val: 2.0284 | Val PCK@0.05: 59.54% | PCK@0.10: 86.02% | Mean Err: 11.30px
Epoch [05/150] (52.8s) | Train: 2.2498 | Val: 2.0297 | Val PCK@0.05: 59.76% | PCK@0.10: 86.24% | Mean Err: 11.12px
  [*] New best checkpoint saved! (Mean Err: 11.12px, PCK@0.05: 59.76%)
Epoch [06/150] (53.4s) | Train: 2.2330 | Val: 2.0156 | Val PCK@0.05: 60.39% | PCK@0.10: 86.56% | Mean Err: 11.15px
Epoch [07/150] (53.6s) | Train: 2.2241 | Val: 2.0076 | Val PCK@0.05: 60.33% | PCK@0.10: 86.51% | Mean Err: 11.03px
  [*] New best checkpoint saved! (Mean Err: 11.03px, PCK@0.05: 60.33%)
Epoch [08/150] (52.7s) | Train: 2.1988 | Val: 2.0021 | Val PCK@0.05: 60.82% | PCK@0.10: 86.66% | Mean Err: 10.91px
  [*] New best checkpoint saved! (Mean Err: 10.91px, PCK@0.05: 60.82%)
Epoch [09/150] (51.4s) | Train: 2.1844 | Val: 1.9998 | Val PCK@0.05: 60.69% | PCK@0.10: 86.67% | Mean Err: 10.93px
Epoch [10/150] (52.1s) | Train: 2.1907 | Val: 1.9989 | Val PCK@0.05: 60.56% | PCK@0.10: 86.18% | Mean Err: 11.03px
Epoch [11/150] (52.0s) | Train: 2.1822 | Val: 1.9893 | Val PCK@0.05: 60.86% | PCK@0.10: 86.65% | Mean Err: 10.86px
  [*] New best checkpoint saved! (Mean Err: 10.86px, PCK@0.05: 60.86%)
Epoch [12/150] (51.1s) | Train: 2.1578 | Val: 1.9920 | Val PCK@0.05: 60.84% | PCK@0.10: 86.51% | Mean Err: 10.88px
Epoch [13/150] (50.2s) | Train: 2.1502 | Val: 1.9822 | Val PCK@0.05: 60.85% | PCK@0.10: 86.80% | Mean Err: 10.91px
Epoch [14/150] (51.3s) | Train: 2.1398 | Val: 1.9806 | Val PCK@0.05: 61.10% | PCK@0.10: 86.87% | Mean Err: 10.86px
Epoch [15/150] (50.5s) | Train: 2.1469 | Val: 1.9885 | Val PCK@0.05: 61.15% | PCK@0.10: 86.67% | Mean Err: 10.95px
Epoch [16/150] (50.6s) | Train: 2.1318 | Val: 1.9791 | Val PCK@0.05: 61.52% | PCK@0.10: 86.95% | Mean Err: 10.88px
Epoch [17/150] (50.6s) | Train: 2.1123 | Val: 1.9784 | Val PCK@0.05: 61.48% | PCK@0.10: 86.83% | Mean Err: 10.81px
  [*] New best checkpoint saved! (Mean Err: 10.81px, PCK@0.05: 61.48%)
Epoch [18/150] (50.1s) | Train: 2.1002 | Val: 1.9733 | Val PCK@0.05: 61.42% | PCK@0.10: 86.70% | Mean Err: 10.81px
Epoch [19/150] (51.0s) | Train: 2.1031 | Val: 1.9729 | Val PCK@0.05: 61.33% | PCK@0.10: 86.92% | Mean Err: 10.74px
  [*] New best checkpoint saved! (Mean Err: 10.74px, PCK@0.05: 61.33%)
Epoch [20/150] (51.5s) | Train: 2.0989 | Val: 1.9590 | Val PCK@0.05: 61.78% | PCK@0.10: 87.05% | Mean Err: 10.70px
  [*] New best checkpoint saved! (Mean Err: 10.70px, PCK@0.05: 61.78%)
Epoch [21/150] (50.9s) | Train: 2.1041 | Val: 1.9631 | Val PCK@0.05: 62.07% | PCK@0.10: 87.06% | Mean Err: 10.74px
Epoch [22/150] (49.4s) | Train: 2.0819 | Val: 1.9600 | Val PCK@0.05: 61.94% | PCK@0.10: 87.11% | Mean Err: 10.71px
Epoch [23/150] (50.0s) | Train: 2.0716 | Val: 1.9630 | Val PCK@0.05: 62.17% | PCK@0.10: 87.28% | Mean Err: 10.77px
Epoch [24/150] (51.4s) | Train: 2.0664 | Val: 1.9504 | Val PCK@0.05: 62.07% | PCK@0.10: 87.04% | Mean Err: 10.71px
Epoch [25/150] (50.6s) | Train: 2.0587 | Val: 1.9530 | Val PCK@0.05: 62.00% | PCK@0.10: 87.21% | Mean Err: 10.73px
Epoch [26/150] (50.2s) | Train: 2.0595 | Val: 1.9656 | Val PCK@0.05: 61.68% | PCK@0.10: 87.00% | Mean Err: 10.83px
Epoch [27/150] (50.2s) | Train: 2.0470 | Val: 1.9460 | Val PCK@0.05: 62.27% | PCK@0.10: 87.33% | Mean Err: 10.62px
  [*] New best checkpoint saved! (Mean Err: 10.62px, PCK@0.05: 62.27%)
Epoch [28/150] (52.7s) | Train: 2.0485 | Val: 1.9424 | Val PCK@0.05: 62.70% | PCK@0.10: 87.49% | Mean Err: 10.65px
Epoch [29/150] (54.2s) | Train: 2.0456 | Val: 1.9444 | Val PCK@0.05: 62.63% | PCK@0.10: 87.60% | Mean Err: 10.64px
Epoch [30/150] (53.1s) | Train: 2.0455 | Val: 1.9397 | Val PCK@0.05: 62.45% | PCK@0.10: 87.39% | Mean Err: 10.64px
Epoch [31/150] (53.0s) | Train: 2.0256 | Val: 1.9463 | Val PCK@0.05: 62.74% | PCK@0.10: 87.30% | Mean Err: 10.73px
Epoch [32/150] (53.0s) | Train: 2.0178 | Val: 1.9501 | Val PCK@0.05: 62.42% | PCK@0.10: 87.22% | Mean Err: 10.71px
Epoch [33/150] (52.4s) | Train: 2.0231 | Val: 1.9331 | Val PCK@0.05: 62.93% | PCK@0.10: 87.48% | Mean Err: 10.57px
  [*] New best checkpoint saved! (Mean Err: 10.57px, PCK@0.05: 62.93%)
Epoch [34/150] (51.0s) | Train: 2.0094 | Val: 1.9391 | Val PCK@0.05: 62.82% | PCK@0.10: 87.49% | Mean Err: 10.65px
Epoch [35/150] (50.9s) | Train: 2.0023 | Val: 1.9495 | Val PCK@0.05: 62.47% | PCK@0.10: 87.27% | Mean Err: 10.62px
Epoch [36/150] (50.6s) | Train: 1.9955 | Val: 1.9326 | Val PCK@0.05: 63.12% | PCK@0.10: 87.44% | Mean Err: 10.60px
Epoch [37/150] (50.4s) | Train: 1.9955 | Val: 1.9425 | Val PCK@0.05: 62.89% | PCK@0.10: 87.41% | Mean Err: 10.65px
Epoch [38/150] (50.1s) | Train: 1.9774 | Val: 1.9278 | Val PCK@0.05: 62.89% | PCK@0.10: 87.79% | Mean Err: 10.55px
  [*] New best checkpoint saved! (Mean Err: 10.55px, PCK@0.05: 62.89%)
Epoch [39/150] (49.9s) | Train: 1.9919 | Val: 1.9178 | Val PCK@0.05: 63.49% | PCK@0.10: 87.92% | Mean Err: 10.51px
  [*] New best checkpoint saved! (Mean Err: 10.51px, PCK@0.05: 63.49%)
Epoch [40/150] (53.0s) | Train: 1.9738 | Val: 1.9378 | Val PCK@0.05: 63.07% | PCK@0.10: 87.52% | Mean Err: 10.63px
Epoch [41/150] (53.7s) | Train: 1.9845 | Val: 1.9290 | Val PCK@0.05: 63.07% | PCK@0.10: 87.79% | Mean Err: 10.53px
Epoch [42/150] (53.9s) | Train: 1.9634 | Val: 1.9300 | Val PCK@0.05: 62.86% | PCK@0.10: 87.59% | Mean Err: 10.60px
Epoch [43/150] (54.1s) | Train: 1.9774 | Val: 1.9263 | Val PCK@0.05: 63.24% | PCK@0.10: 87.34% | Mean Err: 10.59px
Epoch [44/150] (54.3s) | Train: 1.9624 | Val: 1.9242 | Val PCK@0.05: 63.65% | PCK@0.10: 87.56% | Mean Err: 10.57px
Epoch [45/150] (53.0s) | Train: 1.9607 | Val: 1.9273 | Val PCK@0.05: 63.33% | PCK@0.10: 87.29% | Mean Err: 10.62px
Epoch [46/150] (52.8s) | Train: 1.9507 | Val: 1.9178 | Val PCK@0.05: 63.34% | PCK@0.10: 87.67% | Mean Err: 10.56px
Epoch [47/150] (52.3s) | Train: 1.9491 | Val: 1.9085 | Val PCK@0.05: 63.54% | PCK@0.10: 88.11% | Mean Err: 10.46px
  [*] New best checkpoint saved! (Mean Err: 10.46px, PCK@0.05: 63.54%)
Epoch [48/150] (52.9s) | Train: 1.9452 | Val: 1.9168 | Val PCK@0.05: 63.27% | PCK@0.10: 87.56% | Mean Err: 10.52px
Epoch [49/150] (52.7s) | Train: 1.9391 | Val: 1.9158 | Val PCK@0.05: 63.31% | PCK@0.10: 87.69% | Mean Err: 10.63px
Epoch [50/150] (51.1s) | Train: 1.9318 | Val: 1.9120 | Val PCK@0.05: 63.86% | PCK@0.10: 87.74% | Mean Err: 10.53px
Epoch [51/150] (50.0s) | Train: 1.9276 | Val: 1.9068 | Val PCK@0.05: 64.09% | PCK@0.10: 87.94% | Mean Err: 10.45px
  [*] New best checkpoint saved! (Mean Err: 10.45px, PCK@0.05: 64.09%)
Epoch [52/150] (49.8s) | Train: 1.9254 | Val: 1.9111 | Val PCK@0.05: 63.55% | PCK@0.10: 87.91% | Mean Err: 10.54px
Epoch [53/150] (49.8s) | Train: 1.9249 | Val: 1.9068 | Val PCK@0.05: 63.62% | PCK@0.10: 88.15% | Mean Err: 10.43px
  [*] New best checkpoint saved! (Mean Err: 10.43px, PCK@0.05: 63.62%)
Epoch [54/150] (51.5s) | Train: 1.9235 | Val: 1.9054 | Val PCK@0.05: 63.64% | PCK@0.10: 87.88% | Mean Err: 10.44px
Epoch [55/150] (52.7s) | Train: 1.9156 | Val: 1.9049 | Val PCK@0.05: 63.84% | PCK@0.10: 87.94% | Mean Err: 10.46px
Epoch [56/150] (52.8s) | Train: 1.9172 | Val: 1.9069 | Val PCK@0.05: 64.16% | PCK@0.10: 87.92% | Mean Err: 10.45px
Epoch [57/150] (53.2s) | Train: 1.9048 | Val: 1.9104 | Val PCK@0.05: 63.88% | PCK@0.10: 87.62% | Mean Err: 10.56px
Epoch [58/150] (53.6s) | Train: 1.9073 | Val: 1.8961 | Val PCK@0.05: 64.24% | PCK@0.10: 87.85% | Mean Err: 10.45px
Epoch [59/150] (52.2s) | Train: 1.9090 | Val: 1.8935 | Val PCK@0.05: 63.56% | PCK@0.10: 88.11% | Mean Err: 10.36px
  [*] New best checkpoint saved! (Mean Err: 10.36px, PCK@0.05: 63.56%)
Epoch [60/150] (51.9s) | Train: 1.8949 | Val: 1.9029 | Val PCK@0.05: 64.09% | PCK@0.10: 87.87% | Mean Err: 10.49px
Epoch [61/150] (52.0s) | Train: 1.9021 | Val: 1.8955 | Val PCK@0.05: 64.25% | PCK@0.10: 87.82% | Mean Err: 10.46px
Epoch [62/150] (52.1s) | Train: 1.8873 | Val: 1.8969 | Val PCK@0.05: 63.94% | PCK@0.10: 88.02% | Mean Err: 10.47px
Epoch [63/150] (51.0s) | Train: 1.8908 | Val: 1.9012 | Val PCK@0.05: 64.12% | PCK@0.10: 87.91% | Mean Err: 10.48px
Epoch [64/150] (51.2s) | Train: 1.8915 | Val: 1.8970 | Val PCK@0.05: 64.45% | PCK@0.10: 87.89% | Mean Err: 10.44px
Epoch [65/150] (51.1s) | Train: 1.8859 | Val: 1.8963 | Val PCK@0.05: 64.32% | PCK@0.10: 88.06% | Mean Err: 10.49px
Epoch [66/150] (52.9s) | Train: 1.8862 | Val: 1.8855 | Val PCK@0.05: 64.26% | PCK@0.10: 88.28% | Mean Err: 10.31px
  [*] New best checkpoint saved! (Mean Err: 10.31px, PCK@0.05: 64.26%)
Epoch [67/150] (53.1s) | Train: 1.8749 | Val: 1.8944 | Val PCK@0.05: 64.49% | PCK@0.10: 88.06% | Mean Err: 10.36px
Epoch [68/150] (52.1s) | Train: 1.8659 | Val: 1.8844 | Val PCK@0.05: 64.52% | PCK@0.10: 88.11% | Mean Err: 10.31px
  [*] New best checkpoint saved! (Mean Err: 10.31px, PCK@0.05: 64.52%)
Epoch [69/150] (52.2s) | Train: 1.8722 | Val: 1.8954 | Val PCK@0.05: 64.09% | PCK@0.10: 88.11% | Mean Err: 10.39px
Epoch [70/150] (51.7s) | Train: 1.8653 | Val: 1.8943 | Val PCK@0.05: 64.60% | PCK@0.10: 88.03% | Mean Err: 10.40px
Epoch [71/150] (51.1s) | Train: 1.8709 | Val: 1.8852 | Val PCK@0.05: 64.78% | PCK@0.10: 87.96% | Mean Err: 10.32px
Epoch [72/150] (49.8s) | Train: 1.8634 | Val: 1.8833 | Val PCK@0.05: 64.95% | PCK@0.10: 88.28% | Mean Err: 10.29px
  [*] New best checkpoint saved! (Mean Err: 10.29px, PCK@0.05: 64.95%)
Epoch [73/150] (51.2s) | Train: 1.8593 | Val: 1.8850 | Val PCK@0.05: 65.00% | PCK@0.10: 88.17% | Mean Err: 10.30px
Epoch [74/150] (49.8s) | Train: 1.8563 | Val: 1.8916 | Val PCK@0.05: 64.72% | PCK@0.10: 88.09% | Mean Err: 10.41px
Epoch [75/150] (54.0s) | Train: 1.8511 | Val: 1.8863 | Val PCK@0.05: 64.78% | PCK@0.10: 88.12% | Mean Err: 10.33px
Epoch [76/150] (54.4s) | Train: 1.8530 | Val: 1.8919 | Val PCK@0.05: 64.38% | PCK@0.10: 87.75% | Mean Err: 10.45px
Epoch [77/150] (55.6s) | Train: 1.8530 | Val: 1.8821 | Val PCK@0.05: 65.14% | PCK@0.10: 88.40% | Mean Err: 10.29px
  [*] New best checkpoint saved! (Mean Err: 10.29px, PCK@0.05: 65.14%)
Epoch [78/150] (53.9s) | Train: 1.8506 | Val: 1.8854 | Val PCK@0.05: 64.40% | PCK@0.10: 88.16% | Mean Err: 10.39px
Epoch [79/150] (53.0s) | Train: 1.8326 | Val: 1.8852 | Val PCK@0.05: 64.73% | PCK@0.10: 88.03% | Mean Err: 10.36px
Epoch [80/150] (52.8s) | Train: 1.8459 | Val: 1.8827 | Val PCK@0.05: 65.03% | PCK@0.10: 88.16% | Mean Err: 10.30px
Epoch [81/150] (52.0s) | Train: 1.8422 | Val: 1.8745 | Val PCK@0.05: 64.89% | PCK@0.10: 87.99% | Mean Err: 10.36px
Epoch [82/150] (52.1s) | Train: 1.8350 | Val: 1.8773 | Val PCK@0.05: 64.82% | PCK@0.10: 88.11% | Mean Err: 10.32px
Epoch [83/150] (51.6s) | Train: 1.8354 | Val: 1.8784 | Val PCK@0.05: 64.86% | PCK@0.10: 88.26% | Mean Err: 10.32px
Epoch [84/150] (51.6s) | Train: 1.8412 | Val: 1.8762 | Val PCK@0.05: 64.95% | PCK@0.10: 88.39% | Mean Err: 10.24px
  [*] New best checkpoint saved! (Mean Err: 10.24px, PCK@0.05: 64.95%)
Epoch [85/150] (52.3s) | Train: 1.8258 | Val: 1.8748 | Val PCK@0.05: 64.96% | PCK@0.10: 88.62% | Mean Err: 10.19px
  [*] New best checkpoint saved! (Mean Err: 10.19px, PCK@0.05: 64.96%)
Epoch [86/150] (52.3s) | Train: 1.8364 | Val: 1.8746 | Val PCK@0.05: 65.18% | PCK@0.10: 88.25% | Mean Err: 10.29px
Epoch [87/150] (51.8s) | Train: 1.8256 | Val: 1.8808 | Val PCK@0.05: 65.29% | PCK@0.10: 88.39% | Mean Err: 10.28px
Epoch [88/150] (52.8s) | Train: 1.8270 | Val: 1.8750 | Val PCK@0.05: 65.17% | PCK@0.10: 88.01% | Mean Err: 10.27px
Epoch [89/150] (52.7s) | Train: 1.8233 | Val: 1.8734 | Val PCK@0.05: 65.23% | PCK@0.10: 88.43% | Mean Err: 10.30px
Epoch [90/150] (52.3s) | Train: 1.8079 | Val: 1.8764 | Val PCK@0.05: 65.02% | PCK@0.10: 88.22% | Mean Err: 10.34px
Epoch [91/150] (51.5s) | Train: 1.8182 | Val: 1.8724 | Val PCK@0.05: 65.13% | PCK@0.10: 88.16% | Mean Err: 10.30px
Epoch [92/150] (52.3s) | Train: 1.8175 | Val: 1.8699 | Val PCK@0.05: 65.14% | PCK@0.10: 88.10% | Mean Err: 10.31px
Epoch [93/150] (50.0s) | Train: 1.8092 | Val: 1.8684 | Val PCK@0.05: 65.30% | PCK@0.10: 88.26% | Mean Err: 10.29px
Epoch [94/150] (47.2s) | Train: 1.8186 | Val: 1.8658 | Val PCK@0.05: 65.21% | PCK@0.10: 88.53% | Mean Err: 10.24px
Epoch [95/150] (47.1s) | Train: 1.8146 | Val: 1.8718 | Val PCK@0.05: 65.21% | PCK@0.10: 88.18% | Mean Err: 10.37px
Epoch [96/150] (48.8s) | Train: 1.8093 | Val: 1.8707 | Val PCK@0.05: 64.93% | PCK@0.10: 88.20% | Mean Err: 10.37px
Epoch [97/150] (49.0s) | Train: 1.8107 | Val: 1.8676 | Val PCK@0.05: 65.32% | PCK@0.10: 88.48% | Mean Err: 10.27px
Epoch [98/150] (48.9s) | Train: 1.8025 | Val: 1.8666 | Val PCK@0.05: 65.44% | PCK@0.10: 88.36% | Mean Err: 10.25px
Epoch [99/150] (50.7s) | Train: 1.7983 | Val: 1.8655 | Val PCK@0.05: 65.53% | PCK@0.10: 88.26% | Mean Err: 10.30px
Epoch [100/150] (50.7s) | Train: 1.8027 | Val: 1.8678 | Val PCK@0.05: 65.35% | PCK@0.10: 88.49% | Mean Err: 10.21px
Epoch [101/150] (51.6s) | Train: 1.8044 | Val: 1.8731 | Val PCK@0.05: 65.55% | PCK@0.10: 88.20% | Mean Err: 10.33px
Epoch [102/150] (52.2s) | Train: 1.8110 | Val: 1.8647 | Val PCK@0.05: 65.60% | PCK@0.10: 88.35% | Mean Err: 10.26px
Epoch [103/150] (52.6s) | Train: 1.7996 | Val: 1.8654 | Val PCK@0.05: 65.30% | PCK@0.10: 88.68% | Mean Err: 10.23px
Epoch [104/150] (51.5s) | Train: 1.7965 | Val: 1.8655 | Val PCK@0.05: 65.36% | PCK@0.10: 88.40% | Mean Err: 10.26px
Epoch [105/150] (51.1s) | Train: 1.7947 | Val: 1.8643 | Val PCK@0.05: 65.35% | PCK@0.10: 88.56% | Mean Err: 10.19px
Epoch [106/150] (50.5s) | Train: 1.7968 | Val: 1.8670 | Val PCK@0.05: 65.39% | PCK@0.10: 88.43% | Mean Err: 10.24px
Epoch [107/150] (50.6s) | Train: 1.7854 | Val: 1.8640 | Val PCK@0.05: 65.41% | PCK@0.10: 88.59% | Mean Err: 10.21px
Epoch [108/150] (50.9s) | Train: 1.7920 | Val: 1.8618 | Val PCK@0.05: 65.45% | PCK@0.10: 88.56% | Mean Err: 10.24px
Epoch [109/150] (50.8s) | Train: 1.7953 | Val: 1.8628 | Val PCK@0.05: 65.47% | PCK@0.10: 88.42% | Mean Err: 10.25px
Epoch [110/150] (49.9s) | Train: 1.7987 | Val: 1.8626 | Val PCK@0.05: 65.61% | PCK@0.10: 88.39% | Mean Err: 10.20px
Epoch [111/150] (50.2s) | Train: 1.7864 | Val: 1.8581 | Val PCK@0.05: 65.52% | PCK@0.10: 88.49% | Mean Err: 10.15px
  [*] New best checkpoint saved! (Mean Err: 10.15px, PCK@0.05: 65.52%)
Epoch [112/150] (51.3s) | Train: 1.7925 | Val: 1.8654 | Val PCK@0.05: 65.49% | PCK@0.10: 88.39% | Mean Err: 10.22px
Epoch [113/150] (50.2s) | Train: 1.7890 | Val: 1.8615 | Val PCK@0.05: 65.60% | PCK@0.10: 88.49% | Mean Err: 10.25px
Epoch [114/150] (50.4s) | Train: 1.7852 | Val: 1.8620 | Val PCK@0.05: 65.67% | PCK@0.10: 88.44% | Mean Err: 10.25px
Epoch [115/150] (50.8s) | Train: 1.7897 | Val: 1.8549 | Val PCK@0.05: 65.65% | PCK@0.10: 88.51% | Mean Err: 10.18px
Epoch [116/150] (50.3s) | Train: 1.7845 | Val: 1.8583 | Val PCK@0.05: 65.81% | PCK@0.10: 88.55% | Mean Err: 10.19px
Epoch [117/150] (50.8s) | Train: 1.7835 | Val: 1.8580 | Val PCK@0.05: 65.78% | PCK@0.10: 88.63% | Mean Err: 10.24px
Epoch [118/150] (52.2s) | Train: 1.7776 | Val: 1.8573 | Val PCK@0.05: 65.41% | PCK@0.10: 88.56% | Mean Err: 10.16px
Epoch [119/150] (55.8s) | Train: 1.7857 | Val: 1.8624 | Val PCK@0.05: 65.50% | PCK@0.10: 88.45% | Mean Err: 10.21px
Epoch [120/150] (55.3s) | Train: 1.7839 | Val: 1.8642 | Val PCK@0.05: 65.60% | PCK@0.10: 88.38% | Mean Err: 10.24px
Epoch [121/150] (55.6s) | Train: 1.7807 | Val: 1.8588 | Val PCK@0.05: 65.89% | PCK@0.10: 88.49% | Mean Err: 10.24px
Epoch [122/150] (55.1s) | Train: 1.7854 | Val: 1.8576 | Val PCK@0.05: 65.50% | PCK@0.10: 88.52% | Mean Err: 10.15px
Epoch [123/150] (50.3s) | Train: 1.7832 | Val: 1.8580 | Val PCK@0.05: 65.79% | PCK@0.10: 88.46% | Mean Err: 10.21px
Epoch [124/150] (52.9s) | Train: 1.7790 | Val: 1.8590 | Val PCK@0.05: 65.86% | PCK@0.10: 88.45% | Mean Err: 10.24px
Epoch [125/150] (54.8s) | Train: 1.7740 | Val: 1.8577 | Val PCK@0.05: 65.78% | PCK@0.10: 88.74% | Mean Err: 10.12px
  [*] New best checkpoint saved! (Mean Err: 10.12px, PCK@0.05: 65.78%)
Epoch [126/150] (55.2s) | Train: 1.7748 | Val: 1.8577 | Val PCK@0.05: 66.01% | PCK@0.10: 88.56% | Mean Err: 10.21px
Epoch [127/150] (54.0s) | Train: 1.7721 | Val: 1.8582 | Val PCK@0.05: 65.90% | PCK@0.10: 88.56% | Mean Err: 10.20px
Epoch [128/150] (54.1s) | Train: 1.7726 | Val: 1.8562 | Val PCK@0.05: 66.11% | PCK@0.10: 88.53% | Mean Err: 10.17px
Epoch [129/150] (54.6s) | Train: 1.7782 | Val: 1.8580 | Val PCK@0.05: 65.92% | PCK@0.10: 88.50% | Mean Err: 10.16px
Epoch [130/150] (54.6s) | Train: 1.7742 | Val: 1.8574 | Val PCK@0.05: 66.03% | PCK@0.10: 88.55% | Mean Err: 10.17px
Epoch [131/150] (54.5s) | Train: 1.7761 | Val: 1.8552 | Val PCK@0.05: 66.17% | PCK@0.10: 88.53% | Mean Err: 10.16px
Epoch [132/150] (53.8s) | Train: 1.7713 | Val: 1.8552 | Val PCK@0.05: 65.88% | PCK@0.10: 88.63% | Mean Err: 10.18px
Epoch [133/150] (54.1s) | Train: 1.7746 | Val: 1.8567 | Val PCK@0.05: 65.81% | PCK@0.10: 88.59% | Mean Err: 10.19px
Epoch [134/150] (54.3s) | Train: 1.7709 | Val: 1.8575 | Val PCK@0.05: 65.73% | PCK@0.10: 88.52% | Mean Err: 10.19px
Epoch [135/150] (55.2s) | Train: 1.7738 | Val: 1.8541 | Val PCK@0.05: 65.70% | PCK@0.10: 88.53% | Mean Err: 10.19px
Epoch [136/150] (55.5s) | Train: 1.7722 | Val: 1.8578 | Val PCK@0.05: 65.81% | PCK@0.10: 88.56% | Mean Err: 10.22px
Epoch [137/150] (54.8s) | Train: 1.7668 | Val: 1.8590 | Val PCK@0.05: 65.88% | PCK@0.10: 88.44% | Mean Err: 10.21px
Epoch [138/150] (53.9s) | Train: 1.7734 | Val: 1.8581 | Val PCK@0.05: 65.83% | PCK@0.10: 88.64% | Mean Err: 10.19px
Epoch [139/150] (51.9s) | Train: 1.7769 | Val: 1.8581 | Val PCK@0.05: 65.95% | PCK@0.10: 88.59% | Mean Err: 10.17px
Epoch [140/150] (49.2s) | Train: 1.7679 | Val: 1.8554 | Val PCK@0.05: 65.96% | PCK@0.10: 88.55% | Mean Err: 10.18px
Epoch [141/150] (55.0s) | Train: 1.7744 | Val: 1.8576 | Val PCK@0.05: 65.99% | PCK@0.10: 88.46% | Mean Err: 10.21px
Epoch [142/150] (55.6s) | Train: 1.7786 | Val: 1.8584 | Val PCK@0.05: 65.95% | PCK@0.10: 88.54% | Mean Err: 10.21px
Epoch [143/150] (55.8s) | Train: 1.7681 | Val: 1.8562 | Val PCK@0.05: 66.00% | PCK@0.10: 88.57% | Mean Err: 10.18px
Epoch [144/150] (55.9s) | Train: 1.7671 | Val: 1.8593 | Val PCK@0.05: 65.98% | PCK@0.10: 88.62% | Mean Err: 10.19px
Epoch [145/150] (55.4s) | Train: 1.7689 | Val: 1.8594 | Val PCK@0.05: 66.01% | PCK@0.10: 88.64% | Mean Err: 10.22px
Epoch [146/150] (53.6s) | Train: 1.7778 | Val: 1.8601 | Val PCK@0.05: 66.01% | PCK@0.10: 88.54% | Mean Err: 10.21px
Epoch [147/150] (56.8s) | Train: 1.7654 | Val: 1.8583 | Val PCK@0.05: 65.92% | PCK@0.10: 88.53% | Mean Err: 10.17px
Epoch [148/150] (55.2s) | Train: 1.7710 | Val: 1.8545 | Val PCK@0.05: 65.91% | PCK@0.10: 88.61% | Mean Err: 10.15px
Epoch [149/150] (52.9s) | Train: 1.7665 | Val: 1.8589 | Val PCK@0.05: 65.81% | PCK@0.10: 88.58% | Mean Err: 10.17px
Epoch [150/150] (55.0s) | Train: 1.7717 | Val: 1.8580 | Val PCK@0.05: 65.86% | PCK@0.10: 88.59% | Mean Err: 10.24px

[OK] Training completed. Best checkpoint: checkpoints/rtmpose_13kp_official_coco_full_data/best.pt (Mean Err: 10.12px, PCK@0.05: 65.78%)