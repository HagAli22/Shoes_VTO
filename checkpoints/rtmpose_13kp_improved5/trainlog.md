===========================================================================
  EXPERIMENT 2: Qualcomm RTMPose-13KP Foot Pose (SimCC + GAU) Training
  Dataset:         /content/Shoes_VTO/data/merged_v6_with_negatives
  Backbone:        cspnext (Pretrained: True)
  Device:          cuda (AMP: True)
  Epochs:          100 (Warmup: 3) | Batch: 32
  Backbone LR:     0.000060 | Head LR: 0.000300
  Resolution:      256x256 (SimCC Bins: 512x512, Dec Win: 12)
  Data Workers:    4 (Pin Memory: True, Negatives: True)
  Warm-Start Init: /content/Shoes_VTO/checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt
===========================================================================
[*] Warm-starting model weights from: /content/Shoes_VTO/checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt
[+] Model parameters successfully warm-started from checkpoint.
/content/Shoes_VTO/src/models/qualcomm_experiments/rtmpose/train.py:179: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = torch.cuda.amp.GradScaler(enabled=amp and (device.type == "cuda"))
/content/Shoes_VTO/src/models/qualcomm_experiments/rtmpose/train.py:196: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=amp and (device.type == "cuda")):
Epoch [01/100] (87.4s) | Train: 2.2236 | Val: 2.1528 | Val PCK@0.05: 60.29% | PCK@0.10: 84.96% | Mean Err: 11.77px | Hand Acc: 91.9%
  [*] New best checkpoint saved! (Mean Err: 11.77px, PCK@0.05: 60.29%)
Epoch [02/100] (39.1s) | Train: 2.1964 | Val: 2.1288 | Val PCK@0.05: 60.22% | PCK@0.10: 85.34% | Mean Err: 11.67px | Hand Acc: 92.6%
  [*] New best checkpoint saved! (Mean Err: 11.67px, PCK@0.05: 60.22%)
Epoch [03/100] (38.8s) | Train: 2.1883 | Val: 2.1111 | Val PCK@0.05: 60.10% | PCK@0.10: 85.21% | Mean Err: 11.58px | Hand Acc: 93.4%
  [*] New best checkpoint saved! (Mean Err: 11.58px, PCK@0.05: 60.10%)
Epoch [04/100] (38.7s) | Train: 2.1692 | Val: 2.0981 | Val PCK@0.05: 60.34% | PCK@0.10: 85.49% | Mean Err: 11.53px | Hand Acc: 94.2%
  [*] New best checkpoint saved! (Mean Err: 11.53px, PCK@0.05: 60.34%)
Epoch [05/100] (38.7s) | Train: 2.1531 | Val: 2.0865 | Val PCK@0.05: 60.11% | PCK@0.10: 85.32% | Mean Err: 11.52px | Hand Acc: 94.0%
  [*] New best checkpoint saved! (Mean Err: 11.52px, PCK@0.05: 60.11%)
Epoch [06/100] (38.3s) | Train: 2.1389 | Val: 2.0991 | Val PCK@0.05: 59.86% | PCK@0.10: 85.71% | Mean Err: 11.38px | Hand Acc: 94.6%
  [*] New best checkpoint saved! (Mean Err: 11.38px, PCK@0.05: 59.86%)
Epoch [07/100] (38.8s) | Train: 2.1349 | Val: 2.0775 | Val PCK@0.05: 60.80% | PCK@0.10: 86.10% | Mean Err: 11.23px | Hand Acc: 94.7%
  [*] New best checkpoint saved! (Mean Err: 11.23px, PCK@0.05: 60.80%)
Epoch [08/100] (38.8s) | Train: 2.1171 | Val: 2.0769 | Val PCK@0.05: 60.38% | PCK@0.10: 85.76% | Mean Err: 11.34px | Hand Acc: 94.5%
Epoch [09/100] (37.9s) | Train: 2.1033 | Val: 2.0632 | Val PCK@0.05: 60.77% | PCK@0.10: 85.97% | Mean Err: 11.12px | Hand Acc: 94.8%
  [*] New best checkpoint saved! (Mean Err: 11.12px, PCK@0.05: 60.77%)
Epoch [10/100] (38.5s) | Train: 2.1008 | Val: 2.0680 | Val PCK@0.05: 60.68% | PCK@0.10: 85.92% | Mean Err: 11.11px | Hand Acc: 95.1%
  [*] New best checkpoint saved! (Mean Err: 11.11px, PCK@0.05: 60.68%)
Epoch [11/100] (39.1s) | Train: 2.0837 | Val: 2.0797 | Val PCK@0.05: 60.67% | PCK@0.10: 86.22% | Mean Err: 11.20px | Hand Acc: 95.0%
Epoch [12/100] (38.9s) | Train: 2.0872 | Val: 2.0644 | Val PCK@0.05: 60.75% | PCK@0.10: 86.17% | Mean Err: 11.21px | Hand Acc: 94.8%
Epoch [13/100] (38.5s) | Train: 2.0709 | Val: 2.0469 | Val PCK@0.05: 61.04% | PCK@0.10: 86.62% | Mean Err: 11.05px | Hand Acc: 95.2%
  [*] New best checkpoint saved! (Mean Err: 11.05px, PCK@0.05: 61.04%)
Epoch [14/100] (38.5s) | Train: 2.0722 | Val: 2.0451 | Val PCK@0.05: 60.79% | PCK@0.10: 86.55% | Mean Err: 11.08px | Hand Acc: 95.1%
Epoch [15/100] (37.6s) | Train: 2.0537 | Val: 2.0475 | Val PCK@0.05: 61.48% | PCK@0.10: 86.41% | Mean Err: 11.11px | Hand Acc: 95.2%
Epoch [16/100] (38.9s) | Train: 2.0525 | Val: 2.0527 | Val PCK@0.05: 61.15% | PCK@0.10: 86.59% | Mean Err: 11.11px | Hand Acc: 95.0%
Epoch [17/100] (38.4s) | Train: 2.0439 | Val: 2.0343 | Val PCK@0.05: 61.33% | PCK@0.10: 86.71% | Mean Err: 10.99px | Hand Acc: 95.2%
  [*] New best checkpoint saved! (Mean Err: 10.99px, PCK@0.05: 61.33%)
Epoch [18/100] (38.5s) | Train: 2.0364 | Val: 2.0360 | Val PCK@0.05: 61.81% | PCK@0.10: 86.80% | Mean Err: 10.91px | Hand Acc: 95.2%
  [*] New best checkpoint saved! (Mean Err: 10.91px, PCK@0.05: 61.81%)
Epoch [19/100] (38.7s) | Train: 2.0389 | Val: 2.0297 | Val PCK@0.05: 61.79% | PCK@0.10: 86.62% | Mean Err: 10.93px | Hand Acc: 95.7%
Epoch [20/100] (38.9s) | Train: 2.0275 | Val: 2.0299 | Val PCK@0.05: 61.89% | PCK@0.10: 86.82% | Mean Err: 10.97px | Hand Acc: 95.6%
Epoch [21/100] (38.6s) | Train: 2.0192 | Val: 2.0297 | Val PCK@0.05: 62.04% | PCK@0.10: 86.69% | Mean Err: 10.98px | Hand Acc: 94.9%
Epoch [22/100] (38.9s) | Train: 2.0114 | Val: 2.0282 | Val PCK@0.05: 61.10% | PCK@0.10: 86.94% | Mean Err: 10.92px | Hand Acc: 95.7%
Epoch [23/100] (39.4s) | Train: 2.0106 | Val: 2.0307 | Val PCK@0.05: 61.48% | PCK@0.10: 86.49% | Mean Err: 11.00px | Hand Acc: 95.5%
Epoch [24/100] (38.4s) | Train: 1.9933 | Val: 2.0245 | Val PCK@0.05: 62.28% | PCK@0.10: 86.84% | Mean Err: 10.86px | Hand Acc: 95.5%
  [*] New best checkpoint saved! (Mean Err: 10.86px, PCK@0.05: 62.28%)
Epoch [25/100] (38.6s) | Train: 2.0003 | Val: 2.0200 | Val PCK@0.05: 61.86% | PCK@0.10: 86.94% | Mean Err: 10.83px | Hand Acc: 95.8%
  [*] New best checkpoint saved! (Mean Err: 10.83px, PCK@0.05: 61.86%)
Epoch [26/100] (38.5s) | Train: 1.9952 | Val: 2.0259 | Val PCK@0.05: 62.42% | PCK@0.10: 87.00% | Mean Err: 10.81px | Hand Acc: 95.1%
  [*] New best checkpoint saved! (Mean Err: 10.81px, PCK@0.05: 62.42%)
Epoch [27/100] (39.0s) | Train: 1.9838 | Val: 2.0219 | Val PCK@0.05: 62.17% | PCK@0.10: 87.06% | Mean Err: 10.79px | Hand Acc: 95.5%
  [*] New best checkpoint saved! (Mean Err: 10.79px, PCK@0.05: 62.17%)
Epoch [28/100] (39.0s) | Train: 1.9753 | Val: 2.0194 | Val PCK@0.05: 62.54% | PCK@0.10: 87.09% | Mean Err: 10.85px | Hand Acc: 95.0%
Epoch [29/100] (38.3s) | Train: 1.9799 | Val: 2.0153 | Val PCK@0.05: 62.10% | PCK@0.10: 87.10% | Mean Err: 10.89px | Hand Acc: 95.7%
Epoch [30/100] (39.1s) | Train: 1.9674 | Val: 2.0096 | Val PCK@0.05: 62.39% | PCK@0.10: 87.15% | Mean Err: 10.78px | Hand Acc: 95.5%
  [*] New best checkpoint saved! (Mean Err: 10.78px, PCK@0.05: 62.39%)
Epoch [31/100] (38.5s) | Train: 1.9659 | Val: 2.0134 | Val PCK@0.05: 62.13% | PCK@0.10: 86.99% | Mean Err: 10.79px | Hand Acc: 96.1%
Epoch [32/100] (39.0s) | Train: 1.9738 | Val: 2.0079 | Val PCK@0.05: 62.16% | PCK@0.10: 86.79% | Mean Err: 10.80px | Hand Acc: 96.2%
Epoch [33/100] (38.0s) | Train: 1.9590 | Val: 2.0010 | Val PCK@0.05: 62.61% | PCK@0.10: 87.47% | Mean Err: 10.69px | Hand Acc: 95.9%
  [*] New best checkpoint saved! (Mean Err: 10.69px, PCK@0.05: 62.61%)
Epoch [34/100] (38.3s) | Train: 1.9552 | Val: 2.0165 | Val PCK@0.05: 62.54% | PCK@0.10: 87.15% | Mean Err: 10.73px | Hand Acc: 95.9%
Epoch [35/100] (38.8s) | Train: 1.9534 | Val: 1.9990 | Val PCK@0.05: 62.42% | PCK@0.10: 87.44% | Mean Err: 10.65px | Hand Acc: 96.1%
  [*] New best checkpoint saved! (Mean Err: 10.65px, PCK@0.05: 62.42%)
Epoch [36/100] (38.6s) | Train: 1.9392 | Val: 1.9915 | Val PCK@0.05: 62.89% | PCK@0.10: 87.66% | Mean Err: 10.69px | Hand Acc: 95.7%
Epoch [37/100] (38.8s) | Train: 1.9451 | Val: 2.0115 | Val PCK@0.05: 62.02% | PCK@0.10: 87.42% | Mean Err: 10.71px | Hand Acc: 95.9%
Epoch [38/100] (38.5s) | Train: 1.9395 | Val: 1.9871 | Val PCK@0.05: 62.85% | PCK@0.10: 87.24% | Mean Err: 10.65px | Hand Acc: 96.1%
Epoch [39/100] (38.8s) | Train: 1.9330 | Val: 1.9851 | Val PCK@0.05: 63.19% | PCK@0.10: 87.48% | Mean Err: 10.51px | Hand Acc: 95.9%
  [*] New best checkpoint saved! (Mean Err: 10.51px, PCK@0.05: 63.19%)
Epoch [40/100] (37.9s) | Train: 1.9328 | Val: 2.0092 | Val PCK@0.05: 62.78% | PCK@0.10: 87.14% | Mean Err: 10.73px | Hand Acc: 95.5%
Epoch [41/100] (39.1s) | Train: 1.9256 | Val: 1.9887 | Val PCK@0.05: 62.86% | PCK@0.10: 87.36% | Mean Err: 10.68px | Hand Acc: 96.1%
Epoch [42/100] (38.7s) | Train: 1.9235 | Val: 1.9947 | Val PCK@0.05: 62.92% | PCK@0.10: 87.35% | Mean Err: 10.63px | Hand Acc: 96.2%
Epoch [43/100] (39.0s) | Train: 1.9166 | Val: 1.9795 | Val PCK@0.05: 63.10% | PCK@0.10: 87.56% | Mean Err: 10.57px | Hand Acc: 96.0%
Epoch [44/100] (39.0s) | Train: 1.9104 | Val: 1.9957 | Val PCK@0.05: 62.86% | PCK@0.10: 87.55% | Mean Err: 10.64px | Hand Acc: 95.8%
Epoch [45/100] (38.7s) | Train: 1.9002 | Val: 2.0035 | Val PCK@0.05: 62.95% | PCK@0.10: 87.60% | Mean Err: 10.58px | Hand Acc: 96.1%
Epoch [46/100] (38.4s) | Train: 1.9106 | Val: 2.0009 | Val PCK@0.05: 62.98% | PCK@0.10: 87.53% | Mean Err: 10.65px | Hand Acc: 96.2%
Epoch [47/100] (38.9s) | Train: 1.8978 | Val: 1.9928 | Val PCK@0.05: 63.03% | PCK@0.10: 87.77% | Mean Err: 10.56px | Hand Acc: 95.8%
Epoch [48/100] (38.0s) | Train: 1.8922 | Val: 1.9970 | Val PCK@0.05: 63.25% | PCK@0.10: 87.66% | Mean Err: 10.49px | Hand Acc: 96.3%
  [*] New best checkpoint saved! (Mean Err: 10.49px, PCK@0.05: 63.25%)
Epoch [49/100] (38.9s) | Train: 1.8957 | Val: 1.9835 | Val PCK@0.05: 63.17% | PCK@0.10: 87.61% | Mean Err: 10.49px | Hand Acc: 96.2%
Epoch [50/100] (38.2s) | Train: 1.8942 | Val: 1.9860 | Val PCK@0.05: 62.88% | PCK@0.10: 87.63% | Mean Err: 10.60px | Hand Acc: 96.0%
Epoch [51/100] (38.5s) | Train: 1.8943 | Val: 1.9942 | Val PCK@0.05: 63.11% | PCK@0.10: 87.50% | Mean Err: 10.59px | Hand Acc: 95.8%
Epoch [52/100] (38.6s) | Train: 1.8806 | Val: 1.9801 | Val PCK@0.05: 63.74% | PCK@0.10: 87.63% | Mean Err: 10.57px | Hand Acc: 95.9%
Epoch [53/100] (39.0s) | Train: 1.8813 | Val: 1.9838 | Val PCK@0.05: 63.19% | PCK@0.10: 87.39% | Mean Err: 10.56px | Hand Acc: 96.1%
Epoch [54/100] (38.5s) | Train: 1.8724 | Val: 1.9710 | Val PCK@0.05: 63.48% | PCK@0.10: 87.89% | Mean Err: 10.48px | Hand Acc: 96.2%
  [*] New best checkpoint saved! (Mean Err: 10.48px, PCK@0.05: 63.48%)
Epoch [55/100] (38.5s) | Train: 1.8722 | Val: 1.9780 | Val PCK@0.05: 63.54% | PCK@0.10: 87.71% | Mean Err: 10.49px | Hand Acc: 95.9%
Epoch [56/100] (38.5s) | Train: 1.8691 | Val: 1.9765 | Val PCK@0.05: 63.27% | PCK@0.10: 87.56% | Mean Err: 10.53px | Hand Acc: 96.6%
Epoch [57/100] (38.5s) | Train: 1.8647 | Val: 1.9851 | Val PCK@0.05: 63.34% | PCK@0.10: 87.62% | Mean Err: 10.52px | Hand Acc: 96.2%
Epoch [58/100] (38.2s) | Train: 1.8633 | Val: 1.9827 | Val PCK@0.05: 63.42% | PCK@0.10: 87.54% | Mean Err: 10.59px | Hand Acc: 96.3%
Epoch [59/100] (39.0s) | Train: 1.8567 | Val: 1.9763 | Val PCK@0.05: 63.29% | PCK@0.10: 87.74% | Mean Err: 10.52px | Hand Acc: 96.1%
Epoch [60/100] (39.0s) | Train: 1.8614 | Val: 1.9793 | Val PCK@0.05: 63.63% | PCK@0.10: 87.79% | Mean Err: 10.49px | Hand Acc: 96.3%
Epoch [61/100] (37.8s) | Train: 1.8518 | Val: 1.9807 | Val PCK@0.05: 63.79% | PCK@0.10: 87.81% | Mean Err: 10.47px | Hand Acc: 96.0%
  [*] New best checkpoint saved! (Mean Err: 10.47px, PCK@0.05: 63.79%)
Epoch [62/100] (38.2s) | Train: 1.8569 | Val: 1.9779 | Val PCK@0.05: 63.80% | PCK@0.10: 87.80% | Mean Err: 10.51px | Hand Acc: 96.3%
Epoch [63/100] (38.5s) | Train: 1.8508 | Val: 1.9697 | Val PCK@0.05: 63.66% | PCK@0.10: 88.09% | Mean Err: 10.41px | Hand Acc: 96.6%
  [*] New best checkpoint saved! (Mean Err: 10.41px, PCK@0.05: 63.66%)
Epoch [64/100] (38.4s) | Train: 1.8568 | Val: 1.9752 | Val PCK@0.05: 63.82% | PCK@0.10: 88.02% | Mean Err: 10.47px | Hand Acc: 96.1%
Epoch [65/100] (39.0s) | Train: 1.8420 | Val: 1.9680 | Val PCK@0.05: 63.39% | PCK@0.10: 88.22% | Mean Err: 10.50px | Hand Acc: 96.2%
Epoch [66/100] (38.8s) | Train: 1.8451 | Val: 1.9700 | Val PCK@0.05: 63.57% | PCK@0.10: 88.02% | Mean Err: 10.50px | Hand Acc: 96.1%
Epoch [67/100] (38.8s) | Train: 1.8355 | Val: 1.9655 | Val PCK@0.05: 64.04% | PCK@0.10: 88.19% | Mean Err: 10.43px | Hand Acc: 96.3%
Epoch [68/100] (37.9s) | Train: 1.8328 | Val: 1.9655 | Val PCK@0.05: 63.76% | PCK@0.10: 88.23% | Mean Err: 10.41px | Hand Acc: 96.3%
  [*] New best checkpoint saved! (Mean Err: 10.41px, PCK@0.05: 63.76%)
Epoch [69/100] (37.9s) | Train: 1.8300 | Val: 1.9665 | Val PCK@0.05: 63.94% | PCK@0.10: 88.03% | Mean Err: 10.48px | Hand Acc: 96.3%
Epoch [70/100] (38.8s) | Train: 1.8337 | Val: 1.9616 | Val PCK@0.05: 63.75% | PCK@0.10: 88.05% | Mean Err: 10.45px | Hand Acc: 96.3%
Epoch [71/100] (38.7s) | Train: 1.8368 | Val: 1.9648 | Val PCK@0.05: 63.89% | PCK@0.10: 87.89% | Mean Err: 10.40px | Hand Acc: 96.1%
  [*] New best checkpoint saved! (Mean Err: 10.40px, PCK@0.05: 63.89%)
Epoch [72/100] (37.8s) | Train: 1.8262 | Val: 1.9661 | Val PCK@0.05: 64.13% | PCK@0.10: 87.89% | Mean Err: 10.42px | Hand Acc: 96.2%
Epoch [73/100] (38.6s) | Train: 1.8271 | Val: 1.9671 | Val PCK@0.05: 63.89% | PCK@0.10: 87.89% | Mean Err: 10.45px | Hand Acc: 96.3%
Epoch [74/100] (38.5s) | Train: 1.8298 | Val: 1.9637 | Val PCK@0.05: 64.22% | PCK@0.10: 88.10% | Mean Err: 10.35px | Hand Acc: 96.2%
  [*] New best checkpoint saved! (Mean Err: 10.35px, PCK@0.05: 64.22%)
Epoch [75/100] (38.5s) | Train: 1.8216 | Val: 1.9594 | Val PCK@0.05: 64.35% | PCK@0.10: 87.97% | Mean Err: 10.30px | Hand Acc: 96.2%
  [*] New best checkpoint saved! (Mean Err: 10.30px, PCK@0.05: 64.35%)
Epoch [76/100] (39.5s) | Train: 1.8188 | Val: 1.9739 | Val PCK@0.05: 63.83% | PCK@0.10: 87.91% | Mean Err: 10.41px | Hand Acc: 96.1%
Epoch [77/100] (38.3s) | Train: 1.8274 | Val: 1.9594 | Val PCK@0.05: 64.04% | PCK@0.10: 88.11% | Mean Err: 10.33px | Hand Acc: 96.2%
Epoch [78/100] (38.6s) | Train: 1.8114 | Val: 1.9568 | Val PCK@0.05: 64.10% | PCK@0.10: 88.11% | Mean Err: 10.40px | Hand Acc: 96.4%
Epoch [79/100] (38.3s) | Train: 1.8110 | Val: 1.9621 | Val PCK@0.05: 64.23% | PCK@0.10: 88.14% | Mean Err: 10.37px | Hand Acc: 96.4%
Epoch [80/100] (38.8s) | Train: 1.8150 | Val: 1.9576 | Val PCK@0.05: 64.48% | PCK@0.10: 88.26% | Mean Err: 10.31px | Hand Acc: 96.1%
Epoch [81/100] (38.8s) | Train: 1.8103 | Val: 1.9580 | Val PCK@0.05: 64.13% | PCK@0.10: 88.15% | Mean Err: 10.36px | Hand Acc: 96.3%
Epoch [82/100] (39.1s) | Train: 1.8186 | Val: 1.9609 | Val PCK@0.05: 64.12% | PCK@0.10: 88.28% | Mean Err: 10.36px | Hand Acc: 96.2%
Epoch [83/100] (39.0s) | Train: 1.8116 | Val: 1.9634 | Val PCK@0.05: 64.21% | PCK@0.10: 88.17% | Mean Err: 10.36px | Hand Acc: 96.1%
Epoch [84/100] (38.8s) | Train: 1.8146 | Val: 1.9632 | Val PCK@0.05: 64.37% | PCK@0.10: 88.25% | Mean Err: 10.37px | Hand Acc: 96.3%
Epoch [85/100] (38.9s) | Train: 1.8123 | Val: 1.9609 | Val PCK@0.05: 64.23% | PCK@0.10: 88.12% | Mean Err: 10.38px | Hand Acc: 96.3%
Epoch [86/100] (38.5s) | Train: 1.8111 | Val: 1.9599 | Val PCK@0.05: 64.37% | PCK@0.10: 88.19% | Mean Err: 10.36px | Hand Acc: 96.3%
Epoch [87/100] (38.6s) | Train: 1.8058 | Val: 1.9592 | Val PCK@0.05: 64.22% | PCK@0.10: 88.12% | Mean Err: 10.32px | Hand Acc: 96.5%
Epoch [88/100] (39.2s) | Train: 1.8094 | Val: 1.9596 | Val PCK@0.05: 64.28% | PCK@0.10: 88.19% | Mean Err: 10.36px | Hand Acc: 96.5%
Epoch [89/100] (38.7s) | Train: 1.8143 | Val: 1.9621 | Val PCK@0.05: 64.20% | PCK@0.10: 88.15% | Mean Err: 10.35px | Hand Acc: 96.2%
Epoch [90/100] (38.8s) | Train: 1.8133 | Val: 1.9594 | Val PCK@0.05: 64.31% | PCK@0.10: 88.18% | Mean Err: 10.36px | Hand Acc: 96.3%
Epoch [91/100] (39.1s) | Train: 1.8043 | Val: 1.9583 | Val PCK@0.05: 64.25% | PCK@0.10: 88.15% | Mean Err: 10.34px | Hand Acc: 96.2%
Epoch [92/100] (39.0s) | Train: 1.8095 | Val: 1.9590 | Val PCK@0.05: 64.29% | PCK@0.10: 88.15% | Mean Err: 10.32px | Hand Acc: 96.3%
Epoch [93/100] (39.1s) | Train: 1.8075 | Val: 1.9601 | Val PCK@0.05: 64.34% | PCK@0.10: 88.19% | Mean Err: 10.32px | Hand Acc: 96.2%
Epoch [94/100] (40.1s) | Train: 1.8013 | Val: 1.9586 | Val PCK@0.05: 64.27% | PCK@0.10: 88.31% | Mean Err: 10.28px | Hand Acc: 96.1%
  [*] New best checkpoint saved! (Mean Err: 10.28px, PCK@0.05: 64.27%)
Epoch [95/100] (38.2s) | Train: 1.8051 | Val: 1.9589 | Val PCK@0.05: 64.31% | PCK@0.10: 88.29% | Mean Err: 10.34px | Hand Acc: 96.3%
Epoch [96/100] (38.7s) | Train: 1.8036 | Val: 1.9616 | Val PCK@0.05: 64.08% | PCK@0.10: 88.33% | Mean Err: 10.34px | Hand Acc: 96.4%
Epoch [97/100] (38.9s) | Train: 1.7965 | Val: 1.9614 | Val PCK@0.05: 64.11% | PCK@0.10: 88.17% | Mean Err: 10.35px | Hand Acc: 96.4%
Epoch [98/100] (40.0s) | Train: 1.8097 | Val: 1.9600 | Val PCK@0.05: 64.20% | PCK@0.10: 88.29% | Mean Err: 10.32px | Hand Acc: 96.4%
Epoch [99/100] (38.6s) | Train: 1.8008 | Val: 1.9601 | Val PCK@0.05: 64.21% | PCK@0.10: 88.21% | Mean Err: 10.35px | Hand Acc: 96.4%
Epoch [100/100] (38.7s) | Train: 1.7996 | Val: 1.9604 | Val PCK@0.05: 64.28% | PCK@0.10: 88.18% | Mean Err: 10.34px | Hand Acc: 96.4%

[OK] Training completed. Best checkpoint: /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best.pt (Mean Err: 10.28px, PCK@0.05: 64.27%)
