(yolo) PS F:\Machine learning\Shoes_VTO> python src/models/qualcomm_experiments/rtmpose/train.py --data data/shuffled_v4_python src/models/qualcomm_experiments/rtmpose/train.py --data data/shuffled_v4_13kp --output checkpoints/rtmpose_13kp_pretrained --epochs 100 --batch 32 --lr 0.0005 --device cuda --backbone mobilenetv3 --pretrained
======================================================================
  EXPERIMENT 2: Qualcomm RTMPose-13KP Foot Pose (SimCC) Training
  Dataset:     data\shuffled_v4_13kp
  Backbone:    mobilenetv3 (Pretrained: True)
  Device:      cuda
  Epochs:      100 | Batch: 32 | LR: 0.0005
  Resolution:  256x256 (SimCC Bins: 512x512)
======================================================================
Epoch [01/100] (68.9s) | Train Loss: 5.7540 | Val Loss: 6.7843
  [*] New best checkpoint saved (Val Loss: 6.7843)
Epoch [02/100] (81.7s) | Train Loss: 5.0682 | Val Loss: 5.0959
  [*] New best checkpoint saved (Val Loss: 5.0959)
Epoch [03/100] (91.4s) | Train Loss: 4.7568 | Val Loss: 4.7764
  [*] New best checkpoint saved (Val Loss: 4.7764)
Epoch [04/100] (84.7s) | Train Loss: 4.4852 | Val Loss: 4.6622
  [*] New best checkpoint saved (Val Loss: 4.6622)
Epoch [05/100] (72.5s) | Train Loss: 4.2384 | Val Loss: 4.3884
  [*] New best checkpoint saved (Val Loss: 4.3884)
Epoch [06/100] (71.9s) | Train Loss: 4.0847 | Val Loss: 4.1909
  [*] New best checkpoint saved (Val Loss: 4.1909)
Epoch [07/100] (72.5s) | Train Loss: 3.9693 | Val Loss: 4.0894
  [*] New best checkpoint saved (Val Loss: 4.0894)
Epoch [08/100] (71.9s) | Train Loss: 3.8723 | Val Loss: 3.9754
  [*] New best checkpoint saved (Val Loss: 3.9754)
Epoch [09/100] (71.8s) | Train Loss: 3.7957 | Val Loss: 3.9088
  [*] New best checkpoint saved (Val Loss: 3.9088)
Epoch [10/100] (76.9s) | Train Loss: 3.7203 | Val Loss: 3.8819
  [*] New best checkpoint saved (Val Loss: 3.8819)
Epoch [11/100] (81.3s) | Train Loss: 3.6380 | Val Loss: 3.8015
  [*] New best checkpoint saved (Val Loss: 3.8015)
Epoch [12/100] (88.6s) | Train Loss: 3.5789 | Val Loss: 3.7511
  [*] New best checkpoint saved (Val Loss: 3.7511)
Epoch [13/100] (86.3s) | Train Loss: 3.4966 | Val Loss: 3.7384
  [*] New best checkpoint saved (Val Loss: 3.7384)
Epoch [14/100] (77.9s) | Train Loss: 3.4387 | Val Loss: 3.6903
  [*] New best checkpoint saved (Val Loss: 3.6903)
Epoch [15/100] (75.7s) | Train Loss: 3.3734 | Val Loss: 3.6674
  [*] New best checkpoint saved (Val Loss: 3.6674)
Epoch [16/100] (75.3s) | Train Loss: 3.3185 | Val Loss: 3.6473
  [*] New best checkpoint saved (Val Loss: 3.6473)
Epoch [17/100] (74.9s) | Train Loss: 3.2762 | Val Loss: 3.5992
  [*] New best checkpoint saved (Val Loss: 3.5992)
Epoch [18/100] (75.3s) | Train Loss: 3.2115 | Val Loss: 3.6557
Epoch [19/100] (75.9s) | Train Loss: 3.1568 | Val Loss: 3.5964
  [*] New best checkpoint saved (Val Loss: 3.5964)
Epoch [20/100] (74.8s) | Train Loss: 3.1017 | Val Loss: 3.5550
  [*] New best checkpoint saved (Val Loss: 3.5550)
Epoch [21/100] (75.4s) | Train Loss: 3.0462 | Val Loss: 3.5161
  [*] New best checkpoint saved (Val Loss: 3.5161)
Epoch [22/100] (74.7s) | Train Loss: 2.9995 | Val Loss: 3.5019
  [*] New best checkpoint saved (Val Loss: 3.5019)
Epoch [23/100] (76.1s) | Train Loss: 2.9397 | Val Loss: 3.5067
Epoch [24/100] (74.1s) | Train Loss: 2.8963 | Val Loss: 3.4596
  [*] New best checkpoint saved (Val Loss: 3.4596)
Epoch [25/100] (76.7s) | Train Loss: 2.8439 | Val Loss: 3.4837
Epoch [26/100] (76.1s) | Train Loss: 2.8067 | Val Loss: 3.4536
  [*] New best checkpoint saved (Val Loss: 3.4536)
Epoch [27/100] (77.3s) | Train Loss: 2.7511 | Val Loss: 3.4424
  [*] New best checkpoint saved (Val Loss: 3.4424)
Epoch [28/100] (75.1s) | Train Loss: 2.7265 | Val Loss: 3.4466
Epoch [29/100] (75.4s) | Train Loss: 2.6824 | Val Loss: 3.4292
  [*] New best checkpoint saved (Val Loss: 3.4292)
Epoch [30/100] (73.9s) | Train Loss: 2.6415 | Val Loss: 3.4089
  [*] New best checkpoint saved (Val Loss: 3.4089)
Epoch [31/100] (75.3s) | Train Loss: 2.5859 | Val Loss: 3.3754
  [*] New best checkpoint saved (Val Loss: 3.3754)
Epoch [32/100] (75.3s) | Train Loss: 2.5446 | Val Loss: 3.4136
Epoch [33/100] (75.4s) | Train Loss: 2.5216 | Val Loss: 3.3972
Epoch [34/100] (75.9s) | Train Loss: 2.4932 | Val Loss: 3.3732
  [*] New best checkpoint saved (Val Loss: 3.3732)
Epoch [35/100] (76.7s) | Train Loss: 2.4487 | Val Loss: 3.3808
Epoch [36/100] (74.5s) | Train Loss: 2.4158 | Val Loss: 3.3617
  [*] New best checkpoint saved (Val Loss: 3.3617)
Epoch [37/100] (76.4s) | Train Loss: 2.3865 | Val Loss: 3.3692
Epoch [38/100] (73.8s) | Train Loss: 2.3452 | Val Loss: 3.3610
  [*] New best checkpoint saved (Val Loss: 3.3610)
Epoch [39/100] (76.5s) | Train Loss: 2.3219 | Val Loss: 3.4027
Epoch [40/100] (75.7s) | Train Loss: 2.3005 | Val Loss: 3.3962
Epoch [41/100] (75.3s) | Train Loss: 2.2589 | Val Loss: 3.4050
Epoch [42/100] (75.9s) | Train Loss: 2.2122 | Val Loss: 3.3566
  [*] New best checkpoint saved (Val Loss: 3.3566)
Epoch [43/100] (76.1s) | Train Loss: 2.2004 | Val Loss: 3.3625
Epoch [44/100] (75.2s) | Train Loss: 2.1622 | Val Loss: 3.4051
Epoch [45/100] (75.0s) | Train Loss: 2.1427 | Val Loss: 3.3631
Epoch [46/100] (75.0s) | Train Loss: 2.1267 | Val Loss: 3.3745
Epoch [47/100] (75.7s) | Train Loss: 2.0912 | Val Loss: 3.3753
Epoch [48/100] (80.1s) | Train Loss: 2.0578 | Val Loss: 3.3550
  [*] New best checkpoint saved (Val Loss: 3.3550)
Epoch [49/100] (74.9s) | Train Loss: 2.0318 | Val Loss: 3.3226
  [*] New best checkpoint saved (Val Loss: 3.3226)
Epoch [50/100] (77.5s) | Train Loss: 2.0202 | Val Loss: 3.3314
Epoch [51/100] (76.8s) | Train Loss: 1.9936 | Val Loss: 3.3450
Epoch [52/100] (75.8s) | Train Loss: 1.9729 | Val Loss: 3.3491
Epoch [53/100] (75.8s) | Train Loss: 1.9306 | Val Loss: 3.3450
Epoch [54/100] (76.5s) | Train Loss: 1.9222 | Val Loss: 3.3685
Epoch [55/100] (78.0s) | Train Loss: 1.9155 | Val Loss: 3.3416
Epoch [56/100] (75.5s) | Train Loss: 1.8850 | Val Loss: 3.3655
Epoch [57/100] (77.7s) | Train Loss: 1.8805 | Val Loss: 3.3353
Epoch [58/100] (75.1s) | Train Loss: 1.8547 | Val Loss: 3.3583
Epoch [59/100] (79.5s) | Train Loss: 1.8489 | Val Loss: 3.3391
Epoch [60/100] (74.8s) | Train Loss: 1.8377 | Val Loss: 3.3635
Epoch [61/100] (77.7s) | Train Loss: 1.8048 | Val Loss: 3.3276
Epoch [62/100] (75.9s) | Train Loss: 1.7923 | Val Loss: 3.3624
Epoch [63/100] (75.0s) | Train Loss: 1.7692 | Val Loss: 3.3672
Epoch [64/100] (73.8s) | Train Loss: 1.7730 | Val Loss: 3.3574
Epoch [65/100] (77.7s) | Train Loss: 1.7526 | Val Loss: 3.3531
Epoch [66/100] (76.1s) | Train Loss: 1.7289 | Val Loss: 3.3463
Epoch [67/100] (78.3s) | Train Loss: 1.7103 | Val Loss: 3.3513
Epoch [68/100] (74.7s) | Train Loss: 1.7183 | Val Loss: 3.3590
Epoch [69/100] (74.7s) | Train Loss: 1.6915 | Val Loss: 3.3363
Epoch [70/100] (75.2s) | Train Loss: 1.7038 | Val Loss: 3.3592
Epoch [71/100] (75.8s) | Train Loss: 1.6739 | Val Loss: 3.3659
Epoch [72/100] (74.2s) | Train Loss: 1.6648 | Val Loss: 3.3618
Epoch [73/100] (76.3s) | Train Loss: 1.6737 | Val Loss: 3.3665
Epoch [74/100] (75.4s) | Train Loss: 1.6806 | Val Loss: 3.3626
Epoch [75/100] (78.9s) | Train Loss: 1.6410 | Val Loss: 3.3587
Epoch [76/100] (75.1s) | Train Loss: 1.6351 | Val Loss: 3.3660
Epoch [77/100] (70.7s) | Train Loss: 1.6337 | Val Loss: 3.3660
Epoch [78/100] (75.6s) | Train Loss: 1.6158 | Val Loss: 3.3602
Epoch [79/100] (74.6s) | Train Loss: 1.6162 | Val Loss: 3.3599
Epoch [80/100] (75.6s) | Train Loss: 1.6095 | Val Loss: 3.3707
Epoch [81/100] (73.3s) | Train Loss: 1.6024 | Val Loss: 3.3718
Epoch [82/100] (67.2s) | Train Loss: 1.6007 | Val Loss: 3.3695
Epoch [83/100] (73.6s) | Train Loss: 1.5952 | Val Loss: 3.3712
Epoch [84/100] (73.2s) | Train Loss: 1.5858 | Val Loss: 3.3687
Epoch [85/100] (72.9s) | Train Loss: 1.5878 | Val Loss: 3.3720
Epoch [86/100] (74.3s) | Train Loss: 1.5679 | Val Loss: 3.3713
Epoch [87/100] (71.6s) | Train Loss: 1.5829 | Val Loss: 3.3689
Epoch [88/100] (75.0s) | Train Loss: 1.5778 | Val Loss: 3.3689
Epoch [89/100] (71.4s) | Train Loss: 1.5724 | Val Loss: 3.3658
Epoch [90/100] (74.9s) | Train Loss: 1.5737 | Val Loss: 3.3756
Epoch [91/100] (72.6s) | Train Loss: 1.5603 | Val Loss: 3.3722
Epoch [92/100] (75.1s) | Train Loss: 1.5652 | Val Loss: 3.3759
Epoch [93/100] (73.6s) | Train Loss: 1.5686 | Val Loss: 3.3731
Epoch [94/100] (75.2s) | Train Loss: 1.5663 | Val Loss: 3.3741
Epoch [95/100] (73.8s) | Train Loss: 1.5704 | Val Loss: 3.3792
Epoch [96/100] (76.1s) | Train Loss: 1.5779 | Val Loss: 3.3766
Epoch [97/100] (71.6s) | Train Loss: 1.5597 | Val Loss: 3.3803
Epoch [98/100] (73.0s) | Train Loss: 1.5703 | Val Loss: 3.3820
Epoch [99/100] (75.6s) | Train Loss: 1.5648 | Val Loss: 3.3800
Epoch [100/100] (75.1s) | Train Loss: 1.5429 | Val Loss: 3.3753

[OK] Training completed. Best checkpoint: checkpoints\rtmpose_13kp_pretrained\best.pt
(yolo) PS F:\Machine learning\Shoes_VTO> 