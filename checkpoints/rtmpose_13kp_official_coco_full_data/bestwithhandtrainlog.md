F:\Machine learning\Shoes_VTO\src\models\qualcomm_experiments\rtmpose\train_handedness.py:154: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
F:\Machine learning\Shoes_VTO\src\models\qualcomm_experiments\rtmpose\train_handedness.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):

===========================================================================
  PHASE 2: FAST FROZEN HANDEDNESS (LEFT/RIGHT) HEAD CALIBRATION
  Init Checkpoint: checkpoints/rtmpose_13kp_official_coco_full_data/best.pt
  Output Model:    checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt
  Dataset:         data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3
  Device:          cuda (Batch: 64, LR: 0.001)
===========================================================================
[*] Loaded base weights from checkpoints/rtmpose_13kp_official_coco_full_data/best.pt
    Missing keys (expected for new hand head): ['head.hand.1.weight', 'head.hand.1.bias', 'head.hand.4.weight', 'head.hand.4.bias']
[+] Frozen parameters: 3,299,628 (eval mode, zero drift on keypoints)
[+] Trainable parameters: 426,241 (head.hand only)
[*] Training started: 10366 train crops, 1294 val crops

Epoch [01/08] (101.9s) | Train Loss: 0.2880 | Train Acc: 87.54% | Val Loss: 0.1699 | Val Acc: 93.82% [*] BEST!
Epoch [02/08] (110.4s) | Train Loss: 0.1940 | Train Acc: 92.19% | Val Loss: 0.1655 | Val Acc: 93.89% [*] BEST!
Epoch [03/08] (101.5s) | Train Loss: 0.1805 | Train Acc: 93.27% | Val Loss: 0.1544 | Val Acc: 94.13% [*] BEST!
Epoch [04/08] (99.2s) | Train Loss: 0.1673 | Train Acc: 93.88% | Val Loss: 0.1460 | Val Acc: 95.21% [*] BEST!
Epoch [05/08] (100.1s) | Train Loss: 0.1592 | Train Acc: 94.20% | Val Loss: 0.1463 | Val Acc: 94.51%
Epoch [06/08] (105.8s) | Train Loss: 0.1512 | Train Acc: 94.67% | Val Loss: 0.1501 | Val Acc: 94.36%
Epoch [07/08] (102.0s) | Train Loss: 0.1545 | Train Acc: 94.50% | Val Loss: 0.1464 | Val Acc: 95.29% [*] BEST!
Epoch [08/08] (102.3s) | Train Loss: 0.1492 | Train Acc: 94.85% | Val Loss: 0.1476 | Val Acc: 94.82%

===========================================================================
  CALIBRATION COMPLETE: Best Val Hand Acc: 95.29%
  Checkpoint saved: checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt
===========================================================================