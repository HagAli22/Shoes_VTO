===========================================================================
  PHASE 2: FAST FROZEN HANDEDNESS (LEFT/RIGHT) HEAD CALIBRATION
  Init Checkpoint: /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best.pt
  Output Model:    /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best_with_hand.pt
  Dataset:         /content/Shoes_VTO/data/merged_v6_with_negatives
  Device:          cuda (Batch: 64, LR: 0.001)
===========================================================================
[*] Loaded base weights from /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best.pt
    Missing keys (expected for new hand head): []
[+] Frozen parameters: 3,299,628 (eval mode, zero drift on keypoints)
[+] Trainable parameters: 426,241 (head.hand only)
[*] Training started: 13842 train crops, 1725 val crops

/content/Shoes_VTO/src/models/qualcomm_experiments/rtmpose/train_handedness.py:154: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
/content/Shoes_VTO/src/models/qualcomm_experiments/rtmpose/train_handedness.py:174: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
Epoch [01/40] (38.8s) | Train Loss: 0.0802 | Train Acc: 97.13% | Val Loss: 0.1652 | Val Acc: 95.42% [*] BEST!
Epoch [02/40] (37.6s) | Train Loss: 0.0797 | Train Acc: 97.31% | Val Loss: 0.1553 | Val Acc: 95.83% [*] BEST!
Epoch [03/40] (38.5s) | Train Loss: 0.0803 | Train Acc: 97.28% | Val Loss: 0.1462 | Val Acc: 96.17% [*] BEST!
Epoch [04/40] (37.6s) | Train Loss: 0.0822 | Train Acc: 97.12% | Val Loss: 0.1594 | Val Acc: 95.71%
Epoch [05/40] (37.2s) | Train Loss: 0.0797 | Train Acc: 97.16% | Val Loss: 0.1602 | Val Acc: 95.88%
Epoch [06/40] (37.6s) | Train Loss: 0.0809 | Train Acc: 97.30% | Val Loss: 0.1553 | Val Acc: 95.88%
Epoch [07/40] (37.2s) | Train Loss: 0.0777 | Train Acc: 97.37% | Val Loss: 0.1671 | Val Acc: 95.54%
Epoch [08/40] (37.9s) | Train Loss: 0.0789 | Train Acc: 97.34% | Val Loss: 0.1529 | Val Acc: 95.94%
Epoch [09/40] (38.4s) | Train Loss: 0.0784 | Train Acc: 97.40% | Val Loss: 0.1513 | Val Acc: 95.83%
Epoch [10/40] (37.7s) | Train Loss: 0.0788 | Train Acc: 97.25% | Val Loss: 0.1739 | Val Acc: 95.25%
Epoch [11/40] (37.7s) | Train Loss: 0.0782 | Train Acc: 97.27% | Val Loss: 0.1530 | Val Acc: 95.59%
Epoch [12/40] (37.5s) | Train Loss: 0.0737 | Train Acc: 97.57% | Val Loss: 0.1474 | Val Acc: 96.17%
Epoch [13/40] (37.7s) | Train Loss: 0.0763 | Train Acc: 97.39% | Val Loss: 0.1817 | Val Acc: 95.48%
Epoch [14/40] (37.8s) | Train Loss: 0.0776 | Train Acc: 97.40% | Val Loss: 0.1569 | Val Acc: 95.83%
Epoch [15/40] (36.9s) | Train Loss: 0.0759 | Train Acc: 97.51% | Val Loss: 0.1547 | Val Acc: 96.41% [*] BEST!
Epoch [16/40] (37.7s) | Train Loss: 0.0762 | Train Acc: 97.33% | Val Loss: 0.1522 | Val Acc: 96.06%
Epoch [17/40] (37.4s) | Train Loss: 0.0749 | Train Acc: 97.61% | Val Loss: 0.1609 | Val Acc: 95.94%
Epoch [18/40] (37.8s) | Train Loss: 0.0769 | Train Acc: 97.56% | Val Loss: 0.1627 | Val Acc: 95.65%
Epoch [19/40] (38.4s) | Train Loss: 0.0709 | Train Acc: 97.66% | Val Loss: 0.1569 | Val Acc: 95.94%
Epoch [20/40] (37.3s) | Train Loss: 0.0737 | Train Acc: 97.55% | Val Loss: 0.1544 | Val Acc: 96.35%
Epoch [21/40] (37.5s) | Train Loss: 0.0746 | Train Acc: 97.51% | Val Loss: 0.1480 | Val Acc: 96.06%
Epoch [22/40] (37.7s) | Train Loss: 0.0776 | Train Acc: 97.39% | Val Loss: 0.1597 | Val Acc: 95.42%
Epoch [23/40] (37.4s) | Train Loss: 0.0771 | Train Acc: 97.43% | Val Loss: 0.1486 | Val Acc: 95.77%
Epoch [24/40] (37.4s) | Train Loss: 0.0742 | Train Acc: 97.59% | Val Loss: 0.1495 | Val Acc: 96.06%
Epoch [25/40] (37.6s) | Train Loss: 0.0687 | Train Acc: 97.62% | Val Loss: 0.1635 | Val Acc: 95.59%
Epoch [26/40] (37.3s) | Train Loss: 0.0696 | Train Acc: 97.62% | Val Loss: 0.1613 | Val Acc: 96.06%
Epoch [27/40] (38.5s) | Train Loss: 0.0714 | Train Acc: 97.51% | Val Loss: 0.1623 | Val Acc: 96.00%
Epoch [28/40] (37.7s) | Train Loss: 0.0718 | Train Acc: 97.60% | Val Loss: 0.1682 | Val Acc: 95.42%
Epoch [29/40] (37.8s) | Train Loss: 0.0771 | Train Acc: 97.36% | Val Loss: 0.1652 | Val Acc: 95.36%
Epoch [30/40] (37.6s) | Train Loss: 0.0735 | Train Acc: 97.46% | Val Loss: 0.1749 | Val Acc: 95.48%
Epoch [31/40] (37.5s) | Train Loss: 0.0735 | Train Acc: 97.48% | Val Loss: 0.1680 | Val Acc: 95.54%
Epoch [32/40] (37.1s) | Train Loss: 0.0698 | Train Acc: 97.54% | Val Loss: 0.1698 | Val Acc: 95.65%
Epoch [33/40] (37.7s) | Train Loss: 0.0723 | Train Acc: 97.42% | Val Loss: 0.1693 | Val Acc: 95.77%
Epoch [34/40] (38.2s) | Train Loss: 0.0735 | Train Acc: 97.56% | Val Loss: 0.1564 | Val Acc: 96.29%
Epoch [35/40] (37.2s) | Train Loss: 0.0695 | Train Acc: 97.51% | Val Loss: 0.1524 | Val Acc: 96.29%
Epoch [36/40] (37.9s) | Train Loss: 0.0724 | Train Acc: 97.70% | Val Loss: 0.1536 | Val Acc: 96.06%
Epoch [37/40] (37.9s) | Train Loss: 0.0691 | Train Acc: 97.64% | Val Loss: 0.1650 | Val Acc: 96.06%
Epoch [38/40] (37.2s) | Train Loss: 0.0698 | Train Acc: 97.70% | Val Loss: 0.1577 | Val Acc: 96.00%
Epoch [39/40] (37.1s) | Train Loss: 0.0694 | Train Acc: 97.64% | Val Loss: 0.1661 | Val Acc: 95.83%
Epoch [40/40] (37.8s) | Train Loss: 0.0724 | Train Acc: 97.44% | Val Loss: 0.1556 | Val Acc: 95.88%

===========================================================================
  CALIBRATION COMPLETE: Best Val Hand Acc: 96.41%
  Checkpoint saved: /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best_with_hand.pt
===========================================================================
