"""
tools/training/colab_finetune_cells.py
──────────────────────────────────────
Ready-to-run code snippets for Google Colab.
Copy-paste each block into a separate notebook cell.
"""

# ==============================================================================
# CELL 1: Mount Google Drive & Environment Setup
# ==============================================================================
CELL_1 = """
# Cell 1: Mount Drive & Verify GPU
import os, torch
from google.colab import drive
drive.mount('/content/drive')

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
"""

# ==============================================================================
# CELL 2: Merge V6 Dataset & Split Negatives with Empty Labels
# ==============================================================================
CELL_2 = """
# Cell 2: Merge Base + V6 Hard Poses + 782 Negative Images
%cd /content/Shoes_VTO

!python tools/dataset/merge_and_prepare_v6_dataset.py \
  --base_dir /content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3 \
  --v6_dir /content/Shoes_VTO/data/shuffled_v6_raw_13kp \
  --neg_images_dir /content/shoes_negative_part2/images \
  --output_dir /content/Shoes_VTO/data/merged_v6_with_negatives \
  --seed 42
"""

# ==============================================================================
# CELL 3: Safe Fine-Tuning Stage 1 NanoDet-Plus (improved4 -> improved5)
# ==============================================================================
CELL_3 = """
# Cell 3: Safe Fine-Tuning Stage 1 NanoDet (Warm-start from improved4)
# Preserves existing knowledge with differential LR (backbone=3e-5, neck=1.5e-4, head=3e-4)
!python src/models/qualcomm_experiments/nanodet/train.py \
  --data /content/Shoes_VTO/data/merged_v6_with_negatives \
  --output /content/Shoes_VTO/checkpoints/nanodet_stage_a_improved5 \
  --finetune /content/Shoes_VTO/checkpoints/nanodet_stage_a_improved4/best.pt \
  --epochs 40 \
  --batch 16 \
  --lr 3e-4 \
  --neck rep_pan \
  --act relu \
  --reg_max 16 \
  --loss_bbox ciou \
  --neg_penalty 1.5 \
  --amp
"""

# ==============================================================================
# CELL 4: Safe Fine-Tuning Stage 2 RTMPose-13KP (CSPNeXt-Tiny + SimCC)
# ==============================================================================
CELL_4 = """
# Cell 4: Safe Fine-Tuning Stage 2 RTMPose (Warm-start from best_with_hand.pt)
# allow_flip=False preserves chirality; include_negatives=True enforces background suppression
!python src/models/qualcomm_experiments/rtmpose/train.py \
  --data /content/Shoes_VTO/data/merged_v6_with_negatives \
  --output /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5 \
  --init_weights /content/Shoes_VTO/checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt \
  --epochs 40 \
  --batch 32 \
  --lr 3e-4 \
  --warmup 3 \
  --include_negatives \
  --amp
"""

# ==============================================================================
# CELL 5: Phase B Handedness Head Fast Calibration (10 Epochs)
# ==============================================================================
CELL_5 = """
# Cell 5: Calibrate In-Model Left/Right Handedness Head
!python src/models/qualcomm_experiments/rtmpose/train_handedness.py \
  --data /content/Shoes_VTO/data/merged_v6_with_negatives \
  --init_weights /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best.pt \
  --output_path /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best_with_hand.pt \
  --epochs 10 \
  --batch 64 \
  --lr 1e-3
"""

# ==============================================================================
# CELL 6: Export Both Models to Qualcomm ONNX (Opset 12)
# ==============================================================================
CELL_6 = """
# Cell 6: Export to Qualcomm ONNX (Opset 12)
# Export Stage 1 NanoDet-Plus
!python src/models/qualcomm_experiments/nanodet/export_onnx.py \
  --weights /content/Shoes_VTO/checkpoints/nanodet_stage_a_improved5/best.pt \
  --output /content/Shoes_VTO/deliverables/qualcomm_experiments/models/nanodet_stage_a_improved5_opset12.onnx \
  --neck rep_pan \
  --act relu \
  --reg_max 16

# Export Stage 2 RTMPose-13KP with Hand Head
!python src/models/qualcomm_experiments/rtmpose/export_onnx.py \
  --weights /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best_with_hand.pt \
  --output /content/Shoes_VTO/deliverables/qualcomm_experiments/models/rtmpose_13kp_hand_improved5_opset12.onnx
"""

# ==============================================================================
# CELL 7: Option A - Distill RTMPose-Slim (Stride-16, 3.04 MB FP32 / 0.76 MB INT8)
# ==============================================================================
CELL_7 = """
# Cell 7: Option A: Distill RTMPose-Slim (Stride-16, 3.04 MB) from improved5
# Transfers 100% of Stem + Stages 1-3 weights (79.3% of student model warm-started)
# Distills soft SimCC coordinates (tau=2.0) and chirality probability
!python src/models/qualcomm_experiments/rtmpose/train_distill.py \
  --data /content/Shoes_VTO/data/merged_v6_with_negatives \
  --teacher /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best_with_hand.pt \
  --student_arch cspnext_slim \
  --output /content/Shoes_VTO/checkpoints/rtmpose_13kp_slim_distill \
  --epochs 40 \
  --batch 32 \
  --lr 3e-4 \
  --temperature 2.0 \
  --alpha_kd 1.0 \
  --warmup 3 \
  --workers 4 \
  --amp
"""

# ==============================================================================
# CELL 8: Option A+Lite - Distill RTMPose-Slim-Stage4Lite (4.25 MB FP32 / 1.06 MB INT8)
# ==============================================================================
CELL_8 = """
# Cell 8: Option A+Lite: Distill RTMPose-Slim-Stage4Lite (Stride-32, 4.25 MB)
# Includes lightweight Depthwise-Separable Stage 4 (cutting 81.6% of Stage 4 params)
!python src/models/qualcomm_experiments/rtmpose/train_distill.py \
  --data /content/Shoes_VTO/data/merged_v6_with_negatives \
  --teacher /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best_with_hand.pt \
  --student_arch cspnext_slim_s4lite \
  --output /content/Shoes_VTO/checkpoints/rtmpose_13kp_slim_s4lite_distill \
  --epochs 40 \
  --batch 32 \
  --lr 3e-4 \
  --temperature 2.0 \
  --alpha_kd 1.0 \
  --warmup 3 \
  --workers 4 \
  --amp
"""

# ==============================================================================
# CELL 9: Export Compressed RTMPose Models to ONNX (FP32, FP16, and INT8)
# ==============================================================================
CELL_9 = """
# Cell 9: Export Compressed Models to ONNX (FP32, FP16, and INT8)
# Export Option A: RTMPose-Slim (3.04 MB FP32 / 1.52 MB FP16 / 0.76 MB INT8)
!python src/models/qualcomm_experiments/rtmpose/export_onnx.py \
  --weights /content/Shoes_VTO/checkpoints/rtmpose_13kp_slim_distill/best_with_hand.pt \
  --output /content/Shoes_VTO/deliverables/qualcomm_experiments/models/rtmpose_slim_improved5_opset12.onnx \
  --backbone cspnext_slim

# Export Option A+Lite: RTMPose-Slim-Stage4Lite (4.25 MB FP32 / 2.12 MB FP16 / 1.06 MB INT8)
!python src/models/qualcomm_experiments/rtmpose/export_onnx.py \
  --weights /content/Shoes_VTO/checkpoints/rtmpose_13kp_slim_s4lite_distill/best_with_hand.pt \
  --output /content/Shoes_VTO/deliverables/qualcomm_experiments/models/rtmpose_slim_s4lite_improved5_opset12.onnx \
  --backbone cspnext_slim_s4lite
"""

# ==============================================================================
# CELL 10: Benchmark Compressed Models vs Baseline on Validation Set
# ==============================================================================
CELL_10 = """
# Cell 10: Comparative Accuracy & Latency Benchmark
!python tools/evaluation/benchmark_rtmpose_compressed.py \
  --data /content/Shoes_VTO/data/merged_v6_with_negatives/valid/images \
  --teacher /content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best_with_hand.pt \
  --output /content/Shoes_VTO/deliverables/qualcomm_experiments/rtmpose_compression_benchmark.json
"""


