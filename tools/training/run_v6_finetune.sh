#!/usr/bin/env bash
# ==============================================================================
# tools/training/run_v6_finetune.sh
# ──────────────────────────────────
# Complete End-to-End Pipeline for:
#   1. Merging Base Dataset + V6 Hard Poses + Negative Part 2 Distractors
#   2. Safe Fine-Tuning of Stage 1: NanoDet-Plus (improved4 -> improved5)
#   3. Safe Fine-Tuning of Stage 2: RTMPose-13KP (best_with_hand -> improved5)
#   4. Handedness Calibration (Phase B)
#   5. Exporting Stage 1 & Stage 2 to Qualcomm ONNX (Opset 12)
# ==============================================================================

set -e

# Default directory variables (configurable)
BASE_DIR="${BASE_DIR:-/content/Shoes_VTO/data/merged_v4_1_v4_2_v5_raw_13kp_with_negative3}"
V6_DIR="${V6_DIR:-/content/Shoes_VTO/data/shuffled_v6_raw_13kp}"
NEG_DIR="${NEG_DIR:-/content/shoes_negative_part2/images}"
MERGED_DIR="${MERGED_DIR:-/content/Shoes_VTO/data/merged_v6_with_negatives}"

CKPT_STAGE1_PREV="${CKPT_STAGE1_PREV:-/content/Shoes_VTO/checkpoints/nanodet_stage_a_improved4/best.pt}"
CKPT_STAGE2_PREV="${CKPT_STAGE2_PREV:-/content/Shoes_VTO/checkpoints/rtmpose_13kp_official_coco_full_data/best_with_hand.pt}"

OUT_STAGE1_DIR="${OUT_STAGE1_DIR:-/content/Shoes_VTO/checkpoints/nanodet_stage_a_improved5}"
OUT_STAGE2_DIR="${OUT_STAGE2_DIR:-/content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5}"

ONNX_STAGE1_OUT="${ONNX_STAGE1_OUT:-/content/Shoes_VTO/deliverables/qualcomm_experiments/models/nanodet_stage_a_improved5_opset12.onnx}"
ONNX_STAGE2_OUT="${ONNX_STAGE2_OUT:-/content/Shoes_VTO/deliverables/qualcomm_experiments/models/rtmpose_13kp_hand_improved5_opset12.onnx}"

echo "========================================================================"
echo "  SHOES VTO: V6 DATASET MERGE & SAFE FINE-TUNING PIPELINE"
echo "========================================================================"
echo "  Base Dataset:       $BASE_DIR"
echo "  V6 Dataset:         $V6_DIR"
echo "  Negative Part 2:    $NEG_DIR"
echo "  Target Merged Data: $MERGED_DIR"
echo "  Stage 1 Pretrained: $CKPT_STAGE1_PREV"
echo "  Stage 2 Pretrained: $CKPT_STAGE2_PREV"
echo "========================================================================"

# ------------------------------------------------------------------------------
# STEP 1: Merge Datasets and Generate Negative Labels
# ------------------------------------------------------------------------------
echo -e "\n[*] STEP 1: Merging Datasets & Generating Empty Negative Labels..."
python tools/dataset/merge_and_prepare_v6_dataset.py \
  --base_dir "$BASE_DIR" \
  --v6_dir "$V6_DIR" \
  --neg_images_dir "$NEG_DIR" \
  --output_dir "$MERGED_DIR" \
  --seed 42

# ------------------------------------------------------------------------------
# STEP 2: Stage 1 NanoDet-Plus Foot Detector Safe Fine-Tuning
# ------------------------------------------------------------------------------
echo -e "\n[*] STEP 2: Fine-Tuning Stage 1 NanoDet-Plus (improved4 -> improved5)..."
python src/models/qualcomm_experiments/nanodet/train.py \
  --data "$MERGED_DIR" \
  --output "$OUT_STAGE1_DIR" \
  --finetune "$CKPT_STAGE1_PREV" \
  --epochs 40 \
  --batch 16 \
  --lr 3e-4 \
  --neck rep_pan \
  --act relu \
  --reg_max 16 \
  --loss_bbox ciou \
  --loss_bbox_weight 3.0 \
  --loss_dfl_weight 0.15 \
  --neg_penalty 1.5 \
  --amp

# ------------------------------------------------------------------------------
# STEP 3: Stage 2 RTMPose-13KP Pose & Chirality Safe Fine-Tuning
# ------------------------------------------------------------------------------
echo -e "\n[*] STEP 3: Fine-Tuning Stage 2 RTMPose-13KP (CSPNeXt-Tiny + SimCC)..."
python src/models/qualcomm_experiments/rtmpose/train.py \
  --data "$MERGED_DIR" \
  --output "$OUT_STAGE2_DIR" \
  --init_weights "$CKPT_STAGE2_PREV" \
  --epochs 40 \
  --batch 32 \
  --lr 3e-4 \
  --warmup 3 \
  --include_negatives \
  --amp

# ------------------------------------------------------------------------------
# STEP 4: Phase B Handedness Head Fast Calibration (10 Epochs)
# ------------------------------------------------------------------------------
echo -e "\n[*] STEP 4: Fine-Tuning In-Model Handedness Head (Phase B Calibration)..."
python src/models/qualcomm_experiments/rtmpose/train_handedness.py \
  --data "$MERGED_DIR" \
  --init_weights "$OUT_STAGE2_DIR/best.pt" \
  --output_path "$OUT_STAGE2_DIR/best_with_hand.pt" \
  --epochs 10 \
  --batch 64 \
  --lr 1e-3

# ------------------------------------------------------------------------------
# STEP 5: Export Fine-Tuned Models to Qualcomm ONNX (Opset 12)
# ------------------------------------------------------------------------------
echo -e "\n[*] STEP 5: Exporting Fine-Tuned Models to Qualcomm ONNX (Opset 12)..."

mkdir -p "$(dirname "$ONNX_STAGE1_OUT")"
mkdir -p "$(dirname "$ONNX_STAGE2_OUT")"

# Export Stage 1 NanoDet
python src/models/qualcomm_experiments/nanodet/export_onnx.py \
  --weights "$OUT_STAGE1_DIR/best.pt" \
  --output "$ONNX_STAGE1_OUT" \
  --neck rep_pan \
  --act relu \
  --reg_max 16

# Export Stage 2 RTMPose
python src/models/qualcomm_experiments/rtmpose/export_onnx.py \
  --weights "$OUT_STAGE2_DIR/best_with_hand.pt" \
  --output "$ONNX_STAGE2_OUT" \
  --include_handedness

echo -e "\n========================================================================"
echo "  V6 MERGE & FINE-TUNING PIPELINE SUCCESSFULLY COMPLETED!"
echo "  Stage 1 Best Weights: $OUT_STAGE1_DIR/best.pt"
echo "  Stage 2 Best Weights: $OUT_STAGE2_DIR/best_with_hand.pt"
echo "  Stage 1 ONNX:         $ONNX_STAGE1_OUT"
echo "  Stage 2 ONNX:         $ONNX_STAGE2_OUT"
echo "========================================================================"

