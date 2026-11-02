#!/usr/bin/env bash
# ==============================================================================
# tools/training/run_colab_distill.sh
# ────────────────────────────────────
# Turnkey End-to-End Distillation Runner for Google Colab:
#   1. Distills Option A (RTMPose-Slim Stride 16, 3.04 MB FP32 / 0.76 MB INT8)
#   2. Distills Option A+Lite (RTMPose-Slim-Stage4Lite, 4.25 MB FP32 / 1.06 MB INT8)
#   3. Exports both models to ONNX (FP32, FP16, and INT8)
#   4. Benchmarks accuracy & latency on Colab GPU
# ==============================================================================

set -e

DATA_DIR="${DATA_DIR:-/content/Shoes_VTO/data/merged_v6_with_negatives}"
TEACHER_CKPT="${TEACHER_CKPT:-/content/Shoes_VTO/checkpoints/rtmpose_13kp_improved5/best_with_hand.pt}"

OUT_SLIM_DIR="${OUT_SLIM_DIR:-/content/Shoes_VTO/checkpoints/rtmpose_13kp_slim_distill}"
OUT_S4LITE_DIR="${OUT_S4LITE_DIR:-/content/Shoes_VTO/checkpoints/rtmpose_13kp_slim_s4lite_distill}"

echo "========================================================================"
echo "  SHOES VTO: RTMPOSE-13KP COMPRESSION & KNOWLEDGE DISTILLATION"
echo "========================================================================"
echo "  Dataset:            $DATA_DIR"
echo "  Teacher Checkpoint: $TEACHER_CKPT"
echo "  Option A Output:    $OUT_SLIM_DIR"
echo "  Option A+Lite Out:  $OUT_S4LITE_DIR"
echo "========================================================================"

# Check if teacher exists
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "[!] Error: Teacher checkpoint not found at $TEACHER_CKPT"
    exit 1
fi

# ------------------------------------------------------------------------------
# STEP 1: Option A Distillation (RTMPose-Slim, Stride 16, 3.04 MB)
# ------------------------------------------------------------------------------
echo -e "\n[*] STEP 1: Distilling Option A: RTMPose-Slim (Stride 16, 3.04 MB FP32)..."
python src/models/qualcomm_experiments/rtmpose/train_distill.py \
  --data "$DATA_DIR" \
  --teacher "$TEACHER_CKPT" \
  --student_arch cspnext_slim \
  --output "$OUT_SLIM_DIR" \
  --epochs 40 \
  --batch 32 \
  --lr 3e-4 \
  --temperature 2.0 \
  --alpha_kd 1.0 \
  --warmup 3 \
  --workers 4 \
  --amp

# ------------------------------------------------------------------------------
# STEP 2: Option A+Lite Distillation (RTMPose-Slim-Stage4Lite, 4.25 MB)
# ------------------------------------------------------------------------------
echo -e "\n[*] STEP 2: Distilling Option A+Lite: RTMPose-Slim-Stage4Lite (4.25 MB FP32)..."
python src/models/qualcomm_experiments/rtmpose/train_distill.py \
  --data "$DATA_DIR" \
  --teacher "$TEACHER_CKPT" \
  --student_arch cspnext_slim_s4lite \
  --output "$OUT_S4LITE_DIR" \
  --epochs 40 \
  --batch 32 \
  --lr 3e-4 \
  --temperature 2.0 \
  --alpha_kd 1.0 \
  --warmup 3 \
  --workers 4 \
  --amp

# ------------------------------------------------------------------------------
# STEP 3: Export Compressed Models to ONNX (FP32, FP16, and INT8)
# ------------------------------------------------------------------------------
echo -e "\n[*] STEP 3: Exporting Compressed Models to Qualcomm ONNX (Opset 12)..."

mkdir -p /content/Shoes_VTO/deliverables/qualcomm_experiments/models

python src/models/qualcomm_experiments/rtmpose/export_onnx.py \
  --weights "$OUT_SLIM_DIR/best_with_hand.pt" \
  --output /content/Shoes_VTO/deliverables/qualcomm_experiments/models/rtmpose_slim_improved5_opset12.onnx \
  --backbone cspnext_slim

python src/models/qualcomm_experiments/rtmpose/export_onnx.py \
  --weights "$OUT_S4LITE_DIR/best_with_hand.pt" \
  --output /content/Shoes_VTO/deliverables/qualcomm_experiments/models/rtmpose_slim_s4lite_improved5_opset12.onnx \
  --backbone cspnext_slim_s4lite

# ------------------------------------------------------------------------------
# STEP 4: Comprehensive Benchmark
# ------------------------------------------------------------------------------
echo -e "\n[*] STEP 4: Running Comprehensive Benchmark..."
python tools/evaluation/benchmark_rtmpose_compressed.py \
  --data "$DATA_DIR/valid/images" \
  --teacher "$TEACHER_CKPT" \
  --output /content/Shoes_VTO/deliverables/qualcomm_experiments/rtmpose_compression_benchmark.json

echo -e "\n[OK] Distillation pipeline completed successfully!"

