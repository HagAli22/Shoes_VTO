@echo off
REM ──────────────────────────────────────────────────────────────────────────
REM  finetune_stage_a.bat
REM  Fine-tune Stage A foot detector on the new shoes_v3 dataset.
REM
REM  Usage (from Shoes_VTO root):
REM     scripts\finetune_stage_a.bat
REM
REM  To change epochs or LR, edit the variables below or pass them on CLI:
REM     scripts\finetune_stage_a.bat --epochs 50 --freeze 10
REM ──────────────────────────────────────────────────────────────────────────

set ROOT=%~dp0..
cd /d "%ROOT%"

echo.
echo ===========================================================
echo   Stage A Fine-Tuning on shoes_v3 Dataset
echo ===========================================================
echo.

call conda activate yolo 2>nul || echo [WARN] Could not activate conda env 'yolo'

python -m src.models.detector.finetune_stage_a ^
    --weights "outputs/stage_a/run_v2_1/weights/best.pt" ^
    --data    "data/shoes_v3/data.yaml" ^
    --epochs  100 ^
    --batch   16 ^
    --imgsz   320 ^
    --device  0 ^
    --lr0     0.0001 ^
    --lrf     0.01 ^
    --patience 25 ^
    --freeze  0 ^
    --project "outputs/stage_a" ^
    --name    "finetune_v3" ^
    %*

echo.
echo [Done] Fine-tuning finished.
echo Results saved to: outputs\stage_a\finetune_v3\
pause

