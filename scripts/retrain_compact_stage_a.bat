@echo off
REM ==============================================================================
REM  One-Click Retraining & Packaging Script for Stage A Compact Detector
REM  Environment: Conda YOLO (Python 3.10 with PyTorch 2.5 + CUDA)
REM ==============================================================================

set CONDA_PYTHON=C:\Users\miniconda3\envs\yolo\python.exe

if not exist "%CONDA_PYTHON%" (
    echo [ERROR] Python not found at %CONDA_PYTHON%
    echo Please verify your conda environment path.
    pause
    exit /b 1
)

echo ==============================================================================
echo   Starting Stage A Compact (YOLOv8-Pico) Retraining Pipeline...
echo ==============================================================================

"%CONDA_PYTHON%" tools\train_and_package_compact_stage_a.py %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Compact Stage A models trained, sliced, quantized, and packaged!
) else (
    echo.
    echo [ERROR] Pipeline encountered an error.
)

pause

