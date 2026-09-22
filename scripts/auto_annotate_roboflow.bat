@echo off
REM ==============================================================================
REM  Auto-Annotate Any Folder of Raw Foot Images for Roboflow Upload
REM  Usage: Drag and drop a folder onto this script, or run:
REM         scripts\auto_annotate_roboflow.bat "data\uploadtoroboflow\7"
REM ==============================================================================

set CONDA_PYTHON=C:\Users\miniconda3\envs\yolo\python.exe

if not exist "%CONDA_PYTHON%" (
    echo [ERROR] Python not found at %CONDA_PYTHON%
    pause
    exit /b 1
)

set TARGET_DIR=%~1
if "%TARGET_DIR%"=="" (
    set TARGET_DIR=data\uploadtoroboflow\7
)

echo ==============================================================================
echo   Auto-Annotating Images in: %TARGET_DIR%
echo ==============================================================================

"%CONDA_PYTHON%" tools\auto_annotate_roboflow.py --input_dir "%TARGET_DIR%" %*

if %ERRORLEVEL% equ 0 (
    echo.
    echo [SUCCESS] Annotations created in exact Roboflow YOLO pose format!
    echo You can now upload the folder directly to Roboflow.
) else (
    echo.
    echo [ERROR] Auto-annotation encountered an error.
)

pause

