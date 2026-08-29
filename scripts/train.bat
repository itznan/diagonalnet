@echo off
setlocal EnableDelayedExpansion

:: 1. Self-Elevation to Administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo ====================================================================
    echo   [DiagonNet] Requesting Administrator Privileges...
    echo ====================================================================
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -Verb RunAs -FilePath '%~f0' -WorkingDirectory '%~dp0'"
    exit /b
)

:: 2. Set Working Directory to Repository Root
cd /d "%~dp0\.."

echo ====================================================================
echo   DiagonNet Deep Learning Model Training Pipeline
echo   Mode: Administrator Execution (Multi-Core CPU Data-Parallelism)
echo ====================================================================
echo.

:: 3. Build executable if missing
if not exist "bin\diagonnet.exe" (
    echo [Info] Building binary 'bin\diagonnet.exe'...
    go build -o bin\diagonnet.exe .
    if %errorlevel% neq 0 (
        echo [Error] Failed to compile DiagonNet binary.
        pause
        exit /b %errorlevel%
    )
)

:: 4. Run Training with parameters or defaults
if "%~1"=="" (
    echo [Info] Running default training: 10 Epochs, LR=0.002, Batch=32
    bin\diagonnet.exe -train -data data -model weights\diagonnet_model.bin -epochs 10 -lr 0.002 -batch 32
) else (
    echo [Info] Running custom training parameters: %*
    bin\diagonnet.exe -train %*
)

echo.
echo ====================================================================
echo   Training Finished.
echo ====================================================================
pause
