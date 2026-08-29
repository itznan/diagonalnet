@echo off
setlocal EnableDelayedExpansion
title DiagonNet Deep Learning Training Hub

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
cd /d "%~dp0"
if not exist "main.go" (
    if exist "%~dp0\.." cd /d "%~dp0\.."
)

if not exist "weights" mkdir "weights"

:TRAIN_MENU
cls
echo ================================================================================
echo                    DIAGONNET TRAINING TEMPLATE HUB
echo              Pure Go Multi-Core CPU Data-Parallel Architecture
echo ================================================================================
echo.
echo   [1] Fast Training     (4 Epochs, Batch: 64, LR: 0.0025)    ~1-2 mins
echo   [2] Normal Training   (12 Epochs, Batch: 32, LR: 0.0020)   ~3-4 mins [Recommended]
echo   [3] Hardcore Training (30 Epochs, Batch: 32, LR: 0.0020)   ~8 mins   [98%%+ Accuracy]
echo   [4] Manual Training   (Custom Configuration: Epochs, Batch, LR, Data)
echo   [5] Back / Exit
echo.
echo ================================================================================
set /p TCHOICE="Select a training mode [1-5]: "

if "%TCHOICE%"=="1" goto DO_FAST
if "%TCHOICE%"=="2" goto DO_NORMAL
if "%TCHOICE%"=="3" goto DO_HARDCORE
if "%TCHOICE%"=="4" goto DO_MANUAL
if "%TCHOICE%"=="5" exit /b
echo [Error] Invalid selection. Choose 1-5.
timeout /t 2 >nul
goto TRAIN_MENU

:DO_FAST
cls
echo [Info] Launching FAST Training Profile (4 Epochs, Batch: 64, LR: 0.0025)...
echo.
go run . -train -profile fast -data data -model weights\diagonnet_model.bin
echo.
pause
goto TRAIN_MENU

:DO_NORMAL
cls
echo [Info] Launching NORMAL Standard Recommended Training (12 Epochs, Batch: 32, LR: 0.0020)...
echo.
go run . -train -profile normal -data data -model weights\diagonnet_model.bin
echo.
pause
goto TRAIN_MENU

:DO_HARDCORE
cls
echo [Info] Launching HARDCORE Maximum Accuracy Deep Training (30 Epochs, Batch: 32, LR: 0.0020, 15x Augmentation)...
echo.
go run . -train -profile hardcore -data data -model weights\diagonnet_model.bin
echo.
pause
goto TRAIN_MENU

:DO_MANUAL
cls
echo ================================================================================
echo                       MANUAL CUSTOM TRAINING CONFIGURATION
echo ================================================================================
echo.
set /p DATA_DIR="Dataset directory path [default: data]: "
if "!DATA_DIR!"=="" set DATA_DIR=data

set /p EP="Number of Epochs [default: 15]: "
if "!EP!"=="" set EP=15

set /p BS="Mini-Batch Size [default: 32]: "
if "!BS!"=="" set BS=32

set /p LR="Learning Rate [default: 0.002]: "
if "!LR!"=="" set LR=0.002

set /p OUT_MODEL="Output model weights path [default: weights\diagonnet_model.bin]: "
if "!OUT_MODEL!"=="" set OUT_MODEL=weights\diagonnet_model.bin

echo.
echo [Info] Starting manual training with !EP! epochs, LR=!LR!, Batch=!BS!...
go run . -train -data !DATA_DIR! -model !OUT_MODEL! -epochs !EP! -batch !BS! -lr !LR!
echo.
pause
goto TRAIN_MENU
