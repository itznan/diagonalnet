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
echo   DiagonNet Architecture Benchmark Runner (DiagonNet vs CNN vs MLP)
echo   Mode: Administrator Execution (Comparative Evaluation Suite)
echo ====================================================================
echo.

:: 3. Run Benchmark
if "%~1"=="" (
    echo [Info] Running 15-epoch comparative benchmark...
    go run . -benchmark -epochs 15
) else (
    echo [Info] Running benchmark with custom parameters: %*
    go run . -benchmark %*
)

echo.
echo ====================================================================
echo   Benchmark Complete. Results exported to assets\comparison_results.csv
echo ====================================================================
pause
