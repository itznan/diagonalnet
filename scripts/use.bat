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
echo   DiagonNet Real-Time Neural Drawing Canvas & Web Inference Engine
echo   Mode: Administrator Execution (Sub-8ms Real-Time Inference)
echo ====================================================================
echo.

:: 3. Launch Interactive HTTP Server and Auto-Launch Browser
echo [Info] Launching inference server on http://localhost:8081 ...
go run . -serve -port 8081 -model weights\diagonnet_model.bin

echo.
pause
