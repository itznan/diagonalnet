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

:: 4. Launch Interactive HTTP Server and Auto-Launch Browser
echo [Info] Launching inference server on http://localhost:8081 ...
bin\diagonnet.exe -serve -port 8081 -model weights\diagonnet_model.bin

echo.
pause
