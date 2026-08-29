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
echo   DiagonNet Dataset Health, Quality & Bounding Box Auditor
echo   Mode: Administrator Execution (Pure Go Zero-Dep Engine)
echo ====================================================================
echo.

:: 3. Run Dataset Audit
go run . -audit -data data

echo.
echo ====================================================================
echo   Audit Complete.
echo ====================================================================
pause
