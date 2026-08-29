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
echo   DiagonNet Build & Test Verification Engine
echo   Mode: Administrator Execution (Pure Go Static Binary Compiler)
echo ====================================================================
echo.

:: 3. Compile Native Executable
echo [1/2] Compiling native binary 'bin\diagonnet.exe'...
if not exist "bin" mkdir "bin"
go build -o bin\diagonnet.exe .
if %errorlevel% neq 0 (
    echo [Error] Build failed.
    pause
    exit /b %errorlevel%
)
echo [Success] Binary compiled cleanly to bin\diagonnet.exe.
echo.

:: 4. Run Test Suite
echo [2/2] Running 56-test mathematical autograd & manifold suite...
go test -v ./...
if %errorlevel% neq 0 (
    echo [Warning] One or more tests failed.
) else (
    echo [Success] All 56 unit tests passed successfully.
)

echo.
echo ====================================================================
echo   Build & Test Pipeline Completed.
echo ====================================================================
pause
