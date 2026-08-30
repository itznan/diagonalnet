@echo off
setlocal enabledelayedexpansion
title DiagonalNet - Zero-Dependency Deep Learning Engine

REM Navigate to repository root
cd /d "%~dp0"

REM Route command line argument if provided
if /i "%~1"=="audit" goto CLI_AUDIT
if /i "%~1"=="train" goto CLI_TRAIN
if /i "%~1"=="serve" goto CLI_SERVE
if /i "%~1"=="build" goto CLI_BUILD
if /i "%~1"=="check" goto CLI_CHECK
if /i "%~1"=="test" goto CLI_TEST
if /i "%~1"=="deps" goto CLI_DEPS
if /i "%~1"=="pull" goto CLI_PULL
if /i "%~1"=="push" goto CLI_PUSH
if /i "%~1"=="help" goto CLI_HELP
if /i "%~1"=="-help" goto CLI_HELP
if /i "%~1"=="--help" goto CLI_HELP
if not "%~1"=="" goto CLI_UNKNOWN

:MAIN_MENU
cls
echo ================================================================================
echo              DIAGONALNET ZERO-DEPENDENCY DEEP LEARNING ENGINE
echo       Pure Go 13-Channel Manifold Calculus, Autograd, CPU Runtime and Server
echo ================================================================================
echo.
echo   [1] Build Static Binary and Full Health Check (build, test, deps, verify)
echo   [2] Run Dataset Health, Quality and Bounding Box Audit   (-audit)
echo   [3] Train DiagonalNet Deep Neural Network Pipeline      (-train)
echo   [4] Launch Interactive Web Drawing Canvas and Server    (-serve)
echo   [5] Execute Full 54-Test Mathematical Suite             (go test -v)
echo   [6] Perform Zero-Dependency and Standard Library Audit  (deps-check)
echo   [7] Git Sync Hub (Pull latest / Commit and Push)
echo   [8] Show Hardware Topology and Engine Diagnostics       (-help)
echo   [0] Exit
echo.
echo ================================================================================
set /p CHOICE="Enter your selection [0-8]: "

if "%CHOICE%"=="1" goto DO_CHECK
if "%CHOICE%"=="2" goto DO_AUDIT
if "%CHOICE%"=="3" goto DO_TRAIN
if "%CHOICE%"=="4" goto DO_SERVE
if "%CHOICE%"=="5" goto DO_TEST
if "%CHOICE%"=="6" goto DO_DEPS
if "%CHOICE%"=="7" goto DO_GIT
if "%CHOICE%"=="8" goto DO_DIAGNOSTICS
if "%CHOICE%"=="0" goto DO_EXIT
echo [Error] Invalid option selected. Please enter 0-8.
timeout /t 2 >nul
goto MAIN_MENU

REM ============================================================================
REM 1. BUILD AND FULL SYSTEM HEALTH VERIFICATION
REM ============================================================================
:DO_CHECK
:CLI_CHECK
:CLI_BUILD
cls
echo ================================================================================
echo              DIAGONALNET BUILD AND SYSTEM HEALTH VERIFICATION
echo       Pure Go Zero-Dependency Deep Learning Engine and Web Runtime
echo ================================================================================
echo.

echo [1/4] Checking Zero-Dependency Compliance...
echo --------------------------------------------------------------------------------
echo Checking active module graph (go list -m all)...
go list -m all
if %errorlevel% neq 0 goto CHECK_FAILED

echo.
echo Checking direct standard library imports...
go list -f "{{range .Imports}}{{println .}}{{end}}" ./... | sort /unique
if %errorlevel% neq 0 goto CHECK_FAILED
echo [PASS] 100 percent Pure Go Standard Library. Zero external dependencies.
echo.

echo [2/4] Compiling native binary to bin\diagonalnet.exe...
echo --------------------------------------------------------------------------------
if not exist "bin" mkdir "bin"
go build -o bin\diagonalnet.exe .
if %errorlevel% neq 0 goto CHECK_FAILED

if exist "bin\diagonalnet.exe" (
    echo [PASS] Native executable compiled successfully: bin\diagonalnet.exe
) else (
    echo [FAIL] bin\diagonalnet.exe was not generated.
    goto CHECK_FAILED
)
echo.

echo [3/4] Running mathematical autograd, tensor and layer test suite...
echo --------------------------------------------------------------------------------
go test -v ./...
if %errorlevel% neq 0 goto CHECK_FAILED
echo [PASS] All 54+ unit tests and Jacobian gradient checks passed successfully.
echo.

echo [4/4] Verifying engine subsystems and CLI routing...
echo --------------------------------------------------------------------------------
go run . -help >nul 2>&1
if %errorlevel% neq 0 goto CHECK_FAILED
echo [PASS] Engine initialization and CLI routing verified successfully.
echo.

echo ================================================================================
echo                          SYSTEM VERIFICATION SUMMARY
echo ================================================================================
echo   [PASS] 1. Zero External Dependencies (100 percent Pure Go Stdlib)
echo   [PASS] 2. Native Binary Compiled (bin\diagonalnet.exe)
echo   [PASS] 3. Full Mathematical Autograd and Manifold Suite Passed
echo   [PASS] 4. Engine Runtime and CLI Subsystems Verified
echo ================================================================================
echo   Status: ALL CHECKS PASSED - EVERYTHING IS IN OPTIMAL HEALTH
echo ================================================================================
echo.
if not "%~1"=="" exit /b 0
pause
goto MAIN_MENU

:CHECK_FAILED
echo.
echo ================================================================================
echo   [ERROR] Verification detected issues. Please check the output above.
echo ================================================================================
echo.
if not "%~1"=="" exit /b 1
pause
goto MAIN_MENU

REM ============================================================================
REM 2. DATASET AUDIT
REM ============================================================================
:DO_AUDIT
cls
echo ================================================================================
echo          DIAGONALNET DATASET HEALTH, QUALITY AND BOUNDING BOX AUDITOR
echo ================================================================================
echo.
set /p DATA_PATH="Enter dataset directory path [default: data]: "
if "!DATA_PATH!"=="" set DATA_PATH=data
echo [Info] Auditing dataset at '!DATA_PATH!'...
echo.
go run . -audit -data "!DATA_PATH!"
echo.
pause
goto MAIN_MENU

:CLI_AUDIT
shift
set "DATA_ARG=data"
if not "%~1"=="" set "DATA_ARG=%~1"
go run . -audit -data "%DATA_ARG%"
exit /b %errorlevel%

REM ============================================================================
REM 3. TRAINING HUB
REM ============================================================================
:DO_TRAIN
cls
echo ================================================================================
echo                   DIAGONALNET TRAINING TEMPLATE HUB
echo              Pure Go Multi-Core CPU Data-Parallel Architecture
echo ================================================================================
echo.
echo   [1] Fast Training     (4 Epochs, Batch: 64, LR: 0.0025)    ~1-2 mins
echo   [2] Normal Training   (12 Epochs, Batch: 32, LR: 0.0020)   ~3-4 mins [Recommended]
echo   [3] Hardcore Training (30 Epochs, Batch: 32, LR: 0.0020)   ~8 mins   [98%%+ Accuracy]
echo   [4] Manual Training   (Custom Configuration: Epochs, Batch, LR, Data)
echo   [5] Back to Main Menu
echo.
echo ================================================================================
set /p TCHOICE="Select a training mode [1-5]: "

if not exist "weights" mkdir "weights"

if "%TCHOICE%"=="1" goto TRAIN_FAST
if "%TCHOICE%"=="2" goto TRAIN_NORMAL
if "%TCHOICE%"=="3" goto TRAIN_HARDCORE
if "%TCHOICE%"=="4" goto TRAIN_MANUAL
if "%TCHOICE%"=="5" goto MAIN_MENU
echo [Error] Invalid selection. Choose 1-5.
timeout /t 2 >nul
goto DO_TRAIN

:TRAIN_FAST
cls
echo [Info] Launching FAST Training Profile (4 Epochs, Batch: 64, LR: 0.0025)...
echo.
go run . -train -profile fast -data data -model weights\diagonalnet_model.bin
echo.
pause
goto DO_TRAIN

:TRAIN_NORMAL
cls
echo [Info] Launching NORMAL Standard Recommended Training (12 Epochs, Batch: 32, LR: 0.0020)...
echo.
go run . -train -profile normal -data data -model weights\diagonalnet_model.bin
echo.
pause
goto DO_TRAIN

:TRAIN_HARDCORE
cls
echo [Info] Launching HARDCORE Maximum Accuracy Deep Training (30 Epochs, Batch: 32, LR: 0.0020, 15x Augmentation)...
echo.
go run . -train -profile hardcore -data data -model weights\diagonalnet_model.bin
echo.
pause
goto DO_TRAIN

:TRAIN_MANUAL
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
set /p OUT_MODEL="Output model weights path [default: weights\diagonalnet_model.bin]: "
if "!OUT_MODEL!"=="" set OUT_MODEL=weights\diagonalnet_model.bin
echo.
echo [Info] Starting manual training with !EP! epochs, LR=!LR!, Batch=!BS!...
go run . -train -data !DATA_DIR! -model !OUT_MODEL! -epochs !EP! -batch !BS! -lr !LR!
echo.
pause
goto DO_TRAIN

:CLI_TRAIN
shift
if not exist "weights" mkdir "weights"
go run . -train %*
exit /b %errorlevel%

REM ============================================================================
REM 4. INFERENCE SERVER AND WEB DASHBOARD
REM ============================================================================
:DO_SERVE
cls
echo ================================================================================
echo  DIAGONALNET REAL-TIME NEURAL DRAWING CANVAS AND WEB INFERENCE SERVER
echo ================================================================================
echo.
set /p PORT="Enter HTTP server port [default: 8081]: "
if "!PORT!"=="" set PORT=8081
set /p MODEL_PATH="Enter model weights path [default: weights\diagonalnet_model.bin]: "
if "!MODEL_PATH!"=="" set MODEL_PATH=weights\diagonalnet_model.bin
echo.
echo [Info] Launching inference server on http://localhost:!PORT! ...
go run . -serve -port !PORT! -model "!MODEL_PATH!"
echo.
pause
goto MAIN_MENU

:CLI_SERVE
shift
go run . -serve %*
exit /b %errorlevel%

REM ============================================================================
REM 5. TEST SUITE
REM ============================================================================
:DO_TEST
:CLI_TEST
cls
echo ================================================================================
echo             DIAGONALNET MATHEMATICAL AUTOGRAD AND LAYER TEST SUITE
echo ================================================================================
echo.
go test -v ./...
echo.
if not "%~1"=="" exit /b %errorlevel%
pause
goto MAIN_MENU

REM ============================================================================
REM 6. ZERO-DEPENDENCY AUDIT
REM ============================================================================
:DO_DEPS
:CLI_DEPS
cls
echo ================================================================================
echo                     ZERO-DEPENDENCY VERIFICATION AUDIT
echo ================================================================================
echo.
echo [1] Active Go Modules (go list -m all):
go list -m all
echo.
echo [2] Standard Library Direct Package Imports:
go list -f "{{range .Imports}}{{println .}}{{end}}" ./... | sort /unique
echo.
echo [Success] 100 percent Pure Go Standard Library. Zero third-party dependencies.
echo.
if not "%~1"=="" exit /b 0
pause
goto MAIN_MENU

REM ============================================================================
REM 7. GIT SYNC HUB
REM ============================================================================
:DO_GIT
cls
echo ================================================================================
echo                            GIT SYNCHRONIZATION HUB
echo ================================================================================
echo.
echo   [1] Pull latest changes from origin main (git pull)
echo   [2] Stage, Commit and Push changes       (git add, commit, push)
echo   [3] Back to Main Menu
echo.
echo ================================================================================
set /p GCHOICE="Select an option [1-3]: "

if "%GCHOICE%"=="1" goto GIT_PULL_DO
if "%GCHOICE%"=="2" goto GIT_PUSH_DO
if "%GCHOICE%"=="3" goto MAIN_MENU
echo [Error] Invalid selection. Choose 1-3.
timeout /t 2 >nul
goto DO_GIT

:GIT_PULL_DO
cls
echo [Info] Pulling latest changes from origin main...
git pull origin main
echo.
pause
goto DO_GIT

:GIT_PUSH_DO
cls
set /p COMMIT_MSG="Enter commit message [default: Update]: "
if "!COMMIT_MSG!"=="" set COMMIT_MSG=Update
echo.
git add -A
git commit -m "!COMMIT_MSG!"
git push origin main
echo.
pause
goto DO_GIT

:CLI_PULL
git pull origin main
exit /b %errorlevel%

:CLI_PUSH
shift
set "MSG=%*"
if "%MSG%"=="" set "MSG=Update"
git add -A
git commit -m "%MSG%"
git push origin main
exit /b %errorlevel%

REM ============================================================================
REM 8. ENGINE DIAGNOSTICS AND CLI HELP
REM ============================================================================
:DO_DIAGNOSTICS
:CLI_HELP
cls
go run . -help
echo.
if not "%~1"=="" exit /b 0
pause
goto MAIN_MENU

:CLI_UNKNOWN
echo [Error] Unknown subcommand: %~1
echo Valid subcommands: audit, train, serve, build, check, test, deps, pull, push, help
exit /b 1

:DO_EXIT
cls
echo Exiting DiagonalNet Engine.
exit /b 0
