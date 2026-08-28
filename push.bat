@echo off
setlocal enabledelayedexpansion

set "msg=%*"
if "%msg%"=="" (
    set /p "msg=Enter commit message (or press enter for 'Update'): "
)
if "%msg%"=="" (
    set "msg=Update"
)

git add -A
git commit -m "%msg%"
git push origin main
pause
