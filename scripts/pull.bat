@echo off
cd /d "%~dp0\.."
echo Pulling latest changes from origin main...
git pull origin main
pause
