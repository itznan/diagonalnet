@echo off
REM Forward to diagonalnet.bat
call "%~dp0diagonalnet.bat" %*
exit /b %errorlevel%
