@echo off
REM Piper TTS Trainer - WSL Installation Script for Windows
REM This script installs Windows Subsystem for Linux (WSL) with Ubuntu

echo =========================================================
echo Piper TTS Trainer - WSL Installation Script
echo =========================================================
echo.
echo This script will install Windows Subsystem for Linux (WSL)
echo with Ubuntu, which is required to run Piper TTS Trainer.
echo.
echo Prerequisites:
echo - Windows 10 version 2004 or higher (Build 19041+) or Windows 11
echo - Administrator privileges
echo.
echo Press any key to continue or Ctrl+C to cancel...
pause > nul

REM Check if running as administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: This script requires administrator privileges.
    echo Please right-click and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo.
echo Installing WSL with Ubuntu (this may take some time)...
echo.

REM Install WSL with Ubuntu (default)
wsl --install

echo.
echo =========================================================
echo WSL installation has been initiated.
echo.
echo IMPORTANT:
echo 1. Your computer will RESTART after the installation completes.
echo 2. After restart, Ubuntu will continue installation.
echo 3. You will be prompted to create a username and password.
echo.
echo After Ubuntu setup completes:
echo 1. Open Ubuntu from the Start menu
echo 2. Navigate to your Piper TTS Trainer directory:
echo    cd /mnt/c/path/to/Piper-TTS-Trainer
echo 3. Run the setup script:
echo    ./setup.sh
echo.
echo =========================================================
echo.
pause 