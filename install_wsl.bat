@echo off
echo ========================================================
echo Piper TTS Trainer - WSL Setup Helper
echo ========================================================
echo.
echo This script will help you set up WSL for Piper TTS Trainer.
echo.

REM Check if WSL is installed
wsl --status >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WSL is not installed. Installing WSL...
    echo.
    echo This will require administrator privileges.
    echo.
    echo Press any key to continue or Ctrl+C to cancel.
    pause >nul
    
    REM Install WSL
    powershell -Command "Start-Process powershell -ArgumentList '-Command wsl --install -d Ubuntu-22.04' -Verb RunAs"
    
    echo.
    echo Please restart your computer after the installation completes.
    echo After restarting, open Ubuntu from the Start menu to complete setup.
    echo Then return to this project to set up Piper TTS Trainer.
    echo.
    pause
    exit /b 0
) else (
    echo WSL is already installed.
    echo.
    
    REM Check if Ubuntu is installed
    wsl -d Ubuntu-22.04 echo "WSL Ubuntu check" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo Ubuntu 22.04 is not installed. Installing Ubuntu...
        echo.
        wsl --install -d Ubuntu-22.04
        
        echo.
        echo Please complete the Ubuntu setup when it opens.
        echo After setting up your username and password, run this script again.
        echo.
        pause
        exit /b 0
    ) else (
        echo Ubuntu 22.04 is installed and ready.
        echo.
    )
)

echo WSL is set up correctly!
echo.
echo ========================================================
echo NEXT STEPS (in Ubuntu):
echo ========================================================
echo.
echo 1. Open Ubuntu from the Start menu
echo 2. Navigate to this project directory:
echo    cd /mnt/c/Users/your-username/path/to/Piper-TTS-Trainer
echo    Example: cd /mnt/c/Users/josep/OneDrive/Documentos/GitHub/Piper-TTS-Trainer
echo 3. Run the Linux setup script:
echo    chmod +x setup.sh
echo    ./setup.sh
echo 4. Launch the interface:
echo    ~/piper_tts_trainer/launch.sh
echo.
echo The web interface will be accessible at: http://localhost:7860
echo.

echo Opening Ubuntu for you now...
start wsl -d Ubuntu-22.04

pause 