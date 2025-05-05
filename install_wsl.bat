@echo off
echo ========================================================
echo Piper TTS Trainer - WSL Setup Helper
echo ========================================================
echo.
echo This script will help you set up WSL for Piper TTS Trainer.
echo.

REM Check if running as administrator
net session >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Administrator privileges required. Restarting with admin rights...
    powershell -Command "Start-Process cmd -ArgumentList '/c %~dpnx0' -Verb RunAs"
    exit /b
)

REM Enable required Windows features
echo Enabling required Windows features...
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart

REM Set WSL default version to 2
echo Setting WSL default version to 2...
wsl --set-default-version 2

REM Download and install WSL kernel update
echo Downloading WSL kernel update...
powershell -Command "Invoke-WebRequest -Uri 'https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi' -OutFile 'wsl_update_x64.msi'"
echo Installing WSL kernel update...
start /wait msiexec /i wsl_update_x64.msi /quiet
del wsl_update_x64.msi

REM Check if Ubuntu is installed
wsl -l | find "Ubuntu-22.04" >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Ubuntu 22.04 is not installed.
    echo.
    echo Installing Ubuntu 22.04. This may take several minutes...
    echo.
    
    REM Download Ubuntu directly
    echo Downloading Ubuntu 22.04 from Microsoft servers...
    powershell -Command "Invoke-WebRequest -Uri 'https://aka.ms/wslubuntu2204' -OutFile 'Ubuntu2204.appx'"
    
    echo Installing Ubuntu from package...
    powershell -Command "Add-AppxPackage -Path 'Ubuntu2204.appx'"
    
    echo Cleaning up...
    del Ubuntu2204.appx
    
    echo Waiting for Ubuntu to register with WSL...
    timeout /t 10 /nobreak >nul
    
    REM Verify installation
    wsl -l | find "Ubuntu-22.04" >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ====================================================
        echo ERROR: Ubuntu installation failed.
        echo ====================================================
        echo.
        echo Please try manual installation:
        echo.
        echo 1. Open Microsoft Store
        echo 2. Search for "Ubuntu 22.04"
        echo 3. Install and set up Ubuntu 22.04
        echo 4. Run this script again
        echo.
        choice /c YN /m "Would you like to open Microsoft Store now?"
        if %ERRORLEVEL% EQU 1 (
            start ms-windows-store://search/?query=Ubuntu
        )
        pause
        exit /b 1
    ) else (
        echo Ubuntu 22.04 has been successfully installed.
    )
) else (
    echo Ubuntu 22.04 is already installed.
)

REM Launch Ubuntu for first time setup if needed
echo.
echo Launching Ubuntu 22.04...
echo (You may need to create a username and password if this is the first launch)
echo.

REM Try to launch Ubuntu
powershell -Command "Get-AppxPackage -Name *Ubuntu* | ForEach-Object { Start-Process shell:AppsFolder\$($_.PackageFamilyName)!Ubuntu }"

echo.
echo If Ubuntu doesn't open automatically:
echo 1. Search for "Ubuntu 22.04" in your Start menu and launch it
echo 2. Complete the first-time setup if prompted
echo.

echo ========================================================
echo NEXT STEPS:
echo ========================================================
echo.
echo Once Ubuntu is set up:
echo.
echo 1. Navigate to this project directory in Ubuntu:
echo    cd /mnt/c/Users/your-username/path/to/Piper-TTS-Trainer
echo    Example: cd %~dp0
echo.
echo 2. Run the Linux setup script:
echo    chmod +x setup.sh
echo    ./setup.sh
echo.
echo    NOTE: The setup script installs compatible versions of dependencies
echo    to resolve the pytorch-lightning version issue.
echo.
echo 3. Launch the interface:
echo    ~/piper_tts_trainer/launch.sh
echo.
echo The web interface will be accessible at: http://localhost:7860
echo.

pause 