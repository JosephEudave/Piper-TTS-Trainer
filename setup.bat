@echo off
echo =========================================================
echo Piper TTS Trainer - Windows Setup Script
echo =========================================================
echo.

REM Create necessary directories
set PIPER_HOME=%USERPROFILE%\piper_tts_trainer
mkdir "%PIPER_HOME%\checkpoints" 2>nul
mkdir "%PIPER_HOME%\datasets" 2>nul
mkdir "%PIPER_HOME%\training" 2>nul
mkdir "%PIPER_HOME%\models" 2>nul

REM Create and activate Python virtual environment
echo Setting up Python environment...
if not exist "%PIPER_HOME%\.venv" (
    python -m venv "%PIPER_HOME%\.venv"
)

REM Activate virtual environment and install packages
call "%PIPER_HOME%\.venv\Scripts\activate.bat"

echo Installing Python packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM Clone Piper repository if it doesn't exist
if not exist "%PIPER_HOME%\piper" (
    echo Cloning Piper repository...
    git clone https://github.com/rhasspy/piper.git "%PIPER_HOME%\piper"
)

REM Install piper_train
echo Installing piper_train...
cd "%PIPER_HOME%\piper\src\python"
pip install -e .
pip install torchmetrics==0.11.4

REM Create launcher script
echo @echo off > "%PIPER_HOME%\launch.bat"
echo call "%%USERPROFILE%%\piper_tts_trainer\.venv\Scripts\activate.bat" >> "%PIPER_HOME%\launch.bat"
echo cd /d "%%~dp0" >> "%PIPER_HOME%\launch.bat"
echo python -m piper_train.ui %%* >> "%PIPER_HOME%\launch.bat"

echo.
echo =========================================================
echo Setup Complete!
echo =========================================================
echo.
echo To start the Piper TTS Trainer web interface:
echo   %PIPER_HOME%\launch.bat
echo.
echo The web interface will be accessible at: http://localhost:7860
echo.
echo =========================================================

pause 