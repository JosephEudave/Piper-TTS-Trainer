@echo off
echo Starting Piper TTS Preprocessor...

:: Activate the environment
call conda activate piperTrain

:: Check if environment exists
if errorlevel 1 (
    echo Environment not found. Creating it...
    call setup_environment.bat
)

:: Start the GUI
python preprocess_gui.py

pause 