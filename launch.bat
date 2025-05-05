@echo off
echo Starting Piper TTS Preprocessor...

:: Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found. Creating it...
    call setup_environment.bat
)

:: Activate the environment
call venv\Scripts\activate

:: Start the GUI
python preprocess_gui.py

pause 