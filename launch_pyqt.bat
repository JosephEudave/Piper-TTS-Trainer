@echo off
echo Starting Piper TTS PyQt Preprocessor...

:: Check if virtual environment exists
if not exist venv (
    echo Virtual environment not found. Creating it...
    call setup_environment.bat
) else (
    call venv\Scripts\activate
)

:: Launch PyQt interface
echo Starting PyQt Preprocessor...
python preprocess_gui.py

pause 