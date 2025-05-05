@echo off
echo Starting Piper TTS Trainer...

:: Check if virtual environment exists and setup if needed
if not exist venv (
    echo Virtual environment not found. Setting up...
    call setup.bat
) else (
    :: Check if dependencies need to be updated
    echo Checking environment...
    call venv\Scripts\activate
    python -c "try: import gradio; print('Environment check passed.'); except ImportError: print('Dependencies missing, running setup...'); exit(1)"
    if errorlevel 1 (
        echo Running setup to fix missing dependencies...
        call setup.bat
    )
)

:: Activate the environment
call venv\Scripts\activate

:: Launch Gradio interface
echo Starting Gradio Web Interface...
python gradio_interface.py

pause 