@echo off
REM Piper TTS Trainer - GUI Launcher for Windows
REM This script activates the virtual environment and launches the Piper TTS Trainer GUI

echo Starting Piper TTS Trainer GUI...
echo The interface will be available at: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.

REM Activate the virtual environment and run the GUI script
call ".venv\Scripts\activate.bat" && python piper_trainer_gui.py

REM If we get here, the GUI has been closed
echo.
echo GUI server has stopped.
pause
