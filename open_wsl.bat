@echo off
REM Piper TTS Trainer - WSL Launcher for Windows
REM This script simply launches WSL to access the Linux environment

echo Opening Windows Subsystem for Linux (WSL)...
echo.
echo Once in WSL, please run the following commands:
echo 1. cd /mnt/c/Users/User/Documents/GitHub/Piper-TTS-Trainer
echo 2. source .venv/bin/activate
echo 3. python piper_trainer_gui.py
echo.
echo The GUI will be available at: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.

REM Launch WSL directly
wsl

REM If we get here, WSL has been closed
echo.
echo WSL session ended.
pause 