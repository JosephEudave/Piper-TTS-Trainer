@echo off
REM Launch the Piper TTS Trainer GUI for Windows

cd /d "%~dp0"
call piper\src\python\.venv\Scripts\activate.bat
python piper_trainer_gui.py

pause 