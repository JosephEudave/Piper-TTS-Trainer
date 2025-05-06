@echo off
REM Launch the Piper TTS Data Preparation GUI for Windows

cd /d "%~dp0"
call piper\src\python\.venv\Scripts\activate.bat
python piper_data_prep_gui.py

pause