@echo off
REM Piper TTS Trainer GUI - Windows Launcher
echo ========================================================
echo Piper TTS Trainer GUI - Windows Launcher
echo ========================================================

set PIPER_HOME=%~dp0
cd %PIPER_HOME%

REM Activate Python virtual environment
call "%PIPER_HOME%\piper\src\python\.venv\Scripts\activate.bat"

REM Launch the GUI application
python "%PIPER_HOME%\piper_trainer_gui.py"

REM Keep console window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while launching the GUI.
    echo Please check that all dependencies are installed.
    echo.
    pause
) 