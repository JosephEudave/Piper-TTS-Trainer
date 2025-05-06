@echo off
REM Piper Data Preparation GUI - Windows Launcher
echo ========================================================
echo Piper Data Preparation GUI - Windows Launcher
echo ========================================================

set PIPER_HOME=%~dp0
cd %PIPER_HOME%

REM Activate Python virtual environment
call "%PIPER_HOME%\piper\src\python\.venv\Scripts\activate.bat"

REM Launch the Data Preparation GUI application
python "%PIPER_HOME%\piper_data_prep_gui.py"

REM Keep console window open if there's an error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo An error occurred while launching the GUI.
    echo Please check that all dependencies are installed.
    echo.
    pause
)