@echo off
echo =============================
echo Piper TTS Trainer Setup
echo =============================

:: Check if Python is installed
python --version
if errorlevel 1 (
    echo Python not found. Please install Python 3.9 or higher.
    exit /b 1
)

:: Check if virtual environment exists
if not exist venv (
    echo Creating new virtual environment...
    python -m venv venv
) else (
    echo Virtual environment found.
)

:: Activate the environment
call venv\Scripts\activate
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

:: Update pip
echo Updating pip...
pip install --upgrade pip

:: Install dependencies from requirements file
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

:: Install PyTorch with CUDA support (if available)
echo Installing PyTorch...
pip install torch torchvision torchaudio

:: Fix problematic packages
echo Fixing specific package dependencies...
pip uninstall -y cffi soundfile
pip install cffi>=1.17.0
pip install soundfile==0.13.1

:: Install Gradio explicitly
echo Ensuring Gradio is installed...
pip install gradio==4.8.0

:: Fix any potential conflicts
echo Ensuring all dependencies are properly installed...
pip install --force-reinstall Pillow==10.2.0
pip install --force-reinstall pandas==2.0.3
pip install --force-reinstall numpy>=1.24.0

:: Check for piper_train module
echo Checking for piper_train module...
python -c "try: import piper_train; print('SUCCESS: piper_train module found!'); except ImportError: print('WARNING: piper_train module not found. Training functionality will be limited.\nSee PIPER_TRAIN_INFO.md for installation instructions.')"

echo.
echo =============================
echo Setup complete!
echo =============================
echo.
echo To launch the application, run: launch.bat
echo.

pause 