@echo off
REM Piper TTS Trainer - Windows Setup Script
REM This script installs all dependencies for Piper TTS Training

echo =========================================================
echo Piper TTS Trainer - Windows Setup Script
echo =========================================================
echo.

REM Set the current directory as PIPER_HOME
set PIPER_HOME=%cd%
echo Using directory: %PIPER_HOME%

REM Create necessary directories
if not exist "%PIPER_HOME%\checkpoints" mkdir "%PIPER_HOME%\checkpoints"
if not exist "%PIPER_HOME%\datasets" mkdir "%PIPER_HOME%\datasets"
if not exist "%PIPER_HOME%\training" mkdir "%PIPER_HOME%\training"
if not exist "%PIPER_HOME%\models" mkdir "%PIPER_HOME%\models"
if not exist "%PIPER_HOME%\normalized_wavs" mkdir "%PIPER_HOME%\normalized_wavs"

REM Check for Python installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or newer and try again.
    pause
    exit /b 1
)

REM Check for NVIDIA GPU
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo NVIDIA GPU detected. Will configure for GPU training.
    set GPU_AVAILABLE=true
    
    REM Get CUDA version
    for /f "tokens=3" %%i in ('nvidia-smi --query-gpu^=driver_version --format^=csv,noheader') do (
        set CUDA_VERSION=%%i
    )
    echo Detected CUDA driver version: %CUDA_VERSION%
) else (
    echo No NVIDIA GPU detected. Will configure for CPU training (slower).
    set GPU_AVAILABLE=false
)

REM Setup Python virtual environment
echo Setting up Python environment...
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python packages...
python -m pip install --upgrade pip
python -m pip install --upgrade wheel setuptools

REM Install PyTorch based on GPU availability
if "%GPU_AVAILABLE%"=="true" (
    echo Installing PyTorch with CUDA support...
    
    REM This is a simplified version since Windows batch doesn't handle floating point comparison well
    REM Users may need to manually adjust this based on their CUDA version
    echo Installing PyTorch nightly build with CUDA 12.8 support...
    python -m pip install --pre torch==2.8.0.dev20250325+cu128 torchvision==0.22.0.dev20250325+cu128 torchaudio==2.6.0.dev20250325+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
) else (
    echo Installing PyTorch for CPU (training will be slower)...
    python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
)

REM Verify PyTorch installation
echo Verifying PyTorch installation...
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device count:', torch.cuda.device_count()); print('GPU device name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A')"

REM Install specific versions required
echo Installing required dependencies with specific versions...
python -m pip install onnxruntime==1.14.1
python -m pip install piper-phonemize==1.1.0
python -m pip install torchmetrics==0.11.4

REM Install Gradio and other dependencies for the GUI
echo Installing GUI dependencies...
python -m pip install gradio==4.8.0 librosa soundfile numpy==1.26.4 tqdm

REM Clone Piper repository if it doesn't exist
if not exist "piper" (
    echo Cloning Piper repository...
    git clone https://github.com/rhasspy/piper.git
)

REM Setup Piper
cd piper\src\python
python -m pip install -e .
python build_monotonic_align.py

REM Return to main directory
cd %PIPER_HOME%

echo.
echo =========================================================
echo Setup Complete!
echo =========================================================
echo.
if "%GPU_AVAILABLE%"=="true" (
    echo GPU training is enabled for faster voice training.
) else (
    echo WARNING: No GPU detected. Training will use CPU and be slow.
    echo For optimal performance, install on a system with NVIDIA GPU.
)
echo.
echo IMPORTANT: This script only installs dependencies. To train models:
echo 1. Use Windows Subsystem for Linux (WSL) with Ubuntu
echo 2. Run install_wsl.bat if you haven't installed WSL yet
echo 3. In Ubuntu, navigate to this directory and run setup.sh
echo.
echo To start the Piper TTS Trainer GUI:
echo   run_gui.bat
echo.
echo =========================================================

pause 