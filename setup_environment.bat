@echo off
echo Creating virtual environment for Piper TTS training...

:: Check if Python is installed
python --version
if errorlevel 1 (
    echo Python not found. Please install Python 3.9 or higher.
    exit /b 1
)

:: Create the virtual environment
python -m venv venv

:: Activate the environment
call venv\Scripts\activate

:: Install dependencies using pip
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scipy librosa tqdm
pip install piper-phonemize==1.1.0
pip install onnxruntime>=1.15.0
pip install pytorch-lightning
pip install faster-whisper
pip install tensorboard
pip install cython>=0.29.0
pip install soundfile>=0.12.1
pip install PyQt5>=5.15.0

echo Environment setup complete!
echo To activate the environment, run: venv\Scripts\activate 