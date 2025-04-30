@echo off
echo Creating Anaconda environment for Piper TTS training...

:: Create the environment
conda create -n piperTrain python=3.9 -y

:: Activate the environment
call conda activate piperTrain

:: Install basic dependencies
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y numpy pandas scipy librosa tqdm

:: Install additional packages via pip
pip install piper-phonemize==1.1.0
pip install onnxruntime>=1.15.0
pip install pytorch-lightning
pip install faster-whisper
pip install tensorboard
pip install cython>=0.29.0

echo Environment setup complete!
echo To activate the environment, run: conda activate piperTrain 