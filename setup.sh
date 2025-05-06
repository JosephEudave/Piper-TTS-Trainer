#!/bin/bash
# Piper TTS Trainer - Setup Script
# Based on guide from https://ssamjh.nz/create-custom-piper-tts-voice/

set -e  # Exit on error
echo "========================================================="
echo "Piper TTS Trainer - Setup Script"
echo "========================================================="
echo

# Function to check command existence
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is required but not installed."
        exit 1
    fi
}

# Check if running in WSL
if grep -q Microsoft /proc/version; then
    echo "WSL environment detected."
else
    echo "This script is designed for WSL. It may work on native Linux but is not tested."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for required commands
echo "Checking required commands..."
check_command python3
check_command pip3
check_command git

# Use current directory
PIPER_HOME="$(pwd)"
echo "Using directory: $PIPER_HOME"

# Create necessary directories
echo "Creating project directories..."
mkdir -p "$PIPER_HOME/checkpoints"
mkdir -p "$PIPER_HOME/datasets"
mkdir -p "$PIPER_HOME/training"
mkdir -p "$PIPER_HOME/models"
mkdir -p "$PIPER_HOME/logs"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected!"
    GPU_INFO=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader,nounits)
    echo "GPU Model: $GPU_INFO"
    echo "GPU support will be enabled for training."
    HAS_GPU=true
else
    echo "No NVIDIA GPU detected. Training will be CPU-only (this will be very slow)."
    HAS_GPU=false
fi

# Update system and install required packages
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-dev python3-venv espeak-ng ffmpeg build-essential git libespeak-ng-dev 
sudo apt install -y autoconf automake libtool

# Setup ONNX Runtime (required for piper-phonemize)
echo "Setting up ONNX Runtime..."
cd "$PIPER_HOME"
mkdir -p "$PIPER_HOME/lib/Linux-x86_64/onnxruntime/include"
mkdir -p "$PIPER_HOME/lib/Linux-x86_64/onnxruntime/lib"
if [ ! -f "onnxruntime-linux-x64-1.14.1.tgz" ]; then
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz
fi
tar xf onnxruntime-linux-x64-1.14.1.tgz
cp -r onnxruntime-linux-x64-1.14.1/include/* "$PIPER_HOME/lib/Linux-x86_64/onnxruntime/include/"
cp -r onnxruntime-linux-x64-1.14.1/lib/* "$PIPER_HOME/lib/Linux-x86_64/onnxruntime/lib/"

# Setup Piper Recording Studio
echo "Setting up Piper Recording Studio..."
if [ ! -d "piper-recording-studio" ]; then
    git clone https://github.com/rhasspy/piper-recording-studio.git
fi

cd piper-recording-studio
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_export.txt
deactivate
cd "$PIPER_HOME"

# Setup Piper-Phonemize (required dependency for Piper)
echo "Setting up Piper-Phonemize..."
if [ ! -d "piper-phonemize" ]; then
    git clone https://github.com/rhasspy/piper-phonemize.git
fi

# Setup Piper TTS
echo "Setting up Piper TTS..."
if [ ! -d "piper" ]; then
    git clone https://github.com/rhasspy/piper.git
fi

cd piper/src/python
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade wheel setuptools

# Install piper-phonemize with proper dependencies
echo "Installing piper-phonemize..."
python3 -m pip install "$PIPER_HOME/piper-phonemize"

# Install Piper in editable mode
python3 -m pip install -e .

# Install specific version of torchmetrics to avoid compatibility issues
python3 -m pip install torchmetrics==0.11.4

# Install PyTorch with CUDA support if GPU is available
if [ "$HAS_GPU" = true ]; then
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    python3 -m pip install torch torchvision torchaudio
fi

# Install Gradio and dependencies for the GUI interface
echo "Installing Gradio for GUI interface..."
python3 -m pip install gradio requests 

bash build_monotonic_align.sh
deactivate
cd "$PIPER_HOME"

# Create launcher scripts
echo "Creating launcher scripts..."

# Recording Studio launcher
cat > "$PIPER_HOME/launch_recording.sh" << 'EOL'
#!/bin/bash
cd "$(dirname "$0")/piper-recording-studio"
source .venv/bin/activate
python3 -m piper_recording_studio
EOL

# Training launcher with GPU support detection
cat > "$PIPER_HOME/launch_training.sh" << 'EOL'
#!/bin/bash
cd "$(dirname "$0")/piper/src/python"
source .venv/bin/activate

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "Using GPU for training..."
    ACCELERATOR="gpu"
    DEVICES=1
    BATCH_SIZE=32
else
    echo "No GPU detected, using CPU (this will be very slow)..."
    ACCELERATOR="cpu"
    DEVICES=1
    BATCH_SIZE=4
fi

# Default training parameters
python3 -m piper_train \
    --accelerator "$ACCELERATOR" \
    --devices "$DEVICES" \
    --batch-size "$BATCH_SIZE" \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 6000 \
    --checkpoint-epochs 1 \
    --precision 32 \
    "$@"
EOL

# GUI Launcher script
cat > "$PIPER_HOME/launch_trainer_gui.sh" << 'EOL'
#!/bin/bash
# Launch the Piper TTS Trainer GUI
cd "$(dirname "$0")"
source piper/src/python/.venv/bin/activate
python3 piper_trainer_gui.py
EOL

# Make launchers executable
chmod +x "$PIPER_HOME/launch_recording.sh"
chmod +x "$PIPER_HOME/launch_training.sh"
chmod +x "$PIPER_HOME/launch_trainer_gui.sh"

# Create a README with instructions
cat > "$PIPER_HOME/SETUP_INSTRUCTIONS.md" << 'EOL'
# Piper TTS Trainer Setup Instructions

## Quick Start

1. Start the Recording Studio:
   ```bash
   ./launch_recording.sh
   ```

2. Record your voice samples (aim for at least 100 recordings)

3. Export your dataset:
   ```bash
   cd piper-recording-studio
   source .venv/bin/activate
   python3 -m export_dataset output/<your-language>/ ~/piper/my-dataset
   ```

4. Download a checkpoint:
   ```bash
   wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt -O checkpoints/epoch=2164-step=1355540.ckpt
   ```

5. Start training:
   ```bash
   ./launch_training.sh --dataset-dir ~/piper/my-training --resume_from_checkpoint checkpoints/epoch=2164-step=1355540.ckpt
   ```

## GUI Interface

For a more user-friendly experience, you can use the GUI interface:
```bash
./launch_trainer_gui.sh
```

This will open a web interface where you can:
- Prepare your dataset
- Preprocess your data
- Train your model

## Important Notes

- Training requires significant computational resources. A GPU is highly recommended.
- The training process can take several hours to days depending on your hardware.
- Monitor the training progress in the logs directory.
- For best results, record at least 100 high-quality voice samples.

## Troubleshooting

If you encounter issues with the installation:

1. libcuda error: Try running:
   ```bash
   sudo rm -r /usr/lib/wsl/lib/libcuda.so.1 && sudo rm -r /usr/lib/wsl/lib/libcuda.so && sudo ln -s /usr/lib/wsl/lib/libcuda.so.1.1 /usr/lib/wsl/lib/libcuda.so.1 && sudo ln -s /usr/lib/wsl/lib/libcuda.so.1.1 /usr/lib/wsl/lib/libcuda.so && sudo ldconfig
   ```

2. If piper-phonemize fails to install, make sure all dependencies are installed and try:
   ```bash
   cd piper/src/python && source .venv/bin/activate && python3 -m pip install ~/piper-phonemize/
   ```

For more detailed instructions, visit: https://ssamjh.nz/create-custom-piper-tts-voice/
EOL

echo
echo "========================================================="
echo "Setup Complete!"
echo "========================================================="
echo
echo "Please read SETUP_INSTRUCTIONS.md for detailed instructions"
echo "on how to record your voice and train your model."
echo
echo "=========================================================" 