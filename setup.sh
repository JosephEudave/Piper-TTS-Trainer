#!/bin/bash
# Piper TTS Trainer - Setup Script
# This script installs all dependencies for Piper TTS Training

set -e  # Exit on error
echo "=============================================="
echo "Setting up Piper TTS Trainer..."
echo "=============================================="

# Create necessary directories
PIPER_HOME="$HOME/piper_tts_trainer"
mkdir -p "$PIPER_HOME/checkpoints"
mkdir -p "$PIPER_HOME/datasets"
mkdir -p "$PIPER_HOME/training"
mkdir -p "$PIPER_HOME/models"

# Update system and install required packages
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-dev python3-pip python3.10-venv espeak-ng ffmpeg build-essential git wget

# Navigate to project directory
cd "$PIPER_HOME"

# Create and activate Python virtual environment
echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Clone Piper repository if it doesn't exist
if [ ! -d "piper" ]; then
    echo "Cloning Piper repository..."
    git clone https://github.com/rhasspy/piper.git
fi

# Install Python dependencies
echo "Installing Python packages..."
pip install --upgrade pip wheel setuptools
pip install gradio==4.8.0 requests packaging
pip install onnxruntime tqdm matplotlib pandas

# Install piper_train
echo "Installing piper_train..."
cd "$PIPER_HOME/piper/src/python"
pip install -e .
pip install torchmetrics==0.11.4

# Build monotonic_align
echo "Building monotonic_align..."
bash build_monotonic_align.sh

# Install PyTorch with CUDA if available
echo "Checking for CUDA support..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing PyTorch with CUDA support..."
    pip install torch==2.1.0+cu118 torchaudio==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
else
    echo "No NVIDIA GPU detected. Installing CPU-only PyTorch..."
    pip install torch torchaudio
fi

# Copy the Gradio app if it doesn't exist yet
if [ ! -f "$PIPER_HOME/gradio_app.py" ]; then
    echo "Creating Gradio web interface..."
    cd "$PIPER_HOME"
    # The gradio_app.py will be created separately
fi

# Create launcher script
cat > "$PIPER_HOME/launch.sh" << 'EOF'
#!/bin/bash
# Launcher script for Piper TTS Trainer

cd "$HOME/piper_tts_trainer"
source .venv/bin/activate
python3 gradio_app.py
EOF
chmod +x "$PIPER_HOME/launch.sh"

echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo "To start the Piper TTS Trainer, run: ~/piper_tts_trainer/launch.sh"
echo "Access the web interface at: http://localhost:7860"
echo "==============================================" 