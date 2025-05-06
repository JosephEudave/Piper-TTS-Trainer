#!/bin/bash
# Piper TTS Trainer - Setup Script
# This script installs all dependencies for Piper TTS Training

set -e  # Exit on error
echo "========================================================="
echo "Piper TTS Trainer - Linux Setup Script"
echo "========================================================="
echo

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

# Create necessary directories
PIPER_HOME="$HOME/piper_tts_trainer"
mkdir -p "$PIPER_HOME/checkpoints"
mkdir -p "$PIPER_HOME/datasets"
mkdir -p "$PIPER_HOME/training"
mkdir -p "$PIPER_HOME/models"

# Update system and install required packages
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git build-essential

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
pip install --upgrade pip
pip install torch>=2.2.0
pip install pytorch-lightning==2.5.1.post0
pip install piper-phonemize
pip install piper-train
pip install gradio

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
cat > "$PIPER_HOME/launch.sh" << 'EOL'
#!/bin/bash
source "$HOME/piper_tts_trainer/.venv/bin/activate"
cd "$(dirname "$0")"
python3 -m piper_train.ui "$@"
EOL

chmod +x "$PIPER_HOME/launch.sh"

echo
echo "========================================================="
echo "Setup Complete!"
echo "========================================================="
echo
echo "To start the Piper TTS Trainer web interface:"
echo "  ~/piper_tts_trainer/launch.sh"
echo
echo "The web interface will be accessible at: http://localhost:7860"
echo
echo "=========================================================" 