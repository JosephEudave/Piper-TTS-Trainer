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

# Use current directory instead of home directory
PIPER_HOME="$(pwd)"
echo "Using directory: $PIPER_HOME"

# Create necessary directories
mkdir -p "$PIPER_HOME/checkpoints"
mkdir -p "$PIPER_HOME/datasets"
mkdir -p "$PIPER_HOME/training"
mkdir -p "$PIPER_HOME/models"

# Update system and install required packages
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-dev python3-venv espeak-ng ffmpeg build-essential git

# Create and activate Python virtual environment
echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade wheel setuptools

# Clone Piper repository if it doesn't exist
if [ ! -d "piper" ]; then
    echo "Cloning Piper repository..."
    git clone https://github.com/rhasspy/piper.git
fi

# Setup Piper
cd piper/src/python/
python3 -m pip install -e .
bash build_monotonic_align.sh

# Install specific version of torchmetrics
python3 -m pip install torchmetrics==0.11.4

# Return to main directory
cd "$PIPER_HOME"

# Create launcher script
cat > "$PIPER_HOME/launch.sh" << 'EOL'
#!/bin/bash
source "$(dirname "$0")/.venv/bin/activate"
cd "$(dirname "$0")"
python3 -m piper_train.ui "$@"
EOL

chmod +x "$PIPER_HOME/launch.sh"

echo
echo "========================================================="
echo "Setup Complete!"
echo "========================================================="
echo
echo "To start the Piper TTS Trainer:"
echo "  ./launch.sh"
echo
echo "For recording studio, install it separately with:"
echo "  git clone https://github.com/rhasspy/piper-recording-studio.git"
echo "  cd piper-recording-studio"
echo "  python3 -m venv .venv"
echo "  source .venv/bin/activate"
echo "  pip install -r requirements.txt"
echo "  pip install -r requirements_export.txt"
echo
echo "=========================================================" 