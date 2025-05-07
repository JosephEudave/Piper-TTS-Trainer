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
mkdir -p "$PIPER_HOME/normalized_wavs"

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Will configure for GPU training."
    GPU_AVAILABLE=true
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1,2 | head -n 1)
    echo "Detected CUDA driver version: $CUDA_VERSION"
else
    echo "No NVIDIA GPU detected. Will configure for CPU training (slower)."
    GPU_AVAILABLE=false
fi

# Update system and install required packages
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y python3-dev python3-venv espeak-ng ffmpeg build-essential git
# Additional build dependencies for piper_phonemize
sudo apt install -y libespeak-ng-dev pkg-config cmake

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

# Install PyTorch based on GPU availability
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Installing PyTorch with CUDA support..."
    
    # Choose PyTorch version based on CUDA version
    if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l) -eq 1 ]]; then
        echo "Installing PyTorch nightly build with CUDA 12.8 support..."
        python3 -m pip install --pre torch==2.8.0.dev20250325+cu128 torchvision==0.22.0.dev20250325+cu128 torchaudio==2.6.0.dev20250325+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
    elif [[ $(echo "$CUDA_VERSION >= 11.8" | bc -l) -eq 1 ]]; then
        echo "Installing PyTorch with CUDA 11.8 support..."
        python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    elif [[ $(echo "$CUDA_VERSION >= 11.6" | bc -l) -eq 1 ]]; then
        echo "Installing PyTorch with CUDA 11.6 support..."
        python3 -m pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116
    else
        echo "Warning: Your CUDA version ($CUDA_VERSION) may not be compatible with current PyTorch versions."
        echo "Attempting to install latest PyTorch with CUDA support..."
        python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo "Installing PyTorch for CPU (training will be slower)..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device count:', torch.cuda.device_count()); print('GPU device name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A')"

# Install all other dependencies from requirements.txt
echo "Installing remaining dependencies from requirements.txt..."
python3 -m pip install -r requirements.txt --no-deps

# Clone Piper repository if it doesn't exist
if [ ! -d "piper" ]; then
    echo "Cloning Piper repository..."
    git clone https://github.com/rhasspy/piper.git
fi

# Setup Piper
cd piper/src/python/
python3 -m pip install -e .
bash build_monotonic_align.sh

# Install piper_phonemize with simpler approach based on guide
echo "Installing piper_phonemize dependencies..."
sudo apt-get install -y libespeak-ng-dev

# Go to src directory
cd ..

# Clone piper_phonemize if not exists
if [ ! -d "piper_phonemize" ]; then
    echo "Cloning piper_phonemize repository..."
    git clone https://github.com/rhasspy/piper-phonemize.git piper_phonemize
fi

# Install piper_phonemize
cd piper_phonemize
echo "Installing piper_phonemize..."
python3 -m pip install -e .

# Verify installation
if python3 -c "from piper_phonemize import phonemize_espeak; print('✅ piper_phonemize successfully imported')"; then
    echo "✅ piper_phonemize installed successfully"
else
    echo "⚠️ Installation issue detected, trying alternative installation method..."
    python3 -m pip install --no-build-isolation -e .
    
    # Check again
    if python3 -c "from piper_phonemize import phonemize_espeak; print('✅ piper_phonemize successfully imported')"; then
        echo "✅ piper_phonemize installed successfully with alternative method"
    else
        echo "❌ piper_phonemize installation failed"
        echo "Please manually install using: pip install --no-build-isolation -e ."
    fi
fi

# Return to python directory
cd ../python

# Ensure PYTHONPATH includes both piper_train and piper_phonemize
export PYTHONPATH="$PIPER_HOME/piper/src/python:$PIPER_HOME/piper/src:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Create an environment script for piper_phonemize
echo "Creating environment script for piper_phonemize..."
ENV_SCRIPT="$PIPER_HOME/set_piper_env.sh"

cat > "$ENV_SCRIPT" << 'EOL'
#!/bin/bash
# Environment variables for Piper TTS

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPER_SRC="$SCRIPT_DIR/piper/src"
PIPER_PYTHON="$PIPER_SRC/python"

# Set PYTHONPATH to include Piper modules
export PYTHONPATH="$PIPER_SRC:$PIPER_PYTHON:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"
EOL

chmod +x "$ENV_SCRIPT"
echo "✅ Created environment script at $ENV_SCRIPT"
echo "To use piper_phonemize, run:"
echo "source $ENV_SCRIPT"

# Return to main directory
cd "$PIPER_HOME"

# Create GUI launcher script
cat > "$PIPER_HOME/run_gui.sh" << 'EOL'
#!/bin/bash
# Piper TTS Trainer - GUI Launcher
# This script activates the virtual environment and launches the Piper TTS Trainer GUI

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Set PYTHONPATH to include Piper modules
export PYTHONPATH="$SCRIPT_DIR/piper/src/python:$SCRIPT_DIR/piper/src:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

echo "Starting Piper TTS Trainer GUI..."
echo "The interface will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop the server"
echo

# Run the GUI script
python "$SCRIPT_DIR/piper_trainer_gui.py"
EOL

chmod +x "$PIPER_HOME/run_gui.sh"

# Create Windows batch launcher
cat > "$PIPER_HOME/run_gui.bat" << 'EOL'
@echo off
REM Piper TTS Trainer - GUI Launcher for Windows
REM This script activates the virtual environment and launches the Piper TTS Trainer GUI

echo Starting Piper TTS Trainer GUI...
echo The interface will be available at: http://localhost:7860
echo Press Ctrl+C to stop the server
echo.

REM Set PYTHONPATH to include Piper modules
set "SCRIPT_DIR=%~dp0"
set "PYTHONPATH=%SCRIPT_DIR%piper\src\python;%SCRIPT_DIR%piper\src;%PYTHONPATH%"
echo PYTHONPATH set to: %PYTHONPATH%

REM Activate the virtual environment and run the GUI script
call ".venv\Scripts\activate.bat" && python piper_trainer_gui.py

REM If we get here, the GUI has been closed
echo.
echo GUI server has stopped.
pause
EOL

echo
echo "========================================================="
echo "Setup Complete!"
echo "========================================================="
echo
if [ "$GPU_AVAILABLE" = true ]; then
    echo "GPU training is enabled for faster voice training."
else
    echo "WARNING: No GPU detected. Training will use CPU and be slow."
    echo "For optimal performance, install on a system with NVIDIA GPU."
fi
echo
echo "To start the Piper TTS Trainer GUI:"
echo "  Linux/WSL: ./run_gui.sh"
echo "  Windows:   run_gui.bat"
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