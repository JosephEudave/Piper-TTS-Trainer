#!/bin/bash
# Piper TTS Trainer - Setup Script
# This script installs all dependencies for Piper TTS Training based on the guide at https://ssamjh.nz/create-custom-piper-tts-voice/

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

# Clone Piper repository if it doesn't exist
if [ ! -d "piper" ]; then
    echo "Cloning Piper repository..."
    git clone https://github.com/rhasspy/piper.git
fi

# Create and activate Python virtual environment in the root directory
echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install Python dependencies
echo "Installing Python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade wheel setuptools
python3 -m pip install pybind11  # Required for piper_phonemize

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

# Install torchmetrics (specific version that works well with Piper)
python3 -m pip install torchmetrics==0.11.4

# Install all other dependencies from requirements.txt
echo "Installing remaining dependencies from requirements.txt..."
python3 -m pip install -r requirements.txt

# Install Piper in development mode
echo "Installing Piper in development mode..."
cd "$PIPER_HOME/piper/src/python"
python3 -m pip install -e .

# Build monotonic align
echo "Building monotonic align module..."
bash build_monotonic_align.sh

# Set up PYTHONPATH for proper module discovery
export PYTHONPATH="$PIPER_HOME/piper/src/python:$PIPER_HOME/piper/src:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Create or update run_gui.sh script
echo "Updating GUI launcher scripts..."
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
echo "All functionality is accessible through the GUI interface."
echo
echo "=========================================================" 