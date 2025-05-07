#!/bin/bash
# Piper TTS Trainer - Poetry Setup Script
# This script installs all dependencies for Piper TTS Training using Poetry
# Based on the guide at https://ssamjh.nz/create-custom-piper-tts-voice/

set -e  # Exit on error
echo "========================================================="
echo "Piper TTS Trainer - Poetry Setup Script"
echo "========================================================="
echo

# Setup log directory
mkdir -p logs
LOG_DIR="logs"
echo "Logs will be saved to: $PIPER_HOME/$LOG_DIR"

# Check if running in WSL
if grep -q Microsoft /proc/version; then
    echo "WSL environment detected."
    WSL_ENVIRONMENT=true
else
    WSL_ENVIRONMENT=false
    echo "This script is designed for WSL. It may work on native Linux but is not tested."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Use current directory as home
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
sudo apt dist-upgrade -y
sudo apt install -y python3-dev espeak-ng ffmpeg build-essential git curl
# Additional build dependencies for piper_phonemize
sudo apt install -y libespeak-ng-dev pkg-config cmake

# Fix for WSL CUDA issues if needed
if [ "$WSL_ENVIRONMENT" = true ] && [ "$GPU_AVAILABLE" = true ]; then
    echo "Applying WSL-specific CUDA fixes..."
    
    # Fix for libcuda.so error in WSL
    if [ -f "/usr/lib/wsl/lib/libcuda.so.1" ]; then
        echo "Fixing libcuda.so symlinks for WSL..."
        sudo rm -f /usr/lib/wsl/lib/libcuda.so.1
        sudo rm -f /usr/lib/wsl/lib/libcuda.so
        sudo ln -s /usr/lib/wsl/lib/libcuda.so.1.1 /usr/lib/wsl/lib/libcuda.so.1
        sudo ln -s /usr/lib/wsl/lib/libcuda.so.1.1 /usr/lib/wsl/lib/libcuda.so
        sudo ldconfig
        echo "CUDA symlinks fixed."
    else
        echo "No libcuda.so found in /usr/lib/wsl/lib/ - skipping symlink fix."
    fi
fi

# Step 1: Install Poetry if not installed
if ! command -v poetry &> /dev/null; then
    echo "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    # Add Poetry to PATH
    export PATH="$HOME/.local/bin:$PATH"
fi

# Step 2: Clone repositories
echo "Setting up repositories..."
# Clone Piper Recording Studio if needed
if [ ! -d "piper-recording-studio" ]; then
    echo "Cloning Piper Recording Studio repository..."
    git clone https://github.com/rhasspy/piper-recording-studio.git
fi

# Clone Piper repository if it doesn't exist
if [ ! -d "piper" ]; then
    echo "Cloning Piper repository..."
    git clone https://github.com/rhasspy/piper.git
fi

# Setup piper-phonemize
echo "Setting up piper-phonemize with ONNX runtime..."
# Clone piper-phonemize if it doesn't exist
if [ ! -d "piper-phonemize" ]; then
    echo "Cloning piper-phonemize repository..."
    git clone https://github.com/rhasspy/piper-phonemize.git
fi

cd "$PIPER_HOME/piper-phonemize"
# Remove old ONNX runtime if it exists and install new version
rm -rf onnxruntime-linux-x64-1.8.1* || true
wget https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz
tar xf onnxruntime-linux-x64-1.14.1.tgz

# Ensure directory structure exists
mkdir -p lib/Linux-x86_64/onnxruntime/include/
mkdir -p lib/Linux-x86_64/onnxruntime/lib/

# Copy ONNX runtime files
cp -r onnxruntime-linux-x64-1.14.1/include/* lib/Linux-x86_64/onnxruntime/include/
cp onnxruntime-linux-x64-1.14.1/lib/* lib/Linux-x86_64/onnxruntime/lib/

# Return to project root
cd "$PIPER_HOME"

# Initialize Poetry project if not already
if [ ! -f "pyproject.toml" ]; then
    echo "Initializing Poetry project..."
    poetry init --name "piper-tts-trainer" --description "GUI for training Piper TTS models" --author "Piper TTS Team" --license "MIT" --python ">=3.10,<4.0" 2>&1 | tee "$PIPER_HOME/$LOG_DIR/poetry_init.log" | grep -e "Created\|Adding\|Using\|What" --line-buffered
    
    # Adding PyTorch mirror
    echo "Adding PyTorch mirror..."
    poetry source add --priority supplemental torch_nightly https://download.pytorch.org/whl/nightly/cu128 2>&1 | tee "$PIPER_HOME/$LOG_DIR/poetry_source.log" | grep -e "Adding\|Source" --line-buffered
    
    # Adding PyTorch dependencies
    echo "Adding PyTorch dependencies (this may take a while)..."
    echo "Full logs will be saved to $PIPER_HOME/$LOG_DIR/pytorch_install.log"
    echo "Installing PyTorch packages - please be patient..."
    poetry add torch==2.8.0.dev20250325+cu128 torchvision==0.22.0.dev20250325+cu128 torchaudio==2.6.0.dev20250325+cu128 --source torch_nightly 2>&1 | tee "$PIPER_HOME/$LOG_DIR/pytorch_install.log" | grep -e "Installing\|Resolving\|Downloading\|complete\|Package\|Added" --line-buffered | uniq
fi

# Install core dependencies
echo "Installing core dependencies via Poetry (this may take a while)..."
echo "Full logs will be saved to $PIPER_HOME/$LOG_DIR/core_deps_install.log"
# Piper core dependencies
poetry add cython==0.29.36 numpy==1.24.3 torchmetrics==0.11.4 pytorch-lightning==2.0.6 onnxruntime==1.14.1 librosa==0.9.2 pybind11 scipy pytaglib onnx numba wheel setuptools soundfile 2>&1 | tee "$PIPER_HOME/$LOG_DIR/core_deps_install.log" | grep -e "Installing\|Resolving\|Downloading\|complete\|Package\|Added" --line-buffered | uniq

# Dataset handling
echo "Installing dataset handling dependencies..."
echo "Full logs will be saved to $PIPER_HOME/$LOG_DIR/dataset_deps_install.log"
poetry add matplotlib requests pandas jiwer unidecode 2>&1 | tee "$PIPER_HOME/$LOG_DIR/dataset_deps_install.log" | grep -e "Installing\|Resolving\|Downloading\|complete\|Package\|Added" --line-buffered | uniq

# Export functionality
echo "Installing export functionality dependencies..."
echo "Full logs will be saved to $PIPER_HOME/$LOG_DIR/export_deps_install.log"
poetry add onnxruntime-extensions 2>&1 | tee "$PIPER_HOME/$LOG_DIR/export_deps_install.log" | grep -e "Installing\|Resolving\|Downloading\|complete\|Package\|Added" --line-buffered | uniq

# GUI dependencies
echo "Installing GUI dependencies (this may take a while)..."
echo "Full logs will be saved to $PIPER_HOME/$LOG_DIR/gui_deps_install.log"
poetry add gradio==5.29.0 gradio_client==1.10.0 httpx>=0.28.1 aiofiles>=22.0,\<25.0 fastapi>=0.115.2,\<1.0 ffmpy groovy@0.1 huggingface-hub>=0.28.1 orjson@3.0 python-dateutil>=2.8.2 tzdata>=2022.7 pytz pydantic==2.11.4 pydantic-core==2.33.2 annotated-types>=0.6.0 typing-inspection>=0.4.0 pydub python-multipart>=0.0.18 ruff>=0.9.3 safehttpx>=0.1.6,\<0.2.0 "semantic-version@2.0" starlette>=0.40.0,\<1.0 tomlkit>=0.12.0,\<0.14.0 typer>=0.12,\<1.0 rich>=10.11.0 shellingham>=1.3.0 uvicorn>=0.14.0 websockets>=10.0,\<16.0 click tqdm 2>&1 | tee "$PIPER_HOME/$LOG_DIR/gui_deps_install.log" | grep -e "Installing\|Resolving\|Downloading\|complete\|Package\|Added" --line-buffered | uniq

# Install piper-phonemize
echo "Installing piper-phonemize from local repository..."
echo "Full logs will be saved to $PIPER_HOME/$LOG_DIR/phonemize_install.log"
poetry run pip install "$PIPER_HOME/piper-phonemize/" 2>&1 | tee "$PIPER_HOME/$LOG_DIR/phonemize_install.log" | grep -e "Installing\|Resolving\|Downloading\|complete\|Successfully" --line-buffered | uniq

# Install Piper in development mode
echo "Installing Piper in development mode..."
echo "Full logs will be saved to $PIPER_HOME/$LOG_DIR/piper_install.log"
cd "$PIPER_HOME/piper/src/python"
poetry run pip install -e . 2>&1 | tee "$PIPER_HOME/$LOG_DIR/piper_install.log" | grep -e "Installing\|Resolving\|Downloading\|complete\|Successfully" --line-buffered | uniq

# Build monotonic align
echo "Building monotonic align module..."
echo "Full logs will be saved to $PIPER_HOME/$LOG_DIR/build_monotonic_align.log"
cd "$PIPER_HOME/piper/src/python"
poetry run bash build_monotonic_align.sh 2>&1 | tee "$PIPER_HOME/$LOG_DIR/build_monotonic_align.log"

# Return to project root
cd "$PIPER_HOME"

# Set up PYTHONPATH for proper module discovery
export PYTHONPATH="$PIPER_HOME/piper/src/python:$PIPER_HOME/piper/src:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Update the run_gui.sh script
echo "Updating GUI launcher scripts..."
cat > "$PIPER_HOME/run_gui.sh" << 'EOL'
#!/bin/bash
# Piper TTS Trainer - GUI Launcher (Poetry version)
# This script uses Poetry to launch the Piper TTS Trainer GUI

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PYTHONPATH to include Piper modules
export PYTHONPATH="$SCRIPT_DIR/piper/src/python:$SCRIPT_DIR/piper/src:$PYTHONPATH"

echo "Starting Piper TTS Trainer GUI..."
echo "The interface will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop the server"
echo

# Run the GUI script with Poetry
cd "$SCRIPT_DIR"
poetry run python piper_trainer_gui.py
EOL

chmod +x "$PIPER_HOME/run_gui.sh"

# Create script to launch recording studio
cat > "$PIPER_HOME/run_recording_studio.sh" << 'EOL'
#!/bin/bash
# Piper Recording Studio Launcher (Poetry version)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Piper Recording Studio..."
echo "The interface will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo

# Run the Recording Studio with Poetry
cd "$SCRIPT_DIR/piper-recording-studio"
poetry run python -m piper_recording_studio
EOL

chmod +x "$PIPER_HOME/run_recording_studio.sh"

# Create script for exporting dataset
cat > "$PIPER_HOME/export_dataset.sh" << 'EOL'
#!/bin/bash
# Script to export dataset from Recording Studio (Poetry version)

# Check if parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <language-code> <output-directory>"
    echo "Example: $0 en-GB my-dataset"
    exit 1
fi

LANG="$1"
OUTPUT_DIR="$2"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Exporting dataset for language: $LANG to directory: $OUTPUT_DIR"

# Run the export command with Poetry
cd "$SCRIPT_DIR/piper-recording-studio"
poetry run python -m export_dataset output/$LANG/ "$SCRIPT_DIR/$OUTPUT_DIR"

echo "Export complete!"
EOL

chmod +x "$PIPER_HOME/export_dataset.sh"

# Make checkpoint downloader executable
chmod +x "$PIPER_HOME/download_checkpoints.sh"

echo
echo "========================================================="
echo "Poetry Setup Complete!"
echo
echo "You can now use the following commands:"
echo "  - ./run_gui.sh                       # Start the Piper TTS Trainer GUI"
echo "  - ./run_recording_studio.sh          # Start the Piper Recording Studio"
echo "  - ./export_dataset.sh <lang> <dir>   # Export recording data"
echo "  - ./download_checkpoints.sh          # Download pre-trained checkpoints"
echo
echo "Poetry commands:"
echo "  - poetry shell                       # Activate the Poetry environment"
echo "  - poetry install                     # Install all dependencies"
echo "  - poetry add <package>               # Add a new dependency"
echo
echo "Installation logs are saved in the $PIPER_HOME/$LOG_DIR directory"
echo
echo "For more information, see the guide at: https://ssamjh.nz/create-custom-piper-tts-voice/"
echo "========================================================="
