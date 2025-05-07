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

# Use current directory instead of home directory
PIPER_HOME="$(pwd)"
echo "Using directory: $PIPER_HOME"

# Create necessary directories
mkdir -p "$PIPER_HOME/checkpoints"
mkdir -p "$PIPER_HOME/datasets"
mkdir -p "$PIPER_HOME/training"
mkdir -p "$PIPER_HOME/models"
mkdir -p "$PIPER_HOME/normalized_wavs"
mkdir -p "$PIPER_HOME/logs"  # Add logs directory

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
sudo apt install -y python3-dev python3-venv espeak-ng ffmpeg build-essential git
# Additional build dependencies for piper_phonemize
sudo apt install -y libespeak-ng-dev pkg-config cmake

# Fix for WSL CUDA issues
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

# Step 1: Clone Piper Recording Studio repository
echo "Setting up Piper Recording Studio..."
if [ ! -d "piper-recording-studio" ]; then
    echo "Cloning Piper Recording Studio repository..."
    git clone https://github.com/rhasspy/piper-recording-studio.git
fi

# Step 2: Clone Piper repository if it doesn't exist
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

# Step 3: Create a single virtual environment in the root directory
echo "Setting up a single Python environment..."
python3 -m venv "$PIPER_HOME/.venv"
source "$PIPER_HOME/.venv/bin/activate"

# Install Python dependencies
echo "Installing Python base packages..."
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade wheel setuptools

# Install specific PyTorch version
echo "Installing specified PyTorch version..."
python3 -m pip install --pre torch==2.8.0.dev20250325+cu128 torchvision==0.22.0.dev20250325+cu128 torchaudio==2.6.0.dev20250325+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('GPU device count:', torch.cuda.device_count()); print('GPU device name:', torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A')"

# Install other requirements
echo "Installing additional dependencies..."
python3 -m pip install -r "$PIPER_HOME/requirements.txt"

# Install piper-phonemize
echo "Installing piper-phonemize from local repository..."
python3 -m pip install "$PIPER_HOME/piper-phonemize/"

# Verify piper-phonemize
if python3 -c "import piper_phonemize" &> /dev/null; then
    echo "piper_phonemize is installed correctly!"
else
    echo "WARNING: piper_phonemize installation failed. Trying alternate method..."
    # Try alternate installation method
    cd "$PIPER_HOME/piper/src"
    mkdir -p build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make
    cd "$PIPER_HOME"
    
    # Verify again
    if python3 -c "import piper_phonemize" &> /dev/null; then
        echo "piper_phonemize installed successfully via alternate method."
    else
        echo "ERROR: Failed to install piper_phonemize. Preprocessing may not work."
    fi
fi

# Install Piper in development mode
echo "Installing Piper in development mode..."
cd "$PIPER_HOME/piper/src/python"
python3 -m pip install -e .

# Build monotonic align
echo "Building monotonic align module..."
mkdir -p "$PIPER_HOME/logs"
LOG_FILE="logs/build_monotonic_align.log"
echo "Full logs will be saved to $PIPER_HOME/$LOG_FILE"
# Using exact path to build script
SCRIPT_PATH="/mnt/c/Users/User/Documents/GitHub/Piper-TTS-Trainer/piper/src/python/build_monotonic_align.sh"
echo "Using build script at: $SCRIPT_PATH"

# Directly run the script without log redirection
if [ -f "$SCRIPT_PATH" ]; then
    echo "Found build script, executing..."
    cd "$PIPER_HOME/piper/src/python"
    sudo bash build_monotonic_align.sh
else
    echo "ERROR: Build script not found at $SCRIPT_PATH"
    echo "Trying to find the script in standard locations..."
    
    # Try alternative paths
    for alt_path in "$PIPER_HOME/piper/src/python/build_monotonic_align.sh" \
                    "./piper/src/python/build_monotonic_align.sh"
    do
        if [ -f "$alt_path" ]; then
            echo "Found at $alt_path, executing..."
            cd "$(dirname "$alt_path")"
            sudo bash "$(basename "$alt_path")"
            break
        fi
    done
fi

# Install Piper Recording Studio dependencies
echo "Installing Piper Recording Studio dependencies..."
cd "$PIPER_HOME/piper-recording-studio"
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements_export.txt

# Return to the home directory
cd "$PIPER_HOME"

# Set up PYTHONPATH for proper module discovery
export PYTHONPATH="$PIPER_HOME/piper/src/python:$PIPER_HOME/piper/src:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Update the run_gui.sh script
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

# Create script to launch recording studio
cat > "$PIPER_HOME/run_recording_studio.sh" << 'EOL'
#!/bin/bash
# Piper Recording Studio Launcher

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Starting Piper Recording Studio..."
echo "The interface will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo

# Run the Recording Studio
cd "$SCRIPT_DIR/piper-recording-studio"
python3 -m piper_recording_studio
EOL

chmod +x "$PIPER_HOME/run_recording_studio.sh"

# Create script for exporting dataset
cat > "$PIPER_HOME/export_dataset.sh" << 'EOL'
#!/bin/bash
# Script to export dataset from Recording Studio

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

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

echo "Exporting dataset for language: $LANG to directory: $OUTPUT_DIR"

# Run the export command
cd "$SCRIPT_DIR/piper-recording-studio"
python3 -m export_dataset output/$LANG/ "$SCRIPT_DIR/$OUTPUT_DIR"

echo "Export complete!"
EOL

chmod +x "$PIPER_HOME/export_dataset.sh"

# Make checkpoint downloader executable
chmod +x "$PIPER_HOME/download_checkpoints.sh"

echo
echo "========================================================="
echo "Setup Complete!"
echo
echo "To run the Piper TTS Trainer GUI, use: ./run_gui.sh"
echo "To run the Piper Recording Studio, use: ./run_recording_studio.sh"
echo "To export recording data, use: ./export_dataset.sh <lang-code> <output-dir>"
echo "To download pre-trained checkpoints, use: ./download_checkpoints.sh"
echo
echo "For more information, see the guide at: https://ssamjh.nz/create-custom-piper-tts-voice/"
echo "=========================================================" 