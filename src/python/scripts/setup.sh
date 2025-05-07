#!/usr/bin/env bash
set -eo pipefail

# Directory of *this* script
this_dir="$( cd "$( dirname "$0" )" && pwd )"

# Base directory of repo
base_dir="$(realpath "${this_dir}/..")"

# Path to virtual environment
: "${venv:=${base_dir}/.venv}"

# Python binary to use
: "${PYTHON=python3}"

python_version="$(${PYTHON} --version)"

# Create virtual environment
echo "Creating virtual environment at ${venv} (${python_version})"
rm -rf "${venv}"
"${PYTHON}" -m venv "${venv}"
source "${venv}/bin/activate"

# Install Python dependencies
echo 'Installing Python dependencies'
pip3 install --upgrade pip
pip3 install --upgrade wheel setuptools

# Check for NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Will configure for GPU training."
    GPU_AVAILABLE=true
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1,2 | head -n 1)
    echo "Detected CUDA driver version: $CUDA_VERSION"
    
    # Choose PyTorch version based on CUDA version
    if [[ $(echo "$CUDA_VERSION >= 12.0" | bc -l) -eq 1 ]]; then
        echo "Installing PyTorch nightly build with CUDA 12.8 support..."
        pip3 install --pre torch==2.8.0.dev20250325+cu128 torchvision==0.22.0.dev20250325+cu128 torchaudio==2.6.0.dev20250325+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
    else
        echo "Using stable PyTorch release for your CUDA version..."
        pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    fi
else
    echo "No NVIDIA GPU detected. Using CPU PyTorch version."
    GPU_AVAILABLE=false
    pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
fi

# Install remaining dependencies from requirements.txt
echo "Installing remaining dependencies from requirements.txt..."
pip3 install -r "${base_dir}/requirements.txt" --no-deps

# -----------------------------------------------------------------------------

echo "OK"
