#!/bin/bash
# Script to properly install piper_phonemize in the virtual environment

set -e  # Exit on error

# Get absolute paths
PIPER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPER_SRC="$PIPER_HOME/piper/src"
PIPER_PYTHON="$PIPER_SRC/python"
PHONEMIZE_PATH="$PIPER_SRC/piper_phonemize"

echo "=== Piper Phonemize Installer ==="
echo "PIPER_HOME: $PIPER_HOME"
echo "PIPER_SRC: $PIPER_SRC"
echo "PIPER_PYTHON: $PIPER_PYTHON"
echo "PHONEMIZE_PATH: $PHONEMIZE_PATH"

# Check if phonemize directory exists
if [ ! -d "$PHONEMIZE_PATH" ]; then
    echo "❌ Phonemize directory not found at $PHONEMIZE_PATH"
    echo "Cloning repository..."
    
    # Create directory if needed
    mkdir -p "$PIPER_SRC"
    cd "$PIPER_SRC"
    
    # Clone the repository
    git clone https://github.com/rhasspy/piper-phonemize.git
    
    if [ ! -d "$PHONEMIZE_PATH" ]; then
        echo "❌ Failed to clone repository"
        exit 1
    fi
    
    echo "✅ Repository cloned successfully"
fi

# Install dependencies
echo -e "\n=== Installing system dependencies ==="
sudo apt update
sudo apt install -y python3-dev libespeak-ng-dev pkg-config cmake build-essential

# Activate virtual environment if it exists
VENV_ACTIVATE="$PIPER_PYTHON/.venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
    echo -e "\n=== Activating virtual environment ==="
    source "$VENV_ACTIVATE"
    echo "✅ Virtual environment activated"
else
    echo "⚠️ Virtual environment not found at $VENV_ACTIVATE"
    echo "Using system Python"
fi

# Install ONNX Runtime
echo -e "\n=== Installing ONNX Runtime ==="
pip install onnxruntime

# Set up ONNX Runtime for C++ compilation
echo -e "\n=== Setting up ONNX Runtime for C++ ==="
cd "$PHONEMIZE_PATH"
mkdir -p download
cd download

# Download ONNX Runtime
if [ "$(uname -m)" == "x86_64" ]; then
    echo "Downloading ONNX Runtime for x86_64..."
    wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.14.1/onnxruntime-linux-x64-1.14.1.tgz
    tar -xzf onnxruntime-linux-x64-1.14.1.tgz
    
    # Create library directories
    mkdir -p ../lib/Linux-x86_64/onnxruntime
    mkdir -p ../lib/Linux-x86_64/onnxruntime/include
    mkdir -p ../lib/Linux-x86_64/onnxruntime/lib
    
    # Copy necessary files
    cp -r onnxruntime-linux-x64-1.14.1/include/* ../lib/Linux-x86_64/onnxruntime/include/
    cp -r onnxruntime-linux-x64-1.14.1/lib/* ../lib/Linux-x86_64/onnxruntime/lib/
    
    echo "✅ ONNX Runtime set up successfully"
else
    echo "⚠️ Unsupported architecture: $(uname -m)"
    echo "Manual installation may be required"
fi

# Install piper_phonemize
echo -e "\n=== Installing piper_phonemize ==="
cd "$PHONEMIZE_PATH"

# Try to install
if ! pip install -e .; then
    echo "⚠️ Standard installation failed, trying alternative method..."
    pip install --no-build-isolation -e .
fi

# Verify installation
echo -e "\n=== Verifying installation ==="
if python -c "from piper_phonemize import phonemize_espeak; print('✅ phonemize_espeak imported successfully')"; then
    echo "✅ piper_phonemize installed successfully"
else
    echo "❌ Installation verification failed"
    echo "Try running with --no-build-isolation:"
    echo "pip install --no-build-isolation -e ."
fi

# Export environment variables
echo -e "\n=== Setting environment variables ==="
export PYTHONPATH="$PIPER_SRC:$PIPER_PYTHON:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

# Create a shell script to set environment variables
echo -e "\n=== Creating environment script ==="
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

echo -e "\n=== Installation Complete ==="
echo "To use piper_phonemize, run:"
echo "source $ENV_SCRIPT" 