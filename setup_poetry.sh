#!/bin/bash
# Piper TTS Trainer - Poetry Setup Script
# This script installs all dependencies for Piper TTS Training using Poetry
# Based on the guide at https://ssamjh.nz/create-custom-piper-tts-voice/

set -e  # Exit on error

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}Piper TTS Trainer - Poetry Setup Script${NC}"
echo -e "${YELLOW}=========================================================${NC}"
echo

# Fix for apt_pkg module issues - run at the beginning
echo -e "${YELLOW}Fixing apt_pkg module issue...${NC}"
if ! python3 -c "import apt_pkg" &>/dev/null; then
    echo -e "${YELLOW}apt_pkg module not found, attempting to fix...${NC}"
    # Try to find the appropriate .so file
    APT_PKG_FILE=$(find /usr/lib/python3/dist-packages -name "apt_pkg.cpython-3*-x86_64-linux-gnu.so" | head -n 1)
    
    if [ -n "$APT_PKG_FILE" ]; then
        echo -e "${YELLOW}Found apt_pkg file: $APT_PKG_FILE${NC}"
        sudo ln -sf "$APT_PKG_FILE" /usr/lib/python3/dist-packages/apt_pkg.so
        echo -e "${GREEN}Created symlink for apt_pkg.so${NC}"
    else
        echo -e "${YELLOW}Could not find apt_pkg.so file, reinstalling python3-apt...${NC}"
        sudo apt-get purge -y python3-apt
        sudo apt-get install -y python3-apt
    fi
    
    # Verify the fix worked
    if python3 -c "import apt_pkg" &>/dev/null; then
        echo -e "${GREEN}✅ apt_pkg module issue fixed!${NC}"
    else
        echo -e "${RED}⚠️ apt_pkg module still not available. Continuing anyway, but apt commands may fail.${NC}"
        # Continue anyway since the script might still work
    fi
fi

# Setup log directory
mkdir -p logs
LOG_DIR="logs"
echo "Logs will be saved to: $PIPER_HOME/$LOG_DIR"

# Check if running in WSL
if grep -q Microsoft /proc/version; then
    echo -e "${GREEN}WSL environment detected.${NC}"
    WSL_ENVIRONMENT=true
else
    WSL_ENVIRONMENT=false
    echo -e "${YELLOW}This script is designed for WSL. It may work on native Linux but is not tested.${NC}"
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
mkdir -p "$PIPER_HOME/downloads"

# Check for NVIDIA GPU
echo -e "${YELLOW}Checking for NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected. Will configure for GPU training.${NC}"
    GPU_AVAILABLE=true
    # Get CUDA version
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1,2 | head -n 1)
    echo "Detected CUDA driver version: $CUDA_VERSION"
else
    echo -e "${YELLOW}No NVIDIA GPU detected. Will configure for CPU training (slower).${NC}"
    GPU_AVAILABLE=false
fi

# Fix for WSL CUDA issues if needed
if [ "$WSL_ENVIRONMENT" = true ] && [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${YELLOW}Applying WSL-specific CUDA fixes...${NC}"
    
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

# Update system and install required packages
echo -e "${YELLOW}Installing system dependencies...${NC}"
sudo apt update

# Check if Python 3.11 is already installed
PYTHON311_INSTALLED=false
if command -v python3.11 &> /dev/null; then
    echo -e "${GREEN}Python 3.11 is already installed.${NC}"
    PYTHON311_INSTALLED=true
else
    echo -e "${YELLOW}Python 3.11 is not installed. Attempting to install...${NC}"
    
    # First try to install from standard repositories
    echo -e "${YELLOW}Trying to install Python 3.11 from standard repositories...${NC}"
    if sudo apt-get install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils; then
        echo -e "${GREEN}Successfully installed Python 3.11 from standard repositories.${NC}"
        PYTHON311_INSTALLED=true
    else
        echo -e "${YELLOW}Failed to install Python 3.11 from standard repositories.${NC}"
        
        # Alternative method using direct PPA addition instead of add-apt-repository
        echo -e "${YELLOW}Adding deadsnakes PPA manually...${NC}"
        echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa.list
        sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776
        sudo apt update
        
        # Try installing Python 3.11 again
        if sudo apt-get install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils; then
            echo -e "${GREEN}Successfully installed Python 3.11 from deadsnakes PPA.${NC}"
            PYTHON311_INSTALLED=true
        else
            echo -e "${RED}Failed to install Python 3.11. Please install it manually.${NC}"
            exit 1
        fi
    fi
fi

# Install other system dependencies
sudo apt install -y build-essential git curl pkg-config cmake ffmpeg \
                   libespeak-ng-dev autoconf automake libtool libsonic-dev

# Create a symbolic link to make python3.11 the default python3
if [ "$PYTHON311_INSTALLED" = true ]; then
    echo -e "${YELLOW}Setting Python 3.11 as default python3...${NC}"
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
    
    # Check that Python 3.11 is installed and working
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}Python version: ${PYTHON_VERSION}${NC}"
    
    if [[ ! "$PYTHON_VERSION" =~ ^3\.11\. ]]; then
        echo -e "${RED}Error: Python 3.11 is not the default version. Current version: ${PYTHON_VERSION}${NC}"
        echo -e "${YELLOW}Attempting to continue but there may be issues...${NC}"
    fi
fi

# Step 1: Install Poetry if not installed
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}Installing Poetry...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
    # Add Poetry to PATH
    export PATH="$HOME/.local/bin:$PATH"
fi

# Set Poetry to use Python 3.11
echo -e "${YELLOW}Configuring Poetry to use Python 3.11...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf .venv
fi
poetry env use python3.11

# Step 2: Clone repositories
echo -e "${YELLOW}Setting up repositories...${NC}"
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

# Step 3: Install the rhasspy fork of espeak-ng
echo -e "${YELLOW}=== Installing and configuring the rhasspy fork of espeak-ng ===${NC}"
echo -e "${YELLOW}This fork contains the required TextToPhonemesWithTerminator function${NC}"

# Check if the special rhasspy fork is already installed
if command -v espeak-ng > /dev/null; then
    echo -e "${YELLOW}espeak-ng is already installed, checking if it's the rhasspy fork...${NC}"
    
    # Try to check for the special function by running a test
    TEST_OUTPUT=$(python3 -c "
import ctypes
import sys
try:
    espeak_lib = ctypes.cdll.LoadLibrary('libespeak-ng.so.1')
    # Check if the special function exists by attempting to get its address
    func = espeak_lib.espeak_TextToPhonemesWithTerminator
    print('rhasspy_fork_detected')
except (AttributeError, OSError):
    print('standard_espeak_detected')
" 2>/dev/null)
    
    if [ "$TEST_OUTPUT" = "rhasspy_fork_detected" ]; then
        echo -e "${GREEN}✅ The rhasspy fork of espeak-ng is already installed${NC}"
        ESPEAK_RHASSPY_INSTALLED=true
    else
        echo -e "${YELLOW}Standard espeak-ng detected. Will install the rhasspy fork...${NC}"
        ESPEAK_RHASSPY_INSTALLED=false
    fi
else
    echo -e "${YELLOW}espeak-ng not found. Will install the rhasspy fork...${NC}"
    ESPEAK_RHASSPY_INSTALLED=false
fi

# Install the rhasspy fork of espeak-ng if needed
if [ "$ESPEAK_RHASSPY_INSTALLED" = false ]; then
    echo -e "${YELLOW}Cloning the rhasspy/espeak-ng repository...${NC}"
    cd "$PIPER_HOME/downloads"
    if [ ! -d "espeak-ng" ]; then
        git clone https://github.com/rhasspy/espeak-ng.git
    else
        echo -e "${YELLOW}Using existing espeak-ng repository, updating...${NC}"
        cd espeak-ng
        git pull
        cd ..
    fi
    
    # Build and install espeak-ng
    cd espeak-ng
    echo -e "${YELLOW}Configuring espeak-ng build...${NC}"
    ./autogen.sh
    ./configure --prefix=/usr
    
    echo -e "${YELLOW}Building espeak-ng...${NC}"
    make -j$(nproc)
    
    echo -e "${YELLOW}Installing espeak-ng (requires sudo)...${NC}"
    sudo make install
    sudo ldconfig
    
    # Return to the project directory
    cd "$PIPER_HOME"
    
    # Verify installation
    if command -v espeak-ng > /dev/null; then
        echo -e "${GREEN}✅ rhasspy/espeak-ng installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install rhasspy/espeak-ng${NC}"
        exit 1
    fi
fi

# Step 4: Setup piper-phonemize
echo -e "${YELLOW}Setting up piper-phonemize...${NC}"
# Create downloads directory if doesn't exist
mkdir -p "$PIPER_HOME/downloads"

# Check for existing wheel files - DON'T download new ones
echo -e "${YELLOW}Checking for existing piper_phonemize wheels in downloads directory...${NC}"

cd "$PIPER_HOME/downloads"

# Check for existing wheel files
WHEEL_FILES=$(find . -name "piper_phonemize*.whl")

if [ -z "$WHEEL_FILES" ]; then
    echo -e "${YELLOW}No wheel files found in downloads directory.${NC}"
    echo -e "${YELLOW}Using wheels from: C:/Users/User/Documents/GitHub/Piper-TTS-Trainer/downloads${NC}"
    
    # Copy wheels from Windows path to WSL path
    if [ "$WSL_ENVIRONMENT" = true ]; then
        echo -e "${YELLOW}Copying wheels from Windows path to WSL path...${NC}"
        cp -f /mnt/c/Users/User/Documents/GitHub/Piper-TTS-Trainer/downloads/piper_phonemize*.whl ./ 2>/dev/null || true
        
        # Check again after copying
        WHEEL_FILES=$(find . -name "piper_phonemize*.whl")
        if [ -z "$WHEEL_FILES" ]; then
            echo -e "${RED}No wheel files found after copying from Windows path.${NC}"
            echo -e "${YELLOW}Please make sure wheel files exist in:${NC}"
            echo -e "${YELLOW}C:/Users/User/Documents/GitHub/Piper-TTS-Trainer/downloads${NC}"
        else
            echo -e "${GREEN}Found wheel files after copying: ${WHEEL_FILES}${NC}"
        fi
    else
        echo -e "${RED}No wheel files found in downloads directory.${NC}"
        echo -e "${YELLOW}Please place wheel files in:${NC}"
        echo -e "${YELLOW}$PIPER_HOME/downloads${NC}"
    fi
else
    echo -e "${GREEN}Found existing wheel files: ${WHEEL_FILES}${NC}"
fi

# Return to main directory
cd "$PIPER_HOME"

# Step 5: Download and generate helper scripts
echo -e "${YELLOW}Creating helper scripts...${NC}"

# Create the simplified run_gui.sh script with improved phonemize handling
cat > "$PIPER_HOME/run_gui.sh" << 'EOL'
#!/bin/bash
# Piper TTS Trainer - GUI Launcher
# This script uses Poetry to launch the Piper TTS Trainer GUI
# It tries to install wheel files from the downloads directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}Piper TTS Trainer - GUI Launcher${NC}"
echo -e "${YELLOW}=========================================================${NC}"

# Set PYTHONPATH to include Piper modules
export PYTHONPATH="$SCRIPT_DIR/piper/src/python:$SCRIPT_DIR/piper/src:$SCRIPT_DIR/piper/src/piper_phonemize:$PYTHONPATH"

# 1. Check if piper_phonemize is already installed
echo -e "${YELLOW}Checking piper_phonemize installation...${NC}"
if poetry run python -c "import piper_phonemize; print('piper_phonemize is installed')" &>/dev/null; then
    echo -e "${GREEN}✅ piper_phonemize is already installed${NC}"
else
    echo -e "${YELLOW}piper_phonemize not found, looking for wheel files...${NC}"
    
    # 2. Find wheel files in downloads directory
    WHEEL_FILES=$(find "$SCRIPT_DIR/downloads" -name "piper_phonemize*.whl" 2>/dev/null)
    
    if [ -z "$WHEEL_FILES" ]; then
        echo -e "${RED}❌ No wheel files found in downloads directory!${NC}"
        echo -e "${YELLOW}Please download wheel files from:${NC}"
        echo -e "${YELLOW}https://github.com/rhasspy/piper-phonemize/releases/tag/v1.0.0${NC}"
        echo -e "${YELLOW}Place them in: $SCRIPT_DIR/downloads${NC}"
        echo -e "${YELLOW}You need the Python 3.11 compatible wheel file.${NC}"
        exit 1
    else
        echo -e "${GREEN}Found wheel files: ${WHEEL_FILES}${NC}"
        
        # 3. Try to install each wheel file until one succeeds
        INSTALL_SUCCESS=false
        
        # First uninstall any existing piper_phonemize
        poetry run pip uninstall -y piper_phonemize &>/dev/null || true
        
        for wheel_file in $WHEEL_FILES; do
            echo -e "${YELLOW}Trying to install: $(basename "$wheel_file")${NC}"
            if poetry run pip install --force-reinstall "$wheel_file" 2>/dev/null; then
                echo -e "${GREEN}✅ Successfully installed $(basename "$wheel_file")${NC}"
                INSTALL_SUCCESS=true
                break
            else
                echo -e "${RED}❌ Failed to install $(basename "$wheel_file")${NC}"
            fi
        done
        
        if [ "$INSTALL_SUCCESS" = false ]; then
            echo -e "${RED}❌ Failed to install any wheel files.${NC}"
            echo -e "${YELLOW}Please download the appropriate wheel file for Python 3.11 from:${NC}"
            echo -e "${YELLOW}https://github.com/rhasspy/piper-phonemize/releases/tag/v1.0.0${NC}"
            exit 1
        fi
    fi
fi

# 4. Set required environment variables
export PHONEMIZE_ESPEAK_DATA="/usr/share/espeak-ng-data"
export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# 5. Test if phonemize works
echo -e "${YELLOW}Testing piper_phonemize...${NC}"
TEST_OUTPUT=$(poetry run python -c "import os; os.environ['PHONEMIZE_ESPEAK_DATA']='/usr/share/espeak-ng-data'; import piper_phonemize; print(piper_phonemize.phonemize_espeak('Hello world', 'en-us', True))" 2>&1)

if [[ "$TEST_OUTPUT" == *"["* && "$TEST_OUTPUT" == *"]"* ]]; then
    echo -e "${GREEN}✅ piper_phonemize is working correctly!${NC}"
    echo -e "${GREEN}Test output: $TEST_OUTPUT${NC}"
else
    echo -e "${RED}⚠️ piper_phonemize may not be working correctly.${NC}"
    echo -e "${RED}Test output: $TEST_OUTPUT${NC}"
    echo -e "${YELLOW}This may be because espeak-ng is not installed or not the rhasspy fork.${NC}"
    echo -e "${YELLOW}Continuing anyway, but expect issues with preprocessing.${NC}"
fi

echo -e "${GREEN}Starting Piper TTS Trainer GUI...${NC}"
echo "The interface will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop the server"
echo

# Run the GUI script with Poetry
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

# Create PowerShell script for Windows users
cat > "$PIPER_HOME/run_piper_gui.ps1" << 'EOL'
# PowerShell script to run Piper TTS Trainer GUI in WSL

Write-Host "Piper TTS Trainer - GUI Launcher" -ForegroundColor Cyan

# Check if WSL is available
$wslCheck = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL does not appear to be installed or working correctly." -ForegroundColor Red
    Write-Host "Please install WSL by running: wsl --install" -ForegroundColor Yellow
    Exit 1
}

# Get project directory path
$projectDir = $PWD.Path.Replace('\', '/')
$wslPath = "/mnt/c" + $projectDir.Substring(2)
Write-Host "Project directory: $wslPath" -ForegroundColor Cyan

# Make sure the script is executable
Write-Host "Setting execute permissions on scripts..." -ForegroundColor Cyan
wsl -e chmod +x run_gui.sh

# Run the GUI script in WSL
Write-Host "Starting Piper TTS Trainer GUI in WSL..." -ForegroundColor Green
Write-Host "The interface will be available at: http://localhost:7860" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
wsl -e bash -c "cd '$wslPath' && ./run_gui.sh"
EOL

# Step 6: Install packages with Poetry
echo -e "${YELLOW}Installing packages with Poetry...${NC}"
# Remove keyring reference and just install
poetry install --no-interaction

# Step 7: Install Cython explicitly (needed for monotonic align)
echo -e "${YELLOW}Installing Cython explicitly...${NC}"
poetry run pip install cython
echo -e "${GREEN}✅ Cython installed${NC}"

# Step 8: Try to install pre-compiled phonemize wheel
echo -e "${YELLOW}Attempting to install piper_phonemize wheel for Python 3.11...${NC}"
cd "$PIPER_HOME/downloads"
WHEEL_FILES=$(find . -name "piper_phonemize*.whl")

if [ ! -z "$WHEEL_FILES" ]; then
    echo -e "${YELLOW}Using existing wheel files: ${WHEEL_FILES}${NC}"
    
    # First uninstall any existing piper_phonemize
    poetry run pip uninstall -y piper_phonemize &>/dev/null || true
    
    INSTALL_SUCCESS=false
    for wheel_file in $WHEEL_FILES; do
        echo -e "${YELLOW}Trying to install: $wheel_file${NC}"
        if poetry run pip install --force-reinstall "$wheel_file" 2>/dev/null; then
            echo -e "${GREEN}✅ Successfully installed $wheel_file${NC}"
            INSTALL_SUCCESS=true
            break
        else
            echo -e "${RED}❌ Failed to install $wheel_file${NC}"
        fi
    done
    
    if [ "$INSTALL_SUCCESS" = false ]; then
        echo -e "${RED}❌ Failed to install any wheel files.${NC}"
        echo -e "${YELLOW}You may need compatible wheel files for Python 3.11 from:${NC}"
        echo -e "${YELLOW}https://github.com/rhasspy/piper-phonemize/releases/tag/v1.0.0${NC}"
    fi
else
    echo -e "${YELLOW}No wheel files found in downloads directory.${NC}"
    echo -e "${YELLOW}Please add wheel files to:${NC}"
    echo -e "${YELLOW}C:/Users/User/Documents/GitHub/Piper-TTS-Trainer/downloads${NC}"
    echo -e "${YELLOW}or $PIPER_HOME/downloads${NC}"
fi

# Step 9: Build monotonic align
echo -e "${YELLOW}Building monotonic align module...${NC}"
echo "Full logs will be saved to $PIPER_HOME/$LOG_DIR/build_monotonic_align.log"
cd "$PIPER_HOME/piper/src/python"
poetry run bash build_monotonic_align.sh 2>&1 | tee "$PIPER_HOME/$LOG_DIR/build_monotonic_align.log"

# Return to project root
cd "$PIPER_HOME"

echo
echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}Poetry Setup Complete!${NC}"
echo
echo -e "${YELLOW}NOTE: You need piper_phonemize for Python 3.11 to use the trainer${NC}"
echo -e "${YELLOW}If it wasn't installed automatically, download from:${NC}"
echo -e "${GREEN}https://github.com/rhasspy/piper-phonemize/releases/tag/v1.0.0${NC}"
echo
echo -e "${YELLOW}You can now use the following commands:${NC}"
echo -e "${GREEN}  - ./run_gui.sh                       # Start the Piper TTS Trainer GUI${NC}"
echo -e "${GREEN}  - ./run_recording_studio.sh          # Start the Piper Recording Studio${NC}"
echo -e "${GREEN}  - ./export_dataset.sh <lang> <dir>   # Export recording data${NC}"
echo
echo -e "${YELLOW}For Windows users:${NC}"
echo -e "${GREEN}  - run_piper_gui.ps1                  # PowerShell script to run the GUI${NC}"
echo
echo -e "${YELLOW}Installation logs are saved in the $PIPER_HOME/$LOG_DIR directory${NC}"
echo -e "${GREEN}=========================================================${NC}"
