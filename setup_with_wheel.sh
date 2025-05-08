#!/bin/bash
# Complete setup script for Piper TTS Trainer that:
# 1. Fixes common issues (apt_pkg) 
# 2. Installs Rust and dependencies
# 3. Installs the environment with Poetry
# 4. Manually installs piper_phonemize from wheel files

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}Piper TTS Trainer - Complete Setup Script${NC}"
echo -e "${YELLOW}=========================================================${NC}"

# Fix for apt_pkg module issues
echo -e "${YELLOW}Step 1: Fixing apt_pkg module issue...${NC}"
if ! python3 -c "import apt_pkg" &>/dev/null; then
    echo -e "${YELLOW}apt_pkg module not found, attempting to fix...${NC}"
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
    
    if python3 -c "import apt_pkg" &>/dev/null; then
        echo -e "${GREEN}✅ apt_pkg module issue fixed!${NC}"
    else
        echo -e "${RED}⚠️ apt_pkg module still not available. Continuing anyway.${NC}"
    fi
fi

# Step 2: Install Rust and required dependencies
echo -e "${YELLOW}Step 2: Installing Rust and dependencies...${NC}"
sudo apt-get update
sudo apt-get install -y build-essential curl libssl-dev pkg-config libffi-dev python3-dev

# Check if Rust is already installed
if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    echo -e "${GREEN}Rust is already installed: ${RUST_VERSION}${NC}"
else
    echo -e "${YELLOW}Installing Rust...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "${GREEN}Rust installed successfully:${NC}"
    rustc --version
fi

# Step 3: Setup Project Directory
PIPER_HOME="$(pwd)"
echo -e "${YELLOW}Step 3: Setting up project in: ${PIPER_HOME}${NC}"

# Create necessary directories
mkdir -p "$PIPER_HOME/checkpoints"
mkdir -p "$PIPER_HOME/datasets"
mkdir -p "$PIPER_HOME/training"
mkdir -p "$PIPER_HOME/models"
mkdir -p "$PIPER_HOME/normalized_wavs"
mkdir -p "$PIPER_HOME/downloads"
mkdir -p "$PIPER_HOME/logs"

# Step 4: Install Python 3.11 (without using add-apt-repository which can crash)
echo -e "${YELLOW}Step 4: Installing Python 3.11...${NC}"

# Check if Python 3.11 is already installed
if command -v python3.11 &> /dev/null; then
    echo -e "${GREEN}Python 3.11 is already installed.${NC}"
else
    echo -e "${YELLOW}Installing Python 3.11...${NC}"
    # First try to install from standard repositories
    if sudo apt-get install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils; then
        echo -e "${GREEN}Successfully installed Python 3.11.${NC}"
    else
        # Alternative method: add deadsnakes PPA manually
        echo -e "${YELLOW}Adding deadsnakes PPA manually...${NC}"
        echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/deadsnakes-ubuntu-ppa.list
        sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F23C5A6CF475977595C89F51BA6932366A755776
        sudo apt update
        
        # Try installing Python 3.11 again
        if sudo apt-get install -y python3.11 python3.11-dev python3.11-venv python3.11-distutils; then
            echo -e "${GREEN}Successfully installed Python 3.11 from deadsnakes PPA.${NC}"
        else
            echo -e "${RED}Failed to install Python 3.11. Exiting.${NC}"
            exit 1
        fi
    fi
fi

# Set Python 3.11 as default python3
echo -e "${YELLOW}Setting Python 3.11 as default python3...${NC}"
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Verify Python version
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}Python version: ${PYTHON_VERSION}${NC}"

# Step 5: Install other system dependencies
echo -e "${YELLOW}Step 5: Installing system dependencies...${NC}"
sudo apt install -y ffmpeg libespeak-ng-dev autoconf automake libtool libsonic-dev

# Step 6: Install Poetry
echo -e "${YELLOW}Step 6: Installing Poetry...${NC}"
if ! command -v poetry &> /dev/null; then
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
    echo -e "${GREEN}Poetry installed. Adding to PATH.${NC}"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# Ensure poetry is in the path
if ! command -v poetry &> /dev/null; then
    echo -e "${YELLOW}Poetry not found in PATH. Adding temporary path...${NC}"
    export PATH="$HOME/.local/bin:$PATH" 
fi

# Check again
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Poetry still not found in PATH. Please install manually.${NC}"
    echo -e "${YELLOW}Try running: curl -sSL https://install.python-poetry.org | python3 -${NC}"
    exit 1
fi

# Set Poetry to use Python 3.11
echo -e "${YELLOW}Configuring Poetry to use Python 3.11...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf .venv
fi
poetry env use python3.11

# Step 7: Install Poetry dependencies
echo -e "${YELLOW}Step 7: Installing Poetry dependencies...${NC}"
poetry install --no-interaction

# Step 8: Install piper_phonemize wheel files
echo -e "${YELLOW}Step 8: Installing piper_phonemize wheel files...${NC}"

# Check for wheel files in downloads directory
WHEEL_FILES=$(find "$PIPER_HOME/downloads" -name "piper_phonemize*.whl" 2>/dev/null)

if [ -z "$WHEEL_FILES" ]; then
    echo -e "${RED}No wheel files found in $PIPER_HOME/downloads directory.${NC}"
    echo -e "${YELLOW}Checking for wheels in Windows path...${NC}"
    
    # Check if WSL environment
    if grep -q Microsoft /proc/version; then
        echo -e "${YELLOW}WSL environment detected, checking Windows path...${NC}"
        # Try to copy from Windows path
        cp -f /mnt/c/Users/User/Documents/GitHub/Piper-TTS-Trainer/downloads/piper_phonemize*.whl "$PIPER_HOME/downloads/" 2>/dev/null || true
        WHEEL_FILES=$(find "$PIPER_HOME/downloads" -name "piper_phonemize*.whl" 2>/dev/null)
    fi
    
    if [ -z "$WHEEL_FILES" ]; then
        echo -e "${RED}No wheel files found.${NC}"
        echo -e "${YELLOW}Please download wheel files from:${NC}"
        echo -e "${YELLOW}https://github.com/rhasspy/piper-phonemize/releases/tag/v1.0.0${NC}"
        echo -e "${YELLOW}and place them in the downloads directory.${NC}"
        echo -e "${RED}Setup incomplete. Please run this script again after downloading wheel files.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Found wheel files: ${WHEEL_FILES}${NC}"

# Uninstall any existing piper_phonemize
poetry run pip uninstall -y piper_phonemize &>/dev/null || true

# Try to install each wheel file
INSTALL_SUCCESS=false
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
    echo -e "${RED}❌ Failed to install any wheel files. Setup incomplete.${NC}"
    exit 1
fi

# Step 9: Test piper_phonemize
echo -e "${YELLOW}Step 9: Testing piper_phonemize installation...${NC}"
export PHONEMIZE_ESPEAK_DATA="/usr/share/espeak-ng-data"
TEST_OUTPUT=$(poetry run python -c "import os; os.environ['PHONEMIZE_ESPEAK_DATA']='/usr/share/espeak-ng-data'; import piper_phonemize; print(piper_phonemize.phonemize_espeak('Hello world', 'en-us', True))" 2>&1)

if [[ "$TEST_OUTPUT" == *"["* && "$TEST_OUTPUT" == *"]"* ]]; then
    echo -e "${GREEN}✅ piper_phonemize is working correctly!${NC}"
    echo -e "${GREEN}Wheel-based installation successful!${NC}"
    echo -e "${GREEN}Test output: $TEST_OUTPUT${NC}"
else
    echo -e "${RED}⚠️ piper_phonemize is still not working correctly.${NC}"
    echo -e "${RED}Test output: $TEST_OUTPUT${NC}"
    echo -e "${YELLOW}This may be due to an issue with espeak-ng. Checking...${NC}"
    
    # Check that it's the rhasspy fork
    if ! python3 -c "
import ctypes
import sys
try:
    espeak_lib = ctypes.cdll.LoadLibrary('libespeak-ng.so.1')
    func = espeak_lib.espeak_TextToPhonemesWithTerminator
    print('rhasspy_fork_detected')
except (AttributeError, OSError):
    print('standard_espeak_detected')
" 2>/dev/null | grep -q "rhasspy_fork_detected"; then
        echo -e "${YELLOW}Installing rhasspy/espeak-ng fork with TextToPhonemesWithTerminator...${NC}"
        
        cd "$PIPER_HOME/downloads"
        if [ ! -d "espeak-ng" ]; then
            git clone https://github.com/rhasspy/espeak-ng.git
        else
            cd espeak-ng
            git pull
            cd ..
        fi
        
        cd espeak-ng
        ./autogen.sh
        ./configure --prefix=/usr
        make -j$(nproc)
        sudo make install
        cd "$PIPER_HOME"
        
        # Test piper_phonemize again
        TEST_OUTPUT=$(poetry run python -c "import os; os.environ['PHONEMIZE_ESPEAK_DATA']='/usr/share/espeak-ng-data'; import piper_phonemize; print(piper_phonemize.phonemize_espeak('Hello world', 'en-us', True))" 2>&1)
        
        if [[ "$TEST_OUTPUT" == *"["* && "$TEST_OUTPUT" == *"]"* ]]; then
            echo -e "${GREEN}✅ piper_phonemize is now working correctly!${NC}"
            echo -e "${GREEN}Wheel-based installation successful!${NC}"
            echo -e "${GREEN}Test output: $TEST_OUTPUT${NC}"
        else
            echo -e "${RED}⚠️ piper_phonemize still not working correctly.${NC}"
            echo -e "${RED}Test output: $TEST_OUTPUT${NC}"
            echo -e "${YELLOW}Please troubleshoot espeak-ng installation manually.${NC}"
        fi
    fi
fi

# Step 10: Create helper scripts
echo -e "${YELLOW}Step 10: Creating helper scripts...${NC}"

# Create run_gui.sh script
cat > "$PIPER_HOME/run_gui.sh" << 'EOL'
#!/bin/bash
# Piper TTS Trainer - GUI Launcher

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting Piper TTS Trainer GUI...${NC}"

# Set PYTHONPATH to include Piper modules
export PYTHONPATH="$SCRIPT_DIR/piper/src/python:$SCRIPT_DIR/piper/src:$SCRIPT_DIR/piper/src/piper_phonemize:$PYTHONPATH"

# Set espeak data directory for phonemize
export PHONEMIZE_ESPEAK_DATA="/usr/share/espeak-ng-data"
export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

echo "The interface will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop the server"
echo

# Run the GUI script with Poetry
poetry run python piper_trainer_gui.py
EOL

chmod +x "$PIPER_HOME/run_gui.sh"

echo
echo -e "${GREEN}=========================================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo
echo -e "${YELLOW}To run the Piper TTS Trainer GUI, use:${NC}"
echo -e "${GREEN}  ./run_gui.sh${NC}"
echo
echo -e "${YELLOW}If you encounter any issues with piper_phonemize, try:${NC}"
echo -e "${GREEN}  1. Make sure you have the correct wheel files for Python 3.11${NC}"
echo -e "${GREEN}  2. Ensure espeak-ng is properly installed (rhasspy fork)${NC}"
echo -e "${GREEN}  3. Set PHONEMIZE_ESPEAK_DATA environment variable${NC}"
echo
echo -e "${GREEN}=========================================================${NC}" 