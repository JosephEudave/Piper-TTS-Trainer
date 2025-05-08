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

# 1. Check if Cython is installed, install if needed
echo -e "${YELLOW}Checking Cython installation...${NC}"
if ! poetry run python -c "import Cython" &>/dev/null; then
    echo -e "${YELLOW}Cython not found, installing...${NC}"
    poetry run pip install cython
    echo -e "${GREEN}✅ Cython installed${NC}"
else
    echo -e "${GREEN}✅ Cython is already installed${NC}"
fi

# 2. Check if piper_phonemize is already installed
echo -e "${YELLOW}Checking piper_phonemize installation...${NC}"
if poetry run python -c "import piper_phonemize; print('piper_phonemize is installed')" &>/dev/null; then
    echo -e "${GREEN}✅ piper_phonemize is already installed${NC}"
else
    echo -e "${YELLOW}piper_phonemize not found, looking for wheel files...${NC}"
    
    # Find wheel files in downloads directory
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
        
        # Try to install each wheel file until one succeeds
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

# Set required environment variables
export PHONEMIZE_ESPEAK_DATA="/usr/share/espeak-ng-data"
export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Test if phonemize works
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
