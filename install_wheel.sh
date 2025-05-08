#!/bin/bash
# Script to manually install piper_phonemize wheel

# Define colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}Piper TTS Trainer - Manual Wheel Installer${NC}"
echo -e "${YELLOW}=========================================================${NC}"
echo

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}Looking for piper_phonemize wheel files in downloads directory...${NC}"
WHEEL_FILES=$(find "$SCRIPT_DIR/downloads" -name "piper_phonemize*.whl" 2>/dev/null)

if [ -z "$WHEEL_FILES" ]; then
    echo -e "${RED}No wheel files found in $SCRIPT_DIR/downloads directory.${NC}"
    echo -e "${YELLOW}Checking for wheels in Windows path...${NC}"
    
    # Check if WSL environment
    if grep -q Microsoft /proc/version; then
        echo -e "${YELLOW}WSL environment detected, checking Windows path...${NC}"
        # Try to copy from Windows path
        cp -f /mnt/c/Users/User/Documents/GitHub/Piper-TTS-Trainer/downloads/piper_phonemize*.whl "$SCRIPT_DIR/downloads/" 2>/dev/null || true
        WHEEL_FILES=$(find "$SCRIPT_DIR/downloads" -name "piper_phonemize*.whl" 2>/dev/null)
    fi
    
    if [ -z "$WHEEL_FILES" ]; then
        echo -e "${RED}No wheel files found after checking Windows path.${NC}"
        echo -e "${YELLOW}Please place piper_phonemize wheel files in:${NC}"
        echo -e "${YELLOW}$SCRIPT_DIR/downloads${NC}"
        echo -e "${YELLOW}Download from: https://github.com/rhasspy/piper-phonemize/releases/tag/v1.0.0${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}Found wheel files: ${WHEEL_FILES}${NC}"
echo -e "${YELLOW}Attempting to install wheel files with Poetry...${NC}"

# First uninstall any existing piper_phonemize
poetry run pip uninstall -y piper_phonemize &>/dev/null || true

# Try to install each wheel file until one succeeds
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
    echo -e "${RED}❌ Failed to install any wheel files.${NC}"
    echo -e "${YELLOW}You may need compatible wheel files for Python 3.11 from:${NC}"
    echo -e "${YELLOW}https://github.com/rhasspy/piper-phonemize/releases/tag/v1.0.0${NC}"
    exit 1
fi

# Test if piper_phonemize works
echo -e "${YELLOW}Testing piper_phonemize...${NC}"
TEST_OUTPUT=$(python -c "import os; os.environ['PHONEMIZE_ESPEAK_DATA']='/usr/share/espeak-ng-data'; import piper_phonemize; print(piper_phonemize.phonemize_espeak('Hello world', 'en-us', True))" 2>&1)

if [[ "$TEST_OUTPUT" == *"["* && "$TEST_OUTPUT" == *"]"* ]]; then
    echo -e "${GREEN}✅ piper_phonemize is working correctly!${NC}"
    echo -e "${GREEN}Test output: $TEST_OUTPUT${NC}"
else
    echo -e "${RED}⚠️ piper_phonemize may not be working correctly.${NC}"
    echo -e "${RED}Test output: $TEST_OUTPUT${NC}"
    echo -e "${YELLOW}This may be because espeak-ng is not installed or not the rhasspy fork.${NC}"
fi

echo
echo -e "${GREEN}Wheel installation complete!${NC}" 