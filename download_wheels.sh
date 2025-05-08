#!/bin/bash
# Piper TTS Trainer - Wheel Downloader
# Downloads appropriate piper_phonemize wheel files based on Python version

set -e  # Exit on error

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}Piper TTS Trainer - Wheel Downloader${NC}"
echo -e "${YELLOW}=========================================================${NC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create downloads directory if it doesn't exist
mkdir -p downloads
cd downloads

# Clear any empty wheel files (0 byte files)
find . -name "piper_phonemize*.whl" -size 0 -delete

# Get Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_SHORT=$(python3 -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    ARCH_NAME="x86_64"
elif [ "$ARCH" = "aarch64" ]; then
    ARCH_NAME="aarch64"
else
    ARCH_NAME="x86_64"  # Default to x86_64 if unsure
fi

echo -e "${YELLOW}Python version: ${PYTHON_VERSION} (cp${PYTHON_SHORT})${NC}"
echo -e "${YELLOW}Architecture: ${ARCH} (${ARCH_NAME})${NC}"

# URLs to try
URLS=(
    "https://github.com/rhasspy/piper-phonemize/releases/download/v1.0.0/piper_phonemize-1.0.0-cp${PYTHON_SHORT}-cp${PYTHON_SHORT}-manylinux_2_28_${ARCH_NAME}.whl"
    "https://github.com/rhasspy/piper-phonemize/releases/download/v1.0.0/piper_phonemize-1.0.0-py3-none-linux_${ARCH_NAME}.whl"
    "https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper_phonemize-2023.11.14.4-cp${PYTHON_SHORT}-cp${PYTHON_SHORT}-manylinux_2_28_${ARCH_NAME}.whl"
    "https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper_phonemize-2023.11.14.4-py3-none-linux_${ARCH_NAME}.whl"
)

# Download all available wheels for the user's Python version
echo -e "${YELLOW}Downloading wheel files for Python ${PYTHON_VERSION}...${NC}"

for url in "${URLS[@]}"; do
    filename=$(basename "$url")
    echo -e "${YELLOW}Trying to download: ${url}${NC}"
    
    if wget --no-verbose --no-check-certificate "$url" -O "$filename" 2>/dev/null; then
        # Check if download was successful and file is not empty
        if [ -s "$filename" ]; then
            echo -e "${GREEN}✅ Successfully downloaded ${filename}${NC}"
        else
            echo -e "${RED}❌ Downloaded file is empty, removing...${NC}"
            rm -f "$filename"
        fi
    else
        echo -e "${RED}❌ Failed to download ${filename}${NC}"
    fi
done

# If no wheels were downloaded, try the generic approach to get all available ones
if [ ! "$(ls -A *.whl 2>/dev/null)" ]; then
    echo -e "${YELLOW}No specific wheels found. Downloading all available wheels from GitHub releases...${NC}"
    
    # Get release info from GitHub API
    if command -v curl &> /dev/null; then
        releases=$(curl -s https://api.github.com/repos/rhasspy/piper-phonemize/releases)
    elif command -v wget &> /dev/null; then
        releases=$(wget -q -O - https://api.github.com/repos/rhasspy/piper-phonemize/releases)
    else
        echo -e "${RED}❌ Neither curl nor wget available. Cannot fetch releases.${NC}"
        exit 1
    fi
    
    # Parse release URLs (simplified approach)
    if [ -n "$releases" ]; then
        echo "$releases" | grep -o 'https://github.com/rhasspy/piper-phonemize/releases/download/[^"]*\.whl' | while read -r wheel_url; do
            filename=$(basename "$wheel_url")
            if wget --no-verbose --no-check-certificate "$wheel_url" -O "$filename" 2>/dev/null; then
                if [ -s "$filename" ]; then
                    echo -e "${GREEN}✅ Successfully downloaded ${filename}${NC}"
                else
                    echo -e "${RED}❌ Downloaded file is empty, removing...${NC}"
                    rm -f "$filename"
                fi
            else
                echo -e "${RED}❌ Failed to download ${filename}${NC}"
            fi
        done
    fi
fi

# Count downloaded wheels
WHEEL_COUNT=$(ls -1 *.whl 2>/dev/null | wc -l)

if [ "$WHEEL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✅ Downloaded ${WHEEL_COUNT} wheel files${NC}"
    echo -e "${YELLOW}You can now run ./run_gui.sh to try installing these wheels${NC}"
else
    echo -e "${RED}❌ No wheel files were downloaded.${NC}"
    echo -e "${YELLOW}You may need to build piper_phonemize from source:${NC}"
    echo -e "${YELLOW}Run: ./fix_phonemize_poetry.sh${NC}"
fi

# Return to the original directory
cd "$SCRIPT_DIR"

echo -e "${GREEN}Download process complete!${NC}" 