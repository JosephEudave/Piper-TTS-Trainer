#!/bin/bash
# Piper TTS Trainer - Pre-compiled Phonemize Installer
# This script installs the pre-compiled piper_phonemize package from GitHub releases
# And installs the rhasspy fork of espeak-ng with the TextToPhonemesWithTerminator function

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}Installing pre-compiled piper_phonemize package${NC}"
echo -e "${YELLOW}=========================================================${NC}"

# Create a directory for downloads if it doesn't exist
mkdir -p downloads
cd downloads

# Check if any wheel files exist
WHEEL_FILES=$(find . -name "piper_phonemize*.whl")
if [ -z "$WHEEL_FILES" ]; then
    echo -e "${YELLOW}No wheel files found in downloads directory.${NC}"
    echo -e "${YELLOW}Downloading piper_phonemize wheel file...${NC}"
    
    # Try multiple version URLs
    PACKAGE_VERSION="1.0.0"
    PACKAGE_URL="https://github.com/rhasspy/piper-phonemize/releases/download/v${PACKAGE_VERSION}/piper_phonemize-${PACKAGE_VERSION}-py3-none-linux_x86_64.whl"
    PACKAGE_FILE="piper_phonemize-${PACKAGE_VERSION}-py3-none-linux_x86_64.whl"
    
    wget --no-verbose "$PACKAGE_URL" -O "$PACKAGE_FILE" || {
        echo -e "${YELLOW}Failed to download v1.0.0, trying alternative version...${NC}"
        PACKAGE_VERSION="2023.11.14-4"
        PACKAGE_URL="https://github.com/rhasspy/piper-phonemize/releases/download/${PACKAGE_VERSION}/piper_phonemize-${PACKAGE_VERSION}-py3-none-linux_x86_64.whl"
        PACKAGE_FILE="piper_phonemize-${PACKAGE_VERSION}-py3-none-linux_x86_64.whl"
        wget --no-verbose "$PACKAGE_URL" -O "$PACKAGE_FILE" || {
            echo -e "${RED}Failed to download wheel files.${NC}"
            exit 1
        }
    }
    
    WHEEL_FILES=$(find . -name "piper_phonemize*.whl")
fi

echo -e "${YELLOW}Found wheel files: ${WHEEL_FILES}${NC}"

# Check if espeak-ng is installed from the standard package
if command -v espeak-ng &> /dev/null; then
    # The standard espeak-ng is installed, but we need the rhasspy fork
    echo -e "${YELLOW}Standard espeak-ng detected. Piper requires the rhasspy fork with TextToPhonemesWithTerminator.${NC}"
    echo -e "${YELLOW}We will build and install the rhasspy version of espeak-ng...${NC}"
    
    # Install dependencies for building espeak-ng
    echo -e "${YELLOW}Installing build dependencies...${NC}"
    sudo apt update
    sudo apt install -y build-essential git autoconf automake libtool pkg-config libsonic-dev
    
    # Clone rhasspy espeak-ng fork
    cd "$SCRIPT_DIR/downloads"
    if [ ! -d "espeak-ng" ]; then
        echo -e "${YELLOW}Cloning rhasspy/espeak-ng repository...${NC}"
        git clone https://github.com/rhasspy/espeak-ng.git
    else
        echo -e "${YELLOW}Updating existing rhasspy/espeak-ng repository...${NC}"
        cd espeak-ng
        git pull
        cd ..
    fi
    
    # Build and install espeak-ng from the fork
    cd espeak-ng
    if [ ! -f "configure" ]; then
        echo -e "${YELLOW}Preparing build system...${NC}"
        ./autogen.sh
    fi
    
    echo -e "${YELLOW}Configuring espeak-ng build...${NC}"
    ./configure --prefix=/usr
    
    echo -e "${YELLOW}Building espeak-ng...${NC}"
    make -j$(nproc)
    
    echo -e "${YELLOW}Installing espeak-ng (requires sudo)...${NC}"
    sudo make install
    
    # Verify installation
    if command -v espeak-ng &> /dev/null; then
        echo -e "${GREEN}✅ rhasspy/espeak-ng successfully installed!${NC}"
    else
        echo -e "${RED}❌ Failed to install rhasspy/espeak-ng${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}espeak-ng not found, installing rhasspy fork...${NC}"
    
    # Install dependencies for building espeak-ng
    echo -e "${YELLOW}Installing build dependencies...${NC}"
    sudo apt update
    sudo apt install -y build-essential git autoconf automake libtool pkg-config libsonic-dev
    
    # Clone rhasspy espeak-ng fork
    cd "$SCRIPT_DIR/downloads"
    if [ ! -d "espeak-ng" ]; then
        echo -e "${YELLOW}Cloning rhasspy/espeak-ng repository...${NC}"
        git clone https://github.com/rhasspy/espeak-ng.git
    fi
    
    # Build and install espeak-ng from the fork
    cd espeak-ng
    echo -e "${YELLOW}Preparing build system...${NC}"
    ./autogen.sh
    
    echo -e "${YELLOW}Configuring espeak-ng build...${NC}"
    ./configure --prefix=/usr
    
    echo -e "${YELLOW}Building espeak-ng...${NC}"
    make -j$(nproc)
    
    echo -e "${YELLOW}Installing espeak-ng (requires sudo)...${NC}"
    sudo make install
    
    # Verify installation
    if command -v espeak-ng &> /dev/null; then
        echo -e "${GREEN}✅ rhasspy/espeak-ng successfully installed!${NC}"
    else
        echo -e "${RED}❌ Failed to install rhasspy/espeak-ng${NC}"
        exit 1
    fi
fi

# Return to the project directory
cd "$SCRIPT_DIR"

# Remove any existing installations of piper_phonemize
echo -e "${YELLOW}Removing any existing piper_phonemize installations...${NC}"
poetry run pip uninstall -y piper_phonemize 2>/dev/null || true

# Try to install each wheel file until one succeeds
cd downloads
INSTALLATION_SUCCESS=false

for wheel_file in $WHEEL_FILES; do
    echo -e "${YELLOW}Trying to install wheel file: $wheel_file${NC}"
    if poetry run pip install --force-reinstall "$wheel_file" 2>/dev/null; then
        echo -e "${GREEN}✅ Successfully installed $wheel_file${NC}"
        INSTALLATION_SUCCESS=true
        break
    else
        echo -e "${RED}❌ Failed to install $wheel_file${NC}"
    fi
done

# Return to the project directory
cd "$SCRIPT_DIR"

# Verify installation
if [ "$INSTALLATION_SUCCESS" = true ]; then
    echo -e "${YELLOW}Verifying installation...${NC}"
    if poetry run python -c "import piper_phonemize; print('✅ piper_phonemize successfully imported!')" &>/dev/null; then
        echo -e "${GREEN}✅ piper_phonemize successfully installed!${NC}"
        
        # Setup libtashkeel_model.ort (needed for Arabic diacritization)
        echo -e "${YELLOW}Setting up libtashkeel_model.ort...${NC}"
        VENV_DIR=$(poetry env info -p)
        SITE_PACKAGES=$(find "$VENV_DIR" -name "site-packages" -type d | head -n 1)
        PHONEMIZE_DIR=$(find "$SITE_PACKAGES" -name "piper_phonemize" -type d | head -n 1)
        
        if [ -n "$PHONEMIZE_DIR" ] && [ ! -f "$PHONEMIZE_DIR/libtashkeel_model.ort" ]; then
            # Download the model file if needed
            if [ ! -f "downloads/libtashkeel_model.ort" ]; then
                echo -e "${YELLOW}Downloading libtashkeel_model.ort...${NC}"
                wget --no-verbose -O "downloads/libtashkeel_model.ort" "https://github.com/rhasspy/piper-phonemize/raw/master/etc/libtashkeel_model.ort"
            fi
            
            # Copy to the package directory
            cp "downloads/libtashkeel_model.ort" "$PHONEMIZE_DIR/"
            echo -e "${GREEN}✅ Installed libtashkeel_model.ort${NC}"
        fi
        
        # Create environment variable file for runtime
        echo -e "${YELLOW}Creating environment settings for runtime...${NC}"
        cat > ".env_phonemize" << EOL
# This file is generated by install_precompiled_phonemize.sh
# It contains environment settings for piper_phonemize

# Location of the piper_phonemize package
PIPER_PHONEMIZE_DIR="$PHONEMIZE_DIR"

# Set this in your scripts to properly locate espeak-ng data
export PHONEMIZE_ESPEAK_DATA="/usr/share/espeak-ng-data"
EOL
        
        # Test actual phonemization functionality
        echo -e "${YELLOW}Testing piper_phonemize functionality...${NC}"
        export PHONEMIZE_ESPEAK_DATA="/usr/share/espeak-ng-data"
        TEST_RESULT=$(poetry run python -c "import piper_phonemize; print(piper_phonemize.phonemize('Hello world', 'en-us', True))" 2>&1)
        
        if [[ "$TEST_RESULT" == *"phonemes"* ]]; then
            echo -e "${GREEN}✅ piper_phonemize functionality test passed!${NC}"
            echo -e "${GREEN}Test output: $TEST_RESULT${NC}"
        else
            echo -e "${RED}⚠️ piper_phonemize installed but functionality test failed${NC}"
            echo -e "${RED}Test output: $TEST_RESULT${NC}"
            echo -e "${YELLOW}This might be due to missing espeak-ng-data or other runtime dependencies.${NC}"
            echo -e "${YELLOW}Try manually setting: export PHONEMIZE_ESPEAK_DATA=/usr/share/espeak-ng-data${NC}"
        fi
        
        echo -e "${GREEN}✅ Installation complete!${NC}"
        exit 0
    else
        echo -e "${RED}❌ Installation verification failed${NC}"
    fi
else
    echo -e "${RED}❌ Failed to install any wheel files${NC}"
fi

echo -e "${YELLOW}Falling back to building from source...${NC}"

# Try building from source as fallback
if [ -f "rebuild_phonemize.sh" ]; then
    bash "rebuild_phonemize.sh"
    
    # Verify again
    if poetry run python -c "import piper_phonemize; print('✅ piper_phonemize successfully imported!')" &>/dev/null; then
        echo -e "${GREEN}✅ Successfully built and installed piper_phonemize from source${NC}"
        exit 0
    else
        echo -e "${RED}❌ Failed to install piper_phonemize${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ rebuild_phonemize.sh not found${NC}"
    
    # As a last resort, try running fix_phonemize_poetry.sh if it exists
    if [ -f "fix_phonemize_poetry.sh" ]; then
        echo -e "${YELLOW}Trying fix_phonemize_poetry.sh as a last resort...${NC}"
        bash "fix_phonemize_poetry.sh"
        exit $?
    else
        exit 1
    fi
fi 