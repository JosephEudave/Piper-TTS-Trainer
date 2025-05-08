#!/bin/bash
# Piper TTS Trainer - Rebuild Phonemize Script
# This script rebuilds the piper_phonemize module after setup_poetry.sh has been run

set -e  # Exit on error

echo "========================================================="
echo "Piper TTS Trainer - Rebuild Phonemize Script"
echo "========================================================="
echo "This script will rebuild the piper_phonemize module"
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

# Define color codes for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Check if fix_phonemize_poetry.sh exists and run it if it does
if [ -f "fix_phonemize_poetry.sh" ]; then
    echo -e "${YELLOW}Running fix_phonemize_poetry.sh...${NC}"
    bash fix_phonemize_poetry.sh
    exit $?
fi

# Otherwise, continue with a more basic rebuild

echo -e "${YELLOW}==========================================================${NC}"
echo -e "${YELLOW}Rebuilding piper_phonemize C++ extension${NC}"
echo -e "${YELLOW}==========================================================${NC}"

# Install required packages
echo "Installing required packages..."
sudo apt update
sudo apt install -y build-essential cmake python3-dev libespeak-ng-dev pkg-config

# Locate espeak-ng dependencies
echo -e "${YELLOW}Locating espeak-ng dependencies...${NC}"
echo -e "${YELLOW}⚠️ Searching for espeak-ng.h in system...${NC}"

ESPEAK_HEADER=""
for header_path in "/usr/include/espeak-ng.h" "/usr/local/include/espeak-ng.h" "/usr/include/espeak-ng/espeak-ng.h"; do
    if [ -f "$header_path" ]; then
        ESPEAK_HEADER=$header_path
        echo -e "${GREEN}✅ espeak-ng.h header file found at: $ESPEAK_HEADER${NC}"
        break
    fi
done

if [ -z "$ESPEAK_HEADER" ]; then
    echo -e "${RED}❌ Error: Cannot find espeak-ng.h. Please install libespeak-ng-dev${NC}"
    echo -e "${YELLOW}Creating fix_phonemize_poetry.sh to help resolve this issue...${NC}"
    
    # Check if we can create the fix script
    if [ ! -f "fix_phonemize_poetry.sh" ]; then
        wget -q -O fix_phonemize_poetry.sh https://raw.githubusercontent.com/rhasspy/piper/master/scripts/fix_phonemize_poetry.sh 2>/dev/null
        
        if [ $? -ne 0 ]; then
            # If downloading fails, create the file from scratch
            cat > fix_phonemize_poetry.sh << 'EOL'
#!/bin/bash
# Fix script for piper_phonemize installation for Poetry environments

set -e  # Exit on error

# Define colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=========================================================${NC}"
echo -e "${YELLOW}Fixing piper_phonemize installation for Poetry${NC}"
echo -e "${YELLOW}=========================================================${NC}"

# Install required development packages
echo -e "${YELLOW}Installing required development packages...${NC}"
sudo apt update
sudo apt install -y libespeak-ng-dev autoconf automake libtool pkg-config \
                    build-essential git cmake python3-dev libsonic-dev

# Install the rhasspy fork of espeak-ng
echo -e "${YELLOW}Building and installing the rhasspy fork of espeak-ng...${NC}"
mkdir -p downloads
cd downloads

# Clone rhasspy espeak-ng fork if not already cloned
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
echo -e "${YELLOW}Preparing build system...${NC}"
./autogen.sh

echo -e "${YELLOW}Configuring espeak-ng build...${NC}"
./configure --prefix=/usr

echo -e "${YELLOW}Building espeak-ng...${NC}"
make -j$(nproc)

echo -e "${YELLOW}Installing espeak-ng (requires sudo)...${NC}"
sudo make install
sudo ldconfig

# Return to the project directory
cd ../..

# Locate espeak-ng headers and libraries
echo -e "${YELLOW}Locating espeak-ng headers and libraries...${NC}"
ESPEAK_HEADER=""
for header_path in "/usr/include/espeak-ng.h" "/usr/local/include/espeak-ng.h" "/usr/include/espeak-ng/espeak-ng.h"; do
    if [ -f "$header_path" ]; then
        ESPEAK_HEADER=$header_path
        echo -e "${GREEN}✅ espeak-ng.h header file found at: $ESPEAK_HEADER${NC}"
        break
    fi
done

if [ -z "$ESPEAK_HEADER" ]; then
    echo -e "${RED}❌ Error: espeak-ng.h header file not found${NC}"
    exit 1
fi

# Extract espeak include directory from header path
ESPEAK_INCLUDE_DIR=$(dirname "$ESPEAK_HEADER")
echo "ESPEAK_INCLUDE_DIR=$ESPEAK_INCLUDE_DIR"

# Build piper_phonemize from source
echo -e "${YELLOW}Building piper_phonemize from source...${NC}"

# Clone piper-phonemize if it doesn't exist
if [ ! -d "piper-phonemize" ]; then
    echo -e "${YELLOW}Cloning piper-phonemize repository...${NC}"
    git clone https://github.com/rhasspy/piper-phonemize.git
else
    echo -e "${YELLOW}Updating existing piper-phonemize repository...${NC}"
    cd piper-phonemize
    git pull
    cd ..
fi

# Setup directory structure
if [ ! -d "piper" ]; then
    echo -e "${YELLOW}Cloning Piper repository...${NC}"
    git clone https://github.com/rhasspy/piper.git
fi

# Copy the piper-phonemize directory structure to piper/src if it doesn't exist
if [ ! -d "piper/src/piper_phonemize" ]; then
    echo -e "${YELLOW}Creating piper_phonemize in piper/src...${NC}"
    mkdir -p "piper/src/piper_phonemize"
    cp -r "piper-phonemize"/* "piper/src/piper_phonemize/"
fi

# Build the C++ extension
cd piper/src/piper_phonemize
echo -e "${YELLOW}Setting up build environment...${NC}"

# Create a build directory
mkdir -p build
cd build

# Run CMake with correct path to espeak-ng headers
echo -e "${YELLOW}Running CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DESPEAK_INCLUDE_DIR=$ESPEAK_INCLUDE_DIR \
      ..

# Build the extension
echo -e "${YELLOW}Building...${NC}"
make -j$(nproc)

# Return to the project directory
cd ../../../..

# Install the extension using poetry pip
echo -e "${YELLOW}Installing piper_phonemize with Poetry...${NC}"
cd piper/src/piper_phonemize
poetry run pip install --force-reinstall .

# Make sure libtashkeel_model.ort is available
echo -e "${YELLOW}Setting up libtashkeel_model.ort...${NC}"
if [ ! -f "libtashkeel_model.ort" ]; then
    echo -e "${YELLOW}Downloading libtashkeel_model.ort...${NC}"
    wget --no-verbose -O libtashkeel_model.ort "https://github.com/rhasspy/piper-phonemize/raw/master/etc/libtashkeel_model.ort"
fi

# Setup environment variable file for runtime
VENV_DIR=$(poetry env info -p)
SITE_PACKAGES=$(find "$VENV_DIR" -name "site-packages" -type d | head -n 1)

echo -e "${YELLOW}Creating environment settings file...${NC}"
cat > "../../../.env_phonemize" << EOL
# This file is generated by fix_phonemize_poetry.sh
# It contains environment settings for piper_phonemize

# Location of the piper_phonemize package
export PYTHONPATH="$SITE_PACKAGES:$PYTHONPATH"

# Set this for properly locating espeak-ng data
export PHONEMIZE_ESPEAK_DATA="/usr/share/espeak-ng-data"

# Add library paths
export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH"
EOL

# Load the environment settings
source "../../../.env_phonemize"

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
cd ../../..
if poetry run python -c "import piper_phonemize; print('✅ piper_phonemize successfully imported!')" &>/dev/null; then
    echo -e "${GREEN}✅ piper_phonemize successfully installed in Poetry environment!${NC}"
    
    # Test phonemization functionality
    echo -e "${YELLOW}Testing piper_phonemize functionality...${NC}"
    TEST_RESULT=$(poetry run python -c "import piper_phonemize; print(piper_phonemize.phonemize('Hello world', 'en-us', True))" 2>&1)
    
    if [[ "$TEST_RESULT" == *"phonemes"* ]]; then
        echo -e "${GREEN}✅ piper_phonemize functionality test passed!${NC}"
        echo -e "${GREEN}Test output: $TEST_RESULT${NC}"
    else
        echo -e "${RED}⚠️ piper_phonemize installed but functionality test failed${NC}"
        echo -e "${RED}Test output: $TEST_RESULT${NC}"
        echo -e "${YELLOW}Try running: source .env_phonemize${NC}"
    fi
    
    echo -e "${GREEN}✅ Installation complete!${NC}"
    echo
    echo -e "${YELLOW}To use piper_phonemize in your scripts, add these lines:${NC}"
    echo "source .env_phonemize"
    echo
    echo -e "${YELLOW}Now you can run:${NC}"
    echo "./run_gui.sh"
else
    echo -e "${RED}❌ piper_phonemize installation failed in Poetry environment${NC}"
    exit 1
fi
EOL
        fi
        
        chmod +x fix_phonemize_poetry.sh
    fi
    
    echo -e "${YELLOW}Please run this command to fix your installation:${NC}"
    echo -e "${GREEN}./fix_phonemize_poetry.sh${NC}"
    exit 1
fi

# Extract espeak include directory from header path
ESPEAK_INCLUDE_DIR=$(dirname "$ESPEAK_HEADER")
echo "ESPEAK_INCLUDE_DIR=$ESPEAK_INCLUDE_DIR"

# Find or create piper_phonemize directory
if [ -d "piper/src/piper_phonemize" ]; then
    cd piper/src/piper_phonemize
elif [ -d "piper-phonemize" ]; then
    cd piper-phonemize
else
    echo -e "${RED}❌ Error: Cannot find piper_phonemize directory${NC}"
    echo -e "${YELLOW}Please run fix_phonemize_poetry.sh to set up the environment:${NC}"
    echo -e "${GREEN}./fix_phonemize_poetry.sh${NC}"
    exit 1
fi

# Build the C++ extension
echo -e "${YELLOW}Setting up build environment...${NC}"

# Create a build directory
mkdir -p build
cd build

# Run CMake with correct path to espeak-ng headers
echo -e "${YELLOW}Running CMake...${NC}"
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DESPEAK_INCLUDE_DIR=$ESPEAK_INCLUDE_DIR \
      ..

# Build the extension
echo -e "${YELLOW}Building...${NC}"
make -j$(nproc)

# Return to the project directory
cd ../../..
if [ -d "piper-phonemize" ]; then
    cd ..  # One level less if we started from piper-phonemize
fi

# Install the extension using poetry pip
echo -e "${YELLOW}Installing piper_phonemize with Poetry...${NC}"
if [ -d "piper/src/piper_phonemize" ]; then
    cd piper/src/piper_phonemize
elif [ -d "piper-phonemize" ]; then
    cd piper-phonemize
fi

# Make a backup of the setup.py file in case we need to modify it
if [ -f "setup.py" ]; then
    cp setup.py setup.py.bak
fi

poetry run pip install --force-reinstall .

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
cd ../../..
if [ -d "piper-phonemize" ]; then
    cd ..  # One level less if we started from piper-phonemize
fi

if poetry run python -c "import piper_phonemize; print('piper_phonemize imported successfully!')" &>/dev/null; then
    echo -e "${GREEN}✅ piper_phonemize successfully installed!${NC}"
    
    # Test actual phonemization
    echo -e "${YELLOW}Testing phonemization...${NC}"
    TEST_OUTPUT=$(poetry run python -c "import piper_phonemize; result = piper_phonemize.phonemize('Hello world', 'en-us', True); print('success' if 'phonemes' in str(result) else 'failed')" 2>/dev/null)
    
    if [ "$TEST_OUTPUT" = "success" ]; then
        echo -e "${GREEN}✅ Phonemization working correctly!${NC}"
    else
        echo -e "${RED}⚠️ Installed but phonemization not working correctly.${NC}"
        echo -e "${YELLOW}This might be due to espeak-ng issues. Try running fix_phonemize_poetry.sh:${NC}"
        echo -e "${GREEN}./fix_phonemize_poetry.sh${NC}"
    fi
else
    echo -e "${RED}❌ Failed to install piper_phonemize${NC}"
    echo -e "${YELLOW}Please try running fix_phonemize_poetry.sh to fix this issue:${NC}"
    echo -e "${GREEN}./fix_phonemize_poetry.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Rebuild complete!${NC}"
