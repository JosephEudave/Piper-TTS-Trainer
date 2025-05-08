#!/bin/bash
# Script to install and configure espeak-ng for piper_phonemize

set -e  # Exit on error

echo "=== Installing and configuring espeak-ng ==="
echo "This script will install espeak-ng and its development files needed by piper_phonemize"

# Update package repositories
echo "Updating package repositories..."
sudo apt update

# Install espeak-ng and development files
echo "Installing espeak-ng and development files..."
sudo apt install -y espeak-ng libespeak-ng-dev pkg-config cmake build-essential

# Verify installation
echo "Verifying installation..."
if command -v espeak-ng > /dev/null; then
    echo "✅ espeak-ng is installed"
else
    echo "❌ Error: espeak-ng installation failed"
    exit 1
fi

# Verify header files
if [ -f "/usr/include/espeak-ng.h" ]; then
    echo "✅ espeak-ng.h header file found"
else
    echo "❌ Error: espeak-ng.h header file not found"
    echo "Looking for espeak-ng.h in alternative locations..."
    find /usr -name "espeak-ng.h" 2>/dev/null || echo "Not found anywhere in /usr"
    exit 1
fi

# Verify library files
ESPEAK_LIB=""
for lib_path in "/usr/lib/libespeak-ng.so" "/usr/lib/x86_64-linux-gnu/libespeak-ng.so"; do
    if [ -f "$lib_path" ]; then
        ESPEAK_LIB=$lib_path
        echo "✅ libespeak-ng.so found at: $ESPEAK_LIB"
        break
    fi
done

if [ -z "$ESPEAK_LIB" ]; then
    echo "❌ Error: libespeak-ng.so not found"
    echo "Looking for libespeak-ng.so in alternative locations..."
    find /usr -name "libespeak-ng.so" 2>/dev/null || echo "Not found anywhere in /usr"
    exit 1
fi

echo
echo "=== espeak-ng installation successful ==="
echo "The next step is to run the rebuild_phonemize.sh script:"
echo "  chmod +x rebuild_phonemize.sh"
echo "  ./rebuild_phonemize.sh"
echo 