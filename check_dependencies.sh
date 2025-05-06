#!/bin/bash
# Check and install required dependencies for Piper TTS Trainer GUI

echo "========================================================="
echo "Checking dependencies for Piper TTS Trainer GUI"
echo "========================================================="

# Activate virtual environment
source "$(dirname "$0")/piper/src/python/.venv/bin/activate"

# Check and install required packages
echo "Checking and installing required packages..."
pip install --upgrade pip
pip install gradio requests torch torchvision torchaudio

# Check if packages are installed
echo -e "\nChecking installed packages..."
python3 -c "
import sys
packages = ['gradio', 'requests', 'torch']
missing = []
for package in packages:
    try:
        __import__(package)
        print(f'✓ {package} is installed')
    except ImportError:
        missing.append(package)
        print(f'✗ {package} is missing')

if missing:
    print('\nInstalling missing packages...')
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
    print('\nAll required packages are now installed')
else:
    print('\nAll required packages are installed')
"

echo -e "\nDependency check complete!" 