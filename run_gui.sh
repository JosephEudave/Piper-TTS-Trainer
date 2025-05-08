#!/bin/bash
# Piper TTS Trainer - GUI Launcher (Poetry version)
# This script uses Poetry to launch the Piper TTS Trainer GUI

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PYTHONPATH to include Piper modules
export PYTHONPATH="$SCRIPT_DIR/piper/src/python:$SCRIPT_DIR/piper/src:$PYTHONPATH"

echo "Starting Piper TTS Trainer GUI..."
echo "The interface will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop the server"
echo

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry is not installed. Please install Poetry first:"
    echo "curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Build monotonic_align (needed for training)
if [ -f "$SCRIPT_DIR/piper/src/python/build_monotonic_align.sh" ]; then
    echo "Building monotonic_align..."
    (cd "$SCRIPT_DIR/piper/src/python" && bash build_monotonic_align.sh)
fi

# Check if the dependencies are installed
echo "Checking and installing dependencies with Poetry..."
poetry install --no-interaction

# If installation was successful, run the application
if [ $? -eq 0 ]; then
    echo "Starting Piper TTS Trainer GUI..."
    poetry run python piper_trainer_gui.py
else
    echo "Failed to install dependencies. Please check the error messages above."
    exit 1
fi