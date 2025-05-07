#!/bin/bash
# Piper TTS Trainer - GUI Launcher
# This script activates the virtual environment and launches the Piper TTS Trainer GUI

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Set PYTHONPATH to include Piper modules
export PYTHONPATH="$SCRIPT_DIR/piper/src/python:$SCRIPT_DIR/piper/src:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"

echo "Starting Piper TTS Trainer GUI..."
echo "The interface will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop the server"
echo

# Run the GUI script
python "$SCRIPT_DIR/piper_trainer_gui.py"
