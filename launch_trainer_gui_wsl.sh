#!/bin/bash
# Piper TTS Trainer GUI - WSL Launcher

echo "========================================================="
echo "Piper TTS Trainer GUI - WSL Launcher"
echo "========================================================="

# Get the directory where this script is located
PIPER_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PIPER_HOME"

# Activate Python virtual environment
source "$PIPER_HOME/piper/src/python/.venv/bin/activate"

# Launch the GUI application
python3 "$PIPER_HOME/piper_trainer_gui.py"

# Check if there was an error
if [ $? -ne 0 ]; then
    echo
    echo "An error occurred while launching the GUI."
    echo "Please check that all dependencies are installed."
    echo
    read -p "Press Enter to continue..."
fi 