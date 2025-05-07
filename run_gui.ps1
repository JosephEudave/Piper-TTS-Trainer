# Piper TTS Trainer - GUI Launcher (Windows)
# This script activates the virtual environment and launches the Piper TTS Trainer GUI

# Get the directory where this script is located
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

# Activate the virtual environment
& "$SCRIPT_DIR\.venv\Scripts\Activate.ps1"

Write-Host "Starting Piper TTS Trainer GUI..."
Write-Host "The interface will be available at: http://localhost:7860"
Write-Host "Press Ctrl+C to stop the server"
Write-Host ""

# Run the GUI script
python "$SCRIPT_DIR\piper_trainer_gui.py" 