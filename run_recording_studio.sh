#!/bin/bash
# Piper Recording Studio Launcher (Poetry version)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Piper Recording Studio..."
echo "The interface will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"
echo

# Run the Recording Studio with Poetry
cd "$SCRIPT_DIR/piper-recording-studio"
poetry run python -m piper_recording_studio
