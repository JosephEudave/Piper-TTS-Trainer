#!/bin/bash
# Environment variables for Piper TTS

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPER_SRC="$SCRIPT_DIR/piper/src"
PIPER_PYTHON="$PIPER_SRC/python"

# Set PYTHONPATH to include Piper modules
export PYTHONPATH="$PIPER_SRC:$PIPER_PYTHON:$PYTHONPATH"
echo "PYTHONPATH set to: $PYTHONPATH"
