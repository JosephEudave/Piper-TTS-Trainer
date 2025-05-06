#!/bin/bash
# Fix Gradio version compatibility issues

echo "========================================================="
echo "Fixing Gradio version compatibility"
echo "========================================================="

# Activate virtual environment
source "$(dirname "$0")/piper/src/python/.venv/bin/activate"

# Uninstall current Gradio version
echo "Uninstalling current Gradio version..."
pip uninstall -y gradio gradio-client

# Install a known working version
echo "Installing Gradio 4.44.1..."
pip install gradio==4.44.1

echo -e "\nGradio version fix complete!" 