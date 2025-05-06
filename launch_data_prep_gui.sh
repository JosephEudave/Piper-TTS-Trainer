#!/bin/bash
# Launch the Piper TTS Data Preparation GUI

cd "$(dirname "$0")"
source piper/src/python/.venv/bin/activate
python3 piper_data_prep_gui.py 