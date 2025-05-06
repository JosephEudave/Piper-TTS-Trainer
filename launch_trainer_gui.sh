#!/bin/bash
# Launch the Piper TTS Trainer GUI

cd "$(dirname "$0")"
source piper/src/python/.venv/bin/activate
python3 piper_trainer_gui.py 