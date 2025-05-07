#!/usr/bin/env python3
"""
Fix for piper_phonemize module issues.
This script:
1. Creates a mock piper_phonemize module if the real one has issues
2. Adds necessary paths to Python's import path
"""

import os
import sys
import importlib.util
import shutil
from pathlib import Path

# Get the base directory (where this script is located)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Piper paths
PIPER_PATH = os.path.join(BASE_DIR, "piper")
PIPER_SRC = os.path.join(PIPER_PATH, "src")
PIPER_PYTHON = os.path.join(PIPER_SRC, "python")
PIPER_PHONEMIZE = os.path.join(PIPER_SRC, "piper_phonemize")

# Ensure these are in Python's import path
for path in [PIPER_PYTHON, PIPER_SRC]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Check if piper_phonemize module can be imported correctly
try:
    # Try to import the module
    import piper_phonemize
    from piper_phonemize import phonemize_espeak
    print("✓ piper_phonemize module is working correctly!")
    sys.exit(0)  # Success exit
except ImportError as e:
    print(f"✗ Error importing piper_phonemize: {e}")
    print("Creating mock module as workaround...")

# If we got here, we need to create the mock module
os.makedirs("phonemize_fix", exist_ok=True)

# Create __init__.py
with open(os.path.join("phonemize_fix", "__init__.py"), "w") as f:
    f.write("""
# Mock piper_phonemize module
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

def get_espeak_phonemes():
    """Returns a dictionary of phoneme ids for all languages"""
    return {}

def reset_espeak():
    """Reset eSpeak-ng state"""
    pass

def get_lang_phonemes(lang: str):
    """Returns a dictionary of phoneme ids for a specific language"""
    return {}

def phonemize_espeak(
    text: str,
    language: str = "en-us",
    keep_stress: bool = True,
    punctuation: Optional[str] = None,
    phoneme_separator: Optional[str] = None,
    word_separator: str = " ",
    njobs: int = 1,
) -> str:
    """Mock implementation that just returns the input text"""
    print(f"WARNING: Using mock phonemizer for {text} in {language}")
    return text
""")

print("✓ Mock module created at ./phonemize_fix")
print("Add this directory to your PYTHONPATH to use it.")

# Create a simple wrapper script for preprocess.py that uses the mock module
with open("run_preprocess.py", "w") as f:
    f.write("""#!/usr/bin/env python3
import os
import sys
import subprocess

# Add the mock module to the Python path
mock_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "phonemize_fix"))
os.environ["PYTHONPATH"] = f"{mock_dir}:{os.environ.get('PYTHONPATH', '')}"

# Find the original preprocess.py
piper_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "piper"))
preprocess_path = os.path.join(piper_dir, "src", "python", "piper_train", "preprocess.py")

# Get the Python executable
venv_python = os.path.join(piper_dir, "src", "python", ".venv", "bin", "python3")
if not os.path.exists(venv_python):
    venv_python = "python3"

# Run the preprocess script with the mock module
cmd = [venv_python, preprocess_path] + sys.argv[1:]
print(f"Running: {' '.join(cmd)}")
print(f"With PYTHONPATH: {os.environ['PYTHONPATH']}")
os.execvpe(venv_python, [venv_python, preprocess_path] + sys.argv[1:], os.environ)
""")

# Make it executable
os.chmod("run_preprocess.py", 0o755)

print("✓ Created run_preprocess.py wrapper script")
print("Usage: ./run_preprocess.py [same arguments as preprocess.py]") 