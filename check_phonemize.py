#!/usr/bin/env python3
"""
Diagnostic script to check the piper_phonemize module installation.
"""
import os
import sys
import importlib.util
import subprocess

# Get current directory
PIPER_HOME = os.path.abspath(os.path.dirname(__file__))
PIPER_SRC = os.path.join(PIPER_HOME, "piper", "src")
PIPER_PYTHON = os.path.join(PIPER_SRC, "python")
PIPER_VENV_BIN = os.path.join(PIPER_PYTHON, ".venv", "bin")
PHONEMIZE_PATH = os.path.join(PIPER_SRC, "piper_phonemize")

# Print paths for debugging
print("=== PATHS ===")
print(f"PIPER_HOME: {PIPER_HOME}")
print(f"PIPER_SRC: {PIPER_SRC}")
print(f"PIPER_PYTHON: {PIPER_PYTHON}")
print(f"PHONEMIZE_PATH: {PHONEMIZE_PATH}")

# Check if the phonemize directory exists
print("\n=== DIRECTORY CHECK ===")
if os.path.exists(PHONEMIZE_PATH):
    print(f"✓ piper_phonemize directory exists at {PHONEMIZE_PATH}")
    # List content
    print("Directory contents:")
    for item in os.listdir(PHONEMIZE_PATH):
        item_path = os.path.join(PHONEMIZE_PATH, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}")
        else:
            print(f"  📄 {item}")
else:
    print(f"✗ piper_phonemize directory NOT FOUND at {PHONEMIZE_PATH}")

# Add piper src to Python path
sys.path.insert(0, PIPER_SRC)
sys.path.insert(0, PIPER_PYTHON)

# Try to import the module
print("\n=== IMPORT TEST ===")
try:
    import piper_phonemize
    print(f"✓ Successfully imported piper_phonemize module from {piper_phonemize.__file__}")
    
    # Check for specific function
    if hasattr(piper_phonemize, "phonemize_espeak"):
        print("✓ phonemize_espeak function exists in the module")
    else:
        print("✗ phonemize_espeak function NOT FOUND in the module")
        print("Available attributes:")
        for attr in dir(piper_phonemize):
            if not attr.startswith("__"):
                print(f"  - {attr}")
except ImportError as e:
    print(f"✗ Failed to import piper_phonemize: {e}")

# Try to find the module in Python path
print("\n=== MODULE SEARCH ===")
spec = importlib.util.find_spec("piper_phonemize")
if spec:
    print(f"✓ Found module at: {spec.origin}")
else:
    print("✗ Module not found in sys.path")
    print("Current sys.path:")
    for p in sys.path:
        print(f"  - {p}")

# Check if it's installed in the virtual environment
print("\n=== VENV CHECK ===")
python_exe = os.path.join(PIPER_VENV_BIN, "python3")
if not os.path.exists(python_exe):
    python_exe = os.path.join(PIPER_PYTHON, ".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"

try:
    result = subprocess.run(
        [python_exe, "-c", "import piper_phonemize; print('Found at:', piper_phonemize.__file__)"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✓ Module found in venv: {result.stdout.strip()}")
    else:
        print(f"✗ Module not found in venv: {result.stderr.strip()}")
        
        # Try to list installed packages
        pip_cmd = [python_exe, "-m", "pip", "list"]
        pip_result = subprocess.run(pip_cmd, capture_output=True, text=True)
        if pip_result.returncode == 0:
            print("\nInstalled packages:")
            for line in pip_result.stdout.strip().split("\n")[:20]:  # Show first 20 packages
                if "piper" in line.lower():
                    print(f"  > {line}")  # Highlight piper packages
                else:
                    print(f"    {line}")
            if len(pip_result.stdout.strip().split("\n")) > 20:
                print("    ... (more packages not shown)")
except Exception as e:
    print(f"✗ Error checking venv: {e}")

print("\n=== RECOMMENDATION ===")
print("If piper_phonemize directory exists but module can't be imported:")
print("1. Make sure it's installed in the virtual environment:")
print(f"   cd {PHONEMIZE_PATH}")
print(f"   {python_exe} -m pip install -e .")
print("2. Make sure the C++ dependencies are installed:")
print("   sudo apt install -y libespeak-ng-dev")
print("3. Make sure PYTHONPATH includes both directories:")
print(f"   export PYTHONPATH={PIPER_SRC}:{PIPER_PYTHON}:$PYTHONPATH") 