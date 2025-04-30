import os
import subprocess
import sys
from pathlib import Path

def setup_environment():
    # Create necessary directories
    directories = [
        "dataset/wavs",
        "output",
        "cache",
        "piper"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def install_dependencies():
    # Install pip packages
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Clone piper repository if not exists
    if not Path("piper").exists():
        subprocess.run(["git", "clone", "https://github.com/rmcpantoja/piper"], check=True)
    
    # Download resampler
    if not Path("resample.py").exists():
        subprocess.run(["wget", "https://raw.githubusercontent.com/coqui-ai/TTS/dev/TTS/bin/resample.py"], check=True)
    
    # Build monotonic align
    os.chdir("piper/src/python")
    subprocess.run(["bash", "build_monotonic_align.sh"], check=True)
    os.chdir("../..")

def main():
    print("Setting up Piper training environment...")
    setup_environment()
    print("\nInstalling dependencies...")
    install_dependencies()
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Place your .wav files in the dataset/wavs directory")
    print("2. If you have transcriptions, place them in metadata.csv in the dataset directory")
    print("3. Start Jupyter Notebook with: jupyter notebook")
    print("4. Open piper_multilingual_training_notebook_local.ipynb")

if __name__ == "__main__":
    main() 