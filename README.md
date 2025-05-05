# Piper TTS Trainer

A simplified interface for training custom voice models with Piper TTS, built for Linux environments.

## Overview

This project provides a user-friendly way to train custom text-to-speech voices using Piper TTS. It uses a Gradio web interface to make the complex process of voice training more accessible, running completely in Linux for maximum compatibility.

## Requirements

- Linux environment (Native Linux or WSL on Windows)
- Python 3.8+
- Internet connection (for downloading models and dependencies)
- For GPU acceleration (recommended): NVIDIA GPU with CUDA support

## Quick Start

1. Clone this repository:
   ```
   git clone https://github.com/josepheudave/Piper-TTS-Trainer.git
   cd Piper-TTS-Trainer
   ```

2. Run the setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

3. Launch the Gradio interface:
   ```
   ~/piper_tts_trainer/launch.sh
   ```

4. Open your browser and navigate to: http://localhost:7860

## Features

- **Pre-trained Models**: Browse and download checkpoint models from Hugging Face
- **Dataset Configuration**: Preprocess your audio datasets for training
- **Training**: Configure and run model training with GPU acceleration (if available)
- **Export**: Convert trained models to ONNX format for use with Piper

## Using the Interface

### Pre-trained Models

1. Select the language, region, voice, and quality
2. Choose a checkpoint model
3. Click "Download Selected Checkpoint"

### Dataset Configuration

1. Prepare your dataset in LJSpeech, Mycroft, or VCTK format
2. Enter the path to your dataset
3. Configure dataset parameters
4. Click "Preprocess Dataset"

### Training

1. Select your preprocessed dataset
2. Choose a checkpoint to start from
3. Configure training parameters (batch size, epochs, etc.)
4. Click "Start Training"

### Export Model

1. Enter the path to your trained checkpoint
2. Set the export directory and model name
3. Click "Export Model"

## Dataset Format

For LJSpeech format (recommended):
1. Create a folder with:
   - A `metadata.csv` file with `|` as delimiter
   - Format: `id|text`
   - A `wav` directory containing audio files named `id.wav`
2. Place your dataset in the ~/piper_tts_trainer/datasets directory

## For Windows Users (Using WSL)

If you're using Windows, we've made the WSL setup process easier with the included automation scripts:

1. **Run the WSL installation script**:
   - Double-click `install_wsl.bat` in your Windows File Explorer
   - This will install WSL and Ubuntu 22.04 if not already installed
   - Follow any on-screen prompts and restart your computer if required

2. **After WSL is installed and Ubuntu is set up**:
   - Ubuntu should open automatically (if not, open it from the Start menu)
   - Navigate to the project directory using:
     ```
     cd /mnt/c/Users/your-username/path/to/Piper-TTS-Trainer
     ```
     For example:
     ```
     cd /mnt/c/Users/josep/OneDrive/Documentos/GitHub/Piper-TTS-Trainer
     ```

3. **Run the setup script** to install all required dependencies:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```
   - This installs Python, required packages, and sets up the Piper TTS training environment

4. **Launch the application**:
   ```
   ~/piper_tts_trainer/launch.sh
   ```
   - The web interface will be accessible at http://localhost:7860

For manual WSL setup (if the script doesn't work):
1. Install WSL by running `wsl --install -d Ubuntu-22.04` in PowerShell (admin)
2. Complete the Ubuntu setup when prompted
3. Open Ubuntu and follow the steps above from step 2

## Troubleshooting

- If the interface doesn't open, check that the launch script is running correctly
- For GPU training issues, verify that CUDA is properly installed
- Dataset paths should be Linux paths (e.g., `/home/username/piper_tts_trainer/datasets/my-dataset`)

## Notes

- Training can take several hours to days depending on your hardware and dataset size
- GPU acceleration significantly improves training speed

## Resources

- [Piper TTS Documentation](https://github.com/rhasspy/piper)
- [Piper Checkpoints on Hugging Face](https://huggingface.co/datasets/rhasspy/piper-checkpoints)