# Piper TTS Trainer GUI

This graphical user interface provides a complete end-to-end workflow for creating your own text-to-speech voice using the Piper TTS system. It streamlines dataset preparation, audio normalization, preprocessing, training, and model export.

## Features

- **Dataset Preparation**: Import audio files and create properly formatted datasets
- **Audio Normalization**: Fix audio clipping issues to prevent "audio amplitude out of range" warnings
- **Dataset Preprocessing**: Convert raw audio datasets into training-ready format
- **Model Training**: Train your TTS model with customizable parameters
- **Model Export**: Convert your trained models to ONNX format for inference

## Requirements

All dependencies are specified in the main `requirements.txt` file, including:

```
torch>=2.2.0
librosa>=0.9.2
soundfile==0.13.1
numpy>=1.21.0,<=1.26.4
gradio==4.8.0
```

## Getting Started

1. Make sure you have set up the Piper repository according to the main installation instructions
2. Run the GUI with:

```bash
python piper_trainer_gui.py
```

The GUI will be available at http://localhost:7860 in your web browser.

## Workflow

### 1. Prepare Dataset

1. Enter the directory containing your audio files (WAV, MP3, FLAC)
2. Optionally provide a transcription file (format: `filename|text`)
3. Click "Prepare Dataset" to create a properly structured dataset

### 2. Normalize Audio (Important!)

This step scales your audio files to prevent the "audio amplitude out of range, auto clipped" warning:

1. Select a dataset to normalize
2. Set your desired target peak amplitude (0.95 recommended)
3. Click "Normalize Audio" to create a normalized version of your dataset

### 3. Preprocess Dataset

1. Select either a "Regular Dataset" or "Normalized Dataset"
2. Choose the language and format settings
3. Click "Preprocess Dataset" to prepare it for training

### 4. Train Model

1. Select the preprocessed training directory
2. Configure training parameters (batch size, epochs, quality)
3. Optionally select a checkpoint to continue training
4. Click "Start Training" to begin the training process

### 5. Export Model

1. Select a checkpoint to export
2. Enter a name for your model and configure language/quality
3. Click "Export Model" to convert to ONNX format

## Folder Structure

The GUI organizes files into these directories:

- `datasets/` - Raw audio datasets
- `normalized_wavs/` - Normalized audio datasets
- `training/` - Preprocessed data ready for training
- `checkpoints/` - Downloaded or specified starting checkpoints
- `models/` - Exported ONNX models and config files
- `logs/` - Training logs

## Tips

- Use the "Download Starter Checkpoint" option for faster training through transfer learning
- For faster inference, keep "Enable Streaming Mode" checked when exporting models
- GPU training is automatically detected and enabled when available 