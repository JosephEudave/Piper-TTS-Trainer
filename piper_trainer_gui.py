#!/usr/bin/env python3
# Piper TTS Trainer - Focused GUI for Dataset Preparation, Preprocessing and Training

import gradio as gr
import os
import shutil
import subprocess
import glob
import torch
import re
from pathlib import Path

# Define paths
PIPER_HOME = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(PIPER_HOME, "datasets")
TRAINING_DIR = os.path.join(PIPER_HOME, "training")
CHECKPOINTS_DIR = os.path.join(PIPER_HOME, "checkpoints")
MODELS_DIR = os.path.join(PIPER_HOME, "models")
LOGS_DIR = os.path.join(PIPER_HOME, "logs")

# Ensure directories exist
for directory in [DATASETS_DIR, TRAINING_DIR, CHECKPOINTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Language options for preprocessing
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese"
}

# Get Python executable with venv
def get_piper_python():
    if os.name == 'nt':  # Windows
        return os.path.join(PIPER_HOME, "piper\\src\\python\\.venv\\Scripts\\python.exe")
    else:  # Linux/Mac
        return os.path.join(PIPER_HOME, "piper/src/python/.venv/bin/python3")

# Dataset preparation function
def prepare_dataset(audio_dir, transcription_file=None, output_name=None):
    try:
        # If no output name provided, use the last part of audio_dir
        if not output_name:
            output_name = os.path.basename(os.path.normpath(audio_dir))
        
        # Create dataset directory
        dataset_dir = os.path.join(DATASETS_DIR, output_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create wavs directory
        wavs_dir = os.path.join(dataset_dir, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)
        
        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.extend(glob.glob(os.path.join(audio_dir, ext)))
        
        if not audio_files:
            return f"No audio files found in {audio_dir}", None
        
        # Process each audio file
        processed_files = []
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            target_path = os.path.join(wavs_dir, os.path.splitext(filename)[0] + '.wav')
            
            # Convert to WAV if needed
            if not audio_file.lower().endswith('.wav'):
                # Use ffmpeg to convert
                cmd = ['ffmpeg', '-i', audio_file, '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1', target_path]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                shutil.copy2(audio_file, target_path)
            
            processed_files.append(os.path.splitext(filename)[0])
        
        # Create metadata.csv
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        
        # If transcription file is provided, parse it
        transcriptions = {}
        if transcription_file and os.path.exists(transcription_file):
            with open(transcription_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|', 1)
                    if len(parts) == 2:
                        name, text = parts
                        transcriptions[name] = text
        
        # Create metadata.csv file
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for name in processed_files:
                if name in transcriptions:
                    text = transcriptions[name]
                else:
                    # Create placeholder text
                    text = f"This is a sample text for {name}."
                
                f.write(f"{name}|{text}\n")
        
        return f"Successfully prepared dataset with {len(processed_files)} files in '{dataset_dir}'", dataset_dir
    
    except Exception as e:
        return f"Error preparing dataset: {str(e)}", None

# Preprocess dataset function
def preprocess_dataset(dataset_dir, language, sample_rate=22050, format="ljspeech", single_speaker=True):
    try:
        # Get dataset name from directory
        dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        output_dir = os.path.join(TRAINING_DIR, dataset_name)
        
        # Build command
        python_exe = get_piper_python()
        cmd = [
            python_exe, 
            "-m", "piper_train.preprocess",
            "--language", language,
            "--input-dir", dataset_dir,
            "--output-dir", output_dir,
            "--dataset-format", format,
            "--sample-rate", str(sample_rate)
        ]
        
        if single_speaker:
            cmd.append("--single-speaker")
        
        # Run preprocessing
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            log_output = process.stdout
            return True, output_dir, log_output
        else:
            return False, None, process.stderr
    
    except Exception as e:
        return False, None, str(e)

# Train model function
def train_model(training_dir, checkpoint_path=None, batch_size=32, epochs=6000, 
               quality="medium", precision="32"):
    try:
        python_exe = get_piper_python()
        
        # Build command
        cmd = [
            python_exe,
            "-m", "piper_train",
            "--dataset-dir", training_dir,
            "--batch-size", str(batch_size),
            "--max_epochs", str(epochs),
            "--precision", precision,
            "--validation-split", "0.0",
            "--num-test-examples", "0",
            "--checkpoint-epochs", "1"
        ]
        
        # Add quality if specified
        if quality:
            cmd.extend(["--quality", quality])
        
        # Add checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            cmd.extend(["--resume_from_checkpoint", checkpoint_path])
        
        # Check for GPU
        if torch.cuda.is_available():
            cmd.extend(["--accelerator", "gpu", "--devices", "1"])
        else:
            cmd.extend(["--accelerator", "cpu"])
        
        # Start training as a process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream the output
        for line in iter(process.stdout.readline, ""):
            yield line
        
        # Handle process completion
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            yield "Training completed successfully!"
        else:
            yield f"Training failed with return code {return_code}"
    
    except Exception as e:
        yield f"Error during training: {str(e)}"

# Helper functions for UI
def list_datasets():
    try:
        return [d for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))]
    except:
        return []

def list_training_dirs():
    try:
        return [d for d in os.listdir(TRAINING_DIR) if os.path.isdir(os.path.join(TRAINING_DIR, d))]
    except:
        return []

def list_checkpoints():
    try:
        checkpoints = []
        for root, _, files in os.walk(TRAINING_DIR):
            for file in files:
                if file.endswith('.ckpt'):
                    rel_path = os.path.relpath(os.path.join(root, file), PIPER_HOME)
                    checkpoints.append(rel_path)
        return checkpoints
    except:
        return []

def download_checkpoint():
    try:
        # Download a good starting checkpoint
        checkpoint_url = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch%3D2164-step%3D1355540.ckpt"
        output_path = os.path.join(CHECKPOINTS_DIR, "lessac_medium_2164.ckpt")
        
        if not os.path.exists(output_path):
            import requests
            response = requests.get(checkpoint_url, stream=True)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return f"Downloaded checkpoint to {output_path}"
            else:
                return f"Failed to download checkpoint: HTTP {response.status_code}"
        else:
            return f"Checkpoint already exists at {output_path}"
    except Exception as e:
        return f"Error downloading checkpoint: {str(e)}"

# Create the Gradio interface
def create_interface():
    with gr.Blocks(title="Piper TTS Trainer") as app:
        gr.Markdown("# Piper TTS Trainer")
        gr.Markdown("## Create your own text-to-speech voice with Piper")
        
        # Workflow tabs
        with gr.Tabs() as tabs:
            # Tab 1: Dataset Preparation
            with gr.TabItem("1. Prepare Dataset"):
                gr.Markdown("### Prepare your audio files for training")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        audio_dir = gr.Textbox(label="Audio Directory", 
                                             placeholder="Path to directory containing audio files (.wav, .mp3, .flac)")
                        transcription_file = gr.Textbox(label="Transcription File (Optional)", 
                                                     placeholder="Path to text file with format: filename|text")
                        output_name = gr.Textbox(label="Output Dataset Name (optional)", 
                                               placeholder="Name for your dataset")
                        
                        prepare_btn = gr.Button("Prepare Dataset", variant="primary")
                    
                    with gr.Column(scale=2):
                        prepare_output = gr.Textbox(label="Output", interactive=False, lines=2)
                
                # Event handler
                prepare_btn.click(
                    fn=prepare_dataset,
                    inputs=[audio_dir, transcription_file, output_name],
                    outputs=[prepare_output, gr.State()]
                )
            
            # Tab 2: Preprocess Dataset
            with gr.TabItem("2. Preprocess Dataset"):
                gr.Markdown("### Preprocess your dataset for training")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        dataset_dropdown = gr.Dropdown(
                            choices=list_datasets,
                            label="Select Dataset",
                            interactive=True
                        )
                        refresh_datasets_btn = gr.Button("Refresh Datasets")
                        
                        language_dropdown = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="en",
                            label="Language"
                        )
                        
                        sample_rate = gr.Number(value=22050, label="Sample Rate (Hz)")
                        format_dropdown = gr.Dropdown(
                            choices=["ljspeech", "mycroft", "vctk", "mailabs", "commonvoice"],
                            value="ljspeech",
                            label="Dataset Format"
                        )
                        single_speaker = gr.Checkbox(value=True, label="Single Speaker")
                        
                        preprocess_btn = gr.Button("Preprocess Dataset", variant="primary")
                    
                    with gr.Column(scale=2):
                        preprocess_status = gr.Textbox(label="Status", interactive=False)
                        preprocess_log = gr.Textbox(label="Processing Log", interactive=False, lines=10)
                
                # Event handlers
                refresh_datasets_btn.click(
                    lambda: gr.update(choices=list_datasets()),
                    outputs=dataset_dropdown
                )
                
                def run_preprocess(dataset, language, sample_rate, format, single_speaker):
                    dataset_path = os.path.join(DATASETS_DIR, dataset)
                    success, output_dir, log = preprocess_dataset(
                        dataset_path, language, sample_rate, format, single_speaker
                    )
                    
                    if success:
                        return f"Preprocessing completed successfully. Output in {output_dir}", log
                    else:
                        return f"Preprocessing failed. See log for details.", log
                
                preprocess_btn.click(
                    run_preprocess,
                    inputs=[dataset_dropdown, language_dropdown, sample_rate, format_dropdown, single_speaker],
                    outputs=[preprocess_status, preprocess_log]
                )
            
            # Tab 3: Train Model
            with gr.TabItem("3. Train Model"):
                gr.Markdown("### Train your TTS model")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        training_dir_dropdown = gr.Dropdown(
                            choices=list_training_dirs,
                            label="Select Training Directory",
                            interactive=True
                        )
                        refresh_training_btn = gr.Button("Refresh Training Directories")
                        
                        checkpoint_dropdown = gr.Dropdown(
                            choices=list_checkpoints,
                            label="Resume from Checkpoint (Optional)",
                            interactive=True
                        )
                        refresh_checkpoints_btn = gr.Button("Refresh Checkpoints")
                        download_ckpt_btn = gr.Button("Download Starter Checkpoint")
                        
                        batch_size = gr.Slider(minimum=1, maximum=64, step=1, value=32, 
                                             label="Batch Size")
                        epochs = gr.Number(value=6000, label="Maximum Epochs")
                        quality = gr.Dropdown(
                            choices=["low", "medium", "high"],
                            value="medium",
                            label="Model Quality"
                        )
                        precision = gr.Dropdown(
                            choices=["16", "32"],
                            value="32",
                            label="Training Precision"
                        )
                        
                        train_btn = gr.Button("Start Training", variant="primary")
                        
                        checkpoint_status = gr.Textbox(label="Checkpoint Status", interactive=False)
                    
                    with gr.Column(scale=3):
                        training_output = gr.Textbox(
                            label="Training Output",
                            interactive=False,
                            lines=20
                        )
                
                # Event handlers
                refresh_training_btn.click(
                    lambda: gr.update(choices=list_training_dirs()),
                    outputs=training_dir_dropdown
                )
                
                refresh_checkpoints_btn.click(
                    lambda: gr.update(choices=list_checkpoints()),
                    outputs=checkpoint_dropdown
                )
                
                download_ckpt_btn.click(
                    download_checkpoint,
                    outputs=checkpoint_status
                )
                
                def start_train(training_dir, checkpoint, batch_size, epochs, quality, precision):
                    if not training_dir:
                        return "Please select a training directory"
                    
                    full_training_dir = os.path.join(TRAINING_DIR, training_dir)
                    
                    if not os.path.exists(full_training_dir):
                        return f"Training directory not found: {full_training_dir}"
                    
                    if checkpoint:
                        full_checkpoint = os.path.join(PIPER_HOME, checkpoint)
                    else:
                        full_checkpoint = None
                    
                    return train_model(
                        full_training_dir, 
                        full_checkpoint, 
                        int(batch_size), 
                        int(epochs), 
                        quality, 
                        precision
                    )
                
                train_btn.click(
                    start_train,
                    inputs=[training_dir_dropdown, checkpoint_dropdown, batch_size, epochs, quality, precision],
                    outputs=training_output
                )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860) 