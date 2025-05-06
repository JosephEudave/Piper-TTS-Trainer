#!/usr/bin/env python3
# Piper TTS Trainer - Data Preparation Gradio Interface

import gradio as gr
import os
import shutil
import subprocess
import json
import glob
import torch
import sys
from pathlib import Path

# Define paths
PIPER_HOME = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(PIPER_HOME, "datasets")
TRAINING_DIR = os.path.join(PIPER_HOME, "training")
CHECKPOINTS_DIR = os.path.join(PIPER_HOME, "checkpoints")
MODELS_DIR = os.path.join(PIPER_HOME, "models")

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Supported languages
LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "cs": "Czech",
    "pl": "Polish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "ar": "Arabic",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean"
}

# Dataset formats
DATASET_FORMATS = ["ljspeech", "mycroft", "vctk", "mailabs", "commonvoice"]

# Function to organize dataset files
def organize_dataset(wav_files_dir, output_dir, dataset_name):
    """Organize WAV files and prepare for preprocessing"""
    try:
        # Create dataset directory
        dataset_path = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Create wavs directory
        wavs_dir = os.path.join(dataset_path, "wavs")
        os.makedirs(wavs_dir, exist_ok=True)
        
        # Copy WAV files
        wav_files = glob.glob(os.path.join(wav_files_dir, "*.wav"))
        if not wav_files:
            return {
                "success": False,
                "message": f"No WAV files found in {wav_files_dir}"
            }
        
        # Generate metadata.csv file
        metadata_path = os.path.join(dataset_path, "metadata.csv")
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            for wav_file in wav_files:
                filename = os.path.basename(wav_file)
                # Copy the file
                shutil.copy2(wav_file, os.path.join(wavs_dir, filename))
                
                # Write to metadata.csv (filename without extension)
                base_name = os.path.splitext(filename)[0]
                # Placeholder text - would need to be replaced with actual transcription
                f.write(f"{base_name}|This is a placeholder text for {base_name}.\n")
        
        return {
            "success": True,
            "message": f"Successfully organized {len(wav_files)} WAV files to {dataset_path}",
            "dataset_path": dataset_path
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error organizing dataset: {str(e)}"
        }

# Function to preprocess dataset
def preprocess_dataset(dataset_path, output_dir, language, sample_rate, dataset_format, single_speaker):
    """Run piper_train.preprocess on the dataset"""
    try:
        # Use Python executable from the Piper virtual environment
        piper_python = os.path.join(PIPER_HOME, "piper/src/python/.venv/bin/python3")
        if os.name == 'nt':  # Windows
            piper_python = os.path.join(PIPER_HOME, "piper\\src\\python\\.venv\\Scripts\\python.exe")
        
        # Build command
        cmd = [
            piper_python, "-m", "piper_train.preprocess",
            "--language", language,
            "--input-dir", dataset_path,
            "--output-dir", output_dir,
            "--dataset-format", dataset_format,
            "--sample-rate", str(sample_rate)
        ]
        
        if single_speaker:
            cmd.append("--single-speaker")
        
        # Run command
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        if process.returncode == 0:
            return {
                "success": True,
                "message": "Preprocessing successful",
                "output": process.stdout,
                "training_dir": output_dir
            }
        else:
            return {
                "success": False,
                "message": "Preprocessing failed",
                "output": process.stderr
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error during preprocessing: {str(e)}"
        }

# Function to train model
def start_training(training_dir, checkpoint_path, batch_size, epochs, precision, quality):
    """Start the training process"""
    try:
        # Use Python executable from the Piper virtual environment
        piper_python = os.path.join(PIPER_HOME, "piper/src/python/.venv/bin/python3")
        if os.name == 'nt':  # Windows
            piper_python = os.path.join(PIPER_HOME, "piper\\src\\python\\.venv\\Scripts\\python.exe")
        
        # Build command
        cmd = [
            piper_python, "-m", "piper_train",
            "--dataset-dir", training_dir,
            "--batch-size", str(batch_size),
            "--max_epochs", str(epochs),
            "--precision", precision,
            "--validation-split", "0.0",
            "--num-test-examples", "0",
            "--checkpoint-epochs", "1"
        ]
        
        # Add GPU support if available
        if torch.cuda.is_available():
            cmd.extend(["--accelerator", "gpu", "--devices", "1"])
        else:
            cmd.extend(["--accelerator", "cpu"])
        
        # Add optional parameters
        if checkpoint_path and checkpoint_path != "":
            cmd.extend(["--resume_from_checkpoint", checkpoint_path])
        
        if quality and quality != "":
            cmd.extend(["--quality", quality])
        
        # Start training process
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            bufsize=1,
            universal_newlines=True
        )
        
        for line in iter(process.stdout.readline, ""):
            yield line
        
        process.stdout.close()
        process.wait()
        
        if process.returncode == 0:
            yield "Training completed successfully!"
        else:
            yield "Training failed with return code: " + str(process.returncode)
            
    except Exception as e:
        yield f"Error starting training: {str(e)}"

# Function to export trained model
def export_model(checkpoint_path, output_dir, model_name):
    """Export the trained model to ONNX format"""
    try:
        # Use Python executable from the Piper virtual environment
        piper_python = os.path.join(PIPER_HOME, "piper/src/python/.venv/bin/python3")
        if os.name == 'nt':  # Windows
            piper_python = os.path.join(PIPER_HOME, "piper\\src\\python\\.venv\\Scripts\\python.exe")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        cmd = [
            piper_python, "-m", "piper_train.export_onnx",
            checkpoint_path,
            onnx_path
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        
        # Copy config file
        config_source = None
        if os.path.isdir(os.path.dirname(checkpoint_path)):
            training_dir = os.path.dirname(os.path.dirname(checkpoint_path))
            config_source = os.path.join(training_dir, "config.json")
        
        config_dest = f"{onnx_path}.json"
        
        if config_source and os.path.exists(config_source):
            shutil.copy2(config_source, config_dest)
            
            return {
                "success": process.returncode == 0,
                "message": f"Model exported to {onnx_path}",
                "model_path": onnx_path
            }
        else:
            return {
                "success": False,
                "message": f"Config file not found at {config_source}"
            }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error exporting model: {str(e)}"
        }

# Helper functions for the UI
def get_available_datasets():
    """Get list of available datasets"""
    try:
        dataset_dirs = [d for d in os.listdir(DATASETS_DIR) 
                     if os.path.isdir(os.path.join(DATASETS_DIR, d))]
        return dataset_dirs
    except:
        return []

def get_available_checkpoints():
    """Get list of available checkpoints"""
    try:
        checkpoints = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.ckpt"))
        return [os.path.basename(c) for c in checkpoints]
    except:
        return []

def get_available_training_dirs():
    """Get list of available training directories"""
    try:
        training_dirs = [d for d in os.listdir(TRAINING_DIR) 
                      if os.path.isdir(os.path.join(TRAINING_DIR, d))]
        return training_dirs
    except:
        return []

def get_quality_options():
    """Get list of quality options"""
    return ["high", "medium", "low"]

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Piper TTS Data Preparation") as app:
        gr.Markdown("# Piper TTS Data Preparation")
        gr.Markdown("### Prepare, process, and train your custom TTS voice")
        
        with gr.Tab("1. Organize Dataset"):
            gr.Markdown("### Step 1: Organize your audio files for processing")
            
            with gr.Row():
                with gr.Column():
                    wav_dir = gr.Textbox(label="WAV Files Directory", placeholder="Path to directory containing .wav files")
                    dataset_name = gr.Textbox(label="Dataset Name", placeholder="Enter a name for your dataset")
                    organize_btn = gr.Button("Organize Dataset")
                
                with gr.Column():
                    browse_wav_btn = gr.Button("Browse WAV Folder")
                    org_status = gr.Textbox(label="Status", interactive=False)
            
            # Event handlers
            def browse_wav_folder():
                folder = gr.temp_file()
                return folder
            
            def run_organize(wav_dir, dataset_name):
                if not wav_dir or not dataset_name:
                    return "Please provide both WAV directory and dataset name"
                
                result = organize_dataset(wav_dir, DATASETS_DIR, dataset_name)
                return result["message"]
            
            organize_btn.click(run_organize, inputs=[wav_dir, dataset_name], outputs=org_status)
            browse_wav_btn.click(browse_wav_folder, inputs=[], outputs=wav_dir)
        
        with gr.Tab("2. Preprocess Dataset"):
            gr.Markdown("### Step 2: Preprocess your dataset for training")
            
            with gr.Row():
                with gr.Column():
                    dataset_dropdown = gr.Dropdown(choices=get_available_datasets, label="Select Dataset", interactive=True)
                    refresh_datasets_btn = gr.Button("Refresh Datasets")
                    language_dropdown = gr.Dropdown(choices=list(LANGUAGES.keys()), value="en", label="Language")
                    sample_rate = gr.Number(value=22050, label="Sample Rate")
                    dataset_format = gr.Dropdown(choices=DATASET_FORMATS, value="ljspeech", label="Dataset Format")
                    single_speaker = gr.Checkbox(value=True, label="Single Speaker")
                    preprocess_btn = gr.Button("Preprocess Dataset")
                
                with gr.Column():
                    preprocess_status = gr.Textbox(label="Status", interactive=False)
                    preprocess_output = gr.Textbox(label="Output Log", interactive=False, lines=10)
            
            # Event handlers
            def run_preprocess(dataset, language, sample_rate, format, single_speaker):
                if not dataset:
                    return "Please select a dataset", ""
                
                dataset_path = os.path.join(DATASETS_DIR, dataset)
                output_dir = os.path.join(TRAINING_DIR, dataset)
                
                result = preprocess_dataset(dataset_path, output_dir, language, int(sample_rate), format, single_speaker)
                return result["message"], result.get("output", "")
            
            preprocess_btn.click(run_preprocess, 
                              inputs=[dataset_dropdown, language_dropdown, sample_rate, dataset_format, single_speaker], 
                              outputs=[preprocess_status, preprocess_output])
            refresh_datasets_btn.click(lambda: gr.update(choices=get_available_datasets()), inputs=[], outputs=dataset_dropdown)
        
        with gr.Tab("3. Train Model"):
            gr.Markdown("### Step 3: Train your TTS model")
            
            with gr.Row():
                with gr.Column():
                    training_dir_dropdown = gr.Dropdown(choices=get_available_training_dirs, label="Select Training Directory")
                    refresh_training_dirs_btn = gr.Button("Refresh Training Directories")
                    checkpoint_dropdown = gr.Dropdown(choices=get_available_checkpoints, label="Resume from Checkpoint (Optional)")
                    refresh_checkpoints_btn = gr.Button("Refresh Checkpoints")
                    batch_size = gr.Slider(minimum=1, maximum=64, step=1, value=32, label="Batch Size")
                    epochs = gr.Number(value=6000, label="Maximum Epochs")
                    precision = gr.Dropdown(choices=["32", "16"], value="32", label="Precision")
                    quality = gr.Dropdown(choices=get_quality_options(), value="medium", label="Model Quality")
                    train_btn = gr.Button("Start Training")
                
                with gr.Column():
                    training_output = gr.Textbox(label="Training Output", lines=20, interactive=False)
            
            # Event handlers
            def update_training_output_func(training_dir, checkpoint, batch_size, epochs, precision, quality):
                if not training_dir:
                    return "Please select a training directory"
                
                full_training_dir = os.path.join(TRAINING_DIR, training_dir)
                full_checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint) if checkpoint else ""
                
                return start_training(full_training_dir, full_checkpoint_path, int(batch_size), int(epochs), precision, quality)
            
            train_btn.click(update_training_output_func, 
                         inputs=[training_dir_dropdown, checkpoint_dropdown, batch_size, epochs, precision, quality], 
                         outputs=training_output)
            refresh_training_dirs_btn.click(lambda: gr.update(choices=get_available_training_dirs()), inputs=[], outputs=training_dir_dropdown)
            refresh_checkpoints_btn.click(lambda: gr.update(choices=get_available_checkpoints()), inputs=[], outputs=checkpoint_dropdown)
        
        with gr.Tab("4. Export Model"):
            gr.Markdown("### Step 4: Export your trained model")
            
            with gr.Row():
                with gr.Column():
                    export_checkpoint_dropdown = gr.Dropdown(choices=get_available_checkpoints, label="Select Checkpoint to Export")
                    refresh_export_checkpoints_btn = gr.Button("Refresh Checkpoints")
                    model_name = gr.Textbox(label="Model Name", placeholder="Enter a name for your exported model")
                    export_btn = gr.Button("Export Model")
                
                with gr.Column():
                    export_status = gr.Textbox(label="Export Status", interactive=False)
            
            # Event handlers
            def run_export(checkpoint, model_name):
                if not checkpoint or not model_name:
                    return "Please select both a checkpoint and provide a model name"
                
                checkpoint_path = os.path.join(CHECKPOINTS_DIR, checkpoint)
                output_dir = os.path.join(MODELS_DIR, model_name)
                
                result = export_model(checkpoint_path, output_dir, model_name)
                return result["message"]
            
            export_btn.click(run_export, inputs=[export_checkpoint_dropdown, model_name], outputs=export_status)
            refresh_export_checkpoints_btn.click(lambda: gr.update(choices=get_available_checkpoints()), inputs=[], outputs=export_checkpoint_dropdown)
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=8000) 