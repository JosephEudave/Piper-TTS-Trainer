#!/usr/bin/env python3
# Piper TTS Trainer - Focused GUI for Dataset Preparation, Preprocessing and Training

import gradio as gr
import os
import shutil
import subprocess
import glob
import torch
import re
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Define paths
PIPER_HOME = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(PIPER_HOME, "datasets")
TRAINING_DIR = os.path.join(PIPER_HOME, "training")
CHECKPOINTS_DIR = os.path.join(PIPER_HOME, "checkpoints")
MODELS_DIR = os.path.join(PIPER_HOME, "models")
LOGS_DIR = os.path.join(PIPER_HOME, "logs")
NORMALIZED_DIR = os.path.join(PIPER_HOME, "normalized_wavs")

# Add function to convert WSL paths to Windows paths and vice versa
def convert_path_if_needed(path):
    """Convert between WSL and Windows paths if needed"""
    if os.name == 'nt':  # Running on Windows
        # Convert /mnt/c/... to C:/...
        if path.startswith('/mnt/'):
            drive_letter = path[5:6].upper()
            return f"{drive_letter}:{path[6:].replace('/', '\\')}"
    else:  # Running on Linux/WSL
        # Convert C:/... or C:\... to /mnt/c/...
        if re.match(r'^[a-zA-Z]:[/\\]', path):
            drive_letter = path[0].lower()
            path_suffix = path[2:].replace('\\', '/')
            return f"/mnt/{drive_letter}/{path_suffix}"
    return path

# Ensure directories exist
for directory in [DATASETS_DIR, TRAINING_DIR, CHECKPOINTS_DIR, MODELS_DIR, LOGS_DIR, NORMALIZED_DIR]:
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
        venv_python = os.path.join(PIPER_HOME, "piper\\src\\python\\.venv\\Scripts\\python.exe")
    else:  # Linux/Mac
        venv_python = os.path.join(PIPER_HOME, "piper/src/python/.venv/bin/python3")
    
    # Check if the virtual environment exists
    if not os.path.exists(venv_python):
        print(f"Warning: Python virtual environment not found at {venv_python}")
        # Fall back to system Python if venv not found
        return "python" if os.name == 'nt' else "python3"
        
    return venv_python

# Get command to run Piper modules correctly
def get_piper_command(module_name):
    """Create a command that ensures proper Python path for Piper modules"""
    python_exe = get_piper_python()
    
    if os.name == 'nt':  # Windows
        piper_src = os.path.join(PIPER_HOME, "piper\\src\\python")
        piper_parent = os.path.join(PIPER_HOME, "piper\\src")
        path_sep = ";"
    else:  # Linux/Mac
        piper_src = os.path.join(PIPER_HOME, "piper/src/python")
        piper_parent = os.path.join(PIPER_HOME, "piper/src")
        path_sep = ":"
    
    # Convert paths if needed
    piper_src = convert_path_if_needed(piper_src)
    piper_parent = convert_path_if_needed(piper_parent)
    
    # Create environment with PYTHONPATH set to include piper source
    env = os.environ.copy()
    
    # Add piper source and parent dir to PYTHONPATH
    # This ensures both piper_train and piper_phonemize can be found
    python_paths = [piper_src, piper_parent]
    
    if 'PYTHONPATH' in env and env['PYTHONPATH']:
        python_paths.append(env['PYTHONPATH'])
    
    env['PYTHONPATH'] = path_sep.join(python_paths)
    
    # Return the command and environment
    if module_name.startswith("piper_train."):
        module_parts = module_name.split('.')
        script_path = os.path.join(piper_src, *module_parts[:-1], f"{module_parts[-1]}.py")
        script_path = convert_path_if_needed(script_path)
        
        return [python_exe, script_path], env
    else:
        # For modules that aren't part of piper_train (which we don't currently use)
        script_path = os.path.join(piper_src, module_name.replace('.', '/') + '.py')
        script_path = convert_path_if_needed(script_path)
        return [python_exe, script_path], env

# Audio normalization functions
def normalize_audio_file(
    input_file,
    output_file,
    target_peak=0.95,
    sample_rate=22050,
    overwrite=False
):
    """
    Normalize audio file to have a peak amplitude at the specified target level.
    """
    # Skip if output exists and we're not overwriting
    if os.path.exists(output_file) and not overwrite:
        return False
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Load audio without normalization to get original values
        y, sr = librosa.load(input_file, sr=sample_rate, normalize=False)
        
        # Compute current peak amplitude
        current_peak = np.max(np.abs(y))
        
        if current_peak > 0:
            # Calculate scaling factor
            scaling_factor = target_peak / current_peak
            
            # Apply scaling to normalize
            y_normalized = y * scaling_factor
            
            # Save normalized audio
            sf.write(output_file, y_normalized, sample_rate)
            return True
        else:
            return False
            
    except Exception as e:
        return False

def normalize_audio_dataset(input_dir, output_dir, target_peak=0.95, sample_rate=22050, overwrite=False):
    """
    Normalize all audio files in a dataset directory.
    """
    try:
        # Convert paths if needed
        input_dir = convert_path_if_needed(input_dir)
        output_dir = convert_path_if_needed(output_dir)
        
        # Get all WAV files in the input directory
        audio_files = []
        wavs_dir = os.path.join(input_dir, "wavs")
        if os.path.exists(wavs_dir):
            audio_files.extend(glob.glob(os.path.join(wavs_dir, "*.wav")))
        else:
            audio_files.extend(glob.glob(os.path.join(input_dir, "*.wav")))
        
        if not audio_files:
            return f"No audio files found in {input_dir}", 0
        
        # Create output directory structure
        output_wavs_dir = os.path.join(output_dir, "wavs")
        os.makedirs(output_wavs_dir, exist_ok=True)
        
        # Copy metadata.csv if it exists
        metadata_path = os.path.join(input_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            shutil.copy2(metadata_path, os.path.join(output_dir, "metadata.csv"))
        
        # Process each audio file
        processed_count = 0
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            output_file = os.path.join(output_wavs_dir, filename)
            
            if normalize_audio_file(audio_file, output_file, target_peak, sample_rate, overwrite):
                processed_count += 1
        
        return f"Successfully normalized {processed_count} of {len(audio_files)} files", processed_count
    
    except Exception as e:
        return f"Error normalizing audio: {str(e)}", 0

# Dataset preparation function
def prepare_dataset(audio_dir, transcription_file=None, output_name=None):
    try:
        # Convert paths if needed (WSL/Windows compatibility)
        audio_dir = convert_path_if_needed(audio_dir)
        if transcription_file:
            transcription_file = convert_path_if_needed(transcription_file)
            
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
        # First, check that all required modules are available
        module_check = check_piper_modules()
        
        # Check piper_train
        if not module_check.get("piper_train", False) or not module_check.get("preprocess.py", False):
            return False, None, f"Error: piper_train module not found at {module_check.get('piper_train_path', 'unknown path')}"
        
        # Check piper_phonemize
        if not module_check.get("piper_phonemize", False):
            return False, None, f"Error: piper_phonemize module not found at {module_check.get('piper_phonemize_path', 'unknown path')}"
        
        # Convert path if needed
        dataset_dir = convert_path_if_needed(dataset_dir)
        
        # Get dataset name from directory
        dataset_name = os.path.basename(os.path.normpath(dataset_dir))
        output_dir = os.path.join(TRAINING_DIR, dataset_name)
        output_dir = convert_path_if_needed(output_dir)
        
        # Build command
        cmd, env = get_piper_command("piper_train.preprocess")
        
        # Add preprocessing arguments
        cmd.extend([
            "--language", language,
            "--input-dir", dataset_dir,
            "--output-dir", output_dir,
            "--dataset-format", format,
            "--sample-rate", str(sample_rate)
        ])
        
        if single_speaker:
            cmd.append("--single-speaker")
        
        # Run preprocessing
        process = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if process.returncode == 0:
            log_output = process.stdout
            return True, output_dir, log_output
        else:
            error_output = process.stderr
            
            # Debug output to help identify the issue
            debug_info = f"Command: {' '.join(cmd)}\nReturn code: {process.returncode}\nPYTHONPATH: {env.get('PYTHONPATH', 'Not set')}\n"
            return False, None, debug_info + error_output
    
    except Exception as e:
        return False, None, str(e)

# Train model function
def train_model(training_dir, checkpoint_path=None, batch_size=32, epochs=6000, 
               quality="medium", precision="32"):
    try:
        # Convert paths if needed
        training_dir = convert_path_if_needed(training_dir)
        if checkpoint_path:
            checkpoint_path = convert_path_if_needed(checkpoint_path)
        
        # Check for GPU
        gpu_available = torch.cuda.is_available()
        
        # Build command
        cmd, env = get_piper_command("piper_train")
        
        # Add training arguments
        cmd.extend([
            "--dataset-dir", training_dir,
            "--batch-size", str(batch_size),
            "--max_epochs", str(epochs),
            "--precision", precision,
            "--validation-split", "0.0",
            "--num-test-examples", "0",
            "--checkpoint-epochs", "1"
        ])
        
        # Add quality if specified
        if quality:
            cmd.extend(["--quality", quality])
        
        # Add checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            cmd.extend(["--resume_from_checkpoint", checkpoint_path])
        
        # Configure GPU/CPU training
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            yield f"GPU detected: {gpu_name} ({gpu_mem} GB memory)"
            yield "Training will use GPU acceleration for faster processing."
            
            cmd.extend(["--accelerator", "gpu", "--devices", "1"])
        else:
            yield "WARNING: No GPU detected! Training will use CPU (much slower)."
            yield "For faster training, consider running on a system with an NVIDIA GPU."
            
            cmd.extend(["--accelerator", "cpu"])
        
        # Show command being executed for debugging
        yield f"Executing: {' '.join(cmd)}"
        yield f"PYTHONPATH: {env.get('PYTHONPATH', 'Not set')}"
        
        # Start training as a process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
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

# Export model to ONNX
def export_model_to_onnx(checkpoint_path, output_path, streaming=False):
    try:
        # Convert paths if needed
        checkpoint_path = convert_path_if_needed(checkpoint_path)
        output_path = convert_path_if_needed(output_path)
        
        # Determine which exporter to use
        export_module = "piper_train.export_onnx_streaming" if streaming else "piper_train.export_onnx"
        
        # Build command
        cmd, env = get_piper_command(export_module)
        cmd.extend([checkpoint_path, output_path])
        
        # Run export process
        process = subprocess.run(cmd, capture_output=True, text=True, env=env)
        
        if process.returncode == 0:
            return True, f"Successfully exported model to {output_path}", process.stdout
        else:
            debug_info = f"Command: {' '.join(cmd)}\nReturn code: {process.returncode}\nPYTHONPATH: {env.get('PYTHONPATH', 'Not set')}\n"
            return False, f"Failed to export model. See logs for details.", debug_info + process.stderr
    
    except Exception as e:
        return False, f"Error exporting model: {str(e)}", str(e)

# Helper function to create model config
def create_model_config(output_dir, model_name, language="en", quality="medium", speaker_id=0):
    try:
        # Create config
        config = {
            "model": {
                "name": model_name,
                "language": language,
                "quality": quality,
                "speaker_id": speaker_id
            },
            "inference": {
                "noise_scale": 0.667,
                "length_scale": 1.0,
                "noise_w": 0.8
            }
        }
        
        # Write config
        config_path = os.path.join(output_dir, f"{model_name}.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    except Exception as e:
        return None

# Helper functions for UI
def list_datasets():
    try:
        return [d for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))]
    except:
        return []

def list_normalized_datasets():
    try:
        return [d for d in os.listdir(NORMALIZED_DIR) if os.path.isdir(os.path.join(NORMALIZED_DIR, d))]
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

# Add function to check the piper directory structure
def check_piper_modules():
    """Check that all required Piper modules can be found"""
    try:
        # Define critical paths
        if os.name == 'nt':  # Windows
            piper_train_path = os.path.join(PIPER_HOME, "piper\\src\\python\\piper_train")
            piper_phonemize_path = os.path.join(PIPER_HOME, "piper\\src\\piper_phonemize")
        else:  # Linux/Mac
            piper_train_path = os.path.join(PIPER_HOME, "piper/src/python/piper_train")
            piper_phonemize_path = os.path.join(PIPER_HOME, "piper/src/piper_phonemize")
        
        # Convert paths if needed
        piper_train_path = convert_path_if_needed(piper_train_path)
        piper_phonemize_path = convert_path_if_needed(piper_phonemize_path)
        
        # Check if paths exist
        train_exists = os.path.exists(piper_train_path)
        phonemize_exists = os.path.exists(piper_phonemize_path)
        
        # Check for key files
        preprocess_path = os.path.join(piper_train_path, "preprocess.py")
        preprocess_exists = os.path.exists(preprocess_path)
        
        result = {
            "piper_train": train_exists,
            "piper_phonemize": phonemize_exists,
            "preprocess.py": preprocess_exists,
            "piper_train_path": piper_train_path,
            "piper_phonemize_path": piper_phonemize_path
        }
        
        return result
    except Exception as e:
        return {"error": str(e)}

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
            
            # Tab 2: Normalize Audio
            with gr.TabItem("2. Normalize Audio"):
                gr.Markdown("### Normalize audio to fix clipping issues")
                gr.Markdown("This step scales audio files to prevent 'audio amplitude out of range' warnings")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        normalize_dataset_dropdown = gr.Dropdown(
                            choices=list_datasets(),
                            label="Select Dataset to Normalize",
                            interactive=True
                        )
                        refresh_normalize_datasets_btn = gr.Button("Refresh Datasets")
                        
                        normalize_output_name = gr.Textbox(
                            label="Output Dataset Name",
                            placeholder="Name for normalized dataset (defaults to original name)"
                        )
                        
                        target_peak = gr.Slider(
                            minimum=0.1, 
                            maximum=0.99, 
                            step=0.01, 
                            value=0.95,
                            label="Target Peak Amplitude (0-1)"
                        )
                        
                        sample_rate = gr.Number(
                            value=22050, 
                            label="Sample Rate (Hz)"
                        )
                        
                        normalize_btn = gr.Button("Normalize Audio", variant="primary")
                    
                    with gr.Column(scale=2):
                        normalize_status = gr.Textbox(
                            label="Status", 
                            interactive=False,
                            lines=3
                        )
                
                # Event handlers
                refresh_normalize_datasets_btn.click(
                    lambda: gr.update(choices=list_datasets()),
                    outputs=normalize_dataset_dropdown
                )
                
                def run_normalize(dataset, output_name, target_peak, sample_rate):
                    if not dataset:
                        return "Please select a dataset to normalize"
                    
                    # Set output name if not provided
                    if not output_name:
                        output_name = dataset
                    
                    # Set paths
                    input_path = os.path.join(DATASETS_DIR, dataset)
                    output_path = os.path.join(NORMALIZED_DIR, output_name)
                    
                    # Convert paths if needed
                    input_path = convert_path_if_needed(input_path)
                    output_path = convert_path_if_needed(output_path)
                    
                    # Run normalization
                    status, count = normalize_audio_dataset(
                        input_path, 
                        output_path, 
                        target_peak, 
                        sample_rate
                    )
                    
                    return f"{status}\nOutput directory: {output_path}"
                
                normalize_btn.click(
                    run_normalize,
                    inputs=[normalize_dataset_dropdown, normalize_output_name, target_peak, sample_rate],
                    outputs=normalize_status
                )
            
            # Tab 3: Preprocess Dataset
            with gr.TabItem("3. Preprocess Dataset"):
                gr.Markdown("### Preprocess your dataset for training")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Row():
                            dataset_src = gr.Radio(
                                ["Regular Dataset", "Normalized Dataset"], 
                                label="Dataset Source", 
                                value="Regular Dataset"
                            )
                        
                        dataset_dropdown = gr.Dropdown(
                            choices=list_datasets(),
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
                def update_dataset_choices(src):
                    if src == "Regular Dataset":
                        return gr.update(choices=list_datasets(), label="Select Dataset")
                    else:
                        return gr.update(choices=list_normalized_datasets(), label="Select Normalized Dataset")
                
                dataset_src.change(
                    update_dataset_choices,
                    inputs=dataset_src,
                    outputs=dataset_dropdown
                )
                
                refresh_datasets_btn.click(
                    lambda dataset_src: update_dataset_choices(dataset_src),
                    inputs=dataset_src,
                    outputs=dataset_dropdown
                )
                
                def run_preprocess(src, dataset, language, sample_rate, format, single_speaker):
                    if src == "Regular Dataset":
                        dataset_path = os.path.join(DATASETS_DIR, dataset)
                    else:
                        dataset_path = os.path.join(NORMALIZED_DIR, dataset)
                    
                    # Check modules first
                    module_info = check_piper_modules()
                    module_status = ["Piper module status:"]
                    for key, value in module_info.items():
                        if key.endswith("_path"):
                            module_status.append(f"- {key}: {value}")
                        else:
                            module_status.append(f"- {key}: {'Found' if value else 'Missing'}")
                    
                    module_text = "\n".join(module_status)
                    
                    # Run preprocessing
                    success, output_dir, log = preprocess_dataset(
                        dataset_path, language, sample_rate, format, single_speaker
                    )
                    
                    if success:
                        return f"Preprocessing completed successfully. Output in {output_dir}", module_text + "\n\n" + log
                    else:
                        return f"Preprocessing failed. See log for details.", module_text + "\n\n" + log
                
                preprocess_btn.click(
                    run_preprocess,
                    inputs=[dataset_src, dataset_dropdown, language_dropdown, sample_rate, format_dropdown, single_speaker],
                    outputs=[preprocess_status, preprocess_log]
                )
            
            # Tab 4: Train Model
            with gr.TabItem("4. Train Model"):
                gr.Markdown("### Train your TTS model")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        training_dir_dropdown = gr.Dropdown(
                            choices=list_training_dirs(),
                            label="Select Training Directory",
                            interactive=True
                        )
                        refresh_training_btn = gr.Button("Refresh Training Directories")
                        
                        checkpoint_dropdown = gr.Dropdown(
                            choices=list_checkpoints(),
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
                    full_training_dir = convert_path_if_needed(full_training_dir)
                    
                    if not os.path.exists(full_training_dir):
                        return f"Training directory not found: {full_training_dir}"
                    
                    if checkpoint:
                        full_checkpoint = os.path.join(PIPER_HOME, checkpoint)
                        full_checkpoint = convert_path_if_needed(full_checkpoint)
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
            
            # Tab 5: Export Model
            with gr.TabItem("5. Export Model"):
                gr.Markdown("### Export trained model to ONNX format")
                
                with gr.Row():
                    with gr.Column(scale=3):
                        export_checkpoint_dropdown = gr.Dropdown(
                            choices=list_checkpoints(),
                            label="Select Checkpoint to Export",
                            interactive=True
                        )
                        refresh_export_checkpoints_btn = gr.Button("Refresh Checkpoints")
                        
                        model_name = gr.Textbox(
                            label="Model Name", 
                            placeholder="Name for your exported model"
                        )
                        
                        model_language = gr.Dropdown(
                            choices=list(LANGUAGES.keys()),
                            value="en",
                            label="Model Language"
                        )
                        
                        model_quality = gr.Dropdown(
                            choices=["low", "medium", "high"],
                            value="medium",
                            label="Model Quality"
                        )
                        
                        streaming_mode = gr.Checkbox(
                            label="Enable Streaming Mode", 
                            value=True,
                            info="Recommended for faster inference"
                        )
                        
                        export_btn = gr.Button("Export Model", variant="primary")
                    
                    with gr.Column(scale=2):
                        export_status = gr.Textbox(
                            label="Status", 
                            interactive=False,
                            lines=3
                        )
                        export_log = gr.Textbox(
                            label="Export Log", 
                            interactive=False, 
                            lines=10
                        )
                
                # Event handlers
                refresh_export_checkpoints_btn.click(
                    lambda: gr.update(choices=list_checkpoints()),
                    outputs=export_checkpoint_dropdown
                )
                
                def run_export(checkpoint, model_name, language, quality, streaming):
                    if not checkpoint:
                        return "Please select a checkpoint to export", ""
                    
                    if not model_name:
                        # Extract model name from checkpoint path
                        model_name = os.path.basename(os.path.dirname(checkpoint))
                    
                    # Set paths
                    checkpoint_path = os.path.join(PIPER_HOME, checkpoint)
                    model_dir = os.path.join(MODELS_DIR, model_name)
                    os.makedirs(model_dir, exist_ok=True)
                    
                    output_path = os.path.join(model_dir, f"{model_name}.onnx")
                    
                    # Export model
                    success, status, log = export_model_to_onnx(
                        checkpoint_path, 
                        output_path, 
                        streaming
                    )
                    
                    if success:
                        # Create model config file
                        config_path = create_model_config(
                            model_dir, 
                            model_name, 
                            language, 
                            quality
                        )
                        
                        return f"{status}\nModel config created at {config_path}", log
                    else:
                        return status, log
                
                export_btn.click(
                    run_export,
                    inputs=[export_checkpoint_dropdown, model_name, model_language, model_quality, streaming_mode],
                    outputs=[export_status, export_log]
                )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860) 