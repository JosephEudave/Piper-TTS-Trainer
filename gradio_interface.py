import os
import sys
import json
import gradio as gr
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import tempfile
from tqdm import tqdm

from config import Config, default_config
from preprocess_audio import process_directory, validate_metadata, check_audio_format, convert_audio

# Ensure imports for training part
try:
    from piper_train.preprocess import preprocess
    from piper_train import train as piper_train
except ImportError:
    print("Warning: piper_train module not found. Training functionality may be limited.")

def load_config(config_path=None):
    """Load configuration from file or use default"""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
            
        # Create the dataset and training configs
        from config import DatasetConfig, TrainingConfig
        dataset_config = DatasetConfig(**config_dict.get("dataset", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        return Config(dataset=dataset_config, training=training_config)
    return default_config

def save_config(config, config_path="config.json"):
    """Save configuration to file"""
    config_dict = {
        "dataset": {
            "wav_dir": config.dataset.wav_dir,
            "metadata_file": config.dataset.metadata_file,
            "language": config.dataset.language,
            "sample_rate": config.dataset.sample_rate,
            "dataset_format": config.dataset.dataset_format,
            "single_speaker": config.dataset.single_speaker,
            "use_whisper": config.dataset.use_whisper
        },
        "training": {
            "model_name": config.training.model_name,
            "output_dir": config.training.output_dir,
            "batch_size": config.training.batch_size,
            "quality": config.training.quality,
            "max_epochs": config.training.max_epochs,
            "checkpoint_epochs": config.training.checkpoint_epochs,
            "num_ckpt": config.training.num_ckpt,
            "log_every_n_steps": config.training.log_every_n_steps,
            "validation_split": config.training.validation_split,
            "num_test_examples": config.training.num_test_examples,
            "save_last": config.training.save_last,
            "action": config.training.action,
            "pretrained_checkpoint": config.training.pretrained_checkpoint
        }
    }
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    return config_path

def preprocess_files(input_dir, output_dir, metadata_file, sample_rate, single_speaker, progress=gr.Progress()):
    """Process audio files with progress reporting"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate metadata first
    progress(0, desc="Validating metadata...")
    is_valid, errors = validate_metadata(metadata_file, input_dir, single_speaker)
    if not is_valid:
        error_message = "Metadata validation failed:\n" + "\n".join(errors)
        return False, error_message
    
    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    
    if not audio_files:
        return False, "No audio files found in input directory"
    
    # Process audio files
    results = []
    for i, file in enumerate(audio_files):
        progress((i+1)/len(audio_files), desc=f"Processing {file}")
        
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        
        # Check if already in correct format
        is_valid, msg = check_audio_format(input_path)
        if is_valid:
            # Just copy if already correct
            shutil.copy2(input_path, output_path)
            results.append((file, "Already in correct format"))
        else:
            # Convert if needed
            success, msg = convert_audio(input_path, output_path, sample_rate)
            if not success:
                return False, f"Error processing {file}: {msg}"
            results.append((file, msg))
    
    result_text = "Processing completed successfully!\n\nResults:\n"
    for file, msg in results:
        result_text += f"{file}: {msg}\n"
    
    return True, result_text

def setup_training(config, progress=gr.Progress()):
    """Setup training directories and prepare dataset"""
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs("audio_cache", exist_ok=True)
    
    progress(0.2, desc="Preparing dataset...")
    try:
        preprocess(
            language=config.dataset.language,
            input_dir=config.dataset.wav_dir,
            cache_dir="audio_cache",
            output_dir=config.training.output_dir,
            dataset_name=config.training.model_name,
            dataset_format=config.dataset.dataset_format,
            sample_rate=config.dataset.sample_rate,
            single_speaker=config.dataset.single_speaker
        )
        return True, "Dataset preparation completed successfully"
    except Exception as e:
        return False, f"Error preparing dataset: {str(e)}"

def start_training(config, progress=gr.Progress()):
    """Start the training process"""
    progress(0.1, desc="Setting up training...")
    
    # Prepare training arguments
    train_args = {
        "dataset_dir": config.training.output_dir,
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "batch_size": config.training.batch_size,
        "validation_split": config.training.validation_split,
        "num_test_examples": config.training.num_test_examples,
        "quality": config.training.quality,
        "checkpoint_epochs": config.training.checkpoint_epochs,
        "num_ckpt": config.training.num_ckpt,
        "log_every_n_steps": config.training.log_every_n_steps,
        "max_epochs": config.training.max_epochs,
        "precision": 32
    }
    
    # Add action-specific arguments
    if config.training.action == "continue":
        # Find latest checkpoint
        checkpoints = list(Path(config.training.output_dir).glob("lightning_logs/**/checkpoints/last.ckpt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.parent.parent.name.split("_")[1]))
            train_args["resume_from_checkpoint"] = str(latest_checkpoint)
        else:
            return False, "No checkpoints found to continue training from"
    elif config.training.action == "finetune":
        if not config.training.pretrained_checkpoint:
            return False, "Pretrained checkpoint path required for finetuning"
        train_args["resume_from_checkpoint"] = config.training.pretrained_checkpoint
    
    # Add save_last if needed
    if config.training.save_last:
        train_args["save_last"] = True
    
    progress(0.2, desc="Starting training...")
    try:
        # Start training
        piper_train(**train_args)
        return True, "Training completed successfully"
    except Exception as e:
        return False, f"Error during training: {str(e)}"

def upload_files_handler(files, target_dir):
    """Handle file uploads and save to target directory"""
    if not files:
        return "No files uploaded"
    
    os.makedirs(target_dir, exist_ok=True)
    result = []
    
    for file in files:
        filename = os.path.basename(file.name)
        dest_path = os.path.join(target_dir, filename)
        shutil.copy(file.name, dest_path)
        result.append(f"Saved {filename} to {target_dir}")
    
    return "\n".join(result)

def create_interface():
    # Load initial config
    config = load_config("config.json")
    
    with gr.Blocks(title="Piper TTS Trainer") as app:
        gr.Markdown("# Piper TTS Training Interface")
        
        with gr.Tabs():
            # Preprocessing Tab
            with gr.Tab("Preprocess Audio"):
                with gr.Row():
                    with gr.Column():
                        input_dir = gr.Textbox(label="Input Directory", value=config.dataset.wav_dir)
                        output_dir = gr.Textbox(label="Output Directory", value=config.dataset.wav_dir)
                        metadata_file = gr.Textbox(label="Metadata File", value=config.dataset.metadata_file)
                        
                        with gr.Row():
                            sample_rate = gr.Radio(
                                ["16000", "22050"], 
                                label="Sample Rate (Hz)", 
                                value=str(config.dataset.sample_rate)
                            )
                            single_speaker = gr.Checkbox(
                                label="Single Speaker Dataset", 
                                value=config.dataset.single_speaker
                            )
                        
                        preprocess_btn = gr.Button("Preprocess Audio Files")
                    
                    with gr.Column():
                        file_upload = gr.File(
                            label="Upload Audio Files", 
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("Upload to Input Directory")
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                        preprocess_output = gr.Textbox(label="Preprocessing Output", interactive=False)
                
                # Connect buttons
                upload_btn.click(
                    fn=upload_files_handler,
                    inputs=[file_upload, input_dir],
                    outputs=upload_status
                )
                
                preprocess_btn.click(
                    fn=preprocess_files,
                    inputs=[input_dir, output_dir, metadata_file, sample_rate, single_speaker],
                    outputs=preprocess_output
                )
            
            # Configuration Tab
            with gr.Tab("Configuration"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Dataset Configuration")
                        dataset_wav_dir = gr.Textbox(label="WAV Directory", value=config.dataset.wav_dir)
                        dataset_metadata = gr.Textbox(label="Metadata File", value=config.dataset.metadata_file)
                        dataset_language = gr.Textbox(label="Language Code", value=config.dataset.language)
                        dataset_sample_rate = gr.Radio(
                            ["16000", "22050"], 
                            label="Sample Rate", 
                            value=str(config.dataset.sample_rate)
                        )
                        dataset_format = gr.Radio(
                            ["ljspeech", "mycroft"], 
                            label="Dataset Format", 
                            value=config.dataset.dataset_format
                        )
                        dataset_single_speaker = gr.Checkbox(
                            label="Single Speaker", 
                            value=config.dataset.single_speaker
                        )
                        dataset_use_whisper = gr.Checkbox(
                            label="Use Whisper", 
                            value=config.dataset.use_whisper
                        )
                    
                    with gr.Column():
                        gr.Markdown("## Training Configuration")
                        training_model_name = gr.Textbox(label="Model Name", value=config.training.model_name)
                        training_output_dir = gr.Textbox(label="Output Directory", value=config.training.output_dir)
                        training_batch_size = gr.Slider(
                            minimum=1, maximum=64, step=1, 
                            label="Batch Size", 
                            value=config.training.batch_size
                        )
                        training_quality = gr.Radio(
                            ["high", "medium", "x-low"], 
                            label="Quality", 
                            value=config.training.quality
                        )
                        training_max_epochs = gr.Number(
                            label="Max Epochs", 
                            value=config.training.max_epochs
                        )
                        training_checkpoint_epochs = gr.Number(
                            label="Checkpoint Every N Epochs", 
                            value=config.training.checkpoint_epochs
                        )
                        training_action = gr.Radio(
                            ["train", "continue", "finetune"], 
                            label="Training Action", 
                            value=config.training.action
                        )
                        training_pretrained_path = gr.Textbox(
                            label="Pretrained Checkpoint Path (for finetune)", 
                            value=config.training.pretrained_checkpoint or "",
                            visible=(config.training.action == "finetune")
                        )
                        
                        # Show/hide pretrained path based on action
                        def update_pretrained_visibility(action):
                            return gr.Textbox.update(visible=(action == "finetune"))
                        
                        training_action.change(
                            fn=update_pretrained_visibility,
                            inputs=training_action,
                            outputs=training_pretrained_path
                        )
                
                save_config_btn = gr.Button("Save Configuration")
                config_status = gr.Textbox(label="Configuration Status", interactive=False)
                
                # Save config handler
                def save_config_handler(
                    wav_dir, metadata_file, language, sample_rate, dataset_format, 
                    single_speaker, use_whisper, model_name, output_dir, batch_size,
                    quality, max_epochs, checkpoint_epochs, action, pretrained_checkpoint
                ):
                    # Update config
                    temp_config = config
                    
                    # Dataset config
                    temp_config.dataset.wav_dir = wav_dir
                    temp_config.dataset.metadata_file = metadata_file
                    temp_config.dataset.language = language
                    temp_config.dataset.sample_rate = int(sample_rate)
                    temp_config.dataset.dataset_format = dataset_format
                    temp_config.dataset.single_speaker = single_speaker
                    temp_config.dataset.use_whisper = use_whisper
                    
                    # Training config
                    temp_config.training.model_name = model_name
                    temp_config.training.output_dir = output_dir
                    temp_config.training.batch_size = int(batch_size)
                    temp_config.training.quality = quality
                    temp_config.training.max_epochs = int(max_epochs)
                    temp_config.training.checkpoint_epochs = int(checkpoint_epochs)
                    temp_config.training.action = action
                    temp_config.training.pretrained_checkpoint = pretrained_checkpoint if pretrained_checkpoint else None
                    
                    # Save config
                    config_path = save_config(temp_config)
                    return f"Configuration saved to {config_path}"
                
                save_config_btn.click(
                    fn=save_config_handler,
                    inputs=[
                        dataset_wav_dir, dataset_metadata, dataset_language, 
                        dataset_sample_rate, dataset_format, dataset_single_speaker,
                        dataset_use_whisper, training_model_name, training_output_dir,
                        training_batch_size, training_quality, training_max_epochs,
                        training_checkpoint_epochs, training_action, training_pretrained_path
                    ],
                    outputs=config_status
                )
            
            # Training Tab
            with gr.Tab("Training"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Training Controls")
                        setup_btn = gr.Button("Setup & Prepare Dataset")
                        train_btn = gr.Button("Start Training")
                        training_status = gr.Textbox(label="Training Status", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("## Training Metrics")
                        # Placeholder for tensorboard integration
                        gr.Markdown("*Training metrics will be displayed here*")
                        gr.Markdown("You can also view detailed metrics in TensorBoard by running:")
                        gr.Code("tensorboard --logdir=output/lightning_logs", language="python")
                
                # Connect buttons
                setup_btn.click(
                    fn=lambda: setup_training(load_config("config.json")),
                    inputs=[],
                    outputs=training_status
                )
                
                train_btn.click(
                    fn=lambda: start_training(load_config("config.json")),
                    inputs=[],
                    outputs=training_status
                )
    
    return app

def main():
    """Main function to launch the interface"""
    interface = create_interface()
    interface.launch(share=False)

if __name__ == "__main__":
    main() 