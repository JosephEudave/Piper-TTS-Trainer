#!/usr/bin/env python3
# Piper TTS Trainer - Gradio Web Interface

import gradio as gr
import os
import json
import requests
import subprocess
import shutil
from pathlib import Path
import torch

# Constants and configuration
HUGGINGFACE_API_URL = "https://huggingface.co/api/datasets/rhasspy/piper-checkpoints/tree/main"
CHECKPOINT_BASE_URL = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main"
PIPER_HOME = os.path.expanduser("~/piper_tts_trainer")
CHECKPOINTS_DIR = os.path.join(PIPER_HOME, "checkpoints")
DATASETS_DIR = os.path.join(PIPER_HOME, "datasets")
TRAINING_DIR = os.path.join(PIPER_HOME, "training")
MODELS_DIR = os.path.join(PIPER_HOME, "models")

# Create necessary directories
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(TRAINING_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Fetch available models from Hugging Face
def fetch_available_models():
    """Fetch the list of available models from Hugging Face."""
    try:
        response = requests.get(HUGGINGFACE_API_URL)
        if response.status_code == 200:
            data = response.json()
            
            # Filter for checkpoint directories
            languages = {}
            for item in data:
                if item["type"] == "directory" and item["path"].count("/") == 1:
                    lang_code = item["path"]
                    languages[lang_code] = {
                        "name": lang_code,
                        "regions": {}
                    }
            
            # Get regions for each language
            for item in data:
                if item["type"] == "directory" and item["path"].count("/") == 2:
                    parts = item["path"].split("/")
                    lang_code, region_code = parts
                    if lang_code in languages:
                        languages[lang_code]["regions"][region_code] = {
                            "name": region_code,
                            "voices": {}
                        }
            
            # Get voices for each region
            for item in data:
                if item["type"] == "directory" and item["path"].count("/") == 3:
                    parts = item["path"].split("/")
                    lang_code, region_code, voice_name = parts
                    if lang_code in languages and region_code in languages[lang_code]["regions"]:
                        languages[lang_code]["regions"][region_code]["voices"][voice_name] = {
                            "name": voice_name,
                            "qualities": {}
                        }
            
            # Get qualities for each voice
            for item in data:
                if item["type"] == "directory" and item["path"].count("/") == 4:
                    parts = item["path"].split("/")
                    lang_code, region_code, voice_name, quality = parts
                    if (lang_code in languages and 
                        region_code in languages[lang_code]["regions"] and 
                        voice_name in languages[lang_code]["regions"][region_code]["voices"]):
                        languages[lang_code]["regions"][region_code]["voices"][voice_name]["qualities"][quality] = {
                            "name": quality,
                            "checkpoints": []
                        }
            
            # Get checkpoints for each quality
            for item in data:
                if item["type"] == "blob" and item["path"].endswith(".ckpt"):
                    parts = item["path"].split("/")
                    if len(parts) >= 5:
                        lang_code, region_code, voice_name, quality = parts[:4]
                        checkpoint_name = parts[-1]
                        if (lang_code in languages and 
                            region_code in languages[lang_code]["regions"] and 
                            voice_name in languages[lang_code]["regions"][region_code]["voices"] and
                            quality in languages[lang_code]["regions"][region_code]["voices"][voice_name]["qualities"]):
                            checkpoint_info = {
                                "name": checkpoint_name,
                                "path": item["path"],
                                "url": f"{CHECKPOINT_BASE_URL}/{item['path']}"
                            }
                            languages[lang_code]["regions"][region_code]["voices"][voice_name]["qualities"][quality]["checkpoints"].append(checkpoint_info)
            
            return languages
        else:
            return {"error": f"Failed to fetch models: {response.status_code}"}
    except Exception as e:
        return {"error": f"Error fetching models: {str(e)}"}

# Download checkpoint
def download_checkpoint(checkpoint_url, save_path):
    """Download a checkpoint from Hugging Face."""
    try:
        response = requests.get(checkpoint_url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error downloading checkpoint: {str(e)}")
        return False

# Preprocess dataset function
def preprocess_dataset(input_dir, output_dir, language, sample_rate, dataset_format, single_speaker):
    """Run piper_train.preprocess on the dataset."""
    try:
        cmd = [
            "python3", "-m", "piper_train.preprocess",
            "--language", language,
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--dataset-format", dataset_format,
            "--sample-rate", str(sample_rate)
        ]
        if single_speaker:
            cmd.append("--single-speaker")
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        return {"success": process.returncode == 0, "output": process.stdout, "error": process.stderr}
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

# Train model function
def train_model(dataset_dir, checkpoint_path, batch_size, max_epochs, precision, quality):
    """Run piper_train to train the model."""
    try:
        cmd = [
            "python3", "-m", "piper_train",
            "--dataset-dir", dataset_dir,
            "--batch-size", str(batch_size),
            "--max_epochs", str(max_epochs),
            "--precision", precision
        ]
        
        # Add optional parameters
        if checkpoint_path:
            cmd.extend(["--resume_from_checkpoint", checkpoint_path])
            
        if quality:
            cmd.extend(["--quality", quality])
            
        # Check for GPU availability
        if torch.cuda.is_available():
            cmd.extend(["--accelerator", "gpu", "--devices", "1"])
        else:
            cmd.extend(["--accelerator", "cpu"])
            
        cmd.extend([
            "--validation-split", "0.0",
            "--num-test-examples", "0",
            "--checkpoint-epochs", "1"
        ])
        
        # Use Popen to start the process and allow real-time output capture
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        output_lines = []
        for line in iter(process.stdout.readline, ""):
            output_lines.append(line)
            yield {"success": None, "output": line, "error": "", "finished": False}
        
        process.stdout.close()
        return_code = process.wait()
        
        yield {"success": return_code == 0, "output": "".join(output_lines), "error": "", "finished": True}
    except Exception as e:
        yield {"success": False, "output": "", "error": str(e), "finished": True}

# Export model function
def export_model(checkpoint_path, output_dir, model_name):
    """Export the trained model to ONNX format."""
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        export_cmd = [
            "python3", "-m", "piper_train.export_onnx",
            checkpoint_path,
            onnx_path
        ]
        
        export_process = subprocess.run(export_cmd, capture_output=True, text=True)
        
        # Copy config file
        training_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        config_path = os.path.join(training_dir, "config.json")
        config_dest = f"{onnx_path}.json"
        
        if os.path.exists(config_path):
            shutil.copy(config_path, config_dest)
            return {"success": export_process.returncode == 0, "output": export_process.stdout, "onnx_path": onnx_path}
        else:
            return {"success": False, "output": "", "error": f"Config file not found at {config_path}"}
        
    except Exception as e:
        return {"success": False, "output": "", "error": str(e)}

# Gradio interface
def create_interface():
    # Fetch available models
    models_data = fetch_available_models()
    
    # Create lists for dropdown menus
    languages = list(models_data.keys()) if not isinstance(models_data, dict) or "error" not in models_data else []
    
    with gr.Blocks(title="Piper TTS Trainer") as app:
        gr.Markdown("# Piper TTS Trainer")
        gr.Markdown("## Train custom text-to-speech voices with Piper")
        
        with gr.Tab("Pre-trained Models"):
            gr.Markdown("## Select and Download Pre-trained Models")
            
            # Language, region, voice, quality selectors
            language_dropdown = gr.Dropdown(choices=languages, label="Language")
            region_dropdown = gr.Dropdown(label="Region")
            voice_dropdown = gr.Dropdown(label="Voice")
            quality_dropdown = gr.Dropdown(label="Quality")
            checkpoint_dropdown = gr.Dropdown(label="Checkpoint")
            
            download_button = gr.Button("Download Selected Checkpoint")
            download_status = gr.Textbox(label="Download Status", interactive=False)
            
            # Update regions based on selected language
            def update_regions(language):
                if not language or isinstance(models_data, dict) and "error" in models_data:
                    return [], None
                return list(models_data[language]["regions"].keys()), None
            
            # Update voices based on selected region
            def update_voices(language, region):
                if not language or not region or isinstance(models_data, dict) and "error" in models_data:
                    return [], None
                return list(models_data[language]["regions"][region]["voices"].keys()), None
            
            # Update qualities based on selected voice
            def update_qualities(language, region, voice):
                if not language or not region or not voice or isinstance(models_data, dict) and "error" in models_data:
                    return [], None
                return list(models_data[language]["regions"][region]["voices"][voice]["qualities"].keys()), None
            
            # Update checkpoints based on selected quality
            def update_checkpoints(language, region, voice, quality):
                if not language or not region or not voice or not quality or isinstance(models_data, dict) and "error" in models_data:
                    return [], None
                checkpoints = models_data[language]["regions"][region]["voices"][voice]["qualities"][quality]["checkpoints"]
                return [cp["name"] for cp in checkpoints], None
            
            # Download selected checkpoint
            def download_selected_checkpoint(language, region, voice, quality, checkpoint_name):
                if not language or not region or not voice or not quality or not checkpoint_name:
                    return "Please select all options first."
                
                checkpoints = models_data[language]["regions"][region]["voices"][voice]["qualities"][quality]["checkpoints"]
                selected_checkpoint = next((cp for cp in checkpoints if cp["name"] == checkpoint_name), None)
                
                if not selected_checkpoint:
                    return "Checkpoint not found."
                
                # Create directory structure for the checkpoint
                checkpoint_dir = os.path.join(CHECKPOINTS_DIR, language, region, voice, quality)
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Full path to save the checkpoint
                save_path = os.path.join(checkpoint_dir, checkpoint_name)
                
                # Download the checkpoint
                if download_checkpoint(selected_checkpoint["url"], save_path):
                    return f"Successfully downloaded checkpoint to {save_path}"
                else:
                    return "Failed to download checkpoint."
            
            # Connect event handlers
            language_dropdown.change(update_regions, inputs=[language_dropdown], outputs=[region_dropdown, voice_dropdown])
            region_dropdown.change(update_voices, inputs=[language_dropdown, region_dropdown], outputs=[voice_dropdown, quality_dropdown])
            voice_dropdown.change(update_qualities, inputs=[language_dropdown, region_dropdown, voice_dropdown], outputs=[quality_dropdown, checkpoint_dropdown])
            quality_dropdown.change(update_checkpoints, inputs=[language_dropdown, region_dropdown, voice_dropdown, quality_dropdown], outputs=[checkpoint_dropdown, download_status])
            download_button.click(download_selected_checkpoint, inputs=[language_dropdown, region_dropdown, voice_dropdown, quality_dropdown, checkpoint_dropdown], outputs=[download_status])
        
        with gr.Tab("Dataset Configuration"):
            gr.Markdown("## Configure Dataset")
            
            # Dataset configuration
            dataset_input_dir = gr.Textbox(label="Input Directory", value=DATASETS_DIR, info="Directory containing your audio dataset")
            dataset_output_dir = gr.Textbox(label="Output Directory", value=TRAINING_DIR, info="Directory where preprocessed files will be stored")
            dataset_language = gr.Textbox(label="Language", value="en", info="Language code (e.g., en, fr, de)")
            dataset_sample_rate = gr.Number(label="Sample Rate", value=22050, info="Audio sample rate (22050 recommended)")
            dataset_format = gr.Dropdown(label="Dataset Format", choices=["ljspeech", "mycroft", "vctk"], value="ljspeech")
            single_speaker = gr.Checkbox(label="Single Speaker", value=True, info="Check for single speaker datasets")
            
            preprocess_button = gr.Button("Preprocess Dataset")
            preprocess_status = gr.Textbox(label="Preprocessing Status", interactive=False)
            
            # Preprocess function
            def run_preprocess(input_dir, output_dir, language, sample_rate, dataset_format, single_speaker):
                result = preprocess_dataset(input_dir, output_dir, language, sample_rate, dataset_format, single_speaker)
                if result["success"]:
                    return f"Preprocessing completed successfully.\n\nOutput:\n{result['output']}"
                else:
                    return f"Preprocessing failed.\n\nError:\n{result['error']}\n\nOutput:\n{result['output']}"
            
            preprocess_button.click(run_preprocess, inputs=[dataset_input_dir, dataset_output_dir, dataset_language, dataset_sample_rate, dataset_format, single_speaker], outputs=[preprocess_status])
        
        with gr.Tab("Training"):
            gr.Markdown("## Train Model")
            
            # Training configuration
            training_dataset_dir = gr.Textbox(label="Dataset Directory", value=TRAINING_DIR, info="Directory containing preprocessed dataset")
            
            # Checkpoint selection
            checkpoint_dir = gr.Textbox(label="Checkpoint Directory", value=CHECKPOINTS_DIR, info="Directory containing downloaded checkpoints")
            checkpoint_browser = gr.File(label="Select Checkpoint", file_count="single", file_types=[".ckpt"])
            
            # Training parameters
            batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=64, value=32, step=1, info="Larger values use more GPU memory")
            max_epochs = gr.Slider(label="Max Epochs", minimum=100, maximum=10000, value=6000, step=100, info="More epochs means longer training")
            precision = gr.Dropdown(label="Precision", choices=["16", "32"], value="32", info="Lower precision uses less GPU memory but may reduce quality")
            model_quality = gr.Dropdown(label="Model Quality", choices=["low", "medium", "high"], value="medium", info="Higher quality means larger model")
            
            train_button = gr.Button("Start Training")
            train_status = gr.Textbox(label="Training Status", interactive=False)
            
            # Training function
            def run_training(dataset_dir, checkpoint_path, batch_size, max_epochs, precision, quality):
                return gr.Textbox.update(value="Training starting...", interactive=False)
            
            def stream_training_output(dataset_dir, checkpoint_path, batch_size, max_epochs, precision, quality):
                for result in train_model(dataset_dir, checkpoint_path, batch_size, max_epochs, precision, quality):
                    if result["finished"]:
                        if result["success"]:
                            yield "Training completed successfully."
                        else:
                            yield f"Training failed with error: {result['error']}"
                    else:
                        yield result["output"]
            
            train_button.click(run_training, inputs=[training_dataset_dir, checkpoint_browser, batch_size, max_epochs, precision, model_quality], outputs=[train_status])
            train_button.click(stream_training_output, inputs=[training_dataset_dir, checkpoint_browser, batch_size, max_epochs, precision, model_quality], outputs=[train_status])
        
        with gr.Tab("Export Model"):
            gr.Markdown("## Export Trained Model")
            
            # Export configuration
            checkpoint_path = gr.Textbox(label="Checkpoint Path", info="Path to the trained checkpoint (e.g., /home/user/piper_tts_trainer/training/lightning_logs/version_0/checkpoints/epoch=XXX-step=YYY.ckpt)")
            export_dir = gr.Textbox(label="Export Directory", value=MODELS_DIR, info="Directory where the exported model will be saved")
            model_name = gr.Textbox(label="Model Name", value="my-model", info="Name of the exported model")
            
            export_button = gr.Button("Export Model")
            export_status = gr.Textbox(label="Export Status", interactive=False)
            
            # Export function
            def run_export(checkpoint_path, export_dir, model_name):
                result = export_model(checkpoint_path, export_dir, model_name)
                if result["success"]:
                    return f"Model exported successfully to: {result['onnx_path']}"
                else:
                    return f"Export failed: {result['error']}"
            
            export_button.click(run_export, inputs=[checkpoint_path, export_dir, model_name], outputs=[export_status])
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", share=False) 