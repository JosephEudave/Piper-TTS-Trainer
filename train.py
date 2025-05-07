import os
import sys
import argparse
from pathlib import Path
import json
from typing import Optional
import torch

# Add piper_train to Python path
piper_path = os.path.join(os.path.dirname(__file__), "piper", "src", "python")
sys.path.insert(0, piper_path)

from config import Config, default_config

def setup_directories(config: Config) -> None:
    """Create necessary directories for training"""
    os.makedirs(config["training"]["output_dir"], exist_ok=True)
    os.makedirs("audio_cache", exist_ok=True)

def prepare_dataset(config: dict) -> None:
    """Prepare the dataset for training"""
    import subprocess
    import sys
    
    # Check if we need to use Whisper for transcription
    if config.get("dataset", {}).get("use_whisper", False):
        from faster_whisper import WhisperModel
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        whisper = WhisperModel("large-v3", device=device, compute_type="float16")
        # TODO: Implement Whisper transcription
        # This would need to be implemented based on your specific needs
    
    # Set environment variables
    if "espeak_data_path" in config["dataset"]:
        os.environ["ESPEAK_DATA_PATH"] = str(config["dataset"]["espeak_data_path"])
    
    # Set LD_LIBRARY_PATH for onnxruntime
    if "ld_library_path" in config["dataset"]:
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld_path = str(config["dataset"]["ld_library_path"])
        os.environ["LD_LIBRARY_PATH"] = f"{new_ld_path}:{current_ld_path}" if current_ld_path else new_ld_path
    
    # Run preprocessing using the preprocess script as a module
    subprocess.run([
        sys.executable,
        "-m", "piper_train.preprocess",
        "--input-dir", str(config["dataset"]["wav_dir"]),
        "--output-dir", str(config["training"]["output_dir"]),
        "--language", config["dataset"]["language"],
        "--sample-rate", str(config["dataset"]["sample_rate"]),
        "--dataset-format", config["dataset"]["dataset_format"],
        "--cache-dir", "audio_cache",
        "--single-speaker" if config["dataset"].get("single_speaker", False) else "",
        "--max-workers", str(config["dataset"].get("max_workers", 4))
    ], check=True)

def train_model(config: dict) -> None:
    """Train the model with the given configuration"""
    from piper_train.vits.lightning import VitsModel
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    
    # Check if CPU mode is forced
    use_cpu = config["training"].get("use_cpu", False)
    
    # Set up training arguments
    trainer_kwargs = {
        "accelerator": "cpu" if use_cpu else ("gpu" if torch.cuda.is_available() else "cpu"),
        "devices": 1,
        "max_epochs": config["training"]["max_epochs"],
        "default_root_dir": config["training"]["output_dir"]
    }
    
    # Add checkpoint callback if specified
    if config["training"].get("checkpoint_epochs"):
        trainer_kwargs["callbacks"] = [
            ModelCheckpoint(every_n_epochs=config["training"]["checkpoint_epochs"])
        ]
    
    # Create trainer
    trainer = Trainer(**trainer_kwargs)
    
    # Load dataset config
    with open(os.path.join(config["training"]["output_dir"], "config.json"), "r", encoding="utf-8") as f:
        dataset_config = json.load(f)
    
    # Create model
    model_kwargs = {
        "num_symbols": dataset_config["num_symbols"],
        "num_speakers": dataset_config["num_speakers"],
        "sample_rate": dataset_config["audio"]["sample_rate"],
        "dataset": [os.path.join(config["training"]["output_dir"], "dataset.jsonl")],
        "batch_size": config["training"]["batch_size"]
    }
    
    # Adjust model architecture based on quality setting
    if config["training"]["quality"] == "x-low":
        model_kwargs.update({
            "hidden_channels": 96,
            "inter_channels": 96,
            "filter_channels": 384
        })
    elif config["training"]["quality"] == "high":
        model_kwargs.update({
            "resblock": "1",
            "resblock_kernel_sizes": (3, 7, 11),
            "resblock_dilation_sizes": ((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            "upsample_rates": (8, 8, 2, 2),
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": (16, 16, 4, 4)
        })
    
    model = VitsModel(**model_kwargs)
    
    # Start training
    trainer.fit(model)

def main():
    parser = argparse.ArgumentParser(description="Train a Piper TTS model")
    parser.add_argument("--config", type=str, help="Path to config file (JSON)")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = default_config
    
    # Setup directories
    setup_directories(config)
    
    # Prepare dataset
    prepare_dataset(config)
    
    # Train model
    train_model(config)

if __name__ == "__main__":
    main() 