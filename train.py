import os
import sys
import argparse
from pathlib import Path
import json
from typing import Optional

from config import Config, default_config

def setup_directories(config: Config) -> None:
    """Create necessary directories for training"""
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs("audio_cache", exist_ok=True)

def prepare_dataset(config: Config) -> None:
    """Prepare the dataset for training"""
    from piper_train.preprocess import preprocess
    
    # Check if we need to use Whisper for transcription
    if config.dataset.use_whisper:
        from faster_whisper import WhisperModel
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        whisper = WhisperModel("large-v3", device=device, compute_type="float16")
        # TODO: Implement Whisper transcription
        # This would need to be implemented based on your specific needs
        
    # Run preprocessing
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

def train_model(config: Config) -> None:
    """Train the model with the given configuration"""
    from piper_train import train
    
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
            raise ValueError("No checkpoints found to continue training from")
    elif config.training.action == "finetune":
        if not config.training.pretrained_checkpoint:
            raise ValueError("Pretrained checkpoint path required for finetuning")
        train_args["resume_from_checkpoint"] = config.training.pretrained_checkpoint
    
    # Add save_last if needed
    if config.training.save_last:
        train_args["save_last"] = True
    
    # Start training
    train(**train_args)

def main():
    parser = argparse.ArgumentParser(description="Train a Piper TTS model")
    parser.add_argument("--config", type=str, help="Path to config file (JSON)")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = Config(**config_dict)
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