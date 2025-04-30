from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class DatasetConfig:
    """Configuration for dataset handling"""
    wav_dir: str = "wavs"  # Directory containing wav files
    metadata_file: str = "metadata.csv"  # Path to metadata file
    language: str = "es-mx"  # Language code
    sample_rate: int = 22050  # Sample rate for audio
    dataset_format: Literal["ljspeech", "mycroft"] = "ljspeech"
    single_speaker: bool = True
    use_whisper: bool = False  # Whether to use Whisper for transcription

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_name: str = "my_model"  # Name of the model
    output_dir: str = "output"  # Directory to save outputs
    batch_size: int = 12
    quality: Literal["high", "x-low", "medium"] = "medium"
    max_epochs: int = 10000
    checkpoint_epochs: int = 5
    num_ckpt: int = 1
    log_every_n_steps: int = 1000
    validation_split: float = 0.01
    num_test_examples: int = 1
    save_last: bool = False
    action: Literal["train", "continue", "finetune", "convert"] = "train"
    pretrained_checkpoint: Optional[str] = None  # Path to pretrained checkpoint if finetuning

@dataclass
class Config:
    """Main configuration class"""
    dataset: DatasetConfig = DatasetConfig()
    training: TrainingConfig = TrainingConfig()

# Default configuration
default_config = Config() 