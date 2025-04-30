# Piper TTS Training

This is a local adaptation of the Piper TTS training notebook. It provides a more organized way to train TTS models using Piper.

## Project Structure

```
.
├── config.py          # Configuration classes
├── config.json        # Sample configuration file
├── train.py          # Main training script
├── wavs/             # Directory for audio files
└── metadata.csv      # Transcription file
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place your audio files in the `wavs/` directory
   - Create a `metadata.csv` file with transcriptions (or use Whisper for transcription)

## Configuration

The training process can be configured using the `config.json` file. Here are the main options:

### Dataset Configuration
- `wav_dir`: Directory containing WAV files
- `metadata_file`: Path to metadata file
- `language`: Language code (e.g., "en-us")
- `sample_rate`: Audio sample rate (22050 or 16000)
- `dataset_format`: Dataset format ("ljspeech" or "mycroft")
- `single_speaker`: Whether the dataset is single-speaker
- `use_whisper`: Whether to use Whisper for transcription

### Training Configuration
- `model_name`: Name of the model
- `output_dir`: Directory to save outputs
- `batch_size`: Training batch size
- `quality`: Model quality ("high", "medium", or "x-low")
- `max_epochs`: Maximum number of training epochs
- `checkpoint_epochs`: How often to save checkpoints
- `num_ckpt`: Number of checkpoints to keep
- `log_every_n_steps`: How often to log training progress
- `validation_split`: Validation split ratio
- `num_test_examples`: Number of test examples
- `save_last`: Whether to save the last model
- `action`: Training action ("train", "continue", "finetune", or "convert")
- `pretrained_checkpoint`: Path to pretrained checkpoint (for finetuning)

## Usage

1. Modify `config.json` according to your needs
2. Run the training script:
```bash
python train.py --config config.json
```

## Monitoring Training

To monitor training progress, you can use TensorBoard:
```bash
tensorboard --logdir output
```

## Notes

- For GPU training, make sure you have CUDA installed
- The Whisper transcription feature requires additional setup
- Make sure your audio files are in the correct format (WAV, 16-bit, mono)
- For large datasets, consider adjusting batch size and checkpoint frequency 