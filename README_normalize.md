# Audio Normalizer for Piper TTS Training

This tool normalizes audio files to ensure consistent amplitude levels for Piper TTS training, preventing the "audio amplitude out of range, auto clipped" warning.

## Requirements

- Python 3.6+
- librosa
- soundfile
- numpy
- tqdm

Install dependencies with:

```bash
pip install librosa soundfile numpy tqdm
```

## Usage

### Normalize Audio Files

```bash
python normalize_audio.py --input-dir wavs --output-dir normalized_wavs
```

#### Options

- `--input-dir`: Directory containing input audio files (required)
- `--output-dir`: Directory to save normalized audio files (required)
- `--target-peak`: Target peak amplitude between 0-1 (default: 0.95)
- `--sample-rate`: Target sample rate in Hz (default: 22050)
- `--overwrite`: Overwrite existing output files

### Update Config File

```bash
python update_config.py --config your_config.json --normalized-dir normalized_wavs
```

#### Options

- `--config`: Path to original config file (required)
- `--normalized-dir`: Path to normalized audio directory (required)
- `--output`: Path to save updated config (default: overwrite original)

## Complete Workflow

1. **Normalize your audio files**:
   ```bash
   python normalize_audio.py --input-dir wavs --output-dir normalized_wavs
   ```

2. **Update your config file** to use the normalized audio:
   ```bash
   python update_config.py --config your_config.json --normalized-dir normalized_wavs
   ```

3. **Run training** with your usual command:
   ```bash
   python train.py --config your_config.json
   ```

## How It Works

### Audio Normalization

The `normalize_audio.py` script:
1. Scans the input directory for audio files
2. Analyzes each file to find its peak amplitude
3. Scales the audio to reach the target peak amplitude (default 0.95)
4. Saves the normalized audio to the output directory, preserving folder structure

### Config Update

The `update_config.py` script:
1. Loads your existing Piper training configuration
2. Updates the `dataset.wav_dir` path to point to your normalized audio
3. Saves the updated configuration

This approach ensures audio won't be clipped during Piper's preprocessing stage while maintaining the original dynamic range of your recordings. 