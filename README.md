![Piper logo](etc/logo.png)

A fast, local neural text to speech system that sounds great and is optimized for the Raspberry Pi 4.
Piper is used in a [variety of projects](#people-using-piper).

``` sh
echo 'Welcome to the world of speech synthesis!' | \
  ./piper --model en_US-lessac-medium.onnx --output_file welcome.wav
```

[Listen to voice samples](https://rhasspy.github.io/piper-samples) and check out a [video tutorial by Thorsten Müller](https://youtu.be/rjq5eZoWWSo)

Voices are trained with [VITS](https://github.com/jaywalnut310/vits/) and exported to the [onnxruntime](https://onnxruntime.ai/).

[![A library from the Open Home Foundation](https://www.openhomefoundation.org/badges/ohf-library.png)](https://www.openhomefoundation.org/)

## Voices

Our goal is to support Home Assistant and the [Year of Voice](https://www.home-assistant.io/blog/2022/12/20/year-of-voice/).

[Download voices](VOICES.md) for the supported languages:

* Arabic (ar_JO)
* Catalan (ca_ES)
* Czech (cs_CZ)
* Welsh (cy_GB)
* Danish (da_DK)
* German (de_DE)
* Greek (el_GR)
* English (en_GB, en_US)
* Spanish (es_ES, es_MX)
* Finnish (fi_FI)
* French (fr_FR)
* Hungarian (hu_HU)
* Icelandic (is_IS)
* Italian (it_IT)
* Georgian (ka_GE)
* Kazakh (kk_KZ)
* Luxembourgish (lb_LU)
* Nepali (ne_NP)
* Dutch (nl_BE, nl_NL)
* Norwegian (no_NO)
* Polish (pl_PL)
* Portuguese (pt_BR, pt_PT)
* Romanian (ro_RO)
* Russian (ru_RU)
* Serbian (sr_RS)
* Swedish (sv_SE)
* Swahili (sw_CD)
* Turkish (tr_TR)
* Ukrainian (uk_UA)
* Vietnamese (vi_VN)
* Chinese (zh_CN)

You will need two files per voice:

1. A `.onnx` model file, such as [`en_US-lessac-medium.onnx`](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx)
2. A `.onnx.json` config file, such as [`en_US-lessac-medium.onnx.json`](https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json)

The `MODEL_CARD` file for each voice contains important licensing information. Piper is intended for text to speech research, and does not impose any additional restrictions on voice models. Some voices may have restrictive licenses, however, so please review them carefully!


## Installation

You can [run Piper with Python](#running-in-python) or download a binary release:

* [amd64](https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_amd64.tar.gz) (64-bit desktop Linux)
* [arm64](https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_arm64.tar.gz) (64-bit Raspberry Pi 4)
* [armv7](https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_armv7.tar.gz) (32-bit Raspberry Pi 3/4)

If you want to build from source, see the [Makefile](Makefile) and [C++ source](src/cpp).
You must download and extract [piper-phonemize](https://github.com/rhasspy/piper-phonemize) to `lib/Linux-$(uname -m)/piper_phonemize` before building.
For example, `lib/Linux-x86_64/piper_phonemize/lib/libpiper_phonemize.so` should exist for AMD/Intel machines (as well as everything else from `libpiper_phonemize-amd64.tar.gz`).


## Usage

1. [Download a voice](#voices) and extract the `.onnx` and `.onnx.json` files
2. Run the `piper` binary with text on standard input, `--model /path/to/your-voice.onnx`, and `--output_file output.wav`

For example:

``` sh
echo 'Welcome to the world of speech synthesis!' | \
  ./piper --model en_US-lessac-medium.onnx --output_file welcome.wav
```

For multi-speaker models, use `--speaker <number>` to change speakers (default: 0).

See `piper --help` for more options.

### Streaming Audio

Piper can stream raw audio to stdout as its produced:

``` sh
echo 'This sentence is spoken first. This sentence is synthesized while the first sentence is spoken.' | \
  ./piper --model en_US-lessac-medium.onnx --output-raw | \
  aplay -r 22050 -f S16_LE -t raw -
```

This is **raw** audio and not a WAV file, so make sure your audio player is set to play 16-bit mono PCM samples at the correct sample rate for the voice.

### JSON Input

The `piper` executable can accept JSON input when using the `--json-input` flag. Each line of input must be a JSON object with `text` field. For example:

``` json
{ "text": "First sentence to speak." }
{ "text": "Second sentence to speak." }
```

Optional fields include:

* `speaker` - string
    * Name of the speaker to use from `speaker_id_map` in config (multi-speaker voices only)
* `speaker_id` - number
    * Id of speaker to use from 0 to number of speakers - 1 (multi-speaker voices only, overrides "speaker")
* `output_file` - string
    * Path to output WAV file
    
The following example writes two sentences with different speakers to different files:

``` json
{ "text": "First speaker.", "speaker_id": 0, "output_file": "/tmp/speaker_0.wav" }
{ "text": "Second speaker.", "speaker_id": 1, "output_file": "/tmp/speaker_1.wav" }
```


## People using Piper

Piper has been used in the following projects/papers:

* [Home Assistant](https://github.com/home-assistant/addons/blob/master/piper/README.md)
* [Rhasspy 3](https://github.com/rhasspy/rhasspy3/)
* [NVDA - NonVisual Desktop Access](https://www.nvaccess.org/post/in-process-8th-may-2023/#voices)
* [Image Captioning for the Visually Impaired and Blind: A Recipe for Low-Resource Languages](https://www.techrxiv.org/articles/preprint/Image_Captioning_for_the_Visually_Impaired_and_Blind_A_Recipe_for_Low-Resource_Languages/22133894)
* [Open Voice Operating System](https://github.com/OpenVoiceOS/ovos-tts-plugin-piper)
* [JetsonGPT](https://github.com/shahizat/jetsonGPT)
* [LocalAI](https://github.com/go-skynet/LocalAI)
* [Lernstick EDU / EXAM: reading clipboard content aloud with language detection](https://lernstick.ch/)
* [Natural Speech - A plugin for Runelite, an OSRS Client](https://github.com/phyce/rl-natural-speech)
* [mintPiper](https://github.com/evuraan/mintPiper)
* [Vim-Piper](https://github.com/wolandark/vim-piper)

## Training

See the [training guide](TRAINING.md) and the [source code](src/python).

Pretrained checkpoints are available on [Hugging Face](https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main)


## Running in Python

See [src/python_run](src/python_run)

Install with `pip`:

``` sh
pip install piper-tts
```

and then run:

``` sh
echo 'Welcome to the world of speech synthesis!' | piper \
  --model en_US-lessac-medium \
  --output_file welcome.wav
```

This will automatically download [voice files](https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0) the first time they're used. Use `--data-dir` and `--download-dir` to adjust where voices are found/downloaded.

If you'd like to use a GPU, install the `onnxruntime-gpu` package:


``` sh
.venv/bin/pip3 install onnxruntime-gpu
```

and then run `piper` with the `--cuda` argument. You will need to have a functioning CUDA environment, such as what's available in [NVIDIA's PyTorch containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

# Piper TTS Trainer

A comprehensive GUI toolkit for training custom text-to-speech voices using the Piper engine.

## Overview

Piper TTS Trainer provides an end-to-end solution for creating, recording, and training custom voice models for text-to-speech applications. The project combines three key components:

1. **Piper Engine** - The core TTS system
2. **Recording Studio** - A web interface for recording voice datasets
3. **Training GUI** - A graphical interface for model training and testing

## System Requirements

- **Operating System**: Ubuntu 20.04+ (or Windows 10/11 with WSL2)
- **Python**: 3.9+
- **Storage**: At least 20GB free space
- **Memory**: 16GB RAM recommended (8GB minimum)
- **GPU**: NVIDIA GPU with CUDA support recommended for faster training
- **Audio**: Microphone (for recording datasets)

## Installation

### Option 1: Automatic Setup with Poetry (Recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/josepheudave/Piper-TTS-Trainer.git
   cd Piper-TTS-Trainer
   in wls cd /mnt/path/to/Piper-TTS-Trainer
   ```

2. Run the setup script:
   ```bash
   bash setup_poetry.sh
   ```
   
   This script will:
   - Install system dependencies
   - Set up Poetry environment
   - Clone necessary repositories
   - Configure CUDA for GPU acceleration (if available)
   - Build required components
   - Create launcher scripts

3. Download pre-trained checkpoints (optional but recommended):
   ```bash
   bash download_checkpoints.sh
   ```

### Option 2: Manual Setup

For advanced users who prefer manual configuration, follow these steps:

1. Install system dependencies:
   ```bash
   sudo apt update && sudo apt install -y python3-dev python3-pip espeak-ng ffmpeg build-essential git curl libespeak-ng-dev pkg-config cmake
   ```

2. Clone required repositories:
   ```bash
   git clone https://github.com/rhasspy/piper.git
   git clone https://github.com/rhasspy/piper-recording-studio.git
   git clone https://github.com/rhasspy/piper-phonemize.git
   ```

3. Install Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

4. Create and configure your environment manually following the steps in `setup_poetry.sh`

## Usage

### Recording Voice Dataset

1. Start the recording studio:
   ```bash
   ./run_recording_studio.sh
   ```

2. Open your browser and navigate to http://localhost:8000

3. Follow the on-screen instructions to:
   - Create a new speaker profile
   - Record sentences 
   - Review and re-record as needed

4. Export your dataset:
   ```bash
   ./export_dataset.sh en-GB my-dataset
   ```
   Replace `en-GB` with your language code and `my-dataset` with your preferred name.

### Training a Voice Model

1. Launch the training GUI:
   ```bash
   ./run_gui.sh
   ```

2. Open your browser and navigate to http://localhost:7860

3. Use the interface to:
   - Select your dataset
   - Configure training parameters
   - Start training
   - Monitor progress
   - Test your model

### Using Your Trained Model

Once training is complete, you can:

1. Export your model to ONNX format using the GUI
2. Use the model with the Piper TTS engine
3. Integrate with other applications like Home Assistant

## Project Structure

- `checkpoints/` - Pre-trained model checkpoints
- `datasets/` - Voice datasets
- `training/` - Training outputs and logs
- `models/` - Exported TTS models
- `normalized_wavs/` - Processed audio files
- `piper/` - Piper engine source code
- `piper-recording-studio/` - Recording studio interface
- `piper-phonemize/` - Text phonemization module

## Troubleshooting

### Common Issues

1. **CUDA/GPU errors**:
   - Ensure your NVIDIA drivers are up to date
   - Check compatibility between PyTorch and CUDA versions

2. **Audio recording issues**:
   - Verify microphone permissions in browser
   - Check microphone settings in your OS

3. **Training fails to start**:
   - Ensure the dataset is properly exported
   - Check for sufficient disk space

### Getting Help

If you encounter issues not covered here:
- Check the [Piper GitHub repository](https://github.com/rhasspy/piper/issues)
- Visit the [Rhasspy community forums](https://community.rhasspy.org/)

## Additional Resources

- [Piper Documentation](https://github.com/rhasspy/piper/blob/master/README.md)
- [Guide to Creating Custom TTS Voices](https://ssamjh.nz/create-custom-piper-tts-voice/)
- [Rhasspy Project](https://rhasspy.readthedocs.io/)

## License

This project includes components under various open source licenses. See individual repositories for details.

## Acknowledgments

- Based on the [Piper TTS engine](https://github.com/rhasspy/piper) by [Michael Hansen](https://github.com/synesthesiam)
- Recording Studio based on [piper-recording-studio](https://github.com/rhasspy/piper-recording-studio)
- Setup process inspired by the guide at [ssamjh.nz](https://ssamjh.nz/create-custom-piper-tts-voice/)
