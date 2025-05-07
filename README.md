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

A complete end-to-end solution for training custom text-to-speech (TTS) voices using the Piper neural TTS system. This project provides a user-friendly graphical interface for dataset preparation, audio normalization, preprocessing, training, and model export.

## Important: Linux Required

**This project only runs on Linux-based operating systems.** Windows users must use Windows Subsystem for Linux (WSL) to run Piper TTS Trainer.

## Features

- **User-friendly GUI**: Complete graphical interface for all stages of voice creation
- **Dataset Preparation**: Convert raw audio files into properly formatted datasets
- **Audio Normalization**: Fix clipping issues to prevent "audio amplitude out of range" warnings
- **Preprocessing**: Convert datasets into training-ready format
- **Model Training**: Train your TTS model with GPU acceleration
- **Model Export**: Convert trained models to ONNX format for inference

## Installation

### Windows Users (WSL Required)

1. **Install WSL**:
   - Run the `install_wsl.bat` script as administrator
   - Your computer will restart after the installation completes
   - After restart, Ubuntu will continue setup and prompt you to create a username and password

2. **After WSL Setup**:
   - Open Ubuntu from the Start menu
   - Navigate to your Piper TTS Trainer directory:
   ```bash
   cd /mnt/c/path/to/Piper-TTS-Trainer
   ```
   - Run the setup script:
   ```bash
   ./setup.sh
   ```

### Linux Users (Native)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Piper-TTS-Trainer.git
   cd Piper-TTS-Trainer
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## Running the Application

### Start the GUI

```bash
# Linux/WSL
./run_gui.sh

# Windows
# Method 1: Use the run_gui.bat script which provides instructions and launches open_wsl.bat
run_gui.bat

# Method 2: Manual WSL approach
# 1. Run open_wsl.bat to open a WSL terminal
# 2. Navigate to the project directory: cd /mnt/c/path/to/Piper-TTS-Trainer
# 3. Activate the virtual environment: source .venv/bin/activate
# 4. Run the application: python piper_trainer_gui.py
```

The interface will be available at http://localhost:7860 in your web browser.

## Workflow

### 1. Prepare Dataset
Import audio files and transcriptions to create a properly formatted dataset.

### 2. Normalize Audio (Important!)
Fix audio clipping issues to prevent "audio amplitude out of range" warnings during training.

### 3. Preprocess Dataset
Convert your normalized dataset into a format ready for training.

### 4. Train Model
Train your TTS model with GPU acceleration. The training process automatically detects and uses available GPU resources.

### 5. Export Model
Convert your trained model to ONNX format for production use.

## Important Notes on PyTorch and GPU Support

### PyTorch Version

This project uses specific versions of PyTorch to ensure compatibility with the latest CUDA drivers:

```
# For RTX GPU with CUDA 12.8 support (default in setup scripts)
torch==2.8.0.dev20250325+cu128
torchvision==0.22.0.dev20250325+cu128
torchaudio==2.6.0.dev20250325+cu128
```

If you encounter issues with the default PyTorch version, you can modify `setup.sh` or `setup.bat` to use a different version:

- **For older NVIDIA GPUs with CUDA 11.8**:
  ```
  torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```

- **For CPU-only usage** (much slower training):
  ```
  torch>=2.2.0 torchvision>=0.15.0 torchaudio>=2.0.0
  ```

### GPU Acceleration

The training process is configured to use GPU by default. If a CUDA-capable GPU is detected, training will automatically use it. For optimal performance:

- Ensure your NVIDIA drivers are up to date
- Use a GPU with at least 8GB VRAM for reasonable training times
- The batch size can be adjusted in the GUI based on your GPU memory

## Troubleshooting

### PyTorch/CUDA Issues

If you encounter CUDA-related errors:

1. Check your NVIDIA driver version with `nvidia-smi`
2. Modify the PyTorch installation in `setup.sh` to match your CUDA version
3. Reinstall with: `pip install --force-reinstall [pytorch-version]`

### WSL Issues

If you encounter WSL-related issues:

1. Ensure WSL 2 is enabled: `wsl --set-default-version 2`
2. Update the WSL kernel: `wsl --update`
3. Check Ubuntu installation: `wsl --list --verbose`

## License

[Insert your license information here]

## Acknowledgments

- This project builds upon the [Piper TTS system](https://github.com/rhasspy/piper)
- Special thanks to the contributors of the original Piper project
