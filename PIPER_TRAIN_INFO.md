# About the piper_train Module

## Warning: piper_train module not found
If you see this warning when running the Piper TTS Trainer, it means the application cannot find the `piper_train` module which is required for training TTS models. You will still be able to use the preprocessing and configuration features, but you won't be able to train models.

## How to Install piper_train

There are several ways to install the piper_train module:

### Option 1: Install from Official Repository (Recommended)

The module is part of the official Piper project. You can install it by:

```
git clone https://github.com/rhasspy/piper
cd piper
pip install -e .
```

### Option 2: Direct pip install (if available)

If the module is published on PyPI:

```
pip install piper_train
```

### Option 3: Manual Installation

1. Download the piper_train module from the appropriate source
2. Place it in your Python path
3. You can add it to the Python path by:
   ```
   set PYTHONPATH=%PYTHONPATH%;C:\path\to\piper\modules
   ```

## Verifying Installation

To verify that piper_train is installed correctly, run:

```
python -c "import piper_train; print('piper_train found!')"
```

If this runs without error, the module is installed properly.

## Additional Information

The piper_train module provides the following essential functions:
- `preprocess`: Prepares audio data for training
- `train`: Handles the actual training of TTS models

Without these functions, the training tab of the interface will not function correctly. 