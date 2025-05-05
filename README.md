# Piper TTS Training

A simple and organized way to train your own Text-to-Speech (TTS) model using Piper. This guide will walk you through the entire process step by step.

## 🚀 Quick Start Guide

### Step 1: Setup Your Environment
1. Make sure you have Python installed (version 3.9 or higher)
2. Run the setup script:
   ```bash
   setup_environment.bat
   ```
   This will create a virtual environment and install all required dependencies automatically.
3. Activate the virtual environment:
   ```bash
   venv\Scripts\activate
   ```

### Step 2: Prepare Your Audio Files
1. Create a folder named `wavs` in the project directory
2. Add your audio files to the `wavs` folder
   - Files must be in WAV format
   - Use 16-bit audio
   - Use mono channel (not stereo)
   - Recommended sample rate: 22050 Hz or 16000 Hz
   - Keep file names simple (e.g., `audio1.wav`, `audio2.wav`)

### Step 3: Create Your Metadata File
1. Create a file named `metadata.csv` in the project directory
2. Add your transcriptions in this format:
   ```
   filename|transcription
   audio1.wav|This is the first sentence
   audio2.wav|This is the second sentence
   ```
   - Each line should contain the filename and its transcription, separated by a pipe (|)
   - Make sure the transcriptions are accurate
   - You can use a text editor or spreadsheet program to create this file

### Step 4: Configure Your Training
1. Open `config.json` in a text editor
2. Set these important settings:
   ```json
   {
     "language": "en-us",  // Change to your language code
     "sample_rate": 22050, // Match your audio files' sample rate
     "quality": "medium",  // Choose: "high", "medium", or "x-low"
     "batch_size": 32,     // Adjust based on your computer's memory
     "max_epochs": 1000    // Number of training cycles
   }
   ```

### Step 5: Start Training
1. Run the training script:
   ```bash
   python train.py --config config.json
   ```
2. The program will:
   - Check your audio files
   - Process the data
   - Start training
   - Save progress automatically

### Step 6: Monitor Progress
1. Open a new terminal
2. Run TensorBoard to see training graphs:
   ```bash
   tensorboard --logdir output
   ```
3. Open your web browser and go to: `http://localhost:6006`

### Step 7: Using Your Trained Model
1. Once training is complete, your model will be in the `output` folder
2. You can use this model with Piper TTS to generate speech

## 📝 Important Notes

- **Hardware Requirements:**
  - A computer with at least 8GB RAM
  - A GPU is recommended for faster training (CUDA compatible)
  - Sufficient disk space for your dataset and model

- **Training Time:**
  - Small datasets (1-2 hours): 1-2 days
  - Medium datasets (5-10 hours): 3-5 days
  - Large datasets (20+ hours): 1-2 weeks

- **Tips for Better Results:**
  - Use high-quality audio recordings
  - Ensure accurate transcriptions
  - Include a variety of sentences
  - More data usually means better results

## 🆘 Troubleshooting

If you encounter any issues:
1. Check that your audio files are in the correct format
2. Verify your metadata.csv format
3. Make sure all paths in config.json are correct
4. Check that you have enough disk space
5. Ensure the virtual environment is properly activated
6. Make sure all dependencies are installed correctly by running `pip list` in the activated environment

## 📚 Additional Resources

- [Piper TTS Documentation](https://github.com/rhasspy/piper)
- [Audio Format Requirements](https://github.com/rhasspy/piper#audio-format)
- [Training Tips and Best Practices](https://github.com/rhasspy/piper#training)

## 🤝 Need Help?

If you're stuck or have questions:
1. Check the troubleshooting section above
2. Review the error messages carefully
3. Make sure you followed all steps correctly
4. If still having issues, you can open an issue on GitHub 