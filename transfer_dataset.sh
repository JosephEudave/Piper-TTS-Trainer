#!/bin/bash
# Piper TTS Trainer - Dataset Transfer Helper
# This script helps transfer audio datasets from Windows to the Linux environment

echo "========================================================="
echo "Piper TTS Dataset Transfer Helper"
echo "========================================================="
echo

# Set up directories
PIPER_HOME="$HOME/piper_tts_trainer"
DATASETS_DIR="$PIPER_HOME/datasets"

# Make sure directories exist
mkdir -p "$DATASETS_DIR"

# Function to display help
show_help() {
    echo "This script helps you transfer a Windows dataset to the Linux environment."
    echo
    echo "Usage:"
    echo "  ./transfer_dataset.sh <windows_path> <dataset_name>"
    echo
    echo "Example:"
    echo "  ./transfer_dataset.sh /mnt/c/Users/YourName/Documents/my_voice my_voice_dataset"
    echo
    echo "The script will copy files from the Windows path to:"
    echo "  $DATASETS_DIR/<dataset_name>"
    echo
}

# Check for arguments
if [ "$#" -lt 2 ]; then
    show_help
    exit 1
fi

WINDOWS_PATH="$1"
DATASET_NAME="$2"
TARGET_DIR="$DATASETS_DIR/$DATASET_NAME"

# Check if source exists
if [ ! -d "$WINDOWS_PATH" ]; then
    echo "Error: Source directory '$WINDOWS_PATH' does not exist."
    echo
    echo "Windows paths in WSL start with /mnt/ followed by the drive letter,"
    echo "for example: /mnt/c/Users/YourName/Documents"
    echo
    exit 1
fi

# Create target directory
mkdir -p "$TARGET_DIR"

echo "Copying dataset from Windows to Linux..."
echo "From: $WINDOWS_PATH"
echo "To:   $TARGET_DIR"
echo

# Copy files
cp -rv "$WINDOWS_PATH"/* "$TARGET_DIR/"

echo
echo "Transfer complete!"
echo "Your dataset is now available at: $TARGET_DIR"
echo 
echo "For LJSpeech format, make sure you have:"
echo "- metadata.csv file with format: id|text"
echo "- wav directory with audio files named id.wav"
echo
echo "To use this dataset in Piper TTS Trainer:"
echo "1. Go to the Dataset Configuration tab"
echo "2. Enter '$TARGET_DIR' as the Input Directory"
echo "3. Continue with preprocessing and training"
echo 