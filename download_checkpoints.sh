#!/bin/bash
# Helper script to download pre-trained Piper voice models (ONNX)
# These can be used directly with Piper for inference

set -e  # Exit on error
PIPER_HOME="$(pwd)"
MODELS_DIR="$PIPER_HOME/models"

mkdir -p "$MODELS_DIR"

echo "========================================================="
echo "Piper TTS - Voice Model Downloader"
echo "========================================================="
echo "This script downloads pre-trained ONNX models for use with Piper"
echo

# Base URL for voice models
VOICES_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Function to download a voice model and its config
download_model() {
    local lang_code=$1
    local country_code=$2
    local voice=$3
    local quality=$4
    
    local full_lang="${lang_code}_${country_code}"
    local model_name="${full_lang}-${voice}-${quality}"
    local model_path="${lang_code}/${full_lang}/${voice}/${quality}"
    
    # Create directory
    mkdir -p "$MODELS_DIR/${full_lang}"
    
    echo "Downloading ${model_name} model and config..."
    
    # Download ONNX model
    wget -O "$MODELS_DIR/${full_lang}/${model_name}.onnx" \
        "$VOICES_BASE/${model_path}/${model_name}.onnx" || \
        echo "Failed to download ${model_name}.onnx"
    
    # Download JSON config
    wget -O "$MODELS_DIR/${full_lang}/${model_name}.onnx.json" \
        "$VOICES_BASE/${model_path}/${model_name}.onnx.json" || \
        echo "Failed to download ${model_name}.onnx.json"
    
    echo "Downloaded ${model_name} model and config"
}

# Available models list (not exhaustive)
declare -A available_models
available_models=(
    ["es_MX-ald-medium"]="es MX ald medium"
    ["es_ES-css10-medium"]="es ES css10 medium"
    ["en_US-lessac-medium"]="en US lessac medium"
    ["en_US-ryan-medium"]="en US ryan medium"
    ["en_US-libritts-high"]="en US libritts high"
    ["en_US-vctk-medium"]="en US vctk medium"
    ["es_MX-claude-medium"]="es MX claude medium"
    ["es_ES-mls_10246-medium"]="es ES mls_10246 medium"
)

# Display menu and download options
show_menu() {
    echo "Available voice models:"
    local i=1
    declare -a models_array
    for model in "${!available_models[@]}"; do
        echo "  $i. $model"
        models_array[$i]=$model
        ((i++))
    done
    echo "  $i. Download a custom model (specify details)"
    echo "  0. Download all models"
    
    read -p "Enter your choice [0-$i]: " choice
    
    if [[ $choice -eq 0 ]]; then
        for model in "${!available_models[@]}"; do
            IFS=' ' read -r lang country voice quality <<< "${available_models[$model]}"
            download_model "$lang" "$country" "$voice" "$quality"
        done
    elif [[ $choice -eq $i ]]; then
        download_custom_model
    elif [[ $choice -gt 0 && $choice -lt $i ]]; then
        selected=${models_array[$choice]}
        IFS=' ' read -r lang country voice quality <<< "${available_models[$selected]}"
        download_model "$lang" "$country" "$voice" "$quality"
    else
        echo "Invalid choice"
        exit 1
    fi
}

# Function to download a custom model
download_custom_model() {
    echo
    echo "Please provide the details for the model you want to download:"
    echo "Example format: es MX ald medium"
    echo
    
    read -p "Enter language code (e.g., 'en', 'es'): " lang_code
    read -p "Enter country code (e.g., 'US', 'ES', 'MX'): " country_code
    read -p "Enter voice name (e.g., 'lessac', 'ald'): " voice
    read -p "Enter quality level (e.g., 'medium', 'high'): " quality
    
    download_model "$lang_code" "$country_code" "$voice" "$quality"
}

# Main execution
if [[ $# -gt 0 ]]; then
    if [[ $1 == "--model" && $# -eq 5 ]]; then
        # Command line parameters provided
        download_model "$2" "$3" "$4" "$5"
    elif [[ $1 == "--name" && $# -eq 2 ]]; then
        # Download by model name
        model_name=$2
        if [[ -n "${available_models[$model_name]}" ]]; then
            IFS=' ' read -r lang country voice quality <<< "${available_models[$model_name]}"
            download_model "$lang" "$country" "$voice" "$quality"
        else
            echo "Model $model_name not found in available models list."
            echo "Available models: ${!available_models[*]}"
            exit 1
        fi
    else
        echo "Usage:"
        echo "  $0                                     # Interactive mode"
        echo "  $0 --model <lang> <country> <voice> <quality>  # Direct download"
        echo "  $0 --name <model-name>                # Download by name"
        echo
        echo "Example: $0 --model es MX ald medium"
        echo "Example: $0 --name es_MX-ald-medium"
        exit 1
    fi
else
    # Interactive mode
    show_menu
fi

echo
echo "Download complete. Models saved to $MODELS_DIR"
echo "=========================================================" 