import os
import sys
import argparse
import wave
import numpy as np
import librosa
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def check_audio_format(file_path):
    """Check if audio file meets requirements"""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            n_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            
            if n_channels != 1:
                return False, f"Audio is not mono (has {n_channels} channels)"
            if sample_width != 2:  # 16-bit
                return False, f"Audio is not 16-bit (has {sample_width*8}-bit)"
            if sample_rate not in [16000, 22050]:
                return False, f"Sample rate is {sample_rate}Hz (should be 16000 or 22050Hz)"
            return True, "OK"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def convert_audio(input_path, output_path, target_sr=22050):
    """Convert audio to required format"""
    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Resample if needed
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Ensure 16-bit
        y = (y * 32767).astype(np.int16)
        
        # Save with correct format
        sf.write(output_path, y, target_sr, subtype='PCM_16')
        return True, "OK"
    except Exception as e:
        return False, f"Error converting file: {str(e)}"

def validate_metadata(metadata_path, wav_dir, single_speaker=True):
    """Validate metadata format"""
    try:
        # Read metadata
        df = pd.read_csv(metadata_path, sep='|', header=None, 
                        names=['file', 'speaker', 'text'] if not single_speaker else ['file', 'text'])
        
        # Check format
        errors = []
        for idx, row in df.iterrows():
            # Check file exists
            file_path = os.path.join(wav_dir, row['file'])
            if not os.path.exists(file_path):
                errors.append(f"Line {idx+1}: File {row['file']} not found")
            
            # Check text is not empty
            if pd.isna(row['text']) or not row['text'].strip():
                errors.append(f"Line {idx+1}: Empty text")
            
            # For multi-speaker, check speaker column
            if not single_speaker and (pd.isna(row['speaker']) or not row['speaker'].strip()):
                errors.append(f"Line {idx+1}: Empty speaker name")
        
        return len(errors) == 0, errors
    except Exception as e:
        return False, [f"Error reading metadata: {str(e)}"]

def process_directory(input_dir, output_dir, target_sr=22050):
    """Process all audio files in directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files
    audio_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.wav', '.mp3', '.ogg', '.flac'))]
    
    print(f"Found {len(audio_files)} audio files")
    print("Processing files...")
    
    results = []
    for file in tqdm(audio_files):
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(output_dir, file)
        
        # Check if already in correct format
        is_valid, msg = check_audio_format(input_path)
        if is_valid:
            # Just copy if already correct
            import shutil
            shutil.copy2(input_path, output_path)
            results.append((file, "Already in correct format"))
        else:
            # Convert if needed
            success, msg = convert_audio(input_path, output_path, target_sr)
            results.append((file, msg))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Preprocess audio files for Piper TTS training')
    parser.add_argument('--input_dir', required=True, help='Input directory with audio files')
    parser.add_argument('--output_dir', required=True, help='Output directory for processed files')
    parser.add_argument('--metadata', required=True, help='Path to metadata file')
    parser.add_argument('--target_sr', type=int, default=22050, help='Target sample rate (default: 22050)')
    parser.add_argument('--single_speaker', action='store_true', help='Single speaker dataset')
    
    args = parser.parse_args()
    
    # Validate metadata first
    print("Validating metadata...")
    is_valid, errors = validate_metadata(args.metadata, args.input_dir, args.single_speaker)
    if not is_valid:
        print("Metadata validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    print("Metadata validation passed!")
    
    # Process audio files
    print("\nProcessing audio files...")
    results = process_directory(args.input_dir, args.output_dir, args.target_sr)
    
    # Print results
    print("\nProcessing results:")
    for file, msg in results:
        print(f"{file}: {msg}")

if __name__ == "__main__":
    main() 