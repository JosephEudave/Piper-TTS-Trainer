#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa

def normalize_audio(
    input_file: Path,
    output_file: Path,
    target_peak: float = 0.95,
    sample_rate: int = 22050,
    overwrite: bool = False
) -> bool:
    """
    Normalize audio file to have a peak amplitude at the specified target level.
    
    Args:
        input_file: Path to input audio file
        output_file: Path to output normalized audio file
        target_peak: Target peak amplitude (between 0 and 1)
        sample_rate: Target sample rate
        overwrite: Whether to overwrite existing output files
    
    Returns:
        True if file was processed, False otherwise
    """
    # Skip if output exists and we're not overwriting
    if output_file.exists() and not overwrite:
        return False
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load audio without normalization to get original values
        y, sr = librosa.load(input_file, sr=sample_rate, normalize=False)
        
        # Compute current peak amplitude
        current_peak = np.max(np.abs(y))
        
        if current_peak > 0:
            # Calculate scaling factor
            scaling_factor = target_peak / current_peak
            
            # Apply scaling to normalize
            y_normalized = y * scaling_factor
            
            # Save normalized audio
            sf.write(output_file, y_normalized, sample_rate)
            return True
        else:
            print(f"Warning: {input_file} seems to be silent, skipping")
            return False
            
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return False

def process_directory(
    input_dir: Path,
    output_dir: Path,
    target_peak: float = 0.95,
    sample_rate: int = 22050,
    overwrite: bool = False,
    extensions: list = [".wav"]
) -> int:
    """
    Process all audio files in a directory and its subdirectories.
    
    Args:
        input_dir: Directory containing input audio files
        output_dir: Directory to save normalized audio files
        target_peak: Target peak amplitude (between 0 and 1)
        sample_rate: Target sample rate
        overwrite: Whether to overwrite existing output files
        extensions: List of file extensions to process
    
    Returns:
        Number of files processed
    """
    # Get all audio files in input directory and subdirectories
    audio_files = []
    for ext in extensions:
        audio_files.extend(list(input_dir.glob(f"**/*{ext}")))
    
    # Sort files for deterministic processing
    audio_files.sort()
    
    # Process each file
    processed_count = 0
    for input_file in tqdm(audio_files, desc="Normalizing audio files"):
        # Determine output path (preserve directory structure)
        rel_path = input_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        
        # Normalize the audio
        if normalize_audio(input_file, output_file, target_peak, sample_rate, overwrite):
            processed_count += 1
    
    return processed_count

def main():
    parser = argparse.ArgumentParser(description="Normalize audio files for Piper TTS training")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing input audio files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save normalized audio files")
    parser.add_argument("--target-peak", type=float, default=0.95, help="Target peak amplitude (0-1), default: 0.95")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Target sample rate, default: 22050")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory")
        return 1
    
    # Validate target peak
    if args.target_peak <= 0 or args.target_peak > 1:
        print(f"Error: Target peak must be between 0 and 1, got {args.target_peak}")
        return 1
    
    # Process files
    processed_count = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        target_peak=args.target_peak,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite
    )
    
    print(f"Processed {processed_count} audio files")
    print(f"Normalized files saved to {output_dir}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 