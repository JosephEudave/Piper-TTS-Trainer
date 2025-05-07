#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

def update_config(
    config_path: Path,
    normalized_dir: Path,
    output_path: Path = None
) -> bool:
    """
    Update a Piper TTS training config file to use normalized audio.
    
    Args:
        config_path: Path to the original config file
        normalized_dir: Path to the directory with normalized audio
        output_path: Path to save the updated config (if None, overwrites original)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the original config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Store the original wav_dir
        original_wav_dir = config.get('dataset', {}).get('wav_dir', 'wavs')
        
        # Update the wav_dir
        if 'dataset' not in config:
            config['dataset'] = {}
        
        config['dataset']['wav_dir'] = str(normalized_dir)
        
        # Save the config
        output_path = output_path or config_path
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            
        print(f"Updated config: wav_dir changed from '{original_wav_dir}' to '{normalized_dir}'")
        return True
        
    except Exception as e:
        print(f"Error updating config: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Update Piper config to use normalized audio")
    parser.add_argument("--config", type=str, required=True, help="Path to original config file")
    parser.add_argument("--normalized-dir", type=str, required=True, help="Path to normalized audio directory")
    parser.add_argument("--output", type=str, help="Path to save updated config (default: overwrite original)")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    config_path = Path(args.config)
    normalized_dir = Path(args.normalized_dir)
    output_path = Path(args.output) if args.output else None
    
    # Validate paths
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' does not exist")
        return 1
        
    if not normalized_dir.is_dir():
        print(f"Warning: Normalized audio directory '{normalized_dir}' does not exist or is not a directory")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Update the config
    if update_config(config_path, normalized_dir, output_path):
        print(f"Config updated successfully: {output_path or config_path}")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main()) 