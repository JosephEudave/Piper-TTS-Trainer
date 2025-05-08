#!/usr/bin/env python3
import os
import re

# Input and output files
input_file = r'C:\Users\User\Desktop\metadata.csv'
output_file = r'C:\Users\User\Desktop\metadata_reformatted.csv'

# Regular expression to extract the filename from the path
# This will match paths like: /mnt/c/Users/User/Documents/GitHub/Piper-TTS-Trainer/wavs/1.wav
# Or Windows paths like: C:\Users\User\Desktop\wavs\1.wav
path_pattern = r'(?:/mnt/c/|[A-Z]:\\).*?([^/\\]+)\.wav\|'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # Try to match the pattern
        match = re.match(path_pattern, line)
        
        if match:
            # Extract the ID (filename without extension)
            file_id = match.group(1)
            
            # Get the text part (everything after the first |)
            text = line.split('|', 1)[1].strip()
            
            # Write the reformatted line
            outfile.write(f"{file_id}|{text}\n")
        else:
            # If line doesn't match our pattern, write it as is but show a warning
            print(f"Warning: Line does not match expected format: {line.strip()}")
            outfile.write(line)

print(f"Reformatted metadata written to {output_file}")
print("Please verify the file before using it.") 