#!/usr/bin/env python3
# Script to fix the backslash escaping issue in piper_trainer_gui.py

import os

# Path to the file
file_path = 'piper_trainer_gui.py'

# Read the file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the specific line
original = "return f\"{drive_letter}:{path[6:].replace('/', '\\\\')}\""
replacement = "return f\"{drive_letter}:{path[6:].replace('/', '\\\\\\\\')}\""

# Replace the line
new_content = content.replace(original, replacement)

# Write back to the file
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(new_content)

print(f"Updated {file_path} - replaced '{original}' with '{replacement}'") 