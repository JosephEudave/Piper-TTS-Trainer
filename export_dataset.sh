#!/bin/bash
# Script to export dataset from Recording Studio (Poetry version)

# Check if parameters are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <language-code> <output-directory>"
    echo "Example: $0 en-GB my-dataset"
    exit 1
fi

LANG="$1"
OUTPUT_DIR="$2"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Exporting dataset for language: $LANG to directory: $OUTPUT_DIR"

# Run the export command with Poetry
cd "$SCRIPT_DIR/piper-recording-studio"
poetry run python -m export_dataset output/$LANG/ "$SCRIPT_DIR/$OUTPUT_DIR"

echo "Export complete!"
