#!/bin/bash

# Default file
DEFAULT_FILE="run_tensor.mojo"

# Use provided argument or default
TARGET_FILE="${1:-$DEFAULT_FILE}"

# Check if file exists
if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: File '$TARGET_FILE' not found!"
    exit 1
fi

# Run mojo
mojo -I . "$TARGET_FILE"
