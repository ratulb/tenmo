#!/bin/bash

# Default target. debug.mojo is a private scratch file, excluded from the
# published distribution, so fall back to an example that ships everywhere —
# otherwise a bare ./fire.sh fails on a trimmed tree.
DEFAULT_FILE="debug.mojo"
if [ ! -f "$DEFAULT_FILE" ] && [ $# -eq 0 ]; then
    DEFAULT_FILE="examples/xor.mojo"
fi

# Use provided argument or default
TARGET_FILE="${1:-$DEFAULT_FILE}"

# Check if file exists
if [ ! -f "$TARGET_FILE" ]; then
    echo "Error: File '$TARGET_FILE' not found!"
    exit 1
fi

# Run mojo with all args
shift
pixi run mojo -I . "$TARGET_FILE" "$@"
