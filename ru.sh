#!/usr/bin/bash

# Check if an mojo file was specified
if [ $# -eq 0 ]; then
    echo "Error: No file specified"
    exit 1
fi

filename="$1"
name="${filename%.mojo}"

MOJO_ENABLE_STACK_TRACE_ON_ERROR=False mojo build -debug-level=line-tables ${filename}

MOJO_ENABLE_STACK_TRACE_ON_ERROR=True ./${name}


