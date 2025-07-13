#!/usr/bin/bash

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No test specified"
    echo "Usage: $0 [tensor|view]"
    exit 1
fi

# Determine which test to run based on the argument
case $1 in
    tensor)
        mojo -I . tests/test_tensor.mojo
        ;;
    view)
        mojo -I . tests/test_view.mojo
        ;;
    *)
        echo "Error: Unknown test '$1'"
        echo "Available tests: tensor, view"
        exit 1
        ;;
esac
