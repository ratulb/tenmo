#!/usr/bin/bash
clear
# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No example specified"
    echo "Usage: $0 [mnist|xor|spiral|all]"
    exit 1
fi

DEBUG_MODE=""
if [ $# -ge 2 ] && [ "$2" = "d" ]; then
    DEBUG_MODE="-D LOGGING_LEVEL=debug"
fi

# Determine which test to run based on the argument
case $1 in
    mnist)
        echo "Running mnist training loop"
        mojo -I . $DEBUG_MODE examples/mnist.mojo
        ;;
    xor)
        echo "Running xor training loop"
        mojo -I . $DEBUG_MODE examples/xor.mojo
        ;;
    spiral)
        echo "Running mojo spiral training loop"
        mojo -I . $DEBUG_MODE examples/spiral.mojo
        ;;

    all)
        echo "Running mnist training loop"
        mojo -I . $DEBUG_MODE examples/mnist.mojo

        echo "Running xor training loop"
        mojo -I . $DEBUG_MODE examples/xor.mojo

        echo "Running mojo spiral training loop"
        mojo -I . $DEBUG_MODE examples/spiral.mojo
        ;;
    *)
        echo "Error: Unknown test '$1'"
        echo "Available training loops: mnist, xor, spiral, all"
        exit 1
        ;;
esac
