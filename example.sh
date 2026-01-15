#!/usr/bin/bash
clear
# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No example specified"
    echo "Usage: $0 [binary_mnist|mnist|xor|spiral|cifar_10|mnist_conv2d]"
    exit 1
fi

DEBUG_MODE=""
if [ $# -ge 2 ] && [ "$2" = "d" ]; then
    DEBUG_MODE="-D LOGGING_LEVEL=debug"
fi

# Determine which test to run based on the argument
case $1 in
    binary_mnist)
        echo "Running binary mnist training loop"
        mojo -I . $DEBUG_MODE examples/binary_mnist.mojo
        ;;

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
    cifar_10)
        echo "Running mojo cifar_10 training loop"
        mojo -I . $DEBUG_MODE examples/cifar_10.mojo
        ;;
    mnist_conv2d)
        echo "Running mojo mnist_conv2d.mojo training loop"
        mojo -I . $DEBUG_MODE examples/mnist_conv2d.mojo
        ;;

    *)
       echo "Error: Unknown test '$1'"
       echo "Available training loops: binary_mnist, mnist, xor, spiral, cifar_10, mnist_conv2d"
       exit 1
    ;;
esac
