#!/usr/bin/bash

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No test specified"
    echo "Usage: $0 [tensors|buffers|views|intlist|shapes|strides|ancestry|shared|bench|all]"
    exit 1
fi

DEBUG_MODE=""
if [ $# -ge 2 ] && [ "$2" = "d" ]; then
    DEBUG_MODE="-D LOGGING_LEVEL=debug"
fi

# Determine which test to run based on the argument
case $1 in
    tensors)
        echo "Running tensor tests"
        mojo -I . $DEBUG_MODE tests/test_tensors.mojo
        ;;
    buffers)
        mojo -I . $DEBUG_MODE tests/test_buffers.mojo
        ;;

    bench)
        echo "Running tensor multiplication benchmark"
        mojo -I . $DEBUG_MODE tests/test_matmul_bench.mojo
        ;;
    views)
        echo "Running view test cases"
        mojo -I . $DEBUG_MODE tests/test_views.mojo
        ;;
    shapes)
        echo "Running shape test cases"
        mojo -I . $DEBUG_MODE tests/test_shapes.mojo
        ;;
    shared)
        echo "Running shared test cases"
        mojo -I . $DEBUG_MODE tests/test_shared.mojo
        ;;

    strides)
        echo "Running strides test cases"
        mojo -I . $DEBUG_MODE tests/test_strides.mojo
        ;;

    ancestry)
        echo "Running ancestry test cases"
        mojo -I . $DEBUG_MODE tests/test_ancestry.mojo
        ;;

    intlist)
        echo "Running intlist test cases"
        mojo -I . $DEBUG_MODE tests/test_intlist.mojo
        ;;
    all)
        mojo -I . tests/test_tensors.mojo
        mojo -I . tests/test_buffers.mojo
        echo "Running view test cases"
        mojo -I . tests/test_views.mojo
        echo "Running shape test cases"
        mojo -I . tests/test_shapes.mojo
        echo "Running strides test cases"
        mojo -I . tests/test_strides.mojo
        echo "Running shared test cases"
        mojo -I . tests/test_shared.mojo
        echo "Running ancestry test cases"
        mojo -I . tests/test_ancestry.mojo
        echo "Running intList test cases"
        mojo -I . tests/test_intlist.mojo
        echo "Running tensor multiplication benchmark"
        mojo -I . tests/test_matmul_bench.mojo
        ;;
    *)
        echo "Error: Unknown test '$1'"
        echo "Available tests: tensors, buffers, views, shapes, intlists, strides, ancestry, shared, bench, all"
        exit 1
        ;;
esac
