#!/usr/bin/bash

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No test specified"
    echo "Usage: $0 [tensors|views|intlist|shapes|strides|ancestry|shared|bench|all]"
    exit 1
fi

# Determine which test to run based on the argument
case $1 in
    tensors)
        mojo -I . tests/test_tensors.mojo
        ;;
    bench)
        echo "Running tensor multiplication benchmark"
        mojo -I . tests/test_matmul_bench.mojo
        ;;
    views)
        echo "Running view test cases"
        mojo -I . tests/test_views.mojo
        ;;
    shapes)
        echo "Running shape test cases"
        mojo -I . tests/test_shapes.mojo
        ;;
    shared)
        echo "Running shared test cases"
        mojo -I . tests/test_shared.mojo
        ;;

    strides)
        echo "Running strides test cases"
        mojo -I . tests/test_strides.mojo
        ;;

    ancestry)
        echo "Running ancestry test cases"
        mojo -I . tests/test_ancestry.mojo
        ;;

    intlist)
        echo "Running intlist test cases"
        mojo -I . tests/test_intlist.mojo
        ;;
    all)
        mojo -I . tests/test_tensors.mojo
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
        echo "Available tests: tensors, views, shapes, intlists, strides, ancestry, shared, bench, all"
        exit 1
        ;;
esac
