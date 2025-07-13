#!/usr/bin/bash

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No test specified"
    echo "Usage: $0 [tensors|views|graphs|shapes|all]"
    exit 1
fi

# Determine which test to run based on the argument
case $1 in
    tensors)
        mojo -I . tests/test_tensors.mojo
        ;;
    views)
        mojo -I . tests/test_views.mojo
        ;;
    shapes)
        mojo -I . tests/test_shapes.mojo
        ;;

    graphs)
        mojo -I . tests/test_graphs.mojo
        ;;
    all)
        mojo -I . tests/test_tensors.mojo
        mojo -I . tests/test_views.mojo
        mojo -I . tests/test_shapes.mojo
        mojo -I . tests/test_graphs.mojo
        ;;
    *)
        echo "Error: Unknown test '$1'"
        echo "Available tests: tensors, views, shapes, graphs, all"
        exit 1
        ;;
esac
