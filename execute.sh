#!/usr/bin/bash

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No test specified"
    echo "Usage: $0 [tensors|mm2d|argminmax|repeat|tiles|slice|linspace|softmax|relu|shuffle|buffers|flatten|permute|squeeze|unsqueeze|views|gradbox|ndb|transpose|intlist|shapes|strides|ancestry|shared|bench|validators|ce|synth_smoke|synth_mnist|all]"
    exit 1
fi

DEBUG_MODE=""
if [ $# -ge 2 ] && [ "$2" = "d" ]; then
    DEBUG_MODE="-D LOGGING_LEVEL=debug"
fi

# Determine which test to run based on the argument
case $1 in
    tensors)
        echo "Running mojo -I . tests/test_tensors.mojo"
        mojo -I . $DEBUG_MODE tests/test_tensors.mojo
        ;;
    softmax)
        echo "Running mojo -I . tests/test_softmax.mojo"
        mojo -I . $DEBUG_MODE tests/test_softmax.mojo
        ;;
    repeat)
        echo "Running mojo -I . tests/test_repeat.mojo"
        mojo -I . $DEBUG_MODE tests/test_repeat.mojo
        ;;

    mm2d)
        echo "Running mojo -I . tests/test_mm2d.mojo"
        mojo -I . $DEBUG_MODE tests/test_mm2d.mojo
        ;;

    slice)
        echo "Running mojo -I . tests/test_slice.mojo"
        mojo -I . $DEBUG_MODE tests/test_slice.mojo
        ;;
    tiles)
        echo "Running mojo -I . tests/test_tiles.mojo"
        mojo -I . $DEBUG_MODE tests/test_tiles.mojo
        ;;

    linspace)
        echo "Running mojo -I . tests/test_linspace.mojo"
        mojo -I . $DEBUG_MODE tests/test_linspace.mojo
        ;;

    argminmax)
        echo "Running mojo -I . tests/test_argminmax.mojo"
        mojo -I . $DEBUG_MODE tests/test_argminmax.mojo
        ;;
    relu)
        echo "Running mojo -I . tests/test_relu.mojo"
        mojo -I . $DEBUG_MODE tests/test_relu.mojo
        ;;

    shuffle)
        echo "Running mojo -I . tests/test_shuffle.mojo"
        mojo -I . $DEBUG_MODE tests/test_shuffle.mojo
        ;;


    permute)
        echo "Running mojo -I . tests/test_permute.mojo"
        mojo -I . $DEBUG_MODE tests/test_permute.mojo
        ;;

    flatten)
        echo "Running mojo -I . tests/test_flatten.mojo"
        mojo -I . $DEBUG_MODE tests/test_flatten.mojo
        ;;
    squeeze)
        echo "Running mojo -I . tests/test_squeeze.mojo"
        mojo -I . $DEBUG_MODE tests/test_squeeze.mojo
        ;;
    unsqueeze)
        echo "Running mojo -I . tests/test_unsqueeze.mojo"
        mojo -I . $DEBUG_MODE tests/test_unsqueeze.mojo
        ;;

    gradbox)
        echo "Running mojo -I . tests/test_gradbox.mojo"
        mojo -I . $DEBUG_MODE tests/test_gradbox.mojo
        ;;

    transpose)
        echo "Running mojo -I . tests/test_transpose.mojo"
        mojo -I . $DEBUG_MODE tests/test_transpose.mojo
        ;;
    ndb)
        echo "Running mojo -I . tests/test_ndb.mojo"
        mojo -I . $DEBUG_MODE tests/test_ndb.mojo
        ;;
    ce)
        echo "Running mojo -I . tests/test_cross_entropy.mojo"
        mojo -I . $DEBUG_MODE tests/test_cross_entropy.mojo
        ;;
    synth_smoke)
        echo "Running synthetic smoke tests"
        mojo -I . $DEBUG_MODE tests/test_synthetic_smoke.mojo
        ;;
    synth_mnist)
        echo "Running synthetic mnist tests"
        mojo -I . $DEBUG_MODE tests/test_synthetic_mnist.mojo
        ;;

    buffers)
        mojo -I . $DEBUG_MODE tests/test_buffers.mojo
        ;;

    bench)
	    echo "Running tensor multiplication benchmark(tests/test_matmul_bench.mojo)"
        mojo -I . $DEBUG_MODE tests/test_matmul_bench.mojo
        ;;
    views)
        echo "Running view test cases"
        mojo -I . $DEBUG_MODE tests/test_views.mojo
        ;;
    shapes)
        echo "Running mojo -I . tests/test_shapes.mojo"
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
        echo "Running mojo -I . tests/test_ancestry.mojo"
        mojo -I . $DEBUG_MODE tests/test_ancestry.mojo
        ;;

    intlist)
        echo "Running intlist test cases"
        mojo -I . $DEBUG_MODE tests/test_intlist.mojo
        ;;
    validators)
        echo "Running validators test cases"
        mojo -I . $DEBUG_MODE tests/test_validators.mojo
        ;;

    all)
        echo "Running tests/test_mm2d.mojo"
        mojo -I . tests/test_mm2d.mojo

        echo "Running tests/test_repeat.mojo"
        mojo -I . tests/test_repeat.mojo

        echo "Running tests/test_tiles.mojo"
        mojo -I . tests/test_tiles.mojo

        echo "Running tests/test_linspace.mojo"
        mojo -I . tests/test_linspace.mojo

        echo "Running tests/test_slice.mojo"
        mojo -I . tests/test_slice.mojo

        echo "Running tests/test_softmax.mojo"
        mojo -I . tests/test_softmax.mojo

        echo "Running tests/test_relu.mojo"
        mojo -I . tests/test_relu.mojo

        echo "Running tests/test_argminmax.mojo"
        mojo -I . tests/test_argminmax.mojo
        echo "Running tests/test_shuffle.mojo"
        mojo -I . tests/test_shuffle.mojo

        echo "Running tests/test_permute.mojo"
        mojo -I . tests/test_permute.mojo

        echo "Running tensor tests"
        mojo -I . tests/test_tensors.mojo
        echo "Running flatten tests"
        mojo -I . tests/test_flatten.mojo
        echo "Running squeeze tests"
        mojo -I . tests/test_squeeze.mojo
        echo "Running unsqueeze tests"
        mojo -I . tests/test_unsqueeze.mojo
        echo "Running buffer tests"
        mojo -I . tests/test_buffers.mojo
        echo "Running gradbox tests"
        mojo -I . tests/test_gradbox.mojo
        echo "Running ndbuffer tests"
        mojo -I . tests/test_ndb.mojo
        echo "Running tranpose tests"
        mojo -I . tests/test_transpose.mojo
        echo "Running synthetic mnist tests"
        mojo -I . tests/test_synthetic_mnist.mojo
        echo "Running synthetic smoke tests"
        mojo -I . tests/test_synthetic_smoke.mojo
        echo "Running crossentropy loss tests"
        mojo -I . tests/test_cross_entropy.mojo
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
        echo "Running validators test cases"
        mojo -I . tests/test_validators.mojo
        echo "Running tensor multiplication benchmark"
        mojo -I . tests/test_matmul_bench.mojo
        ;;
    *)
        echo "Error: Unknown test '$1'"
        echo "Available tests: mm2d, repeat, tiles, linspace, slice, relu, softmax permute, shuffle, argminmax, tensors, flatten, squeeze, unsqueeze, transpose, gradbox, ndb, buffers, views, shapes, intlist, strides, ancestry, shared, bench, validators, ce, synth_smoke, synth_mnist, all"
        exit 1
        ;;
esac
