#!/usr/bin/bash
clear
# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No test specified"
    echo "Usage: $0 [chunk|fill|pad|logarithm|stack|concat|std_variance|blas|dropout|indexhelper|utils|variance|tanh|losses|data|intarray|tensors|mmnd|mm2d|mv|vm|argminmax|minmax|repeat|tiles|slice|linspace|softmax|relu|shuffle|buffers|flatten|permute|squeeze|unsqueeze|views|gradbox|ndb|transpose|shapes|strides|ancestry|bench|validators|ce|synth_smoke|synth_mnist|shapebroadcast|all]"
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
    fill)
        echo "Running mojo -I . tests/test_fill.mojo"
        mojo -I . $DEBUG_MODE tests/test_fill.mojo
        ;;
    chunk)
        echo "Running mojo -I . tests/test_chunk.mojo"
        mojo -I . $DEBUG_MODE tests/test_chunk.mojo
        ;;

    pad)
        echo "Running mojo -I . tests/test_pad.mojo"
        mojo -I . $DEBUG_MODE tests/test_pad.mojo
        ;;
    blas)
        echo "Running mojo -I . tests/test_blas.mojo"
        mojo -I . $DEBUG_MODE tests/test_blas.mojo
        ;;
    dropout)
        echo "Running mojo -I . tests/test_dropout.mojo"
        mojo -I . $DEBUG_MODE tests/test_dropout.mojo
        ;;
    std_variance)
        echo "Running mojo -I . tests/test_std_variance.mojo"
        mojo -I . $DEBUG_MODE tests/test_std_variance.mojo
        ;;
    stack)
        echo "Running mojo -I . tests/test_stack.mojo"
        mojo -I . $DEBUG_MODE tests/test_stack.mojo
        ;;
    logarithm)
        echo "Running mojo -I . tests/test_logarithm.mojo"
        mojo -I . $DEBUG_MODE tests/test_logarithm.mojo
        ;;

    concat)
        echo "Running mojo -I . tests/test_concat.mojo"
        mojo -I . $DEBUG_MODE tests/test_concat.mojo
        ;;

    variance)
        echo "Running mojo -I . tests/test_variance.mojo"
        mojo -I . $DEBUG_MODE tests/test_variance.mojo
        ;;
    utils)
        echo "Running mojo -I . tests/test_utils.mojo"
        mojo -I . $DEBUG_MODE tests/test_utils.mojo
        ;;
    indexhelper)
        echo "Running mojo -I . tests/test_indexhelper.mojo"
        mojo -I . $DEBUG_MODE tests/test_indexhelper.mojo
        ;;

    losses)
        echo "Running mojo -I . tests/test_losses.mojo"
        mojo -I . $DEBUG_MODE tests/test_losses.mojo
        ;;

    tanh)
        echo "Running mojo -I . tests/test_tanh.mojo"
        mojo -I . $DEBUG_MODE tests/test_tanh.mojo
        ;;

    data)
        echo "Running mojo -I . tests/test_data.mojo"
        mojo -I . $DEBUG_MODE tests/test_data.mojo
        ;;

    softmax)
        echo "Running mojo -I . tests/test_softmax.mojo"
        mojo -I . $DEBUG_MODE tests/test_softmax.mojo
        ;;

    repeat)
        echo "Running mojo -I . tests/test_repeat.mojo"
        mojo -I . $DEBUG_MODE tests/test_repeat.mojo
        ;;

    mmnd)
        echo "Running mojo -I . tests/test_mmnd.mojo"
        mojo -I . $DEBUG_MODE tests/test_mmnd.mojo
        ;;
    intarray)
        echo "Running mojo -I . tests/test_intarray.mojo"
        mojo -I . $DEBUG_MODE tests/test_intarray.mojo
        ;;

    mm2d)
        echo "Running mojo -I . tests/test_mm2d.mojo"
        mojo -I . $DEBUG_MODE tests/test_mm2d.mojo
        ;;
    vm)
        echo "Running mojo -I . tests/test_vm.mojo"
        mojo -I . $DEBUG_MODE tests/test_vm.mojo
        ;;
    mv)
        echo "Running mojo -I . tests/test_mv.mojo"
        mojo -I . $DEBUG_MODE tests/test_mv.mojo
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

    minmax)
        echo "Running mojo -I . tests/test_minmax.mojo"
        mojo -I . $DEBUG_MODE tests/test_minmax.mojo
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

    shapebroadcast)
        echo "Running tests/test_broadcaster.mojo"
        mojo -I . $DEBUG_MODE tests/test_broadcaster.mojo
        ;;

    strides)
        echo "Running strides test cases"
        mojo -I . $DEBUG_MODE tests/test_strides.mojo
        ;;

    ancestry)
        echo "Running mojo -I . tests/test_ancestry.mojo"
        mojo -I . $DEBUG_MODE tests/test_ancestry.mojo
        ;;

    validators)
        echo "Running validators test cases"
        mojo -I . $DEBUG_MODE tests/test_validators.mojo
        ;;

    all)
        echo "Running tests/test_chunk.mojo"
        mojo -I . tests/test_chunk.mojo

        echo "Running tests/test_fill.mojo"
        mojo -I . tests/test_fill.mojo

        echo "Running tests/test_pad.mojo"
        mojo -I . tests/test_pad.mojo

        echo "Running tests/test_logarithm.mojo"
        mojo -I . tests/test_logarithm.mojo

        echo "Running tests/test_stack.mojo"
        mojo -I . tests/test_stack.mojo

        echo "Running tests/test_concat.mojo"
        mojo -I . tests/test_concat.mojo

        echo "Running tests/test_std_variance.mojo"
        mojo -I . tests/test_std_variance.mojo

        echo "Running tests/test_dropout.mojo"
        mojo -I . tests/test_dropout.mojo

        echo "Running tests/test_blas.mojo"
        mojo -I . tests/test_blas.mojo

        echo "Running tests/test_indexhelper.mojo"
        mojo -I . tests/test_indexhelper.mojo

        echo "Running tests/test_utils.mojo"
        mojo -I . tests/test_utils.mojo

        echo "Running tests/test_variance.mojo"
        mojo -I . tests/test_variance.mojo

        echo "Running tests/test_tanh.mojo"
        mojo -I . tests/test_tanh.mojo

        echo "Running tests/test_losses.mojo"
        mojo -I . tests/test_losses.mojo

        echo "Running tests/test_data.mojo"
        mojo -I . tests/test_data.mojo

        echo "Running tests/test_intarray.mojo"
        mojo -I . tests/test_intarray.mojo

        echo "Running tests/test_minmax.mojo"
        mojo -I . tests/test_minmax.mojo

        echo "Running tests/test_broadcaster.mojo"
        mojo -I . tests/test_broadcaster.mojo

        echo "Running tests/test_mv.mojo"
        mojo -I . tests/test_mv.mojo

        echo "Running tests/test_vm.mojo"
        mojo -I . tests/test_vm.mojo

        echo "Running tests/test_mmnd.mojo"
        mojo -I . tests/test_mmnd.mojo

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
        echo "Running validators test cases"
        mojo -I . tests/test_validators.mojo
        echo "Running tensor multiplication benchmark"
        mojo -I . tests/test_matmul_bench.mojo
        ;;
    *)
        echo "Error: Unknown test '$1'"
        echo "Available tests: chunk, fill, pad, logarithm, stack, concat, std_variance, dropout, blas, indexhelper, utils, variance, tanh, losses, data, intarray, mmnd, mm2d, vm, mv, repeat, tiles, linspace, slice, relu, softmax permute, shuffle, argminmax, minmax, tensors, flatten, squeeze, unsqueeze, transpose, gradbox, ndb, buffers, views, shapes, strides, ancestry, shapebroadcast, bench, validators, ce, synth_smoke, synth_mnist, all"
        exit 1
        ;;
esac
