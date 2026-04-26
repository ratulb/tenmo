#!/usr/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Function to print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to run a single test with timing
run_test() {
    local test_name=$1
    local test_file=$2
    local debug_mode=$3
    local log_file="$LOG_DIR/${test_name}.log"

    print_colored "$CYAN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    print_colored "$BOLD" "Running: $test_name"
    print_colored "$CYAN" "File: $test_file"
    echo "────────────────────────────────────────"

    local start_time=$(date +%s%N)

    if [ -n "$debug_mode" ]; then
        mojo -I . $debug_mode "$test_file" 2>&1 | tee "$log_file"
    else
        mojo -I . "$test_file" 2>&1 | tee "$log_file"
    fi

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s%N)
    local duration=$(( (end_time - start_time) / 1000000 )) # milliseconds

    if [ $exit_code -eq 0 ]; then
        print_colored "$GREEN" "✓ PASSED: $test_name (${duration}ms)"
        return 0
    else
        print_colored "$RED" "✗ FAILED: $test_name (${duration}ms)"
        print_colored "$YELLOW" "  Log saved to: $log_file"
        return 1
    fi
}

# Function to run tests in parallel
run_parallel() {
    local tests=("$@")
    local pids=()
    local results=()

    for test in "${tests[@]}"; do
        IFS='|' read -r name file <<< "$test"
        run_test "$name" "$file" "$DEBUG_MODE" &
        pids+=($!)
    done

    local failed=0
    for i in "${!pids[@]}"; do
        wait ${pids[$i]}
        if [ $? -ne 0 ]; then
            failed=$((failed + 1))
        fi
    done

    return $failed
}

# Clear screen
clear

# Print header
print_colored "$MAGENTA" "╔══════════════════════════════════════════════════════════════╗"
print_colored "$MAGENTA" "║                    TENMO TEST SUITE                          ║"
print_colored "$MAGENTA" "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if an argument was provided
if [ $# -eq 0 ]; then
    print_colored "$RED" "Error: No test specified"
    echo ""
    print_colored "$YELLOW" "Usage: $0 [OPTIONS] <test_name>"
    echo ""
    print_colored "$CYAN" "Options:"
    echo "  -p, --parallel    Run tests in parallel (for 'all' mode)"
    echo "  -d, --debug       Enable debug mode (-D LOGGING_LEVEL=debug)"
    echo ""
    print_colored "$CYAN" "Available tests:"
    echo "  product, unary, sqrt, tensors, gpu, item, contiguous, maxmin_scalar"
    echo "  allany, compare, count_unique, transmute, exp, summean, sigmoid"
    echo "  gpusummean, broadcast, scalar, inplace, expand, gpu_expand"
    echo "  sgd, npiop, fill, chunk, cnn, matmul, pad, blas, dropout"
    echo "  std_variance, stack, logarithm, concat, variance, utils, onehot, power"
    echo "  indexhelper, losses, tanh, data, softmax, repeat, mmnd"
    echo "  intarray, mm2d, vm, mv, slice, tiles, linspace, argminmax"
    echo "  minmax, relu, shuffle, permute, flatten, squeeze, unsqueeze"
    echo "  gradbox, ndb, transpose, buffers, views, shapes, strides"
    echo "  shapebroadcast, bench, validators, ce, synth_mnist"
    echo ""
    print_colored "$GREEN" "  all              Run all tests"
    print_colored "$GREEN" "  quick            Run quick sanity tests"
    exit 1
fi

# Parse arguments
DEBUG_MODE=""
PARALLEL=false
TEST_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            DEBUG_MODE="-D LOGGING_LEVEL=debug"
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        *)
            TEST_NAME=$1
            shift
            ;;
    esac
done

if [ -z "$TEST_NAME" ]; then
    print_colored "$RED" "Error: No test specified"
    exit 1
fi

# Record start time
SCRIPT_START=$(date +%s%N)
FAILED_TESTS=()
PASSED_TESTS=()

# Define test mapping (using case statement as before, but with timing)
run_single_test() {
    local test=$1
    case $test in
        product)        run_test "product" "tests/test_product_reduction.mojo" "$DEBUG_MODE" ;;
        unary)          run_test "unary" "tests/test_unary_ops.mojo" "$DEBUG_MODE" ;;
        sqrt)           run_test "sqrt" "tests/test_sqrt.mojo" "$DEBUG_MODE" ;;
        tensors)        run_test "tensors" "tests/test_tensors.mojo" "$DEBUG_MODE" ;;
        gpu)            run_test "gpu" "tests/test_gpu.mojo" "$DEBUG_MODE" ;;
        item)           run_test "item" "tests/test_item.mojo" "$DEBUG_MODE" ;;
        contiguous)     run_test "contiguous" "tests/test_contiguous.mojo" "$DEBUG_MODE" ;;
        maxmin_scalar)  run_test "maxmin_scalar" "tests/test_maxmin_scalar.mojo" "$DEBUG_MODE" ;;
        onehot)         run_test "onehot" "tests/test_onehot.mojo" "$DEBUG_MODE" ;;
        power)          run_test "power" "tests/test_exponentiator.mojo" "$DEBUG_MODE" ;;
        allany)         run_test "allany" "tests/test_all_true_any_true.mojo" "$DEBUG_MODE" ;;
        compare)        run_test "compare" "tests/test_compare.mojo" "$DEBUG_MODE" ;;
        count_unique)   run_test "count_unique" "tests/test_count_unique.mojo" "$DEBUG_MODE" ;;
        transmute)      run_test "transmute" "tests/test_transmutation.mojo" "$DEBUG_MODE" ;;
        exp)            run_test "exp" "tests/test_exponential.mojo" "$DEBUG_MODE" ;;
        summean)        run_test "summean" "tests/test_sum_mean.mojo" "$DEBUG_MODE" ;;
        sigmoid)        run_test "sigmoid" "tests/test_sigmoid.mojo" "$DEBUG_MODE" ;;
        gpusummean)     run_test "gpusummean" "tests/test_gpu_sum_mean.mojo" "$DEBUG_MODE" ;;
        broadcast)      run_test "broadcast" "tests/test_broadcast.mojo" "$DEBUG_MODE" ;;
        scalar)         run_test "scalar" "tests/test_scalar_tensors.mojo" "$DEBUG_MODE" ;;
        inplace)        run_test "inplace" "tests/test_inplace.mojo" "$DEBUG_MODE" ;;
        expand)         run_test "expand" "tests/test_expand.mojo" "$DEBUG_MODE" ;;
        gpu_expand)     run_test "gpu_expand" "tests/test_gpu_expand.mojo" "$DEBUG_MODE" ;;
        sgd)            run_test "sgd" "tests/test_sgd.mojo" "$DEBUG_MODE" ;;
        npiop)          run_test "npiop" "tests/test_numpy_interop.mojo" "$DEBUG_MODE" ;;
        fill)           run_test "fill" "tests/test_fill.mojo" "$DEBUG_MODE" ;;
        chunk)          run_test "chunk" "tests/test_chunk.mojo" "$DEBUG_MODE" ;;
        cnn)            run_test "cnn" "tests/test_cnn.mojo" "$DEBUG_MODE" ;;
        matmul)         run_test "matmul" "tests/test_matmul.mojo" "$DEBUG_MODE" ;;
        pad)            run_test "pad" "tests/test_pad.mojo" "$DEBUG_MODE" ;;
        blas)           run_test "blas" "tests/test_blas.mojo" "$DEBUG_MODE" ;;
        dropout)        run_test "dropout" "tests/test_dropout.mojo" "$DEBUG_MODE" ;;
        std_variance)   run_test "std_variance" "tests/test_std_variance.mojo" "$DEBUG_MODE" ;;
        stack)          run_test "stack" "tests/test_stack.mojo" "$DEBUG_MODE" ;;
        logarithm)      run_test "logarithm" "tests/test_logarithm.mojo" "$DEBUG_MODE" ;;
        concat)         run_test "concat" "tests/test_concat.mojo" "$DEBUG_MODE" ;;
        variance)       run_test "variance" "tests/test_variance.mojo" "$DEBUG_MODE" ;;
        utils)          run_test "utils" "tests/test_utils.mojo" "$DEBUG_MODE" ;;
        indexhelper)    run_test "indexhelper" "tests/test_indexhelper.mojo" "$DEBUG_MODE" ;;
        losses)         run_test "losses" "tests/test_losses.mojo" "$DEBUG_MODE" ;;
        tanh)           run_test "tanh" "tests/test_tanh.mojo" "$DEBUG_MODE" ;;
        data)           run_test "data" "tests/test_data.mojo" "$DEBUG_MODE" ;;
        softmax)        run_test "softmax" "tests/test_softmax.mojo" "$DEBUG_MODE" ;;
        repeat)         run_test "repeat" "tests/test_repeat.mojo" "$DEBUG_MODE" ;;
        mmnd)           run_test "mmnd" "tests/test_mmnd.mojo" "$DEBUG_MODE" ;;
        intarray)       run_test "intarray" "tests/test_intarray.mojo" "$DEBUG_MODE" ;;
        mm2d)           run_test "mm2d" "tests/test_mm2d.mojo" "$DEBUG_MODE" ;;
        vm)             run_test "vm" "tests/test_vm.mojo" "$DEBUG_MODE" ;;
        mv)             run_test "mv" "tests/test_mv.mojo" "$DEBUG_MODE" ;;
        slice)          run_test "slice" "tests/test_slice.mojo" "$DEBUG_MODE" ;;
        tiles)          run_test "tiles" "tests/test_tiles.mojo" "$DEBUG_MODE" ;;
        linspace)       run_test "linspace" "tests/test_linspace.mojo" "$DEBUG_MODE" ;;
        argminmax)      run_test "argminmax" "tests/test_argminmax.mojo" "$DEBUG_MODE" ;;
        minmax)         run_test "minmax" "tests/test_minmax.mojo" "$DEBUG_MODE" ;;
        relu)           run_test "relu" "tests/test_relu.mojo" "$DEBUG_MODE" ;;
        shuffle)        run_test "shuffle" "tests/test_shuffle.mojo" "$DEBUG_MODE" ;;
        permute)        run_test "permute" "tests/test_permute.mojo" "$DEBUG_MODE" ;;
        flatten)        run_test "flatten" "tests/test_flatten.mojo" "$DEBUG_MODE" ;;
        squeeze)        run_test "squeeze" "tests/test_squeeze.mojo" "$DEBUG_MODE" ;;
        unsqueeze)      run_test "unsqueeze" "tests/test_unsqueeze.mojo" "$DEBUG_MODE" ;;
        gradbox)        run_test "gradbox" "tests/test_gradbox.mojo" "$DEBUG_MODE" ;;
        ndb)            run_test "ndb" "tests/test_ndb.mojo" "$DEBUG_MODE" ;;
        transpose)      run_test "transpose" "tests/test_transpose.mojo" "$DEBUG_MODE" ;;
        buffers)        run_test "buffers" "tests/test_buffers.mojo" "$DEBUG_MODE" ;;
        views)          run_test "views" "tests/test_views.mojo" "$DEBUG_MODE" ;;
        shapes)         run_test "shapes" "tests/test_shapes.mojo" "$DEBUG_MODE" ;;
        strides)        run_test "strides" "tests/test_strides.mojo" "$DEBUG_MODE" ;;
        shapebroadcast) run_test "shapebroadcast" "tests/test_broadcaster.mojo" "$DEBUG_MODE" ;;
        bench)          run_test "bench" "tests/test_matmul_bench.mojo" "$DEBUG_MODE" ;;
        validators)     run_test "validators" "tests/test_validators.mojo" "$DEBUG_MODE" ;;
        ce)             run_test "ce" "tests/test_cross_entropy.mojo" "$DEBUG_MODE" ;;
        synth_mnist)    run_test "synth_mnist" "tests/test_synthetic_mnist.mojo" "$DEBUG_MODE" ;;
        quick)
            print_colored "$BLUE" "Running quick sanity tests..."
            run_test "tensors" "tests/test_tensors.mojo" "$DEBUG_MODE"
            run_test "shapes" "tests/test_shapes.mojo" "$DEBUG_MODE"
            run_test "strides" "tests/test_strides.mojo" "$DEBUG_MODE"
            run_test "summean" "tests/test_sum_mean.mojo" "$DEBUG_MODE"
            ;;
        all)
            print_colored "$BLUE" "Running ALL tests..."
            local all_tests=(
                "product|tests/test_product_reduction.mojo"
                "unary|tests/test_unary_ops.mojo"
                "sqrt|tests/test_sqrt.mojo"
                "tensors|tests/test_tensors.mojo"
                "gpu|tests/test_gpu.mojo"
                "item|tests/test_item.mojo"
                "contiguous|tests/test_contiguous.mojo"
                "maxmin_scalar|tests/test_maxmin_scalar.mojo"
                "onehot|tests/test_onehot.mojo"
                "power|tests/test_exponentiator.mojo"
                "allany|tests/test_all_true_any_true.mojo"
                "compare|tests/test_compare.mojo"
                "count_unique|tests/test_count_unique.mojo"
                "transmute|tests/test_transmutation.mojo"
                "exp|tests/test_exponential.mojo"
                "summean|tests/test_sum_mean.mojo"
                "sigmoid|tests/test_sigmoid.mojo"
                "gpusummean|tests/test_gpu_sum_mean.mojo"
                "broadcast|tests/test_broadcast.mojo"
                "scalar|tests/test_scalar_tensors.mojo"
                "inplace|tests/test_inplace.mojo"
                "expand|tests/test_expand.mojo"
                "gpu_expand|tests/test_gpu_expand.mojo"
                "sgd|tests/test_sgd.mojo"
                "npiop|tests/test_numpy_interop.mojo"
                "fill|tests/test_fill.mojo"
                "chunk|tests/test_chunk.mojo"
                "cnn|tests/test_cnn.mojo"
                "matmul|tests/test_matmul.mojo"
                "pad|tests/test_pad.mojo"
                "blas|tests/test_blas.mojo"
                "dropout|tests/test_dropout.mojo"
                "std_variance|tests/test_std_variance.mojo"
                "stack|tests/test_stack.mojo"
                "logarithm|tests/test_logarithm.mojo"
                "concat|tests/test_concat.mojo"
                "variance|tests/test_variance.mojo"
                "utils|tests/test_utils.mojo"
                "indexhelper|tests/test_indexhelper.mojo"
                "losses|tests/test_losses.mojo"
                "tanh|tests/test_tanh.mojo"
                "data|tests/test_data.mojo"
                "softmax|tests/test_softmax.mojo"
                "repeat|tests/test_repeat.mojo"
                "mmnd|tests/test_mmnd.mojo"
                "intarray|tests/test_intarray.mojo"
                "mm2d|tests/test_mm2d.mojo"
                "vm|tests/test_vm.mojo"
                "mv|tests/test_mv.mojo"
                "slice|tests/test_slice.mojo"
                "tiles|tests/test_tiles.mojo"
                "linspace|tests/test_linspace.mojo"
                "argminmax|tests/test_argminmax.mojo"
                "minmax|tests/test_minmax.mojo"
                "relu|tests/test_relu.mojo"
                "shuffle|tests/test_shuffle.mojo"
                "permute|tests/test_permute.mojo"
                "flatten|tests/test_flatten.mojo"
                "squeeze|tests/test_squeeze.mojo"
                "unsqueeze|tests/test_unsqueeze.mojo"
                "gradbox|tests/test_gradbox.mojo"
                "ndb|tests/test_ndb.mojo"
                "transpose|tests/test_transpose.mojo"
                "buffers|tests/test_buffers.mojo"
                "views|tests/test_views.mojo"
                "shapes|tests/test_shapes.mojo"
                "strides|tests/test_strides.mojo"
                "shapebroadcast|tests/test_broadcaster.mojo"
                "validators|tests/test_validators.mojo"
                "ce|tests/test_cross_entropy.mojo"
            )

            if [ "$PARALLEL" = true ]; then
                run_parallel "${all_tests[@]}"
            else
                for test in "${all_tests[@]}"; do
                    IFS='|' read -r name file <<< "$test"
                    if run_test "$name" "$file" "$DEBUG_MODE"; then
                        PASSED_TESTS+=("$name")
                    else
                        FAILED_TESTS+=("$name")
                    fi
                done
            fi
            ;;
        *)
            print_colored "$RED" "Error: Unknown test '$TEST_NAME'"
            exit 1
            ;;
    esac
}

# Run the test
run_single_test "$TEST_NAME"

# Calculate total time
SCRIPT_END=$(date +%s%N)
TOTAL_DURATION=$(( (SCRIPT_END - SCRIPT_START) / 1000000 )) # milliseconds

# Print summary
echo ""
print_colored "$MAGENTA" "═══════════════════════════════════════════════════════════════"
print_colored "$BOLD" "Test Summary"
print_colored "$MAGENTA" "═══════════════════════════════════════════════════════════════"

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    print_colored "$GREEN" "✓ All tests passed!"
else
    print_colored "$RED" "✗ Failed tests: ${#FAILED_TESTS[@]}"
    for test in "${FAILED_TESTS[@]}"; do
        print_colored "$RED" "  - $test"
    done
fi

print_colored "$CYAN" "Total execution time: ${TOTAL_DURATION}ms"
print_colored "$CYAN" "Logs saved to: $LOG_DIR"

# Exit with appropriate code
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    exit 0
else
    exit 1
fi
