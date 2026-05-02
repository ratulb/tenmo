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

# Define the complete ordered list of tests
declare -a ALL_TESTS_IN_ORDER=(
    "product|tests/test_product_reduction.mojo"
    "unary|tests/test_unary_ops.mojo"
    "sqrt|tests/test_sqrt.mojo"
    "attn_matmul|tests/test_attn_matmul.mojo"
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
    "dev_transfer|tests/test_device_transfer_gradflow.mojo"
    "std_variance|tests/test_std_variance.mojo"
    "stack|tests/test_stack.mojo"
    "logarithm|tests/test_logarithm.mojo"
    "concat|tests/test_concat.mojo"
    "variance|tests/test_variance.mojo"
    "variance_and_std|tests/test_variance_and_std.mojo"
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
#    "synth_mnist|tests/test_synthetic_mnist.mojo"
)

declare -a GPU_TESTS=(
    "product|tests/test_product_reduction.mojo"
    "unary|tests/test_unary_ops.mojo"
    "sqrt|tests/test_sqrt.mojo"
    "attn_matmul|tests/test_attn_matmul.mojo"
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
    "gpu_expand|tests/test_gpu_expand.mojo"
    "sgd|tests/test_sgd.mojo"
    "dropout|tests/test_dropout.mojo"
    "dev_transfer|tests/test_device_transfer_gradflow.mojo"
    "logarithm|tests/test_logarithm.mojo"
    "tanh|tests/test_tanh.mojo"
    "softmax|tests/test_softmax.mojo"
    "argminmax|tests/test_argminmax.mojo"
    "minmax|tests/test_minmax.mojo"
    "relu|tests/test_relu.mojo"
    "shuffle|tests/test_shuffle.mojo"
    "permute|tests/test_permute.mojo"
    "flatten|tests/test_flatten.mojo"
    "squeeze|tests/test_squeeze.mojo"
    "ndb|tests/test_ndb.mojo"
    "transpose|tests/test_transpose.mojo"
    "variance_and_std|tests/test_variance_and_std.mojo"
    "tiles|tests/test_tiles.mojo"
    "ce|tests/test_cross_entropy.mojo"
)

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
    print_colored "$YELLOW" "Usage: $0 [OPTIONS] <test_name1> [test_name2 ...]"
    echo "       $0 [OPTIONS] from <test_name>"
    echo "       $0 [OPTIONS] gpu [from <test_name> | test1 test2 ...]"
    echo ""
    print_colored "$CYAN" "Options:"
    echo "  -p, --parallel    Run tests in parallel (for 'all' or 'gpu' mode)"
    echo "  -d, --debug       Enable debug mode (-D LOGGING_LEVEL=debug)"
    echo ""
    print_colored "$CYAN" "Examples:"
    echo "  $0 softmax matmul tensors     - Run only softmax, matmul, and tensors"
    echo "  $0 from softmax               - Run softmax and all tests after it"
    echo "  $0 gpu                        - Run all GPU-guarded tests"
    echo "  $0 gpu from relu              - Run relu and all GPU tests after it"
    echo "  $0 gpu relu tanh              - Run only relu and tanh"
    echo ""
    print_colored "$CYAN" "Available tests:"
    echo "  product, unary, sqrt, tensors, gpu, item, contiguous, maxmin_scalar"
    echo "  allany, compare, count_unique, transmute, exp, summean, sigmoid"
    echo "  gpusummean, broadcast, scalar, inplace, expand, gpu_expand"
    echo "  sgd, npiop, fill, chunk, cnn, matmul, pad, blas, dropout, dev_transfer"
    echo "  std_variance, stack, logarithm, concat, variance, variance_and_std, utils, onehot, power"
    echo "  indexhelper, losses, tanh, data, softmax, repeat, mmnd, attn_matmul"
    echo "  intarray, mm2d, vm, mv, slice, tiles, linspace, argminmax"
    echo "  minmax, relu, shuffle, permute, flatten, squeeze, unsqueeze"
    echo "  gradbox, ndb, transpose, buffers, views, shapes, strides"
    echo "  shapebroadcast, validators, ce, synth_mnist"
    echo ""
    print_colored "$GREEN" "  all              Run all tests"
    print_colored "$GREEN" "  gpu              Run all GPU-guarded tests"
    print_colored "$GREEN" "  quick            Run quick sanity tests"
    exit 1
fi

# Parse arguments
DEBUG_MODE=""
PARALLEL=false
FROM_MODE=false
START_TEST=""
declare -a SPECIFIC_TESTS=()

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
        from)
            FROM_MODE=true
            shift
            ;;
        *)
            if [ "$FROM_MODE" = true ]; then
                START_TEST=$1
                shift
            else
                SPECIFIC_TESTS+=("$1")
                shift
            fi
            ;;
    esac
done

# Record start time
SCRIPT_START=$(date +%s%N)
FAILED_TESTS=()
PASSED_TESTS=()

# Function to run a test by name
run_test_by_name() {
    local test_name=$1
    local exit_code=0

    case $test_name in
        product)        run_test "product" "tests/test_product_reduction.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        unary)          run_test "unary" "tests/test_unary_ops.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        sqrt)           run_test "sqrt" "tests/test_sqrt.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        attn_matmul)    run_test "attn_matmul" "tests/test_attn_matmul.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        tensors)        run_test "tensors" "tests/test_tensors.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        gpu)
            print_colored "$BLUE" "Running all GPU-guarded tests..."
            if [ "$PARALLEL" = true ]; then
                run_parallel "${GPU_TESTS[@]}"
                exit_code=$?
            else
                for test in "${GPU_TESTS[@]}"; do
                    IFS='|' read -r name file <<< "$test"
                    if run_test "$name" "$file" "$DEBUG_MODE"; then
                        PASSED_TESTS+=("$name")
                    else
                        FAILED_TESTS+=("$name")
                        exit_code=1
                    fi
                done
            fi
            ;;
        item)           run_test "item" "tests/test_item.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        contiguous)     run_test "contiguous" "tests/test_contiguous.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        maxmin_scalar)  run_test "maxmin_scalar" "tests/test_maxmin_scalar.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        onehot)         run_test "onehot" "tests/test_onehot.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        power)          run_test "power" "tests/test_exponentiator.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        allany)         run_test "allany" "tests/test_all_true_any_true.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        compare)        run_test "compare" "tests/test_compare.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        count_unique)   run_test "count_unique" "tests/test_count_unique.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        transmute)      run_test "transmute" "tests/test_transmutation.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        exp)            run_test "exp" "tests/test_exponential.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        summean)        run_test "summean" "tests/test_sum_mean.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        sigmoid)        run_test "sigmoid" "tests/test_sigmoid.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        gpusummean)     run_test "gpusummean" "tests/test_gpu_sum_mean.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        broadcast)      run_test "broadcast" "tests/test_broadcast.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        scalar)         run_test "scalar" "tests/test_scalar_tensors.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        inplace)        run_test "inplace" "tests/test_inplace.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        expand)         run_test "expand" "tests/test_expand.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        gpu_expand)     run_test "gpu_expand" "tests/test_gpu_expand.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        sgd)            run_test "sgd" "tests/test_sgd.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        npiop)          run_test "npiop" "tests/test_numpy_interop.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        fill)           run_test "fill" "tests/test_fill.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        chunk)          run_test "chunk" "tests/test_chunk.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        cnn)            run_test "cnn" "tests/test_cnn.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        matmul)         run_test "matmul" "tests/test_matmul.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        pad)            run_test "pad" "tests/test_pad.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        blas)           run_test "blas" "tests/test_blas.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        dropout)        run_test "dropout" "tests/test_dropout.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        dev_transfer)   run_test "dev_transfer" "tests/test_device_transfer_gradflow.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        std_variance)   run_test "std_variance" "tests/test_std_variance.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        stack)          run_test "stack" "tests/test_stack.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        logarithm)      run_test "logarithm" "tests/test_logarithm.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        concat)         run_test "concat" "tests/test_concat.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        variance)       run_test "variance" "tests/test_variance.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        variance_and_std) run_test "variance_and_std" "tests/test_variance_and_std.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        utils)          run_test "utils" "tests/test_utils.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        indexhelper)    run_test "indexhelper" "tests/test_indexhelper.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        losses)         run_test "losses" "tests/test_losses.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        tanh)           run_test "tanh" "tests/test_tanh.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        data)           run_test "data" "tests/test_data.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        softmax)        run_test "softmax" "tests/test_softmax.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        repeat)         run_test "repeat" "tests/test_repeat.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        mmnd)           run_test "mmnd" "tests/test_mmnd.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        intarray)       run_test "intarray" "tests/test_intarray.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        mm2d)           run_test "mm2d" "tests/test_mm2d.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        vm)             run_test "vm" "tests/test_vm.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        mv)             run_test "mv" "tests/test_mv.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        slice)          run_test "slice" "tests/test_slice.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        tiles)          run_test "tiles" "tests/test_tiles.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        linspace)       run_test "linspace" "tests/test_linspace.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        argminmax)      run_test "argminmax" "tests/test_argminmax.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        minmax)         run_test "minmax" "tests/test_minmax.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        relu)           run_test "relu" "tests/test_relu.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        shuffle)        run_test "shuffle" "tests/test_shuffle.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        permute)        run_test "permute" "tests/test_permute.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        flatten)        run_test "flatten" "tests/test_flatten.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        squeeze)        run_test "squeeze" "tests/test_squeeze.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        unsqueeze)      run_test "unsqueeze" "tests/test_unsqueeze.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        gradbox)        run_test "gradbox" "tests/test_gradbox.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        ndb)            run_test "ndb" "tests/test_ndb.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        transpose)      run_test "transpose" "tests/test_transpose.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        buffers)        run_test "buffers" "tests/test_buffers.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        views)          run_test "views" "tests/test_views.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        shapes)         run_test "shapes" "tests/test_shapes.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        strides)        run_test "strides" "tests/test_strides.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        shapebroadcast) run_test "shapebroadcast" "tests/test_broadcaster.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        validators)     run_test "validators" "tests/test_validators.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        ce)             run_test "ce" "tests/test_cross_entropy.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        synth_mnist)    run_test "synth_mnist" "tests/test_synthetic_mnist.mojo" "$DEBUG_MODE"; exit_code=$? ;;
        quick)
            print_colored "$BLUE" "Running quick sanity tests..."
            run_test "tensors" "tests/test_tensors.mojo" "$DEBUG_MODE"; exit_code=$?
            [ $exit_code -eq 0 ] && run_test "shapes" "tests/test_shapes.mojo" "$DEBUG_MODE"; exit_code=$?
            [ $exit_code -eq 0 ] && run_test "strides" "tests/test_strides.mojo" "$DEBUG_MODE"; exit_code=$?
            [ $exit_code -eq 0 ] && run_test "summean" "tests/test_sum_mean.mojo" "$DEBUG_MODE"; exit_code=$?
            ;;
        all)
            print_colored "$BLUE" "Running ALL tests..."
            if [ "$PARALLEL" = true ]; then
                run_parallel "${ALL_TESTS_IN_ORDER[@]}"
                exit_code=$?
            else
                for test in "${ALL_TESTS_IN_ORDER[@]}"; do
                    IFS='|' read -r name file <<< "$test"
                    if run_test "$name" "$file" "$DEBUG_MODE"; then
                        PASSED_TESTS+=("$name")
                    else
                        FAILED_TESTS+=("$name")
                        exit_code=1
                    fi
                done
            fi
            ;;
        *)
            print_colored "$RED" "Error: Unknown test '$test_name'"
            return 1
            ;;
    esac

    return $exit_code
}

# Function to run tests from a starting point
run_from_test() {
    local start_test=$1
    local found=false
    local exit_code=0

    print_colored "$BLUE" "Running from test '$start_test' and all tests after it..."
    echo ""

    for test_entry in "${ALL_TESTS_IN_ORDER[@]}"; do
        IFS='|' read -r name file <<< "$test_entry"

        if [ "$found" = true ]; then
            # Run this test
            if run_test "$name" "$file" "$DEBUG_MODE"; then
                PASSED_TESTS+=("$name")
            else
                FAILED_TESTS+=("$name")
                exit_code=1
            fi
        elif [ "$name" = "$start_test" ]; then
            # Found the starting test, run it
            found=true
            if run_test "$name" "$file" "$DEBUG_MODE"; then
                PASSED_TESTS+=("$name")
            else
                FAILED_TESTS+=("$name")
                exit_code=1
            fi
        fi
    done

    if [ "$found" = false ]; then
        print_colored "$RED" "Error: Test '$start_test' not found in the test list"
        return 1
    fi

    return $exit_code
}

# Function to run GPU tests from a starting point
run_from_gpu_test() {
    local start_test=$1
    local found=false
    local exit_code=0

    print_colored "$BLUE" "Running GPU tests from '$start_test' and all after it..."
    echo ""

    for test_entry in "${GPU_TESTS[@]}"; do
        IFS='|' read -r name file <<< "$test_entry"

        if [ "$found" = true ]; then
            if run_test "$name" "$file" "$DEBUG_MODE"; then
                PASSED_TESTS+=("$name")
            else
                FAILED_TESTS+=("$name")
                exit_code=1
            fi
        elif [ "$name" = "$start_test" ]; then
            found=true
            if run_test "$name" "$file" "$DEBUG_MODE"; then
                PASSED_TESTS+=("$name")
            else
                FAILED_TESTS+=("$name")
                exit_code=1
            fi
        fi
    done

    if [ "$found" = false ]; then
        print_colored "$RED" "Error: GPU test '$start_test' not found in GPU test list"
        return 1
    fi

    return $exit_code
}

# Main execution logic
if [ "${SPECIFIC_TESTS[0]}" = "gpu" ]; then
    if [ "$FROM_MODE" = true ]; then
        if [ -z "$START_TEST" ]; then
            print_colored "$RED" "Error: 'gpu from' mode requires a test name"
            exit 1
        fi
        run_from_gpu_test "$START_TEST"
    elif [ ${#SPECIFIC_TESTS[@]} -gt 1 ]; then
        for ((i=1; i<${#SPECIFIC_TESTS[@]}; i++)); do
            if run_test_by_name "${SPECIFIC_TESTS[$i]}"; then
                PASSED_TESTS+=("${SPECIFIC_TESTS[$i]}")
            else
                FAILED_TESTS+=("${SPECIFIC_TESTS[$i]}")
            fi
        done
    else
        run_test_by_name "gpu"
    fi
elif [ "$FROM_MODE" = true ]; then
    if [ -z "$START_TEST" ]; then
        print_colored "$RED" "Error: 'from' mode requires a test name"
        exit 1
    fi
    run_from_test "$START_TEST"
elif [ ${#SPECIFIC_TESTS[@]} -gt 0 ]; then
    for test_name in "${SPECIFIC_TESTS[@]}"; do
        if run_test_by_name "$test_name"; then
            PASSED_TESTS+=("$test_name")
        else
            FAILED_TESTS+=("$test_name")
        fi
    done
else
    print_colored "$RED" "Error: No test specified"
    exit 1
fi

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
