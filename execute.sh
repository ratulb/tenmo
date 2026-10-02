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
  local mojo_flags=$3
  local log_file="$LOG_DIR/${test_name}.log"

  # A test file can be listed here for local runs yet absent from a trimmed
  # distribution (some GPU suites are excluded from publication). Skip it
  # instead of failing on a missing file, so one absent entry does not break
  # a whole suite. Not a failure, not a pass.
  if [ ! -f "$test_file" ]; then
    print_colored "$YELLOW" "⊘ SKIPPED: ${test_name} (${test_file} not in this tree)"
    return 0
  fi

  print_colored "$CYAN" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  print_colored "$BOLD" "Running: $test_name"
  print_colored "$CYAN" "File: $test_file"
  echo "────────────────────────────────────────"

  local start_time=$(date +%s%N)

  if [ -n "$mojo_flags" ]; then
    pixi run mojo -I . $mojo_flags "$test_file" 2>&1 | tee "$log_file"
  else
    pixi run mojo -I . "$test_file" 2>&1 | tee "$log_file"
  fi

  local exit_code=${PIPESTATUS[0]}
  local end_time=$(date +%s%N)
  local duration=$(((end_time - start_time) / 1000000)) # milliseconds

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
    IFS='|' read -r name file <<<"$test"
    run_test "$name" "$file" "$MOJO_FLAGS" &
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
  "ndb_oop_arith|tests/test_ndbuffer_arithmetic_gpu.mojo"
  "ndb_inp_arith|tests/test_ndbuffer_inplace_gpu.mojo"
  "reshape|tests/test_reshape.mojo"
  "scalar_ops_gpu|tests/test_scalar_gpu.mojo"
  "embedding|tests/test_embedding.mojo"
  "positional|tests/test_positional.mojo"
  "attention|tests/test_attention.mojo"
  "bidirectional|tests/test_bidirectional_attention.mojo"
  "encoder|tests/test_encoder.mojo"
  "causal|tests/test_causal_self_attention_extended.mojo"
  "gpt_stack|tests/test_gpt_stack.mojo"
  "gptflow|tests/test_gpt_gradflow.mojo"
  "dot|tests/test_dot.mojo"
  "division|tests/test_division.mojo"
  "outer|tests/test_outer.mojo"
  "layer_norm|tests/test_layernorm.mojo"
  "reciprocal|tests/test_reciprocal.mojo"
  "product|tests/test_product_reduction.mojo"
  "unary|tests/test_unary_ops.mojo"
  "sqrt|tests/test_sqrt.mojo"
  "abs|tests/test_abs.mojo"
  "round_floor|tests/test_round_floor.mojo"
  "fake_quant|tests/test_fakequant.mojo"
  "clip|tests/test_clip.mojo"
  "tril|tests/test_tril.mojo"
  "triu|tests/test_triu.mojo"
  "where|tests/test_where.mojo"
  "cumsum|tests/test_cumsum.mojo"
  "masked_fill|tests/test_masked_fill.mojo"
  "attn_matmul|tests/test_attn_matmul.mojo"
  "attn_matmul_cpu|tests/test_attn_matmul_cpu.mojo"
  "attn_matmul_gpu|tests/test_attn_matmul_gpu.mojo"
  "bce|tests/test_bce.mojo"
  "tensors|tests/test_tensors.mojo"
  "gpu_cpu|tests/test_gpu.mojo"
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
  "exp_gpu|tests/test_exponential_gpu.mojo"
  "exp_gpu_standalone|tests/gpu/standalone/test_exp_gpu.mojo"
  "summean|tests/test_sum_mean.mojo"
  "sigmoid|tests/test_sigmoid.mojo"
  "gpusummean|tests/test_gpu_sum_mean.mojo"
  "broadcast|tests/test_broadcast.mojo"
  "scalar|tests/test_scalar_tensors.mojo"
  "inplace|tests/test_inplace.mojo"
  "expand|tests/test_expand.mojo"
  "gpu_expand|tests/test_gpu_expand.mojo"
  "sgd|tests/test_sgd.mojo"
  "sparse_sgd|tests/test_sparse_sgd.mojo"
  "adamw|tests/test_adamw.mojo"
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
  "logarithm_gpu|tests/test_logarithm_gpu.mojo"
  "concat|tests/test_concat.mojo"
  "variance|tests/test_variance.mojo"
  "variance_and_std|tests/test_variance_and_std.mojo"
  "utils|tests/test_utils.mojo"
  "accuracy|tests/test_accuracy.mojo"
  "indexhelper|tests/test_indexhelper.mojo"
  "losses|tests/test_losses.mojo"
  "tanh|tests/test_tanh.mojo"
  "tanh_gpu|tests/test_tanh_gpu.mojo"
  "data|tests/test_data.mojo"
  "imdb_bert_data|tests/test_imdb_bert_data.mojo"
  "epochs|tests/test_epochs.mojo"
  "generate|tests/test_generate.mojo"
  "softmax|tests/test_softmax.mojo"
  "repeat|tests/test_repeat.mojo"
  "mmnd|tests/test_mmnd.mojo"
  "intarray|tests/test_intarray.mojo"
  "mm2d|tests/test_mm2d.mojo"
  "mm_cpu|tests/test_matmul_cpu.mojo"
  "vm|tests/test_vm.mojo"
  "mv|tests/test_mv.mojo"
  "slice|tests/test_slice.mojo"
  "tiles|tests/test_tiles.mojo"
  "linspace|tests/test_linspace.mojo"
  "argminmax|tests/test_argminmax.mojo"
  "minmax|tests/test_minmax.mojo"
  "welford|tests/test_welford.mojo"
  "relu|tests/test_relu.mojo"
  "gelu|tests/test_gelu.mojo"
  "shuffle|tests/test_shuffle.mojo"
  "permute|tests/test_permute.mojo"
  "flatten|tests/test_flatten.mojo"
  "fanin|tests/test_fanin_drain.mojo"
  "gather|tests/test_gather.mojo"
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
  "checkpoint|tests/test_checkpoint.mojo"
  "scheduler|tests/test_scheduler.mojo"
  "module_list|tests/test_module_list.mojo"
  "cast_graph|tests/test_cast_graph.mojo"
  "mixedseq|tests/test_mixed_sequential.mojo"
  "staticseq|tests/test_static_seq.mojo"
  "idgen|tests/test_idgen.mojo"
)

declare -a GPU_TESTS=(
  "ndb_oop_arith|tests/test_ndbuffer_arithmetic_gpu.mojo"
  "ndb_inp_arith|tests/test_ndbuffer_inplace_gpu.mojo"
  "reshape|tests/test_reshape.mojo"
  "scalar_ops_gpu|tests/test_scalar_gpu.mojo"
  "embedding|tests/test_embedding.mojo"
  "positional|tests/test_positional.mojo"
  "division|tests/test_division.mojo"
  "dot|tests/test_dot.mojo"
  "outer|tests/test_outer.mojo"
  "layer_norm|tests/test_layernorm.mojo"
  "reciprocal|tests/test_reciprocal.mojo"
  "product|tests/test_product_reduction.mojo"
  "unary|tests/test_unary_ops.mojo"
  "sqrt|tests/test_sqrt.mojo"
  "attn_matmul|tests/test_attn_matmul.mojo"
  "attn_matmul_gpu|tests/test_attn_matmul_gpu.mojo"
  "bce|tests/test_bce.mojo"
  "gpu_cpu|tests/test_gpu.mojo"
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
  "exp_gpu|tests/test_exponential_gpu.mojo"
  "exp_gpu_standalone|tests/gpu/standalone/test_exp_gpu.mojo"
  "summean|tests/test_sum_mean.mojo"
  "sigmoid|tests/test_sigmoid.mojo"
  "gpusummean|tests/test_gpu_sum_mean.mojo"
  "broadcast|tests/test_broadcast.mojo"
  "scalar|tests/test_scalar_tensors.mojo"
  "accuracy|tests/test_accuracy.mojo"
  "scalar_gpu|tests/test_scalar_gpu.mojo"
  "inplace|tests/test_inplace.mojo"
  "gpu_expand|tests/test_gpu_expand.mojo"
  "sgd|tests/test_sgd.mojo"
  "sparse_sgd|tests/test_sparse_sgd.mojo"
  "adamw|tests/test_adamw.mojo"
  "dropout|tests/test_dropout.mojo"
  "dev_transfer|tests/test_device_transfer_gradflow.mojo"
  "logarithm|tests/test_logarithm.mojo"
  "logarithm_gpu|tests/test_logarithm_gpu.mojo"
  "abs|tests/test_abs.mojo"
  "cumsum|tests/test_cumsum.mojo"
  "multinomial|tests/test_multinomial.mojo"
  "tril|tests/test_tril.mojo"
  "triu|tests/test_triu.mojo"
  "where|tests/test_where.mojo"
  "masked_fill|tests/test_masked_fill.mojo"
  "tanh|tests/test_tanh.mojo"
  "tanh_gpu|tests/test_tanh_gpu.mojo"
  "softmax|tests/test_softmax.mojo"
  "argminmax|tests/test_argminmax.mojo"
  "minmax|tests/test_minmax.mojo"
  "welford|tests/test_welford.mojo"
  "relu|tests/test_relu.mojo"
  "gelu|tests/test_gelu.mojo"
  "shuffle|tests/test_shuffle.mojo"
  "permute|tests/test_permute.mojo"
  "flatten|tests/test_flatten.mojo"
  "fanin|tests/test_fanin_drain.mojo"
  "gather|tests/test_gather.mojo"
  "squeeze|tests/test_squeeze.mojo"
  "ndb|tests/test_ndb.mojo"
  "transpose|tests/test_transpose.mojo"
  "variance_and_std|tests/test_variance_and_std.mojo"
  "tiles|tests/test_tiles.mojo"
  "ce|tests/test_cross_entropy.mojo"
  "dtype_cast|tests/test_dtype_cast_gpu.mojo"
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
  echo "  -b, --blas        Enable the -D BLAS=1 opt-in (routes eligible Tensor.matmul through OpenBLAS)"
  echo "      --no-blas     Force native (clears a prior -b/--blas, e.g. for blasinteg)"
  echo ""
  print_colored "$CYAN" "Examples:"
  echo "  $0 softmax matmul tensors     - Run only softmax, matmul, and tensors"
  echo "  $0 -b blasinteg               - Run the BLAS-adaptive integration test with OpenBLAS on"
  echo "  $0 blasinteg                  - Same, but native (BLAS assertions skipped)"
  echo "  $0 gpu                        - Run all GPU-guarded tests"
  echo "  $0 gpu from relu              - Run relu and all GPU tests after it"
  echo "  $0 gpu relu tanh              - Run only relu and tanh"
  echo "  $0 select softmax test_softmax_1d   - Isolate and run a single test function"
  echo ""
  print_colored "$CYAN" "Available tests:"
  echo "  scalar_ops_gpu, reshape, ndb_inp_arith, ndb_oop_arith, dot, division, embedding, positional, layer_norm, reciprocal, product, unary, sqrt, tensors, gpu, item, contiguous, maxmin_scalar"
  echo "  allany, compare, count_unique, transmute, exp, exp_gpu, summean, sigmoid"
  echo "  gpusummean, broadcast, scalar, inplace, expand, gpu_expand, gpu_cpu"
  echo "  sgd, sparse_sgd, npiop, fill, chunk, cnn, matmul, pad, blas, blasinteg, dropout, dev_transfer"
  echo "  std_variance, stack, logarithm, concat, variance, variance_and_std, accuracy, utils, onehot, power"
  echo "  indexhelper, welford, losses, tanh, data, softmax, repeat, mmnd, attn_matmul"
  echo "  attention, causal, attn_matmul_cpu, attn_matmul_gpu, bce, intarray, mm2d, mm_cpu, vm, mv, slice, view_slice, tiles, linspace, argminmax"
  echo "  minmax, relu, gelu, shuffle, permute, flatten, gather, squeeze, unsqueeze"
  echo "  select <test_name> <fn>  Extract and run a single test function from a file"
  echo "  gpu_all [N..M] [N]...  Run GPU tests: monolithic; chunk N; range (2..4); or list (2 4 6)"
  echo "  cpu_all [N..M] [N]...  Run CPU tests: monolithic; chunk N; range (2..4); or list (2 4 6)"
  echo "  idgen          Test IDGen global unique-id counter"
  echo "  scheduler, checkpoint, shapebroadcast, validators, ce, dtype_cast"
  echo ""
  print_colored "$GREEN" "  all              Run all tests"
  print_colored "$GREEN" "  gpu              Run all GPU-guarded tests"
  print_colored "$GREEN" "  quick            Run quick sanity tests"
  exit 1
fi

# Parse arguments
MOJO_FLAGS=""
PARALLEL=false
FROM_MODE=false
START_TEST=""
declare -a SPECIFIC_TESTS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    -d | --debug)
      MOJO_FLAGS="$MOJO_FLAGS -D LOGGING_LEVEL=debug"
      shift
      ;;
    -b | --blas)
      MOJO_FLAGS="$MOJO_FLAGS -D BLAS=1"
      shift
      ;;
    --no-blas)
      # Force native: unset the -D BLAS opt-in if previously added.
      MOJO_FLAGS="$(printf '%s' " $MOJO_FLAGS " | sed 's/ -D BLAS=1 / /g' | sed 's/^ *//;s/ *$//')"
      shift
      ;;
    -p | --parallel)
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

# Function to run a test by name (accepts optional chunk arg for cpu_all)
run_test_by_name() {
  local test_name=$1
  local chunk_arg="${2:-}"
  local exit_code=0

  case $test_name in
    ndb_oop_arith)
      run_test "ndb_oop_arith" "tests/test_ndbuffer_arithmetic_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    ndb_inp_arith)
      run_test "ndb_inp_arith" "tests/test_ndbuffer_inplace_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    reshape)
      run_test "reshape" "tests/test_reshape.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    scalar_ops_gpu)
      run_test "scalar_ops_gpu" "tests/test_scalar_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    embedding)
      run_test "embedding" "tests/test_embedding.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    encoder)
      run_test "encoder" "tests/test_encoder.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    positional)
      run_test "positional" "tests/test_positional.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    attention)
      run_test "attention" "tests/test_attention.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    bidirectional)
      run_test "bidirectional" "tests/test_bidirectional_attention.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    causal)
      run_test "attention" "tests/test_causal_self_attention_extended.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    gpt_stack)
      run_test "gpt_stack" "tests/test_gpt_stack.mojo" "$MOJO_FLAGS"
      ;;
    gptflow)
      run_test "gptflow" "tests/test_gpt_gradflow.mojo" "$MOJO_FLAGS"
      ;;
    dot)
      run_test "dot" "tests/test_dot.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    division)
      run_test "division" "tests/test_division.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    outer)
      run_test "outer" "tests/test_outer.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    layer_norm)
      run_test "layer_norm" "tests/test_layernorm.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    reciprocal)
      run_test "reciprocal" "tests/test_reciprocal.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    product)
      run_test "product" "tests/test_product_reduction.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    unary)
      run_test "unary" "tests/test_unary_ops.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    abs)
      run_test "abs" "tests/test_abs.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    round_floor)
      run_test "round_floor" "tests/test_round_floor.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    fake_quant)
      run_test "fake_quant" "tests/test_fakequant.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    clip)
      run_test "clip" "tests/test_clip.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    sqrt)
      run_test "sqrt" "tests/test_sqrt.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    attn_matmul)
      run_test "attn_matmul" "tests/test_attn_matmul.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    attn_matmul_cpu)
      run_test "attn_matmul_cpu" "tests/test_attn_matmul_cpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    attn_matmul_gpu)
      run_test "attn_matmul_gpu" "tests/test_attn_matmul_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    bce)
      run_test "bce" "tests/test_bce.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    tensors)
      run_test "tensors" "tests/test_tensors.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    gpu)
      print_colored "$BLUE" "Running all GPU-guarded tests..."
      if [ "$PARALLEL" = true ]; then
        run_parallel "${GPU_TESTS[@]}"
        exit_code=$?
      else
        for test in "${GPU_TESTS[@]}"; do
          IFS='|' read -r name file <<<"$test"
          if run_test "$name" "$file" "$MOJO_FLAGS"; then
            PASSED_TESTS+=("$name")
          else
            FAILED_TESTS+=("$name")
            exit_code=1
          fi
        done
      fi
      ;;
    item)
      run_test "item" "tests/test_item.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    contiguous)
      run_test "contiguous" "tests/test_contiguous.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    maxmin_scalar)
      run_test "maxmin_scalar" "tests/test_maxmin_scalar.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    onehot)
      run_test "onehot" "tests/test_onehot.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    power)
      run_test "power" "tests/test_exponentiator.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    allany)
      run_test "allany" "tests/test_all_true_any_true.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    compare)
      run_test "compare" "tests/test_compare.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    count_unique)
      run_test "count_unique" "tests/test_count_unique.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    transmute)
      run_test "transmute" "tests/test_transmutation.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    exp)
      run_test "exp" "tests/test_exponential.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    exp_gpu)
      run_test "exp_gpu" "tests/test_exponential_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    exp_gpu_standalone)
      run_test "exp_gpu_standalone" "tests/gpu/standalone/test_exp_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    summean)
      run_test "summean" "tests/test_sum_mean.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    sigmoid)
      run_test "sigmoid" "tests/test_sigmoid.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    gpu_cpu)
      run_test "gpu_cpu" "tests/test_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    gpusummean)
      run_test "gpusummean" "tests/test_gpu_sum_mean.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    broadcast)
      run_test "broadcast" "tests/test_broadcast.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    scalar)
      run_test "scalar" "tests/test_scalar_tensors.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    scalar_gpu)
      run_test "scalar_gpu" "tests/test_scalar_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    inplace)
      run_test "inplace" "tests/test_inplace.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    expand)
      run_test "expand" "tests/test_expand.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    gpu_expand)
      run_test "gpu_expand" "tests/test_gpu_expand.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    sgd)
      run_test "sgd" "tests/test_sgd.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    sparse_sgd)
      run_test "sparse_sgd" "tests/test_sparse_sgd.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    adamw)
      run_test "adamw" "tests/test_adamw.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    scheduler)
      run_test "scheduler" "tests/test_scheduler.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    npiop)
      run_test "npiop" "tests/test_numpy_interop.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    fill)
      run_test "fill" "tests/test_fill.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    chunk)
      run_test "chunk" "tests/test_chunk.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    cnn)
      run_test "cnn" "tests/test_cnn.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    matmul)
      run_test "matmul" "tests/test_matmul.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    pad)
      run_test "pad" "tests/test_pad.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    pool_tt)
      run_test "pool_tt" "tests/test_pool_tt.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    conv_tt)
      run_test "conv_tt" "tests/test_conv_tt.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    blas)
      run_test "blas" "tests/test_blas.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    blasinteg)
      run_test "blasinteg" "tests/test_matmul_blas_integration.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    blasnet)
      run_test "blasnet" "tests/test_blas_net.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    dropout)
      run_test "dropout" "tests/test_dropout.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    dev_transfer)
      run_test "dev_transfer" "tests/test_device_transfer_gradflow.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    std_variance)
      run_test "std_variance" "tests/test_std_variance.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    stack)
      run_test "stack" "tests/test_stack.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    logarithm)
      run_test "logarithm" "tests/test_logarithm.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    logarithm_gpu)
      run_test "logarithm_gpu" "tests/test_logarithm_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    concat)
      run_test "concat" "tests/test_concat.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    variance)
      run_test "variance" "tests/test_variance.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    variance_and_std)
      run_test "variance_and_std" "tests/test_variance_and_std.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    utils)
      run_test "utils" "tests/test_utils.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    accuracy)
      run_test "accuracy" "tests/test_accuracy.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    assignment)
      run_test "assignment" "tests/test_assignment_semantics.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    indexhelper)
      run_test "indexhelper" "tests/test_indexhelper.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    losses)
      run_test "losses" "tests/test_losses.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    abs)
      run_test "abs" "tests/test_abs.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    cumsum)
      run_test "cumsum" "tests/test_cumsum.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    multinomial)
      run_test "multinomial" "tests/test_multinomial.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    tril)
      run_test "tril" "tests/test_tril.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    triu)
      run_test "triu" "tests/test_triu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    where)
      run_test "where" "tests/test_where.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    masked_fill)
      run_test "masked_fill" "tests/test_masked_fill.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    tanh)
      run_test "tanh" "tests/test_tanh.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    tanh_gpu)
      run_test "tanh_gpu" "tests/test_tanh_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    data)
      run_test "data" "tests/test_data.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    imdb_bert_data)
      run_test "imdb_bert_data" "tests/test_imdb_bert_data.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    epochs)
      run_test "epochs" "tests/test_epochs.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    generate)
      run_test "generate" "tests/test_generate.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    softmax)
      run_test "softmax" "tests/test_softmax.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    repeat)
      run_test "repeat" "tests/test_repeat.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    mmnd)
      run_test "mmnd" "tests/test_mmnd.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    intarray)
      run_test "intarray" "tests/test_intarray.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    mm2d)
      run_test "mm2d" "tests/test_mm2d.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    mm_cpu)
      run_test "mm_cpu" "tests/test_matmul_cpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    vm)
      run_test "vm" "tests/test_vm.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    mv)
      run_test "mv" "tests/test_mv.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    slice)
      run_test "slice" "tests/test_slice.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    view_slice)
      run_test "view_slice" "tests/test_view_slice.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    tiles)
      run_test "tiles" "tests/test_tiles.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    linspace)
      run_test "linspace" "tests/test_linspace.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    argminmax)
      run_test "argminmax" "tests/test_argminmax.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    minmax)
      run_test "minmax" "tests/test_minmax.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    welford)
      run_test "welford" "tests/test_welford.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    relu)
      run_test "relu" "tests/test_relu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
     gelu)
      run_test "gelu" "tests/test_gelu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
   shuffle)
      run_test "shuffle" "tests/test_shuffle.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    permute)
      run_test "permute" "tests/test_permute.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    cast_graph)
      run_test "cast_graph" "tests/test_cast_graph.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    mixedseq)
      run_test "mixedseq" "tests/test_mixed_sequential.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    staticseq)
      run_test "staticseq" "tests/test_static_seq.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    idgen)
      run_test "idgen" "tests/test_idgen.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    flatten)
      run_test "flatten" "tests/test_flatten.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    fanin)
      run_test "fanin" "tests/test_fanin_drain.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    gather)
      run_test "gather" "tests/test_gather.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    squeeze)
      run_test "squeeze" "tests/test_squeeze.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    unsqueeze)
      run_test "unsqueeze" "tests/test_unsqueeze.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    gradbox)
      run_test "gradbox" "tests/test_gradbox.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    ndb)
      run_test "ndb" "tests/test_ndb.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    transpose)
      run_test "transpose" "tests/test_transpose.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    buffers)
      run_test "buffers" "tests/test_buffers.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    views)
      run_test "views" "tests/test_views.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    shapes)
      run_test "shapes" "tests/test_shapes.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    strides)
      run_test "strides" "tests/test_strides.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    shapebroadcast)
      run_test "shapebroadcast" "tests/test_broadcaster.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    validators)
      run_test "validators" "tests/test_validators.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    ce)
      run_test "ce" "tests/test_cross_entropy.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    dtype_cast)
      run_test "dtype_cast" "tests/test_dtype_cast_gpu.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    checkpoint)
      run_test "checkpoint" "tests/test_checkpoint.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    module_list)
      run_test "module_list" "tests/test_module_list.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    gpu_all)
      if [ -n "$chunk_arg" ]; then
        local chunk_file="tests/test_gpu_all_${chunk_arg}.mojo"
        if [ -f "$chunk_file" ]; then
          print_colored "$BLUE" "Running GPU test chunk $chunk_arg..."
          run_test "gpu_all_${chunk_arg}" "$chunk_file" "$MOJO_FLAGS"
        else
          print_colored "$RED" "Error: chunk file $chunk_file does not exist"
          return 1
        fi
      elif [ -f "tests/test_gpu_all.mojo" ]; then
        print_colored "$BLUE" "Running all GPU tests from single file..."
        run_test "gpu_all" "tests/test_gpu_all.mojo" "$MOJO_FLAGS"
      else
        print_colored "$BLUE" "Running all GPU test chunks sequentially..."
        for cf in $(printf '%s\n' tests/test_gpu_all_*.mojo | sort -V); do
          local cname="${cf#tests/test_gpu_all_}"
          cname="${cname%.mojo}"
          run_test "gpu_all_${cname}" "$cf" "$MOJO_FLAGS" || true
        done
      fi
      exit_code=$?
      ;;
    cpu_all)
      if [ -n "$chunk_arg" ]; then
        local chunk_file="tests/test_cpu_all_${chunk_arg}.mojo"
        if [ -f "$chunk_file" ]; then
          print_colored "$BLUE" "Running CPU test chunk $chunk_arg..."
          run_test "cpu_all_${chunk_arg}" "$chunk_file" "$MOJO_FLAGS"
        else
          print_colored "$RED" "Error: chunk file $chunk_file does not exist"
          return 1
        fi
      elif [ -f "tests/test_cpu_all.mojo" ]; then
        print_colored "$BLUE" "Running all CPU tests from single file..."
        run_test "cpu_all" "tests/test_cpu_all.mojo" "$MOJO_FLAGS"
      else
        print_colored "$BLUE" "Running all CPU test chunks sequentially..."
        for cf in $(printf '%s\n' tests/test_cpu_all_*.mojo | sort -V); do
          local cname="${cf#tests/test_cpu_all_}"
          cname="${cname%.mojo}"
          run_test "cpu_all_${cname}" "$cf" "$MOJO_FLAGS" || true
        done
      fi
      exit_code=$?
      ;;
    quick)
      print_colored "$BLUE" "Running quick sanity tests..."
      run_test "tensors" "tests/test_tensors.mojo" "$MOJO_FLAGS"
      exit_code=$?
      [ $exit_code -eq 0 ] && run_test "shapes" "tests/test_shapes.mojo" "$MOJO_FLAGS"
      exit_code=$?
      [ $exit_code -eq 0 ] && run_test "strides" "tests/test_strides.mojo" "$MOJO_FLAGS"
      exit_code=$?
      [ $exit_code -eq 0 ] && run_test "summean" "tests/test_sum_mean.mojo" "$MOJO_FLAGS"
      exit_code=$?
      ;;
    all)
      print_colored "$BLUE" "Running ALL tests..."
      if [ "$PARALLEL" = true ]; then
        run_parallel "${ALL_TESTS_IN_ORDER[@]}"
        exit_code=$?
      else
        for test in "${ALL_TESTS_IN_ORDER[@]}"; do
          IFS='|' read -r name file <<<"$test"
          if run_test "$name" "$file" "$MOJO_FLAGS"; then
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
    IFS='|' read -r name file <<<"$test_entry"

    if [ "$found" = true ]; then
      # Run this test
      if run_test "$name" "$file" "$MOJO_FLAGS"; then
        PASSED_TESTS+=("$name")
      else
        FAILED_TESTS+=("$name")
        exit_code=1
      fi
    elif [ "$name" = "$start_test" ]; then
      # Found the starting test, run it
      found=true
      if run_test "$name" "$file" "$MOJO_FLAGS"; then
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
    IFS='|' read -r name file <<<"$test_entry"

    if [ "$found" = true ]; then
      if run_test "$name" "$file" "$MOJO_FLAGS"; then
        PASSED_TESTS+=("$name")
      else
        FAILED_TESTS+=("$name")
        exit_code=1
      fi
    elif [ "$name" = "$start_test" ]; then
      found=true
      if run_test "$name" "$file" "$MOJO_FLAGS"; then
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
    for ((i = 1; i < ${#SPECIFIC_TESTS[@]}; i++)); do
      if run_test_by_name "${SPECIFIC_TESTS[$i]}"; then
        PASSED_TESTS+=("${SPECIFIC_TESTS[$i]}")
      else
        FAILED_TESTS+=("${SPECIFIC_TESTS[$i]}")
      fi
    done
  else
    run_test_by_name "gpu"
  fi
elif [ "${SPECIFIC_TESTS[0]}" = "select" ]; then
  if [ ${#SPECIFIC_TESTS[@]} -lt 3 ]; then
    print_colored "$RED" "Usage: $0 select <test_name> <test_fn_name>"
    exit 1
  fi
  select_test_name="${SPECIFIC_TESTS[1]}"
  select_test_fn="${SPECIFIC_TESTS[2]}"
  case $select_test_name in
    test_*) select_file="tests/${select_test_name}.mojo" ;;
    *) select_file="tests/test_${select_test_name}.mojo" ;;
  esac
  if [ ! -f "$select_file" ]; then
    print_colored "$RED" "Error: test file not found: $select_file"
    exit 1
  fi
  print_colored "$CYAN" "Extracting $select_test_fn from $(basename $select_file)..."
  select_output=$(python3 scripts/select_test.py "$select_file" "$select_test_fn")
  if [ $? -ne 0 ]; then
    FAILED_TESTS+=("$select_test_fn")
  else
    if run_test "$select_test_fn" "$select_output" "$MOJO_FLAGS"; then
      PASSED_TESTS+=("$select_test_fn")
    else
      FAILED_TESTS+=("$select_test_fn")
    fi
  fi
elif [ "$FROM_MODE" = true ]; then
  if [ -z "$START_TEST" ]; then
    print_colored "$RED" "Error: 'from' mode requires a test name"
    exit 1
  fi
  run_from_test "$START_TEST"
elif [ ${#SPECIFIC_TESTS[@]} -gt 0 ]; then
  for ((i = 0; i < ${#SPECIFIC_TESTS[@]}; i++)); do
    test_name="${SPECIFIC_TESTS[$i]}"
    # Handle "cpu_all/gpu_all [N M..P ...]" — collect chunk args
    if [ "$test_name" = "cpu_all" ] || [ "$test_name" = "gpu_all" ]; then
      chunk_args=()
      j=$((i + 1))
      while [ $j -lt ${#SPECIFIC_TESTS[@]} ]; do
        arg="${SPECIFIC_TESTS[$j]}"
        if [[ "$arg" =~ ^[0-9]+$ ]]; then
          chunk_args+=("$arg")
        elif [[ "$arg" =~ ^[0-9]+\.\.[0-9]+$ ]]; then
          start="${arg%..*}"
          end="${arg#*..}"
          [ "$start" -le "$end" ] || {
            start=$end
            end="${arg%..*}"
          }
          for ((k = start; k <= end; k++)); do
            chunk_args+=("$k")
          done
        else
          break
        fi
        j=$((j + 1))
      done
      i=$((j - 1))
      if [ ${#chunk_args[@]} -eq 0 ]; then
        # No chunk args: run monolithic or all chunks (handled in run_test_by_name)
        if run_test_by_name "$test_name"; then
          PASSED_TESTS+=("$test_name")
        else
          FAILED_TESTS+=("$test_name")
        fi
      else
        for chunk in "${chunk_args[@]}"; do
          if run_test_by_name "$test_name" "$chunk"; then
            PASSED_TESTS+=("${test_name}_${chunk}")
          else
            FAILED_TESTS+=("${test_name}_${chunk}")
          fi
        done
      fi
      continue
    fi
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
TOTAL_DURATION=$(((SCRIPT_END - SCRIPT_START) / 1000000)) # milliseconds

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
