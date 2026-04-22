#!/bin/bash
# Run from repo root: bash fix_imports.sh
# Rewrites "from  <tenmo_module> import" -> "from  .<tenmo_module> import"
# Only for known internal modules — leaves "from std.x import" etc. untouched.

MODULES=(
    addition ancestry argminmax array backpropagation
    binary_inplace_ops_kernel binary_ops_kernel blashandle
    broadcastbackward broadcasthelper buffers clip cnn common_utils
    compare_kernel concate contiguous crossentropy dataloader device
    device_transfer division dotproduct dropout expand exponential
    exponentiator filler flatten forwards gpu_attributes gradbox
    indexhelper intarray logarithm matmul matmul_kernel
    matrixshapevalidator matrixvector matrixvector_kernel maxmin_scalar
    mean_reduction minmax minmax_kernel minmax_reducer mnemonics mse
    multiplication ndbuffer net numpy_interop pad permute pooling
    reduction_kernel relu repeat reshape scalar_inplace_ops_kernel
    scalar_ops_kernel sgd shapes shuffle sigmoid softmax squareroot
    squeeze stack std_deviation strides subtraction summation tanh
    tensor tiles transpose unary_ops_kernel unsqueeze utilities
    validators variance vectormatrix vectormatrix_kernel views walkback
)

# Build sed args for package internals: "from buffers import" -> "from .buffers import"
SED_ARGS_PKG=()
for mod in "${MODULES[@]}"; do
    SED_ARGS_PKG+=(-e "s/^\\(from[[:space:]][[:space:]]*\\)\\(${mod}[[:space:]][[:space:]]*import\\)/\\1.\\2/g")
done

# Build sed args for tests: "from buffers import" -> "from tenmo.buffers import"
SED_ARGS_TEST=()
for mod in "${MODULES[@]}"; do
    SED_ARGS_TEST+=(-e "s/^\\(from[[:space:]][[:space:]]*\\)\\(${mod}[[:space:]][[:space:]]*import\\)/\\1tenmo.\\2/g")
done

# Rewrite internal package files (tenmo/*.mojo)
echo "--- Patching package files ---"
find tenmo -maxdepth 1 -name "*.mojo" | while read -r f; do
    echo "  $f"
    sed -i "${SED_ARGS_PKG[@]}" "$f"
done

# Rewrite test files (tests/*.mojo)
echo "--- Patching test files ---"
find examples -name "*.mojo" | while read -r f; do
    echo "  $f"
    sed -i "${SED_ARGS_TEST[@]}" "$f"
done

echo "Done!"
