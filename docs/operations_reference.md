# Operations Reference — Backward Pass

This document lists all 56 operations supported in Tenmo's autograd system.

## Operation Tags

| Tag | Constant | Forward Op | Backward Struct | Module |
|-----|----------|-----------|-------------|---------------|-------|
| 0 | `BACKWARD_ADD` | `a + b` | `AddBackward` | `addition.mojo` |
| 1 | `BACKWARD_MULTIPLY` | `a * b` | `MultiplyBackward` | `multiplication.mojo` |
| 2 | `BACKWARD_RELU` | `relu(a)` | `ReLUBackward` | `relu.mojo` |
| 3 | `BACKWARD_MATMUL_ND` | `a @ b` (ND) | `MatmulNdBackward` | `matmul.mojo` |
| 4 | `BACKWARD_MATMUL_2D` | `a @ b` (2D) | `Matmul2dBackward` | `matmul.mojo` |
| 5 | `BACKWARD_TRANSPOSE` | `a.T` | `TransposeBackward` | `transpose.mojo` |
| 6 | `BACKWARD_PERMUTE` | `permute(a, axes)` | `PermuteBackward` | `permute.mojo` |
| 7 | `BACKWARD_SIGMOID` | `sigmoid(a)` | `SigmoidBackward` | `sigmoid.mojo` |
| 8 | `BACKWARD_SOFTMAX` | `softmax(a, axis)` | `SoftmaxBackward` | `softmax.mojo` |
| 9 | `BACKWARD_CE_CLASS_INDICES` | `cross_entropy(targets, class_indices)` | `CEClassIndicesBackward` | `crossentropy.mojo` |
| 10 | `BACKWARD_CE_PROBABILITIES` | `cross_entropy(targets, probabilities)` | `CEProbabilitiesBackward` | `crossentropy.mojo` |
| 11 | `BACKWARD_TANH` | `tanh(a)` | `TanhBackward` | `tanh.mojo` |
| 12 | `BACKWARD_SUB` | `a - b` | `SubBackward` | `subtraction.mojo` |
| 13 | `BACKWARD_RESHAPE` | `reshape(a, shape)` | `ReshapeBackward` | `reshape.mojo` |
| 14 | `BACKWARD_VIEW` | `view(a, shape)` | `ViewBackward` | `views.mojo` |
| 15 | `BACKWARD_MEAN` | `mean(a, axis)` | `MeanBackward` | `mean_reduction.mojo` |
| 16 | `BACKWARD_SUM` | `sum(a, axis)` | `SumBackward` | `summation.mojo` |
| 17 | `BACKWARD_LOG_SOFTMAX` | `log_softmax(a, axis)` | `LogSoftmaxBackward` | `softmax.mojo` |
| 18 | `BACKWARD_CONTIGUOUS` | `a.contiguous()` | `ContiguousBackward` | `contiguous.mojo` |
| 19 | `BACKWARD_DIVIDE` | `a / b` | `DivideBackward` | `division.mojo` |
| 20 | `BACKWARD_MATRIX_VECTOR_MUL` | `A @ v` | `MatrixVectorMulNdBackward` | `matrixvector.mojo` |
| 21 | `BACKWARD_VECTOR_MATMUL` | `v @ A` | `VectorMatmulNdBackward` | `vectormatrix.mojo` |
| 22 | `BACKWARD_ADD_SCALAR` | `a + scalar` | `AddBackwardScalar` | `addition.mojo` |
| 23 | `BACKWARD_ADD_BROADCAST` | `a + b` (broadcast) | `AddBroadcastBackward` | `broadcastbackward.mojo` |
| 24 | `BACKWARD_MULTIPLY_SCALAR` | `a * scalar` | `MultiplyBackwardScalar` | `multiplication.mojo` |
| 25 | `BACKWARD_SUB_SCALAR` | `a - scalar` | `SubLeftRightBackwardScalar` | `subtraction.mojo` |
| 26 | `BACKWARD_SUBTRACT_BROADCAST` | `a - b` (broadcast) | `SubtractBroadcastBackward` | `broadcastbackward.mojo` |
| 27 | `BACKWARD_DIV_SCALAR` | `a / scalar` | `TrueDivBackwardScalar` | `division.mojo` |
| 28 | `BACKWARD_RIGHT_DIV_SCALAR` | `scalar / a` | `RightTrueDivBackwardScalar` | `division.mojo` |
| 29 | `BACKWARD_EXPONENTIATION` | `a ** n` | `ExponentiationBackward` | `exponentiator.mojo` |
| 30 | `BACKWARD_DOT` | `dot(a, b)` | `DotBackward` | `dotproduct.mojo` |
| 31 | `BACKWARD_EXPAND` | `expand(a, shape)` | `ExpandBackward` | `expand.mojo` |
| 32 | `BACKWARD_FLATTEN` | `flatten(a)` | `FlattenBackward` | `flatten.mojo` |
| 33 | `BACKWARD_SQUEEZE` | `squeeze(a, axis)` | `SqueezeBackward` | `squeeze.mojo` |
| 34 | `BACKWARD_UNSQUEEZE` | `unsqueeze(a, axis)` | `UnsqueezeBackward` | `unsqueeze.mojo` |
| 35 | `BACKWARD_SHUFFLE` | `shuffle(a, axis, permutation)` | `ShuffleBackward` | `shuffle.mojo` |
| 36 | `BACKWARD_MINMAX` | `min(a, axis) / max(a, axis)` | `MinMaxBackward` | `minmax.mojo` |
| 37 | `BACKWARD_TILE` | `tile(a, repeats)` | `TileBackward` | `tiles.mojo` |
| 38 | `BACKWARD_LOG` | `log(a)` | `LogBackward` | `logarithm.mojo` |
| 39 | `BACKWARD_SQRT` | `sqrt(a)` | `SqrtBackward` | `squareroot.mojo` |
| 40 | `BACKWARD_CLIP` | `clip(a, min, max)` | `ClipBackward` | `clip.mojo` |
| 41 | `BACKWARD_VARIANCE` | `variance(a, axis)` | `VarianceBackward` | `variance.mojo` |
| 42 | `BACKWARD_STD` | `std(a, axis)` | `StdBackward` | `std_deviation.mojo` |
| 43 | `BLAS_BACKWARD_MATMUL_2D` | `a @ b` (BLAS) | `BLASMatmul2dBackward` | `matmul.mojo` |
| 44 | `BACKWARD_CONCAT` | `concat(tensors, axis)` | `ConcatBackward` | `concate.mojo` |
| 45 | `BACKWARD_STACK` | `stack(tensors, axis)` | `StackBackward` | `stack.mojo` |
| 46 | `BACKWARD_PAD` | `pad(a, padding, mode)` | `PadBackward` | `pad.mojo` |
| 47 | `BACKWARD_FUSED_CONV` | Fused conv | `FusedCol2ImBackward` | `cnn.mojo` |
| 48 | `BACKWARD_MAXPOOL2D` | `max_pool2d(a, kernel)` | `MaxPool2dBackward` | `pooling.mojo` |
| 49 | `BACKWARD_DROPOUT` | `dropout(a, p)` | `DropoutBackward` | `dropout.mojo` |
| 50 | `BACKWARD_EXPONENTIAL` | `exp(a)` | `ExponentialBackward` | `exponential.mojo` |
| 51 | `BACKWARD_DEVICE_TRANSFER` | `a.to_gpu()` / `a.to_cpu()` | `DeviceTransferBackward` | `device_transfer.mojo` |
| 52 | `BACKWARD_MAX_SCALAR` | `max(a, scalar)` | `MaxBackwardScalar` | `maxmin_scalar.mojo` |
| 53 | `BACKWARD_MIN_SCALAR` | `min(a, scalar)` | `MinBackwardScalar` | `maxmin_scalar.mojo` |
| 54 | `BACKWARD_MULTIPLY_BROADCAST` | `a * b` (broadcast) | `MultiplyBroadcastBackward` | `broadcastbackward.mojo` |
| 55 | `BACKWARD_PRODUCT` | `product(a)` | `ProductBackward` | `product_reduction.mojo` |

## Argument Types

Each operation stores its backward arguments using `BackwardFnArg` factory methods:

| Factory Method | Payload Type | Used By |
|----------------|-------------|---------|
| `null_arg(op_code)` | `NullArg` | ADD, MULTIPLY, element-wise ops |
| `scalar_arg(op_code, value)` | `ScalarArg` | *_SCALAR ops |
| `boolean_arg(op_code, is_true)` | `Boolean` | DROPOUT |
| `integer_arg(op_code, value)` | `Integer` | RESHAPE |
| `from_intarray(op_code, array)` | `IntArrayArg` | REDUCTION (axes) |
| `from_buffer(op_code, buffer)` | `BufferArg` | CONTIGUOUS |
| `from_ndbuffer(op_code, ndb)` | `NDBufferArg` | VIEW, RESHAPE, EXPAND |

## Related Documentation

- [Autograd Deep Dive](../README_AUTOGRAD.md) — Full architecture explanation
- [tenmo/backpropagation.mojo](../tenmo/backpropagation.mojo) — Source code