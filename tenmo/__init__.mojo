"""
Core Types.

- `Tensor` — the central type. Tracks gradients, owns data, participates in the autograd graph. Supports CPU and GPU.
- `NDBuffer` — shape, strides, offset, and data. Single source of truth for memory layout. Shared between tensors and views via ref-counting.
- `Gradbox` — gradient storage. Independently ref-counted — survives ASAP tensor destruction. Views have their own independent gradboxes.
- `Ancestor` — lightweight parent handle in the autograd graph. Carries id, grad routing, a refcounted gradbox pointer, shape, strides, offset, buffer, and device state. Does not copy full tensors.
- `Ancestors` — the ancestry list for a tensor. Carries the `BackwardFnArg` directly — not stored on `Tensor`.
- `Buffer` — linear, SIMD-capable storage. Ref-counted when shared via views.

## Quick Start

```mojo
from tenmo.tensor import Tensor

def main() raises:
    var a = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a * 2
    var c = a * 3
    var d = b + c
    var loss = d.sum()
    loss.backward()
    # a.grad() == [5.0, 5.0, 5.0]
    a.grad().print()
```

## GPU Example

```mojo
from tenmo.tensor import Tensor
from tenmo.shapes import Shape

def main() raises:
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = Tensor.full(Shape(2, 2), 2.0).to_gpu()
    var c_gpu = a_gpu * b_gpu
    var loss = c_gpu.sum()
    loss.backward()
    a.grad().print()
```

## Training Example

```mojo
from tenmo.tensor import Tensor
from tenmo.net import Sequential, Linear, ReLU
from tenmo.optim import SGD

def main() raises:
    var model = Sequential[DType.float32]()
    model.append(Linear[DType.float32](784, 128).into(), ReLU[DType.float32]().into())
    var optimizer = SGD[DType.float32](model.parameters(), lr=0.01, momentum=0.9)
    # ... training loop
```
"""

# Core Types - explicit exports for documentation links
from .tensor import Tensor
from .ndbuffer import NDBuffer
from .gradbox import Gradbox
from .ancestry import Ancestor, Ancestors

from .buffers import Buffer
from .shapes import Shape
from .strides import Strides
from .intarray import IntArray
from .array import Array

# ── Math Operations ──
from .addition import Adder, AddScalar, AddBackward, AddBackwardScalar, AddBroadcastBackward
from .subtraction import Subtractor, SubtractScalar, SubtractFromScalar, SubBackward, SubLeftRightBackwardScalar, SubtractBroadcastBackward
from .multiplication import Multiplicator, MultiplyScalar, MultiplyBackward, MultiplyBackwardScalar, MultiplyBroadcastBackward
from .division import Divider, DivideScalar, DivideByScalar, DivBuffer, DivElement, DivNdBuffer, TrueDivBackwardScalar, RightTrueDivBackwardScalar, DivideBackward
from .exponentiator import Exponentiator, ExponentiationBackward
from .exponential import Exponential, ExponentialBackward
from .logarithm import Logarithm, LogBackward
from .squareroot import Sqrt, SqrtBackward
from .tanh import TanhBackward
from .sigmoid import SigmoidBackward
from .relu import ReLUBackward
from .softmax import Softmax, LogSoftmax, SoftmaxArg, SoftmaxNdBuffer, SoftmaxBackwardDelegate
from .clip import Clip, ClipArg, ClipBackward
from .minmax import MinMax, MinMaxArg, MinMaxBackward
from .maxmin_scalar import MaxScalar, MinScalar, MaxBackwardScalar, MinBackwardScalar
from .sum_reduction import Summer, SumBackward
from .mean_reduction import Mean, MeanBackward
from .product_reduction import Product, ProductBackward
from .sum_mean_reduction import SumMeanReduction, ReductionArg

# ── Tensor Manipulation ──
from .reshape import Reshape, ReshapeBackward
from .contiguous import Contiguous, ContiguousBackward
from .transpose import Transpose, TransposeBackward
from .permute import Permute, PermuteBackward
from .expand import Expand, ExpandBackward
from .squeeze import Squeeze, SqueezeBackward
from .unsqueeze import Unsqueeze, UnsqueezeBackward
from .flatten import FlattenForward, FlattenBackward
from .concate import Concate, ConcatBackward
from .stack import Stack, StackArg, StackBackward
from .tiles import Tile, TilesArg, TileBackward
from .repeat import Repeat
from .views import View, ViewBackward
from .shuffle import Shuffle, ShuffleArg, ShuffleBackward
from .pad import Pad, PadArg, PadBackward
from .broadcast import Broadcast, BroadcastBackward, BroadcastToBackward
from .broadcasthelper import ShapeBroadcaster

# ── Linear Algebra ──
from .dotproduct import Dot, DotBackward
from .matmul import Matmul, Matmul2d, MatmulNd, Matmul2dBackward, MatmulNdBackward, classify_matmul
from .matmul_cpu import MmCpu2d, MmCpuNd
from .matrixvector import MatrixVectorMulNd, MatrixVectorMulNdBackward
from .vectormatrix import VectorMatmulNd, VectorMatmulNdBackward
from .matrixshapevalidator import MatrixShapeValidator
from .blashandle import BLASHandle, BLASHandleLite, BlasArg, BLASMatmul2dBackward

# ── Neural Network ──
from .net import Linear, LinearBLAS, Profile, Sequential, SequentialBLAS, Module, ModuleList, ModuleListIterator, ReLU, Sigmoid, Tanh, MSELoss, Conv2D, Flatten
from .crossentropy import CrossEntropyLoss, CEClassIndicesForward, CEClassIndicesBackward, CEProbabilitiesForward, CEProbabilitiesBackward, CEValidation, CECommon, ClassIndicesBwdArg, ClassProbabilitiesBwdArg

from .dropout import Dropout, ArgDropout, DropoutBackward
from .cnn import Conv2dFused, FusedIm2Col, FusedIm2ColBwdArg, FusedCol2ImBackward
from .pooling import MaxPool2d, MaxPool2dBackward, MaxPool2dBwdArg
from .filler import Filler
from .accuracy import Accuracy
from .optim import SGD
from .scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from .checkpoint import Checkpoint, save_state, load_state, apply_to_model, save_weights, load_weights, save_best_if_improved, save_step_checkpoint
from .dataloader import DataLoader, Dataset, Batch, TensorDataset, NumpyDataset

# ── Device ──
from .device import Device, CPU, GPU, DeviceState
from .device_transfer import DeviceTransfer, DeviceTransferBackward, DeviceTransferBwdArg, Flow

# ── Autograd ──
from .backpropagation import Backward, BackwardFnArg, ArgumentType, NullArg, Boolean, ScalarArg, Integer, IntArrayArg, BufferArg, NDBufferArg, ViewArg, make_destroyer, make_copier
from .argminmax import Argmin, Argmax, ArgMinMaxReducer
from .minmax_reducer import MinMaxReducer

# ── Utilities ──
from .common_utils import now, panic, id, addr, addrs, copy, is_null, IDGen, Epsilon, One, Zero, inf, isinf, isnan, nan, do_assert, assert_grad, NewAxis, i, il, s, Slicer, str_repeat, print_summary, print_buffer, download, pystr, binary_accuracy, multiclass_accuracy, log_debug, log_info, log_warning
from .indexhelper import IndexCalculator, IndexIterator
from .validators import Validator
from .numpy_interop import to_ndarray, from_ndarray, save, load, save_checkpoint, load_checkpoint, numpy_dtype, list_to_tuple, ndarray_ptr, as_nested_list, test_to_ndarray
from .named_parameter import NamedParameter

# ── Stats ──
from .variance import Variance, VarianceBackward, VarianceBwdArg
from .std_deviation import StdDev, StdBackward, StdBwdArg
from .welford import Welford

# ── Mnemonics (op codes) ──
from .mnemonics import (
    Noop, MulTensor, AddTensor, SubtractTensor, ZeroGrad, ScatterAddTensor,
    Add, Subtract, ReverseSubtract, Multiply, Divide, ReverseDivide,
    Equal, NotEqual, LessThan, LessThanEqual, GreaterThan, GreaterThanEqual, Overwrite,
    RELU_FORWARD, SQRT, SQRT_BACKWARD, LOG, vm, mv, mm, invalid,
    LINEAR, LINEAR_BLAS, RELU, SIGMOID, TANH, DROPOUT, CONV2D, FLATTEN, MAXPOOL2D, LAYER_NORM, EMBEDDING,
    max_rank, EXP, NEGATE, ABS, MAX, MIN, POW,
    TANH_FORWARD, SIGMOID_FORWARD, SIGMOID_BACKWARD, TANH_BACKWARD, LOG_BACKWARD,
    INVERT, SUM, MEAN, PRODUCT, ABS_BACKWARD, DEFAULT_INDEX_DTYPE,
)

# ── GPU Kernels ──
from .kernels.binary_inplace_ops_kernel import BinaryInplaceOperations
from .kernels.binary_ops_kernel import BinaryOperations, arithmetic_ops_both_contiguous, arithmetic_ops_both_contiguous_broadcast, arithmetic_ops_A_contiguous, arithmetic_ops_A_contiguous_lastdim_contiguous_B, arithmetic_ops_B_contiguous, arithmetic_ops_both_strided
from .kernels.compare_kernel import Compare, CompareScalar, compare, compare_scalar, AllClose, all_close, atomic_and
from .kernels.matmul_kernel import MatmulNdGpu, matmul_2d_tiled
from .kernels.matrixvector_kernel import MatrixVectorNdGpu, matrix_vector_nd
from .kernels.vectormatrix_kernel import VectorMatmulNdGpu, vector_matmul_nd
from .kernels.minmax_kernel import ReductionMinMax, reduce_minmax, build_minmax_mask, output_to_input_base, reduction_idx_to_reduced_offset
from .shared import Reduction
from .kernels.reduction_kernel import reduce, product_reduce, excl_product_kernel, log_sum_exp_f32, log_sum_exp_f64, ProductArg, welford_reduce
from .kernels.scalar_inplace_ops_kernel import InplaceScalarOperations, inplace_scalar_ops, inplace_pow_op_f32, inplace_pow_op_f64, inplace_scalar_ops_strided, inplace_pow_op_f32_strided, inplace_pow_op_f64_strided
from .kernels.scalar_ops_kernel import ScalarOperations, scalar_ops, scalar_ops_strided, pow_op_f32, pow_op_f64
from .kernels.unary_ops_kernel import UnaryOpsKernel, unary_ops, float_unary_ops, unary_ops_with_mask, invert_bool


def zeros[dtype: DType = DType.float32](s: Shape) -> Tensor[dtype]:
    return Tensor[dtype].zeros(s)


def zeros[dtype: DType = DType.float32](*indices: Int) -> Tensor[dtype]:
    return Tensor[dtype].zeros(Shape(indices))


def empty[dtype: DType = DType.float32](s: Shape) -> Tensor[dtype]:
    return Tensor[dtype].empty(s)


def empty[dtype: DType = DType.float32](*indices: Int) -> Tensor[dtype]:
    return Tensor[dtype].empty(Shape(indices))


def dot[
    dtype: DType = DType.float32
](a: Tensor[dtype], b: Tensor[dtype]) -> Tensor[dtype]:
    return a.dot(b)


def randint(low: Int, high: Int, shape: List[Int]) -> List[Int]:
    var indexer = Tensor[DType.int64].rand(
        shape, low=Int64(low), high=Int64(high)
    )
    return [Int(index) for _, index in indexer]
