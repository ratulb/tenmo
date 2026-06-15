"""Backpropagation — Autograd Dispatch and Backward Operations.

This module provides the core infrastructure for Tenmo's autograd system:

1. **Operation tags** — 56 compile-time constants (BACKWARD_*) for dispatch
2. **Type-erased arguments** — BackwardFnArg for passing operation-specific data
3. **Backward dispatcher** — Backward.invoke() jump table to backward implementations


Related:
  - [README_AUTOGRAD.md](https://github.com/ratulb/tenmo/blob/document/README_AUTOGRAD.md) — Full autograd architecture
  - [ancestry.mojo](https://github.com/ratulb/tenmo/blob/document/tenmo/ancestry.mojo) — Ancestor and Ancestors types
  - [gradbox.mojo](https://github.com/ratulb/tenmo/blob/document/tenmo/gradbox.mojo) — Gradient storage with refcounting
"""

from .ancestry import Ancestor
from .tensor import Tensor
from std.utils import Variant
from .walkback import *
from .common_utils import panic
from .gradbox import Gradbox
from .buffers import Buffer
from .ndbuffer import NDBuffer
from .intarray import IntArray
from .strides import Strides
from .shapes import Shape
from .blashandle import BLASHandleLite
from tenmo.shared import Reduction

# Centralized backward operation tags

comptime BACKWARD_ADD = 0
comptime BACKWARD_MULTIPLY = 1
comptime BACKWARD_RELU = 2
comptime BACKWARD_MATMUL_ND = 3
comptime BACKWARD_MATMUL_2D = 4
comptime BACKWARD_TRANSPOSE = 5
comptime BACKWARD_PERMUTE = 6
comptime BACKWARD_SIGMOID = 7
comptime BACKWARD_SOFTMAX = 8
comptime BACKWARD_CE_CLASS_INDICES = 9
comptime BACKWARD_CE_PROBABILITIES = 10
comptime BACKWARD_TANH = 11
comptime BACKWARD_SUB = 12
comptime BACKWARD_RESHAPE = 13
comptime BACKWARD_VIEW = 14
comptime BACKWARD_MEAN = 15
comptime BACKWARD_SUM = 16
comptime BACKWARD_LOG_SOFTMAX = 17
comptime BACKWARD_CONTIGUOUS = 18
comptime BACKWARD_DIVIDE = 19
comptime BACKWARD_MATRIX_VECTOR_MUL = 20
comptime BACKWARD_VECTOR_MATMUL = 21
comptime BACKWARD_ADD_SCALAR = 22
comptime BACKWARD_ADD_BROADCAST = 23
comptime BACKWARD_MULTIPLY_SCALAR = 24
comptime BACKWARD_SUB_SCALAR = 25
comptime BACKWARD_SUBTRACT_BROADCAST = 26
comptime BACKWARD_DIV_SCALAR = 27
comptime BACKWARD_RIGHT_DIV_SCALAR = 28
comptime BACKWARD_EXPONENTIATION = 29
comptime BACKWARD_DOT = 30
comptime BACKWARD_EXPAND = 31
comptime BACKWARD_FLATTEN = 32
comptime BACKWARD_SQUEEZE = 33
comptime BACKWARD_UNSQUEEZE = 34
comptime BACKWARD_SHUFFLE = 35
comptime BACKWARD_MINMAX = 36
comptime BACKWARD_TILE = 37
comptime BACKWARD_LOG = 38
comptime BACKWARD_SQRT = 39
comptime BACKWARD_CLIP = 40
comptime BACKWARD_VARIANCE = 41
comptime BACKWARD_STD = 42
comptime BLAS_BACKWARD_MATMUL_2D = 43
comptime BACKWARD_CONCAT = 44
comptime BACKWARD_STACK = 45
comptime BACKWARD_PAD = 46
comptime BACKWARD_FUSED_CONV = 47
comptime BACKWARD_MAXPOOL2D = 48
comptime BACKWARD_DROPOUT = 49
comptime BACKWARD_EXPONENTIAL = 50
comptime BACKWARD_DEVICE_TRANSFER = 51
comptime BACKWARD_MAX_SCALAR = 52
comptime BACKWARD_MIN_SCALAR = 53
comptime BACKWARD_MULTIPLY_BROADCAST = 54
comptime BACKWARD_PRODUCT = 55
comptime BACKWARD_LAYER_NORM = 56
comptime BACKWARD_BROADCAST_TO = 57
comptime BACKWARD_GATHER = 58
comptime BACKWARD_BCE_WITH_LOGITS = 59
comptime BACKWARD_BCE = 60
comptime BACKWARD_ABS = 61
comptime BACKWARD_TRIL = 62


trait ArgumentType(ImplicitlyCopyable & Movable):
    pass


comptime DestroyerFn = def(UnsafePointer[UInt8, MutAnyOrigin]) thin -> None


def make_destroyer[T: ArgumentType]() -> DestroyerFn:
    def destroy(p: UnsafePointer[UInt8, MutAnyOrigin]) -> None:
        p.bitcast[T]().destroy_pointee()
        p.bitcast[T]().free()

    return destroy


comptime CopyFn = def(UnsafePointer[UInt8, MutAnyOrigin]) thin -> UnsafePointer[
    UInt8, MutAnyOrigin
]


def make_copier[T: ArgumentType]() -> CopyFn:
    def copy_it(
        src: UnsafePointer[UInt8, MutAnyOrigin]
    ) -> UnsafePointer[UInt8, MutAnyOrigin]:
        var dst = alloc[T](1)
        dst.init_pointee_copy(src.bitcast[T]()[])
        return dst.bitcast[UInt8]()

    return copy_it


# ============================================================================
# BackwardFnArg — Type-Erased Container
# ============================================================================
# Type-erased container for backward operation arguments.
# Stores op_code, type-erased ptr, destroy, and copy_fn.
# ============================================================================


@fieldwise_init
struct BackwardFnArg[dtype: DType](ImplicitlyCopyable & Movable):
    var op_code: Int
    var ptr: UnsafePointer[UInt8, MutAnyOrigin]  # type-erased arg
    var destroy: DestroyerFn
    var copy_fn: CopyFn
    var needs_parent_data: Bool

    def __init__[T: ArgumentType, //](out self, op_code: Int, var arg: T):
        var p = alloc[T](1)
        p.init_pointee_move(arg^)
        self.op_code = op_code
        self.ptr = p.bitcast[UInt8]()
        self.destroy = make_destroyer[T]()
        self.copy_fn = make_copier[T]()
        self.needs_parent_data = False

    def __del__(deinit self):
        self.destroy(self.ptr)  # calls T.__del__

    def __init__(out self, deinit existing: Self):
        self.op_code = existing.op_code
        self.ptr = existing.ptr
        self.destroy = existing.destroy
        self.copy_fn = existing.copy_fn
        self.needs_parent_data = existing.needs_parent_data

    def __init__(out self, *, copy: Self):
        self.op_code = copy.op_code
        self.destroy = copy.destroy
        self.copy_fn = copy.copy_fn
        self.ptr = self.copy_fn(copy.ptr)  # deep copy via T.__init__
        self.needs_parent_data = copy.needs_parent_data

    def get[T: ArgumentType](ref self) -> ref[self.ptr] T:
        return self.ptr.bitcast[T]()[]

    @staticmethod
    def null_arg(op_code: Int) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, NullArg(0))

    @staticmethod
    def boolean_arg(op_code: Int, is_true: Bool) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, Boolean(is_true))

    @staticmethod
    def scalar_arg(
        op_code: Int, value: Scalar[Self.dtype]
    ) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, ScalarArg[Self.dtype](value))

    @staticmethod
    def integer_arg(op_code: Int, value: Int) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, Integer(value))

    @staticmethod
    def from_intarray(
        op_code: Int, array: IntArray
    ) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, IntArrayArg(array))

    @staticmethod
    def from_buffer(
        op_code: Int, buffer: Buffer[Self.dtype]
    ) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, BufferArg[Self.dtype](buffer))

    @staticmethod
    def from_ndbuffer(
        op_code: Int, ndb: NDBuffer[Self.dtype]
    ) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, NDBufferArg[Self.dtype](ndb))


# ============================================================================
# Argument Payload Types
# ============================================================================
# NullArg: Empty payload (used by BACKWARD_ADD, BACKWARD_MULTIPLY)
# Boolean: Bool (used by BACKWARD_DROPOUT)
# ScalarArg: Scalar value (used by *_SCALAR ops)
# ============================================================================


@fieldwise_init
struct NullArg(ArgumentType):
    var zero: UInt8


# Boolean: Bool (used by BACKWARD_DROPOUT)


@fieldwise_init
struct Boolean(ArgumentType):
    var is_true: Bool


# ScalarArg: Scalar value (used by *_SCALAR ops)


@fieldwise_init
struct ScalarArg[dtype: DType](ArgumentType):
    var value: Scalar[Self.dtype]


@fieldwise_init
struct Integer(ArgumentType):
    var value: Int


@fieldwise_init
struct IntArrayArg(ArgumentType):
    var array: IntArray


@fieldwise_init
struct ReductionArg(ArgumentType):
    var axes: IntArray
    var keepdims: Bool


@fieldwise_init
struct BlasArg[dtype: DType](ArgumentType):
    var transpose_A: Bool
    var transpose_B: Bool
    var blas: BLASHandleLite[Self.dtype]


@fieldwise_init
struct StackArg(ArgumentType):
    var axis: Int
    var num_tensors: Int


@fieldwise_init
struct BufferArg[dtype: DType](ArgumentType):
    var buffer: Buffer[Self.dtype]


@fieldwise_init
struct NDBufferArg[dtype: DType](ArgumentType):
    var ndb: NDBuffer[Self.dtype]


@fieldwise_init
struct ViewArg(ArgumentType):
    var shape: Shape
    var strides: Strides
    var offset: Int


@fieldwise_init
struct ShuffleArg(ArgumentType):
    var axis: Int
    var permutation: List[Int]

    def __init__(out self, *, copy: Self):
        self.axis = copy.axis
        self.permutation = copy.permutation.copy()


@fieldwise_init
struct PadArg(ArgumentType):
    var pad: List[Tuple[Int, Int]]
    var mode: String

    def __init__(out self, *, copy: Self):
        self.pad = copy.pad.copy()
        self.mode = copy.mode.copy()


@fieldwise_init
struct MinMaxArg[dtype: DType](ArgumentType):
    var axes: IntArray
    var keepdims: Bool
    var mask: NDBuffer[Self.dtype]


@fieldwise_init
struct SoftmaxArg[dtype: DType](ArgumentType):
    var axes: IntArray
    var softmax_out: NDBuffer[Self.dtype]


@fieldwise_init
struct ClipArg[dtype: DType](ArgumentType):
    var min_val: Scalar[Self.dtype]
    var max_val: Scalar[Self.dtype]


@fieldwise_init
struct TilesArg(ArgumentType):
    var repeat: IntArray
    var orig_shape: Shape


@fieldwise_init
struct StdArg[dtype: DType](ArgumentType):
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool
    var epsilon: Scalar[Self.dtype]


@fieldwise_init
struct BCEWithLogitsBwdArg[dtype: DType](ArgumentType):
    var sigmoid: NDBuffer[Self.dtype]
    var target: NDBuffer[Self.dtype]
    var reduction: Reduction
    var numels: Int


@fieldwise_init
struct BCELossBwdArg[dtype: DType](ArgumentType):
    var clipped_pred: NDBuffer[Self.dtype]
    var target: NDBuffer[Self.dtype]
    var reduction: Reduction
    var numels: Int


@fieldwise_init
struct TrilArg(ArgumentType):
    var diagonal: Int
    var M: Int
    var N: Int


# ============================================================================
# Backward — Jump Table Dispatcher
# ============================================================================
# Reads op_code from Ancestor's backwardFnArg and dispatches to backward struct.
# Each branch calls a static backward() method from its module.
# ============================================================================


@fieldwise_init
struct Backward[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    @staticmethod
    def invoke(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ) where Self.dtype.is_floating_point():
        if not output.has_ancestry():
            print("Inside Backward invoke: output ancestry is not set")
            return
        ref arg = output.ancestry().backward_fn_arg()
        var op_code = arg.op_code
        if op_code == BACKWARD_ADD_SCALAR:
            AddBackwardScalar[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_SUM:
            SumBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_MEAN:
            MeanBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_RESHAPE:
            ReshapeBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_TRANSPOSE:
            TransposeBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_PERMUTE:
            PermuteBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_RELU:
            ReLUBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_VIEW:
            ViewBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_CE_CLASS_INDICES:
            CEClassIndicesBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_CE_PROBABILITIES:
            CEProbabilitiesBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_CONTIGUOUS:
            ContiguousBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_SIGMOID:
            SigmoidBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_EXPONENTIATION:
            ExponentiationBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_EXPAND:
            ExpandBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_FLATTEN:
            FlattenBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_SQUEEZE:
            SqueezeBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_UNSQUEEZE:
            UnsqueezeBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_SHUFFLE:
            ShuffleBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_MINMAX:
            MinMaxBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_SOFTMAX:
            SoftmaxBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_LOG_SOFTMAX:
            LogSoftmaxBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_TILE:
            TileBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_TANH:
            TanhBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_LOG:
            LogBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_CLIP:
            ClipBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_SQRT:
            SqrtBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_VARIANCE:
            VarianceBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_STD:
            StdBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_PAD:
            PadBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_MAXPOOL2D:
            MaxPool2dBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_DROPOUT:
            DropoutBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_DEVICE_TRANSFER:
            DeviceTransferBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_MAX_SCALAR:
            MaxBackwardScalar[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_MIN_SCALAR:
            MinBackwardScalar[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_EXPONENTIAL:
            ExponentialBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_PRODUCT:
            ProductBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_BROADCAST_TO:
            BroadcastToBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_GATHER:
            GatherBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_SUB_SCALAR:
            SubLeftRightBackwardScalar[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_MULTIPLY_SCALAR:
            MultiplyBackwardScalar[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_DIV_SCALAR:
            TrueDivBackwardScalar[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_RIGHT_DIV_SCALAR:
            RightTrueDivBackwardScalar[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_ADD:
            AddBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_ADD_BROADCAST:
            AddBroadcastBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_SUB:
            SubBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_SUBTRACT_BROADCAST:
            SubtractBroadcastBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_MULTIPLY:
            MultiplyBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_MULTIPLY_BROADCAST:
            MultiplyBroadcastBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_DIVIDE:
            DivideBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_MATMUL_2D:
            Matmul2dBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_MATMUL_ND:
            MatmulNdBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BLAS_BACKWARD_MATMUL_2D:
            BLASMatmul2dBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_VECTOR_MATMUL:
            VectorMatmulNdBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_MATRIX_VECTOR_MUL:
            MatrixVectorMulNdBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_DOT:
            DotBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_FUSED_CONV:
            FusedCol2ImBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_LAYER_NORM:
            LayerNormBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_CONCAT:
            ConcatBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_STACK:
            StackBackward[Self.dtype].backward(output, parent_ids, retain_graph)
        elif op_code == BACKWARD_BCE_WITH_LOGITS:
            BCEWithLogitsBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_BCE:
            BCELossBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_ABS:
            AbsBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
        elif op_code == BACKWARD_TRIL:
            TrilBackward[Self.dtype].backward(
                output, parent_ids, retain_graph
            )
