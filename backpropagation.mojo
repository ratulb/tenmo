from tenmo import Tensor
from std.utils import Variant
from walkback import *
from common_utils import panic
from gradbox import Gradbox
from buffers import Buffer
from ndbuffer import NDBuffer
from intarray import IntArray
from strides import Strides
from shapes import Shape
from device_transfer import Flow
from device import GPU

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
comptime BACKWARD_MULTIPLY_SCALAR = 23
comptime BACKWARD_SUB_SCALAR = 24
comptime BACKWARD_DIV_SCALAR = 25
comptime BACKWARD_RIGHT_DIV_SCALAR = 26
comptime BACKWARD_EXPONENTIATION = 27
comptime BACKWARD_DOT = 28
comptime BACKWARD_EXPAND = 29
comptime BACKWARD_FLATTEN = 30
comptime BACKWARD_SQUEEZE = 31
comptime BACKWARD_UNSQUEEZE = 32
comptime BACKWARD_SHUFFLE = 33
comptime BACKWARD_MINMAX = 34
comptime BACKWARD_TILE = 35
comptime BACKWARD_LOG = 36
comptime BACKWARD_SQRT = 37
comptime BACKWARD_CLIP = 38
comptime BACKWARD_VARIANCE = 39
comptime BACKWARD_STD = 40
comptime BLAS_BACKWARD_MATMUL_2D = 41
comptime BACKWARD_CONCAT = 42
comptime BACKWARD_STACK = 43
comptime BACKWARD_PAD = 44
comptime BACKWARD_FUSED_CONV = 45
comptime BACKWARD_MAXPOOL2D = 46
comptime BACKWARD_DROPOUT = 47
comptime BACKWARD_EXPONENTIAL = 48
comptime BACKWARD_DEVICE_TRANSFER = 49
comptime BACKWARD_MAX_SCALAR = 50
comptime BACKWARD_MIN_SCALAR = 51
# ========== Delegate (Variant) ==========

@fieldwise_init
struct NullArg(RegisterPassable & ImplicitlyCopyable):
    pass

@fieldwise_init
struct ScalarArg[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var scalar: Scalar[Self.dtype]

@fieldwise_init
struct BooleanArg(RegisterPassable & ImplicitlyCopyable):
    var is_true: Bool

@fieldwise_init
struct SubtractArg(RegisterPassable, ImplicitlyCopyable):
    var signs: IntArray

    fn __init__(out self):
        self.signs = IntArray()

    fn __copyinit__(out self, copy: Self):
        self.signs = copy.signs.copy()

    fn negate(mut self, neg: Bool):
        if neg:
            self.signs.append(1)
        else:
            self.signs.append(0)

    fn into_arg[dtype: DType](self) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), BACKWARD_SUB)

@fieldwise_init
struct ReductionArgs(RegisterPassable, ImplicitlyCopyable):
    var axes: IntArray
    var keepdims: Bool
    fn into_arg[dtype: DType](self, tag: Int) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), tag) # BACKWARD_SUM/BACKWARD_MEAN

@fieldwise_init
struct IntArrayArg(RegisterPassable, ImplicitlyCopyable):
    var axes: IntArray
    fn into_arg[dtype: DType](self, tag: Int) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), tag)

@fieldwise_init
struct IntArg(RegisterPassable, ImplicitlyCopyable):
    var value: Int
    fn into_arg[dtype: DType](self, tag: Int) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), tag)


@fieldwise_init
struct BufferArg[dtype: DType](ImplicitlyCopyable & Movable):
    var buffer: Buffer[Self.dtype]

    fn into_arg(self, tag: Int) -> FnArg[Self.dtype]:
        return FnArg[Self.dtype](ArgType[Self.dtype](self), tag)

@fieldwise_init
struct ViewArg(RegisterPassable, ImplicitlyCopyable):
    var shape: Shape
    var strides: Strides
    var offset: Int

    fn into_arg[dtype: DType](self, tag: Int) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), tag)

@fieldwise_init
struct ShuffleArg(ImplicitlyCopyable & Movable):
    var axis: Int
    var permutation: List[Int]

    fn __copyinit__(out self, copy: Self):
        self.axis = copy.axis
        self.permutation = copy.permutation.copy()

    fn __moveinit__(out self, deinit take: Self):
        self.axis = take.axis
        self.permutation = take.permutation^

    fn into_arg[dtype: DType](self) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), BACKWARD_SHUFFLE)


@fieldwise_init
struct MinMaxArg[dtype: DType](ImplicitlyCopyable & Movable):
    var axes: IntArray
    var keepdims: Bool
    var mask: NDBuffer[Self.dtype]  # shape == ancestor.shape, contiguous

    fn into_arg(self) -> FnArg[Self.dtype]:
        return FnArg[Self.dtype](ArgType[Self.dtype](self), BACKWARD_MINMAX)

@fieldwise_init
struct SoftmaxArg[dtype: DType](ImplicitlyCopyable & Movable):
    var axes: IntArray
    var softmax_out: NDBuffer[Self.dtype]

    fn into_arg(self, tag: Int) -> FnArg[Self.dtype]:
        return FnArg[Self.dtype](ArgType[Self.dtype](self), tag)

@fieldwise_init
struct TileArg(RegisterPassable, ImplicitlyCopyable):
    var repeat: IntArray
    var orig_shape: Shape

    fn into_arg[dtype: DType](self) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), BACKWARD_TILE)

@fieldwise_init
struct ClipArgs[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    var min_val: Scalar[Self.dtype]
    var max_val: Scalar[Self.dtype]
    fn into_arg(self) -> FnArg[Self.dtype]:
        return FnArg[Self.dtype](ArgType[Self.dtype](self), BACKWARD_CLIP)

@fieldwise_init
struct VarianceArgs[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool  # Track if user wanted keepdims

    fn into_arg(self) -> FnArg[Self.dtype]:
        return FnArg[Self.dtype](ArgType[Self.dtype](self), BACKWARD_VARIANCE)

@fieldwise_init
struct StdArgs[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool
    var epsilon: Scalar[Self.dtype]

    fn into_arg(self) -> FnArg[Self.dtype]:
        return FnArg[Self.dtype](ArgType[Self.dtype](self), BACKWARD_STD)

@fieldwise_init
struct StackArgs(RegisterPassable, ImplicitlyCopyable):
    var axis: Int
    var num_tensors: Int

    fn into_arg[dtype: DType](self) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), BACKWARD_STACK)


@fieldwise_init
struct PadArgs(ImplicitlyCopyable & Movable):

    var pad: List[Tuple[Int, Int]]
    var mode: String

    fn __copyinit__(out self, copy: Self):
        self.pad = copy.pad.copy()
        self.mode = copy.mode

    fn __moveinit__(out self, deinit take: Self):
        self.pad = take.pad.copy()
        self.mode = take.mode

    fn into_arg[dtype: DType](self) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), BACKWARD_PAD)

@fieldwise_init
struct FusedCol2ImArgs(RegisterPassable, ImplicitlyCopyable):
    var N: Int
    var C_in: Int
    var H_pad: Int
    var W_pad: Int
    var C_out: Int
    var KH: Int
    var KW: Int
    var H_out: Int
    var W_out: Int
    var stride: Int
    var dilation: Int

    fn into_arg[dtype: DType](self) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), BACKWARD_FUSED_CONV)

@fieldwise_init
struct MaxPool2dArgs(ImplicitlyCopyable & Movable):
    comptime TAG = BACKWARD_MAXPOOL2D
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var input_shape: Shape
    var argmax_mask: NDBuffer[DType.int64]

    fn into_arg[dtype: DType](self) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), BACKWARD_MAXPOOL2D)


struct DeviceTransferArgs(ImplicitlyCopyable):
    var flow: Flow
    var gpu: Optional[GPU]

    fn __init__(out self):
        self.flow = Flow.UnMoved
        self.gpu = None

    fn __init__(out self, flow: Flow):
        self.flow = flow
        self.gpu = None

    fn __init__(out self, flow: Flow, gpu: GPU):
        self.flow = flow
        self.gpu = gpu

    fn __copyinit__(out self, copy: Self):
        self.flow = copy.flow.copy()
        self.gpu = copy.gpu.copy()

    fn __moveinit__(out self, deinit take: Self):
        self.flow = take.flow
        self.gpu = take.gpu^

    fn into_arg[dtype: DType](self) -> FnArg[dtype]:
        return FnArg[dtype](ArgType[dtype](self), BACKWARD_DEVICE_TRANSFER)


comptime ArgType[dtype: DType] = Variant[
    NullArg,
    ScalarArg[dtype],
    BooleanArg,
    SubtractArg,
    ReductionArgs,
    BLASMatmul2dBwdArg[dtype],
    BufferArg[dtype],
    ViewArg,
    IntArrayArg,
    CEClassIndicesArg[dtype],
    CEProbabilitiesArg[dtype],
    ShuffleArg,
    MinMaxArg[dtype],
    SoftmaxArg[dtype],
    TileArg,
    ClipArgs[dtype],
    VarianceArgs[dtype],
    StdArgs[dtype],
    IntArg,
    StackArgs,
    PadArgs,
    FusedCol2ImArgs,
    MaxPool2dArgs,
    DeviceTransferArgs,
]

struct FnArg[dtype: DType](Copyable & Movable):
    var arg: ArgType[Self.dtype]
    var tag: Int  # O(1) dispatch key

    fn __init__(out self, var arg: ArgType[Self.dtype], tag: Int):
        self.arg= arg^
        self.tag = tag

    fn __moveinit__(out self, deinit take: Self):
        self.arg = take.arg^
        self.tag = take.tag

    fn __copyinit__(out self, copy: Self):
        self.arg = copy.arg.copy()
        self.tag = copy.tag

    @staticmethod
    fn null(tag: Int) -> FnArg[Self.dtype]:
        return FnArg[Self.dtype](ArgType[Self.dtype](NullArg()), tag)

    @staticmethod
    fn scalar(scalar: Scalar[Self.dtype], tag: Int) -> FnArg[Self.dtype]:
        return FnArg[Self.dtype](ArgType[Self.dtype](ScalarArg[Self.dtype](scalar)), tag)

    @staticmethod
    fn boolean(is_true: Bool, tag: Int) -> FnArg[Self.dtype]:
        return FnArg[Self.dtype](ArgType[Self.dtype](BooleanArg(is_true)), tag)

@fieldwise_init
struct Backward[dtype: DType](RegisterPassable & ImplicitlyCopyable):

    @staticmethod
    fn invoke(
            output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        """O(1) dispatch using integer tag comparison.
        Order: Most common operations first for branch prediction.
        """

        # ========== TIER 1: MOST COMMON ==========
        ref arg = output.fn_arg()
        var tag = arg.tag
        if tag == BACKWARD_ADD:
            return AddBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_ADD_SCALAR:
            return AddBackwardScalar[Self.dtype].backward(output)
        elif tag == BACKWARD_MULTIPLY_SCALAR:
            return MultiplyBackwardScalar[Self.dtype].backward(output)
        elif tag == BACKWARD_MULTIPLY:
            return MultiplyBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_SUB:
            return SubBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_SUB_SCALAR:
            return SubLeftRightBackwardScalar[Self.dtype].backward(output)
        elif tag == BACKWARD_DIV_SCALAR:
            return TrueDivBackwardScalar[Self.dtype].backward(output)
        elif tag == BACKWARD_RIGHT_DIV_SCALAR:
            return RightTrueDivBackwardScalar[Self.dtype].backward(output)
        elif tag == BACKWARD_DIVIDE:
            return DivideBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_SUM:
            return SumBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_MEAN:
            return MeanBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_RESHAPE:
            return ReshapeBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_MATMUL_2D:
            return Matmul2dBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_MATMUL_ND:
            return MatmulNdBackward[Self.dtype].backward(output)
        elif tag == BLAS_BACKWARD_MATMUL_2D:
            return BLASMatmul2dBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_RELU:
            return ReLUBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_VIEW:
            return ViewBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_TRANSPOSE:
            return TransposeBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_CE_CLASS_INDICES:
            return CEClassIndicesBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_CE_PROBABILITIES:
            return CEProbabilitiesBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_CONTIGUOUS:
            return ContiguousBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_SIGMOID:
            return SigmoidBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_VECTOR_MATMUL:
            return VectorMatmulNdBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_MATRIX_VECTOR_MUL:
            return MatrixVectorMulNdBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_EXPONENTIATION:
            return ExponentiationBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_DOT:
            return DotBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_EXPAND:
            return ExpandBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_FLATTEN:
            return FlattenBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_SQUEEZE:
            return SqueezeBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_UNSQUEEZE:
            return UnsqueezeBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_PERMUTE:
            return PermuteBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_SHUFFLE:
            return ShuffleBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_MINMAX:
            return MinMaxBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_SOFTMAX:
            return SoftmaxBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_LOG_SOFTMAX:
            return LogSoftmaxBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_TILE:
            return TileBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_TANH:
            return TanhBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_LOG:
            return LogBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_CLIP:
            return ClipBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_SQRT:
            return SqrtBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_VARIANCE:
            return VarianceBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_STD:
            return StdBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_CONCAT:
            return ConcatBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_STACK:
            return StackBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_PAD:
            return PadBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_FUSED_CONV:
            return FusedCol2ImBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_MAXPOOL2D:
            return MaxPool2dBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_DROPOUT:
            return DropoutBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_EXPONENTIAL:
            return ExponentialBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_DEVICE_TRANSFER:
            return DeviceTransferBackward[Self.dtype].backward(output)
        elif tag == BACKWARD_MAX_SCALAR:
            return MaxBackwardScalar[Self.dtype].backward(output)
        elif tag == BACKWARD_MIN_SCALAR:
            return MinBackwardScalar[Self.dtype].backward(output)

        else: return []


fn main() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1, 0, 3]]], requires_grad=True)
    var b = Tensor[dtype].d2([[1], [2], [3]], requires_grad=True)
    #var r =  a.reshape(3,1)
    #print("r requires_grad? ", r.requires_grad)
    var v = a.into_view()
    var m = v.sigmoid()
    m.backward()
    a.grad().print()
    b.grad().print()
    m.print()

    var A = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var B = A.transpose(-1, 0)
    var C = B * 89
    C.backward()
    A.grad().print()
