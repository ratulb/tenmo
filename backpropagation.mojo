from tenmo import Tensor
from mnemonics import AddTensor, SubtractTensor
from std.utils import Variant
from walkback import *
from common_utils import panic
from gradbox import Gradbox
from buffers import Buffer
from ndbuffer import NDBuffer
from intarray import IntArray

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
comptime BACKWARD_MINMAX_GPU = 50
comptime BACKWARD_MAX_SCALAR = 51
comptime BACKWARD_MIN_SCALAR = 52
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

comptime ArgType[dtype: DType] = Variant[
    NullArg,
    ScalarArg[dtype],
    BooleanArg,
    SubtractArg,
    ReductionArgs,
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
    ]:
        """O(1) dispatch using integer tag comparison.
        Order: Most common operations first for branch prediction.
        """

        # ========== TIER 1: MOST COMMON ==========
        ref arg = output.fn_arg()
        var tag = arg.tag
        print("Tag is: ", tag)
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



        else: return []

comptime Delegate[dtype: DType] = Variant[
    MatmulNdBackward[dtype],
    Matmul2dBackward[dtype],
    BLASMatmul2dBackward[dtype],
    ReLUBackward[dtype],
    ViewBackward[dtype],
    TransposeBackward[dtype],
    CEClassIndicesBackward[dtype],
    CEProbabilitiesBackward[dtype],
    ContiguousBackward[dtype],
    SigmoidBackward[dtype],
    VectorMatmulNdBackward[dtype],
    MatrixVectorMulNdBackward[dtype],
    ExponentiationBackward[dtype],
    DotBackward[dtype],
    ExpandBackward[dtype],
    FlattenBackward[dtype],
    SqueezeBackward[dtype],
    UnsqueezeBackward[dtype],
    PermuteBackward[dtype],
    ShuffleBackward[dtype],
    MinMaxBackward[dtype],
    SoftmaxBackward[dtype],
    LogSoftmaxBackward[dtype],
    TileBackward[dtype],
    TanhBackward[dtype],
    LogBackward[dtype],
    ClipBackward[dtype],
    SqrtBackward[dtype],
    VarianceBackward[dtype],
    StdBackward[dtype],
    ConcatBackward[dtype],
    StackBackward[dtype],
    PadBackward[dtype],
    FusedCol2ImBackward[dtype],
    MaxPool2dBackward[dtype],
    DropoutBackward[dtype],
    ExponentialBackward[dtype],
    DeviceTransferBackward[dtype],
    MinMaxBackwardGPU[dtype],
    MaxBackwardScalar[dtype],
    MinBackwardScalar[dtype],
]

# ========== BackwardFn with Tag-Based Dispatch ==========


struct BackwardFn[dtype: DType](Copyable & Movable):
    var grad_fn: Delegate[Self.dtype]
    var tag: Int  # O(1) lookup key

    fn __init__(out self, grad_fn: Delegate[Self.dtype], tag: Int):
        self.grad_fn = grad_fn
        self.tag = tag

    fn __moveinit__(out self, deinit take: Self):
        self.grad_fn = take.grad_fn^
        self.tag = take.tag

    fn __copyinit__(out self, copy: Self):
        self.grad_fn = copy.grad_fn.copy()
        self.tag = copy.tag

    fn __call__(
        self, output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        """O(1) dispatch using integer tag comparison.
        Order: Most common operations first for branch prediction.
        """

        # ========== TIER 1: MOST COMMON ==========

        if self.tag == BACKWARD_RELU:
            return self.grad_fn[ReLUBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_MATMUL_ND:
            return self.grad_fn[MatmulNdBackward[Self.dtype]].backward(output)

        elif self.tag == BLAS_BACKWARD_MATMUL_2D:
            return self.grad_fn[BLASMatmul2dBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_SIGMOID:
            return self.grad_fn[SigmoidBackward[Self.dtype]].backward(output)

        # ========== TIER 2: MATMUL CHAIN (Called by MatmulNd) ==========
        elif self.tag == BACKWARD_MATMUL_2D:
            return self.grad_fn[Matmul2dBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_TRANSPOSE:
            return self.grad_fn[TransposeBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_PERMUTE:
            return self.grad_fn[PermuteBackward[Self.dtype]].backward(output)

        # ========== TIER 3: COMMON OPERATIONS ==========

        elif self.tag == BACKWARD_SOFTMAX:
            return self.grad_fn[SoftmaxBackward[Self.dtype]].backward(output)
        elif self.tag == BACKWARD_VIEW:
            return self.grad_fn[ViewBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_CE_CLASS_INDICES:
            return self.grad_fn[CEClassIndicesBackward[Self.dtype]].backward(
                output
            )
        elif self.tag == BACKWARD_CE_PROBABILITIES:
            return self.grad_fn[CEProbabilitiesBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_TANH:
            return self.grad_fn[TanhBackward[Self.dtype]].backward(output)

        # ========== TIER 4: MODERATELY COMMON ==========
        elif self.tag == BACKWARD_LOG_SOFTMAX:
            return self.grad_fn[LogSoftmaxBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_CONTIGUOUS:
            return self.grad_fn[ContiguousBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_MATRIX_VECTOR_MUL:
            return self.grad_fn[MatrixVectorMulNdBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_VECTOR_MATMUL:
            return self.grad_fn[VectorMatmulNdBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_EXPAND:
            return self.grad_fn[ExpandBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_FLATTEN:
            return self.grad_fn[FlattenBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_SQUEEZE:
            return self.grad_fn[SqueezeBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_UNSQUEEZE:
            return self.grad_fn[UnsqueezeBackward[Self.dtype]].backward(output)


        # ========== TIER 6: SPECIALIZED OPERATIONS ==========
        elif self.tag == BACKWARD_EXPONENTIATION:
            return self.grad_fn[ExponentiationBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_DOT:
            return self.grad_fn[DotBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_LOG:
            return self.grad_fn[LogBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_SQRT:
            return self.grad_fn[SqrtBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_CLIP:
            return self.grad_fn[ClipBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_SHUFFLE:
            return self.grad_fn[ShuffleBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_MINMAX:
            return self.grad_fn[MinMaxBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_MINMAX_GPU:
            return self.grad_fn[MinMaxBackwardGPU[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_TILE:
            return self.grad_fn[TileBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_VARIANCE:
            return self.grad_fn[VarianceBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_STD:
            return self.grad_fn[StdBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_CONCAT:
            return self.grad_fn[ConcatBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_STACK:
            return self.grad_fn[StackBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_PAD:
            return self.grad_fn[PadBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_FUSED_CONV:
            return self.grad_fn[FusedCol2ImBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_MAXPOOL2D:
            return self.grad_fn[MaxPool2dBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_DROPOUT:
            return self.grad_fn[DropoutBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_EXPONENTIAL:
            return self.grad_fn[ExponentialBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_DEVICE_TRANSFER:
            return self.grad_fn[DeviceTransferBackward[Self.dtype]].backward(
                output
            )
        elif self.tag == BACKWARD_MAX_SCALAR:
            return self.grad_fn[MaxBackwardScalar[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_MIN_SCALAR:
            return self.grad_fn[MinBackwardScalar[Self.dtype]].backward(output)

        else:
            panic("BackwardFn: Unknown backward tag: " + String(self.tag))

        return []


fn main() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1, 2, 3], requires_grad=True)
    var s = Tensor[dtype].scalar(42, requires_grad=True)
    var r =  a.reshape(3,1)
    print("r requires_grad? ", r.requires_grad)
    var m = r * 42
    m.backward()
    a.grad().print()
    s.grad().print()

