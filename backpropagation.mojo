from tenmo import Tensor
from mnemonics import AddTensor, SubtractTensor
from utils import Variant
from walkback import *
from common_utils import panic
from gradbox import Gradbox
from buffers import Buffer
from ndbuffer import NDBuffer

# Centralized backward operation tags

comptime BACKWARD_ADD = 0
comptime BACKWARD_MULTIPLY = 1
comptime BACKWARD_RELU = 2
comptime BACKWARD_MATMUL_ND = 3
comptime BACKWARD_MATMUL_2D = 4
comptime BACKWARD_TRANSPOSE = 5
comptime BACKWARD_PERMUTE = 6
comptime BACKWARD_SIGMOID = 7
comptime BACKWARD_ADD_BROADCAST = 8
comptime BACKWARD_MULTIPLY_BROADCAST = 9
comptime BACKWARD_SOFTMAX = 10
comptime BACKWARD_CROSS_ENTROPY = 11
comptime BACKWARD_TANH = 12
comptime BACKWARD_SUB = 13
comptime BACKWARD_RESHAPE = 14
comptime BACKWARD_VIEW = 15
comptime BACKWARD_MEAN = 16
comptime BACKWARD_SUM = 17
comptime BACKWARD_LOG_SOFTMAX = 18
comptime BACKWARD_CONTIGUOUS = 19
comptime BACKWARD_DIVIDE = 20
comptime BACKWARD_MATRIX_VECTOR_MUL = 21
comptime BACKWARD_VECTOR_MATMUL = 22
comptime BACKWARD_ADD_SCALAR = 23
comptime BACKWARD_MULTIPLY_SCALAR = 24
comptime BACKWARD_SUB_SCALAR = 25
comptime BACKWARD_DIV_SCALAR = 26
comptime BACKWARD_RIGHT_DIV_SCALAR = 27
comptime BACKWARD_EXPONENTIATION = 28
comptime BACKWARD_DOT = 29
comptime BACKWARD_EXPAND = 30
comptime BACKWARD_FLATTEN = 31
comptime BACKWARD_SQUEEZE = 32
comptime BACKWARD_UNSQUEEZE = 33
comptime BACKWARD_SHUFFLE = 34
comptime BACKWARD_MINMAX = 35
comptime BACKWARD_TILE = 36
comptime BACKWARD_LOG = 37
comptime BACKWARD_SQRT = 38
comptime BACKWARD_CLIP = 39
comptime BACKWARD_VARIANCE = 40
comptime BACKWARD_STD = 41
comptime BACKWARD_SUBTRACT_BROADCAST = 42
comptime BLAS_BACKWARD_MATMUL_2D = 43
comptime BACKWARD_CONCAT = 44
comptime BACKWARD_STACK = 45
comptime BACKWARD_PAD = 46
comptime BACKWARD_FUSED_CONV = 47
comptime BACKWARD_MAXPOOL2D = 48
comptime BACKWARD_DROPOUT = 49
comptime BACKWARD_EXPONENTIAL = 50
comptime BACKWARD_DEVICE_TRANSFER = 51
comptime BACKWARD_MINMAX_GPU = 52
comptime BACKWARD_MAX_SCALAR = 53
comptime BACKWARD_MIN_SCALAR = 54
# ========== Delegate (Variant) ==========

comptime Delegate[dtype: DType] = Variant[
    MatmulNdBackward[dtype],
    Matmul2dBackward[dtype],
    BLASMatmul2dBackward[dtype],
    ReLUBackward[dtype],
    AddBackwardScalar[dtype],
    AddBackward[dtype],
    AddBroadcastBackward[dtype],
    SubBackward[dtype],
    SubtractBroadcastBackward[dtype],
    SumBackward[dtype],
    MeanBackward[dtype],
    ReshapeBackward[dtype],
    ViewBackward[dtype],
    TransposeBackward[dtype],
    CrossEntropyBackward[dtype],
    ContiguousBackward[dtype],
    MultiplyBackwardScalar[dtype],
    MultiplyBackward[dtype],
    MultiplyBroadcastBackward[dtype],
    SigmoidBackward[dtype],
    VectorMatmulNdBackward[dtype],
    MatrixVectorMulNdBackward[dtype],
    SubLeftRightBackwardScalar[dtype],
    ExponientionBackward[dtype],
    TrueDivBackwardScalar[dtype],
    RightTrueDivBackwardScalar[dtype],
    DivideBackward[dtype],
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

    fn __moveinit__(out self, deinit other: Self):
        self.grad_fn = other.grad_fn^
        self.tag = other.tag

    fn __copyinit__(out self, other: Self):
        self.grad_fn = other.grad_fn.copy()
        self.tag = other.tag

    fn __call__(
        self, output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        """O(1) dispatch using integer tag comparison.
        Order: Most common operations first for branch prediction.
        """

        # ========== TIER 1: MOST COMMON ==========
        if self.tag == BACKWARD_ADD:
            return self.grad_fn[AddBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_MULTIPLY:
            return self.grad_fn[MultiplyBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_MULTIPLY_SCALAR:
            return self.grad_fn[MultiplyBackwardScalar[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_RELU:
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
        elif self.tag == BACKWARD_ADD_BROADCAST:
            return self.grad_fn[AddBroadcastBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_MULTIPLY_BROADCAST:
            return self.grad_fn[MultiplyBroadcastBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_SOFTMAX:
            return self.grad_fn[SoftmaxBackward[Self.dtype]].backward(output)
        elif self.tag == BACKWARD_SUB:
            return self.grad_fn[SubBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_ADD_SCALAR:
            return self.grad_fn[AddBackwardScalar[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_SUB_SCALAR:
            return self.grad_fn[
                SubLeftRightBackwardScalar[Self.dtype]
            ].backward(output)

        elif self.tag == BACKWARD_RESHAPE:
            return self.grad_fn[ReshapeBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_VIEW:
            return self.grad_fn[ViewBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_CROSS_ENTROPY:
            return self.grad_fn[CrossEntropyBackward[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_TANH:
            return self.grad_fn[TanhBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_MEAN:
            return self.grad_fn[MeanBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_SUM:
            return self.grad_fn[SumBackward[Self.dtype]].backward(output)

        # ========== TIER 4: MODERATELY COMMON ==========
        elif self.tag == BACKWARD_LOG_SOFTMAX:
            return self.grad_fn[LogSoftmaxBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_CONTIGUOUS:
            return self.grad_fn[ContiguousBackward[Self.dtype]].backward(output)

        elif self.tag == BACKWARD_DIVIDE:
            return self.grad_fn[DivideBackward[Self.dtype]].backward(output)

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

        # ========== TIER 5: SCALAR OPERATIONS ==========
        elif self.tag == BACKWARD_DIV_SCALAR:
            return self.grad_fn[TrueDivBackwardScalar[Self.dtype]].backward(
                output
            )

        elif self.tag == BACKWARD_RIGHT_DIV_SCALAR:
            return self.grad_fn[
                RightTrueDivBackwardScalar[Self.dtype]
            ].backward(output)

        # ========== TIER 6: SPECIALIZED OPERATIONS ==========
        elif self.tag == BACKWARD_EXPONENTIATION:
            return self.grad_fn[ExponientionBackward[Self.dtype]].backward(
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

        elif self.tag == BACKWARD_SUBTRACT_BROADCAST:
            return self.grad_fn[SubtractBroadcastBackward[Self.dtype]].backward(
                output
            )

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


fn main():
    pass
