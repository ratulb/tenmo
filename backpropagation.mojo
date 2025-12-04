from tenmo import Tensor
from operators import AddTensor, SubtractTensor
from utils import Variant
from walkback import *
from common_utils import panic
from gradbox import Gradbox

# Centralized backward operation tags

alias BACKWARD_ADD = 0
alias BACKWARD_MULTIPLY = 1
alias BACKWARD_RELU = 2
alias BACKWARD_MATMUL_ND = 3
alias BACKWARD_MATMUL_2D = 4
alias BACKWARD_TRANSPOSE = 5
alias BACKWARD_PERMUTE = 6
alias BACKWARD_SIGMOID = 7
alias BACKWARD_ADD_BROADCAST = 8
alias BACKWARD_MULTIPLY_BROADCAST = 9
alias BACKWARD_SOFTMAX = 10
alias BACKWARD_CROSS_ENTROPY = 11
alias BACKWARD_TANH = 12
alias BACKWARD_SUB = 13
alias BACKWARD_RESHAPE = 14
alias BACKWARD_VIEW = 15
alias BACKWARD_MEAN = 16
alias BACKWARD_SUM = 17
alias BACKWARD_LOG_SOFTMAX = 18
alias BACKWARD_CONTIGUOUS = 19
alias BACKWARD_DIVIDE = 20
alias BACKWARD_MATRIX_VECTOR_MUL = 21
alias BACKWARD_VECTOR_MATMUL = 22
alias BACKWARD_ADD_SCALAR = 23
alias BACKWARD_MULTIPLY_SCALAR = 24
alias BACKWARD_SUB_SCALAR = 25
alias BACKWARD_DIV_SCALAR = 26
alias BACKWARD_RIGHT_DIV_SCALAR = 27
alias BACKWARD_EXPONENTIATION = 28
alias BACKWARD_DOT = 29
alias BACKWARD_EXPAND = 30
alias BACKWARD_FLATTEN = 31
alias BACKWARD_SQUEEZE = 32
alias BACKWARD_UNSQUEEZE = 33
alias BACKWARD_SHUFFLE = 34
alias BACKWARD_MINMAX = 35
alias BACKWARD_TILE = 36
alias BACKWARD_LOG = 37
alias BACKWARD_SQRT = 38
alias BACKWARD_CLIP = 39
alias BACKWARD_VARIANCE = 40
alias BACKWARD_STD = 41
alias BACKWARD_SUBTRACT_BROADCAST = 42

# ========== Delegate (Variant) ==========

alias Delegate[dtype: DType] = Variant[
    MatrixVectorMulNdBackward[dtype],
    VectorMatmulNdBackward[dtype],
    MatmulNdBackward[dtype],
    Matmul2dBackward[dtype],
    CrossEntropyBackward[dtype],
    AddBackwardScalar[dtype],
    AddBackward[dtype],
    SubBackward[dtype],
    SubLeftRightBackwardScalar[dtype],
    SubtractBroadcastBackward[dtype],
    ReshapeBackward[dtype],
    SumBackward[dtype],
    AddBroadcastBackward[dtype],
    MultiplyBackwardScalar[dtype],
    MultiplyBackward[dtype],
    MultiplyBroadcastBackward[dtype],
    ExponientionBackward[dtype],
    TrueDivBackwardScalar[dtype],
    RightTrueDivBackwardScalar[dtype],
    DivideBackward[dtype],
    MeanBackward[dtype],
    ViewBackward[dtype],
    TransposeBackward[dtype],
    DotBackward[dtype],
    ExpandBackward[dtype],
    ContiguousBackward[dtype],
    FlattenBackward[dtype],
    SqueezeBackward[dtype],
    UnsqueezeBackward[dtype],
    PermuteBackward[dtype],
    ShuffleBackward[dtype],
    ReLUBackward[dtype],
    MinMaxBackward[dtype],
    SoftmaxBackward[dtype],
    LogSoftmaxBackward[dtype],
    TileBackward[dtype],
    SigmoidBackward[dtype],
    TanhBackward[dtype],
    LogBackward[dtype],
    ClipBackward[dtype],
    SqrtBackward[dtype],
    VarianceBackward[dtype],
    StdBackward[dtype],
]

# ========== BackwardFn with Tag-Based Dispatch ==========

struct BackwardFn[dtype: DType](Copyable & Movable):
    var grad_fn: Delegate[dtype]
    var tag: Int  # O(1) lookup key

    fn __init__(out self, grad_fn: Delegate[dtype], tag: Int):
        self.grad_fn = grad_fn
        self.tag = tag

    fn __moveinit__(out self, deinit other: Self):
        self.grad_fn = other.grad_fn^
        self.tag = other.tag

    fn __copyinit__(out self, other: Self):
        self.grad_fn = other.grad_fn.copy()
        self.tag = other.tag

    fn __call__(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        """O(1) dispatch using integer tag comparison.

        Compiler optimizes integer comparisons to jump table for true O(1).
        Order: Most common operations first for branch prediction.
        """

        # ========== TIER 1: MOST COMMON ==========
        if self.tag == BACKWARD_ADD:
            return self.grad_fn[AddBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_MULTIPLY:
            return self.grad_fn[MultiplyBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_RELU:
            return self.grad_fn[ReLUBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_MATMUL_ND:
            return self.grad_fn[MatmulNdBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_SIGMOID:
            return self.grad_fn[SigmoidBackward[dtype]].backward(output)

        # ========== TIER 2: MATMUL CHAIN (Called by MatmulNd) ==========
        elif self.tag == BACKWARD_MATMUL_2D:
            return self.grad_fn[Matmul2dBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_TRANSPOSE:
            return self.grad_fn[TransposeBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_PERMUTE:
            return self.grad_fn[PermuteBackward[dtype]].backward(output)

        # ========== TIER 3: COMMON OPERATIONS ==========
        elif self.tag == BACKWARD_ADD_BROADCAST:
            return self.grad_fn[AddBroadcastBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_MULTIPLY_BROADCAST:
            return self.grad_fn[MultiplyBroadcastBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_SOFTMAX:
            return self.grad_fn[SoftmaxBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_CROSS_ENTROPY:
            return self.grad_fn[CrossEntropyBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_TANH:
            return self.grad_fn[TanhBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_SUB:
            return self.grad_fn[SubBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_RESHAPE:
            return self.grad_fn[ReshapeBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_VIEW:
            return self.grad_fn[ViewBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_MEAN:
            return self.grad_fn[MeanBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_SUM:
            return self.grad_fn[SumBackward[dtype]].backward(output)

        # ========== TIER 4: MODERATELY COMMON ==========
        elif self.tag == BACKWARD_LOG_SOFTMAX:
            return self.grad_fn[LogSoftmaxBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_CONTIGUOUS:
            return self.grad_fn[ContiguousBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_DIVIDE:
            return self.grad_fn[DivideBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_MATRIX_VECTOR_MUL:
            return self.grad_fn[MatrixVectorMulNdBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_VECTOR_MATMUL:
            return self.grad_fn[VectorMatmulNdBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_EXPAND:
            return self.grad_fn[ExpandBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_FLATTEN:
            return self.grad_fn[FlattenBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_SQUEEZE:
            return self.grad_fn[SqueezeBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_UNSQUEEZE:
            return self.grad_fn[UnsqueezeBackward[dtype]].backward(output)

        # ========== TIER 5: SCALAR OPERATIONS ==========
        elif self.tag == BACKWARD_ADD_SCALAR:
            return self.grad_fn[AddBackwardScalar[dtype]].backward(output)

        elif self.tag == BACKWARD_MULTIPLY_SCALAR:
            return self.grad_fn[MultiplyBackwardScalar[dtype]].backward(output)

        elif self.tag == BACKWARD_SUB_SCALAR:
            return self.grad_fn[SubLeftRightBackwardScalar[dtype]].backward(output)

        elif self.tag == BACKWARD_DIV_SCALAR:
            return self.grad_fn[TrueDivBackwardScalar[dtype]].backward(output)

        elif self.tag == BACKWARD_RIGHT_DIV_SCALAR:
            return self.grad_fn[RightTrueDivBackwardScalar[dtype]].backward(output)

        # ========== TIER 6: SPECIALIZED OPERATIONS ==========
        elif self.tag == BACKWARD_EXPONENTIATION:
            return self.grad_fn[ExponientionBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_DOT:
            return self.grad_fn[DotBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_LOG:
            return self.grad_fn[LogBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_SQRT:
            return self.grad_fn[SqrtBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_CLIP:
            return self.grad_fn[ClipBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_SUBTRACT_BROADCAST:
            return self.grad_fn[SubtractBroadcastBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_SHUFFLE:
            return self.grad_fn[ShuffleBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_MINMAX:
            return self.grad_fn[MinMaxBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_TILE:
            return self.grad_fn[TileBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_VARIANCE:
            return self.grad_fn[VarianceBackward[dtype]].backward(output)

        elif self.tag == BACKWARD_STD:
            return self.grad_fn[StdBackward[dtype]].backward(output)

        else:
            panic("BackwardFn: Unknown backward tag: " + String(self.tag))

        return []

fn main():
    print("passes")
