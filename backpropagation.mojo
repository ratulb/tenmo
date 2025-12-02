from tenmo import Tensor
from operators import AddTensor, SubtractTensor
from utils import Variant
from walkback import *
from common_utils import panic
from gradbox import Gradbox
from ancestry import Ancestor

# Centralized backward operation tags

alias BACKWARD_ADD = 0
alias BACKWARD_MULTIPLY = 1
alias BACKWARD_RELU = 2
alias BACKWARD_MATMUL_ND = 3
alias BACKWARD_MATMUL_2D = 4
alias BACKWARD_TRANSPOSE = 5
alias BACKWARD_PERMUTE = 6
alias BACKWARD_BCE = 7
alias BACKWARD_SIGMOID = 8
alias BACKWARD_ADD_BROADCAST = 9
alias BACKWARD_MULTIPLY_BROADCAST = 10
alias BACKWARD_SOFTMAX = 11
alias BACKWARD_CROSS_ENTROPY = 12
alias BACKWARD_TANH = 13
alias BACKWARD_SUB = 14
alias BACKWARD_RESHAPE = 15
alias BACKWARD_VIEW = 16
alias BACKWARD_MEAN = 17
alias BACKWARD_SUM = 18
alias BACKWARD_LOG_SOFTMAX = 19
alias BACKWARD_CONTIGUOUS = 20
alias BACKWARD_DIVIDE = 21
alias BACKWARD_MATRIX_VECTOR_MUL = 22
alias BACKWARD_VECTOR_MATMUL = 23
alias BACKWARD_ADD_SCALAR = 24
alias BACKWARD_MULTIPLY_SCALAR = 25
alias BACKWARD_SUB_SCALAR = 26
alias BACKWARD_DIV_SCALAR = 27
alias BACKWARD_RIGHT_DIV_SCALAR = 28
alias BACKWARD_EXPONENTIATION = 29
alias BACKWARD_DOT = 30
alias BACKWARD_EXPAND = 31
alias BACKWARD_FLATTEN = 32
alias BACKWARD_SQUEEZE = 33
alias BACKWARD_UNSQUEEZE = 34
alias BACKWARD_SHUFFLE = 35
alias BACKWARD_MINMAX = 36
alias BACKWARD_TILE = 37
alias BACKWARD_LOG = 38
alias BACKWARD_SQRT = 39
alias BACKWARD_CLIP = 40
alias BACKWARD_VARIANCE = 41
alias BACKWARD_STD = 42
alias BACKWARD_SUBTRACT_BROADCAST = 43


alias Delegate[dtype: DType] = Variant[
    # ========== TIER 1: MOST COMMON (Your network uses these heavily) ==========
    AddBackward[dtype],                    # 1. Used in every layer (bias addition, residuals)
    MultiplyBackward[dtype],               # 2. Very common (scaling, attention, etc.)
    ReLUBackward[dtype],                   # 3. You have 4 ReLU layers
    MatmulNdBackward[dtype],               # 4. You have 5 Linear layers
    BCEBackward[dtype],                    # 5. Your loss function

    Matmul2dBackward[dtype],               # 21. Specific 2D case
    TransposeBackward[dtype],              # 14. Used in matmul backward internally
    PermuteBackward[dtype],                # 24. Advanced indexing


    # ========== TIER 2: COMMON IN MANY NETWORKS ==========
    SigmoidBackward[dtype],                # 6. Your output activation
    AddBroadcastBackward[dtype],           # 7. Broadcasting adds (very common)
    MultiplyBroadcastBackward[dtype],      # 8. Broadcasting multiply (common)

    SoftmaxBackward[dtype],                # 9. Common loss/activation
    CrossEntropyBackward[dtype],           # 10. Common loss

    # ========== TIER 3: MODERATELY COMMON ==========
    TanhBackward[dtype],                   # 11. Alternative activation
    SubBackward[dtype],                    # 12. Residual connections
    ReshapeBackward[dtype],                # 13. Shape manipulation
    ViewBackward[dtype],                   # 15. View operations

    # ========== TIER 4: OCCASIONALLY USED ==========
    MeanBackward[dtype],                   # 16. Pooling, normalization
    SumBackward[dtype],                    # 17. Reductions
    LogSoftmaxBackward[dtype],             # 18. NLLLoss companion
    ContiguousBackward[dtype],             # 19. Memory layout
    DivideBackward[dtype],                 # 20. Normalization

    # ========== TIER 5: SPECIALIZED/LESS COMMON ==========
    MatrixVectorMulNdBackward[dtype],      # 22. Specialized matmul
    VectorMatmulNdBackward[dtype],         # 23. Specialized matmul
    ExpandBackward[dtype],                 # 25. Broadcasting expansion

    # ========== TIER 6: SCALAR OPERATIONS ==========
    AddBackwardScalar[dtype],              # 26. Scalar ops (less common)
    MultiplyBackwardScalar[dtype],         # 27. Scalar ops
    SubLeftRightBackwardScalar[dtype],     # 28. Scalar ops
    TrueDivBackwardScalar[dtype],          # 29. Scalar ops
    RightTrueDivBackwardScalar[dtype],     # 30. Scalar ops

    # ========== TIER 7: RARELY USED ==========
    ExponientionBackward[dtype],           # 31. Specialized
    DotBackward[dtype],                    # 32. 1D specific
    LogBackward[dtype],                    # 33. Specialized
    SqrtBackward[dtype],                   # 34. Specialized
    ClipBackward[dtype],                   # 35. Gradient clipping (if manual)

    # ========== TIER 8: VERY SPECIALIZED ==========
    SubtractBroadcastBackward[dtype],      # 36. Less common than add
    FlattenBackward[dtype],                # 37. Shape manipulation
    SqueezeBackward[dtype],                # 38. Shape manipulation
    UnsqueezeBackward[dtype],              # 39. Shape manipulation
    ShuffleBackward[dtype],                # 40. Rare operation
    MinMaxBackward[dtype],                 # 41. Specialized
    TileBackward[dtype],                   # 42. Rare
    VarianceBackward[dtype],               # 43. Statistics
    StdBackward[dtype],                    # 44. Statistics
]

struct BackwardFn[dtype: DType](Copyable & Movable):
    var grad_fn: Delegate[dtype]

    fn __init__(out self, grad_fn: Delegate[dtype]):
        self.grad_fn = grad_fn

    fn __moveinit__(out self, deinit other: Self):
        self.grad_fn = other.grad_fn^

    fn __copyinit__(out self, other: Self):
        self.grad_fn = other.grad_fn.copy()

    fn __call__(self, output: Tensor[dtype]) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        # ========== TIER 1: MOST COMMON ==========
        if self.grad_fn.isa[AddBackward[dtype]]():
            return self.grad_fn[AddBackward[dtype]].backward(output)

        elif self.grad_fn.isa[MultiplyBackward[dtype]]():
            return self.grad_fn[MultiplyBackward[dtype]].backward(output)

        elif self.grad_fn.isa[ReLUBackward[dtype]]():
            return self.grad_fn[ReLUBackward[dtype]].backward(output)

        elif self.grad_fn.isa[MatmulNdBackward[dtype]]():
            return self.grad_fn[MatmulNdBackward[dtype]].backward(output)

        elif self.grad_fn.isa[BCEBackward[dtype]]():
            return self.grad_fn[BCEBackward[dtype]].backward(output)

        elif self.grad_fn.isa[Matmul2dBackward[dtype]]():
            return self.grad_fn[Matmul2dBackward[dtype]].backward(output)

        elif self.grad_fn.isa[PermuteBackward[dtype]]():
            return self.grad_fn[PermuteBackward[dtype]].backward(output)

         elif self.grad_fn.isa[ReshapeBackward[dtype]]():
            return self.grad_fn[ReshapeBackward[dtype]].backward(output)

        elif self.grad_fn.isa[TransposeBackward[dtype]]():
            return self.grad_fn[TransposeBackward[dtype]].backward(output)

        elif self.grad_fn.isa[ViewBackward[dtype]]():
            return self.grad_fn[ViewBackward[dtype]].backward(output)


        # ========== TIER 2: COMMON ==========
        elif self.grad_fn.isa[SigmoidBackward[dtype]]():
            return self.grad_fn[SigmoidBackward[dtype]].backward(output)

        elif self.grad_fn.isa[AddBroadcastBackward[dtype]]():
            return self.grad_fn[AddBroadcastBackward[dtype]].backward(output)

        elif self.grad_fn.isa[MultiplyBroadcastBackward[dtype]]():
            return self.grad_fn[MultiplyBroadcastBackward[dtype]].backward(output)

        elif self.grad_fn.isa[SoftmaxBackward[dtype]]():
            return self.grad_fn[SoftmaxBackward[dtype]].backward(output)

        elif self.grad_fn.isa[CrossEntropyBackward[dtype]]():
            return self.grad_fn[CrossEntropyBackward[dtype]].backward(output)

        # ========== TIER 3: MODERATELY COMMON ==========
        elif self.grad_fn.isa[TanhBackward[dtype]]():
            return self.grad_fn[TanhBackward[dtype]].backward(output)

        elif self.grad_fn.isa[SubBackward[dtype]]():
            return self.grad_fn[SubBackward[dtype]].backward(output)


        # ========== TIER 4: OCCASIONALLY USED ==========
        elif self.grad_fn.isa[MeanBackward[dtype]]():
            return self.grad_fn[MeanBackward[dtype]].backward(output)

        elif self.grad_fn.isa[SumBackward[dtype]]():
            return self.grad_fn[SumBackward[dtype]].backward(output)

        elif self.grad_fn.isa[LogSoftmaxBackward[dtype]]():
            return self.grad_fn[LogSoftmaxBackward[dtype]].backward(output)

        elif self.grad_fn.isa[ContiguousBackward[dtype]]():
            return self.grad_fn[ContiguousBackward[dtype]].backward(output)

        elif self.grad_fn.isa[DivideBackward[dtype]]():
            return self.grad_fn[DivideBackward[dtype]].backward(output)

        # ========== TIER 5: SPECIALIZED ==========

        elif self.grad_fn.isa[MatrixVectorMulNdBackward[dtype]]():
            return self.grad_fn[MatrixVectorMulNdBackward[dtype]].backward(output)

        elif self.grad_fn.isa[VectorMatmulNdBackward[dtype]]():
            return self.grad_fn[VectorMatmulNdBackward[dtype]].backward(output)

        elif self.grad_fn.isa[ExpandBackward[dtype]]():
            return self.grad_fn[ExpandBackward[dtype]].backward(output)

        # ========== TIER 6: SCALAR OPERATIONS ==========
        elif self.grad_fn.isa[AddBackwardScalar[dtype]]():
            return self.grad_fn[AddBackwardScalar[dtype]].backward(output)

        elif self.grad_fn.isa[MultiplyBackwardScalar[dtype]]():
            return self.grad_fn[MultiplyBackwardScalar[dtype]].backward(output)

        elif self.grad_fn.isa[SubLeftRightBackwardScalar[dtype]]():
            return self.grad_fn[SubLeftRightBackwardScalar[dtype]].backward(output)

        elif self.grad_fn.isa[TrueDivBackwardScalar[dtype]]():
            return self.grad_fn[TrueDivBackwardScalar[dtype]].backward(output)

        elif self.grad_fn.isa[RightTrueDivBackwardScalar[dtype]]():
            return self.grad_fn[RightTrueDivBackwardScalar[dtype]].backward(output)

        # ========== TIER 7: RARELY USED ==========
        elif self.grad_fn.isa[ExponientionBackward[dtype]]():
            return self.grad_fn[ExponientionBackward[dtype]].backward(output)

        elif self.grad_fn.isa[DotBackward[dtype]]():
            return self.grad_fn[DotBackward[dtype]].backward(output)

        elif self.grad_fn.isa[LogBackward[dtype]]():
            return self.grad_fn[LogBackward[dtype]].backward(output)

        elif self.grad_fn.isa[SqrtBackward[dtype]]():
            return self.grad_fn[SqrtBackward[dtype]].backward(output)

        elif self.grad_fn.isa[ClipBackward[dtype]]():
            return self.grad_fn[ClipBackward[dtype]].backward(output)

        # ========== TIER 8: VERY SPECIALIZED ==========
        elif self.grad_fn.isa[SubtractBroadcastBackward[dtype]]():
            return self.grad_fn[SubtractBroadcastBackward[dtype]].backward(output)

        elif self.grad_fn.isa[FlattenBackward[dtype]]():
            return self.grad_fn[FlattenBackward[dtype]].backward(output)

        elif self.grad_fn.isa[SqueezeBackward[dtype]]():
            return self.grad_fn[SqueezeBackward[dtype]].backward(output)

        elif self.grad_fn.isa[UnsqueezeBackward[dtype]]():
            return self.grad_fn[UnsqueezeBackward[dtype]].backward(output)

        elif self.grad_fn.isa[ShuffleBackward[dtype]]():
            return self.grad_fn[ShuffleBackward[dtype]].backward(output)

        elif self.grad_fn.isa[MinMaxBackward[dtype]]():
            return self.grad_fn[MinMaxBackward[dtype]].backward(output)

        elif self.grad_fn.isa[TileBackward[dtype]]():
            return self.grad_fn[TileBackward[dtype]].backward(output)

        elif self.grad_fn.isa[VarianceBackward[dtype]]():
            return self.grad_fn[VarianceBackward[dtype]].backward(output)

        elif self.grad_fn.isa[StdBackward[dtype]]():
            return self.grad_fn[StdBackward[dtype]].backward(output)

        else:
            panic("BackwardFn: Unknown gradient function type")

        return []

fn main():
    print("passed")
