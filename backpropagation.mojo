from tensors import Tensor
from shared import TensorLite
from operators import AddTensor, SubtractTensor
from utils import Variant
from os import abort
from walkback import *

alias Delegate[dtype: DType] = Variant[
    MatmulBackward[dtype],
    BatchedMatmulBackward[dtype],
    ReshapeBackward[dtype],
    ViewBackward[dtype],
    PermuteBackward[dtype],
    SumBackward[dtype],
    MeanBackward[dtype],
    AddBackward[dtype],
    AddBackwardScalar[dtype],
    SubBackward[dtype],
    SubLeftRightBackwardScalar[dtype],
    MultiplyBackward[dtype],
    MulBackwardScalar[dtype],
    TrueDivBackwardScalar[dtype],
    RightTrueDivBackwardScalar[dtype],
    ExponientionBackward[dtype],
    TransposeBackward[dtype],
    BroadcastBackward[dtype, AddTensor, AddTensor, False],
    BroadcastBackward[dtype, AddTensor, AddTensor, True],
    BroadcastBackward[dtype, AddTensor, SubtractTensor, False],
    DotBackward[dtype],
]


struct BackwardFn[dtype: DType](Copyable & Movable):
    var grad_fn: Delegate[dtype]

    fn __init__(out self, grad_fn: Delegate[dtype]):
        self.grad_fn = grad_fn

    fn __moveinit__(out self, deinit other: Self):
        self.grad_fn = other.grad_fn

    fn __copyinit__(out self, other: Self):
        self.grad_fn = other.grad_fn

    fn __call__(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:

        if self.grad_fn.isa[MatmulBackward[dtype]]():
            return self.grad_fn[MatmulBackward[dtype]].backward[dtype](output)

        if self.grad_fn.isa[BatchedMatmulBackward[dtype]]():
            return self.grad_fn[BatchedMatmulBackward[dtype]].backward[dtype](output)

        elif self.grad_fn.isa[ReshapeBackward[dtype]]():
            return self.grad_fn[ReshapeBackward[dtype]].backward[dtype](output)

        elif self.grad_fn.isa[ViewBackward[dtype]]():
            return self.grad_fn[ViewBackward[dtype]].backward[dtype](output)

        elif self.grad_fn.isa[TransposeBackward[dtype]]():
            return self.grad_fn[TransposeBackward[dtype]].backward[dtype](output)

        elif self.grad_fn.isa[PermuteBackward[dtype]]():
            return self.grad_fn[PermuteBackward[dtype]].backward[dtype](output)

        elif self.grad_fn.isa[SumBackward[dtype]]():
            return self.grad_fn[SumBackward[dtype]].backward[dtype](output)

        elif self.grad_fn.isa[MeanBackward[dtype]]():
            return self.grad_fn[MeanBackward[dtype]].backward[dtype](output)

        elif self.grad_fn.isa[AddBackward[dtype]]():
            return self.grad_fn[AddBackward[dtype]].backward[dtype](output)

        elif self.grad_fn.isa[AddBackwardScalar[dtype]]():
            return self.grad_fn[AddBackwardScalar[dtype]].backward[dtype](
                output
            )

        elif self.grad_fn.isa[SubBackward[dtype]]():
            return self.grad_fn[SubBackward[dtype]].backward[dtype](output)

        elif self.grad_fn.isa[SubLeftRightBackwardScalar[dtype]]():
            return self.grad_fn[SubLeftRightBackwardScalar[dtype]].backward[
                dtype
            ](output)

        elif self.grad_fn.isa[MultiplyBackward[dtype]]():
            return self.grad_fn[MultiplyBackward[dtype]].backward[dtype](
                output
            )

        elif self.grad_fn.isa[MulBackwardScalar[dtype]]():
            return self.grad_fn[MulBackwardScalar[dtype]].backward[dtype](
                output
            )
        elif self.grad_fn.isa[DotBackward[dtype]]():
            return self.grad_fn[DotBackward[dtype]].backward[dtype](
                output
            )

        elif self.grad_fn.isa[TrueDivBackwardScalar[dtype]]():
            return self.grad_fn[TrueDivBackwardScalar[dtype]].backward[dtype](
                output
            )

        elif self.grad_fn.isa[RightTrueDivBackwardScalar[dtype]]():
            return self.grad_fn[RightTrueDivBackwardScalar[dtype]].backward[
                dtype
            ](output)

        elif self.grad_fn.isa[ExponientionBackward[dtype]]():
            return self.grad_fn[ExponientionBackward[dtype]].backward[dtype](
                output
            )

        elif self.grad_fn.isa[
            BroadcastBackward[dtype, AddTensor, AddTensor, False]
        ]():
            return self.grad_fn[
                BroadcastBackward[dtype, AddTensor, AddTensor, False]
            ].backward[dtype](output)

        elif self.grad_fn.isa[
            BroadcastBackward[dtype, AddTensor, AddTensor, True]
        ]():
            return self.grad_fn[
                BroadcastBackward[dtype, AddTensor, AddTensor, True]
            ].backward[dtype](output)

        elif self.grad_fn.isa[
            BroadcastBackward[dtype, AddTensor, SubtractTensor, False]
        ]():
            return self.grad_fn[
                BroadcastBackward[dtype, AddTensor, SubtractTensor, False]
            ].backward[dtype](output)

        else:
            abort("I am not here to receive you")
        return []


fn main():
    pass
