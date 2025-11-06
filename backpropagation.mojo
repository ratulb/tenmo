from tenmo import Tensor
from operators import AddTensor, SubtractTensor
from utils import Variant
from walkback import *
from common_utils import panic
from gradbox import Gradbox
from ancestry import Ancestor

alias Delegate[dtype: DType] = Variant[
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
    SumBackward[dtype],
    MeanBackward[dtype],
    ViewBackward[dtype],
    TransposeBackward[dtype],
    DotBackward[dtype],
    ExpandBackward[dtype],
    ContiguousBackward[dtype],
    FlattenBackward[dtype],
]


struct BackwardFn[dtype: DType](Copyable & Movable):
    var grad_fn: Delegate[dtype]

    fn __init__(out self, grad_fn: Delegate[dtype]):
        self.grad_fn = grad_fn

    fn __moveinit__(out self, deinit other: Self):
        self.grad_fn = other.grad_fn^

    fn __copyinit__(out self, other: Self):
        self.grad_fn = other.grad_fn.copy()

    fn __call__(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:

        if self.grad_fn.isa[AddBackwardScalar[dtype]]():
            return self.grad_fn[AddBackwardScalar[dtype]].backward(output)

        elif self.grad_fn.isa[ReshapeBackward[dtype]]():
            return self.grad_fn[ReshapeBackward[dtype]].backward(output)

        elif self.grad_fn.isa[TransposeBackward[dtype]]():
            return self.grad_fn[TransposeBackward[dtype]].backward(output)

        elif self.grad_fn.isa[ViewBackward[dtype]]():
            return self.grad_fn[ViewBackward[dtype]].backward(output)

        elif self.grad_fn.isa[SumBackward[dtype]]():
            return self.grad_fn[SumBackward[dtype]].backward(output)

        elif self.grad_fn.isa[MeanBackward[dtype]]():
            return self.grad_fn[MeanBackward[dtype]].backward(output)

        elif self.grad_fn.isa[AddBackward[dtype]]():
            return self.grad_fn[AddBackward[dtype]].backward(output)

        elif self.grad_fn.isa[AddBroadcastBackward[dtype]]():
            return self.grad_fn[AddBroadcastBackward[dtype]].backward(output)

        elif self.grad_fn.isa[SubBackward[dtype]]():
            return self.grad_fn[SubBackward[dtype]].backward(output)

        elif self.grad_fn.isa[SubLeftRightBackwardScalar[dtype]]():
            return self.grad_fn[SubLeftRightBackwardScalar[dtype]].backward(
                output
            )

        elif self.grad_fn.isa[SubtractBroadcastBackward[dtype]]():
            return self.grad_fn[SubtractBroadcastBackward[dtype]].backward(
                output
            )

        elif self.grad_fn.isa[MultiplyBackward[dtype]]():
            return self.grad_fn[MultiplyBackward[dtype]].backward(output)

        elif self.grad_fn.isa[MultiplyBackwardScalar[dtype]]():
            return self.grad_fn[MultiplyBackwardScalar[dtype]].backward(output)

        elif self.grad_fn.isa[MultiplyBroadcastBackward[dtype]]():
            return self.grad_fn[MultiplyBroadcastBackward[dtype]].backward(
                output
            )

        elif self.grad_fn.isa[ExponientionBackward[dtype]]():
            return self.grad_fn[ExponientionBackward[dtype]].backward(output)

        elif self.grad_fn.isa[TrueDivBackwardScalar[dtype]]():
            return self.grad_fn[TrueDivBackwardScalar[dtype]].backward(output)

        elif self.grad_fn.isa[RightTrueDivBackwardScalar[dtype]]():
            return self.grad_fn[RightTrueDivBackwardScalar[dtype]].backward(
                output
            )

        elif self.grad_fn.isa[DivideBackward[dtype]]():
            return self.grad_fn[DivideBackward[dtype]].backward(output)

        elif self.grad_fn.isa[SumBackward[dtype]]():
            return self.grad_fn[SumBackward[dtype]].backward(output)

        elif self.grad_fn.isa[DotBackward[dtype]]():
            return self.grad_fn[DotBackward[dtype]].backward(output)

        elif self.grad_fn.isa[ExpandBackward[dtype]]():
            return self.grad_fn[ExpandBackward[dtype]].backward(output)

        elif self.grad_fn.isa[ContiguousBackward[dtype]]():
            return self.grad_fn[ContiguousBackward[dtype]].backward(output)

        elif self.grad_fn.isa[FlattenBackward[dtype]]():
            return self.grad_fn[FlattenBackward[dtype]].backward(output)

        else:
            panic("I am not here to receive you")
        return []


fn main():
    pass
