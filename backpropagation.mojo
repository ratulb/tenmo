from tensors import Tensor
from shared import TensorLike
from operators import AddTensor, SubtractTensor
from utils import Variant
from os import abort
from sumbackward import SumBackward
from meanbackward import MeanBackward
from addbackward import AddBackward, AddBackwardScalar
from subbackward import SubBackward, SubLeftRightBackwardScalar
from broadcastbackward import BroadcastBackward
from reshapebackward import ReshapeBackward
from mulbackward import MultiplyBackward, MulBackwardScalar
from exponientionbackward import ExponientionBackward
from divbackwardscalar import TrueDivBackwardScalar, RightTrueDivBackwardScalar

alias Delegate[dtype: DType] = Variant[
    ReshapeBackward[dtype],
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
    BroadcastBackward[dtype, AddTensor, AddTensor, False],
    BroadcastBackward[dtype, AddTensor, AddTensor, True],
    BroadcastBackward[dtype, AddTensor, SubtractTensor, False],
]


struct BackwardFn[dtype: DType](Copyable & Movable):
    var grad_fn: Delegate[dtype]

    fn __init__(out self, grad_fn: Delegate[dtype]):
        self.grad_fn = grad_fn

    fn __moveinit__(out self, owned other: Self):
        self.grad_fn = other.grad_fn

    fn __copyinit__(out self, other: Self):
        self.grad_fn = other.grad_fn

    fn __call__(
        self, out_ptr: UnsafePointer[Tensor[dtype]]
    ) -> List[Tuple[TensorLike[dtype], Tensor[dtype], Int]]:
        if self.grad_fn.isa[ReshapeBackward[dtype]]():
            return self.grad_fn[ReshapeBackward[dtype]].backward[dtype](out_ptr)

        elif self.grad_fn.isa[SumBackward[dtype]]():
            return self.grad_fn[SumBackward[dtype]].backward[dtype](out_ptr)

        elif self.grad_fn.isa[MeanBackward[dtype]]():
            return self.grad_fn[MeanBackward[dtype]].backward[dtype](out_ptr)

        elif self.grad_fn.isa[AddBackward[dtype]]():
            return self.grad_fn[AddBackward[dtype]].backward[dtype](out_ptr)

        elif self.grad_fn.isa[AddBackwardScalar[dtype]]():
            return self.grad_fn[AddBackwardScalar[dtype]].backward[dtype](
                out_ptr
            )

        elif self.grad_fn.isa[SubBackward[dtype]]():
            return self.grad_fn[SubBackward[dtype]].backward[dtype](out_ptr)

        elif self.grad_fn.isa[SubLeftRightBackwardScalar[dtype]]():
            return self.grad_fn[SubLeftRightBackwardScalar[dtype]].backward[
                dtype
            ](out_ptr)

        elif self.grad_fn.isa[MultiplyBackward[dtype]]():
            return self.grad_fn[MultiplyBackward[dtype]].backward[dtype](
                out_ptr
            )

        elif self.grad_fn.isa[MulBackwardScalar[dtype]]():
            return self.grad_fn[MulBackwardScalar[dtype]].backward[dtype](
                out_ptr
            )

        elif self.grad_fn.isa[TrueDivBackwardScalar[dtype]]():
            return self.grad_fn[TrueDivBackwardScalar[dtype]].backward[dtype](
                out_ptr
            )

        elif self.grad_fn.isa[RightTrueDivBackwardScalar[dtype]]():
            return self.grad_fn[RightTrueDivBackwardScalar[dtype]].backward[
                dtype
            ](out_ptr)

        elif self.grad_fn.isa[ExponientionBackward[dtype]]():
            return self.grad_fn[ExponientionBackward[dtype]].backward[dtype](
                out_ptr
            )

        elif self.grad_fn.isa[
            BroadcastBackward[dtype, AddTensor, AddTensor, False]
        ]():
            return self.grad_fn[
                BroadcastBackward[dtype, AddTensor, AddTensor, False]
            ].backward[dtype](out_ptr)

        elif self.grad_fn.isa[
            BroadcastBackward[dtype, AddTensor, AddTensor, True]
        ]():
            return self.grad_fn[
                BroadcastBackward[dtype, AddTensor, AddTensor, True]
            ].backward[dtype](out_ptr)

        elif self.grad_fn.isa[
            BroadcastBackward[dtype, AddTensor, SubtractTensor, False]
        ]():
            return self.grad_fn[
                BroadcastBackward[dtype, AddTensor, SubtractTensor, False]
            ].backward[dtype](out_ptr)

        else:
            abort("I am not here to receive you")
        return []


fn main():
    pass
