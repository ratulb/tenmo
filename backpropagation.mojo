from tensors import Tensor
from shared import TensorLike
from operators import __tensor_op_tensor__, AddTensor, SubtractTensor
from utils import Variant
from os import abort
from sumbackward import SumBackward
from meanbackward import MeanBackward
from addbackward import AddBackward

alias Delegate[dtype: DType] = Variant[
    ReshapeBackward[dtype],
    SumBackward[dtype],
    MeanBackward[dtype],
    AddBackward[dtype],
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

        else:
            abort("I am not here to receive you")
        return []


struct ReshapeBackward[dtype: DType](Copyable & Movable):
    fn __init__(out self):
        pass

    fn __moveinit__(out self, owned other: Self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.grad[]
        ancestor = output.ancestors.get(0)[]
        reshaped = gradients.reshape(ancestor.shape())
        # Deduct already contributed portion
        new_contrib = __tensor_op_tensor__[dtype, SubtractTensor](
            reshaped, output.base[]
        )

        # Update base accumulator
        output.base.init_pointee_move(reshaped^)
        return [(ancestor, new_contrib, AddTensor)]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


fn main():
    print("Yes")
