from tensors import Tensor
from operators import AddTensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn


struct AddBackward[dtype: DType](Copyable & Movable):
    fn __init__(out self):
        pass

    fn __moveinit__(out self, var other: Self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.grad[]
        count = len(output.ancestors)
        grad_outputs = List[Tuple[TensorLike[dtype], Tensor[dtype], Int]](
            capacity=count
        )
        if count == 1:
            ancestor = output.ancestors.get(0)[]
            grad_outputs.append(
                (
                    ancestor,
                    gradients,
                    AddTensor,
                )
            )

        else:
            ancestor1 = output.ancestors.get(0)[]
            ancestor2 = output.ancestors.get(1)[]
            grad_outputs.append(
                (
                    ancestor1,
                    gradients,
                    AddTensor,
                )
            )
            grad_outputs.append(
                (
                    ancestor2,
                    gradients,
                    AddTensor,
                )
            )
        return grad_outputs


fn main():
    print("passes")
