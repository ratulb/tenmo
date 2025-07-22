from tensors import Tensor
from intlist import IntList
from operators import AddTensor, SubtractTensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn


struct SubBackward[dtype: DType](Copyable & Movable):
    var signs: IntList

    fn __init__(out self):
        self.signs = IntList.Empty

    fn __moveinit__(out self, var other: Self):
        self.signs = other.signs

    fn __copyinit__(out self, other: Self):
        self.signs = other.signs

    fn negate(mut self, neg: Bool):
        if neg:
            self.signs.append(1)
        else:
            self.signs.append(0)

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
        for i in range(count):
            ancestor = output.ancestors.get(i)[]
            grad_outputs.append(
                (
                    ancestor,
                    gradients,
                    AddTensor if self.signs[i] == 0 else SubtractTensor,
                )
            )
        return grad_outputs


@fieldwise_init
struct SubLeftRightBackwardScalar[dtype: DType](Copyable & Movable):
    var negate: Bool

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.grad[]
        ancestor = output.ancestors.get(0)[]
        return [
            (ancestor, gradients, SubtractTensor if self.negate else AddTensor)
        ]


fn main():
    print("passes")
