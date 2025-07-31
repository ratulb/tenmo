from tensors import Tensor
from intlist import IntList
from operators import AddTensor, SubtractTensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn


struct SubBackward[dtype: DType](Copyable & Movable & Stringable):
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
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        count = len(output.ancestry())
        grad_outputs = List[Tuple[TensorLike[dtype], Tensor[dtype], Int]](
            capacity=count
        )
        for i in range(count):
            ancestor = output.ancestry().get(i)[]
            grad_outputs.append(
                (
                    ancestor,
                    gradients,
                    AddTensor if self.signs[i] == 0 else SubtractTensor,
                )
            )
        return grad_outputs

    fn __str__(self) -> String:
        return "SubBackward"



@fieldwise_init
struct SubLeftRightBackwardScalar[dtype: DType](Copyable & Movable & Stringable):
    var negate: Bool

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        return [
            (ancestor, gradients, SubtractTensor if self.negate else AddTensor)
        ]

    fn __str__(self) -> String:
        return "SubLeftRightBackwardScalar"

fn main():
    print("passes")
