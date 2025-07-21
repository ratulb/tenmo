from tensors import Tensor
from operators import AddTensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn


@fieldwise_init
struct AddBackwardScalar[dtype: DType](Copyable & Movable):
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
        if ancestor.shape() != gradients.shape:
            gradients = gradients.reshape(ancestor.shape())
        # Gradient of addition is 1 → just pass through incoming grad
        return [(ancestor, gradients, AddTensor)]


@fieldwise_init
struct AddBackward[dtype: DType](Copyable & Movable):
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
            grad_outputs.append((ancestor, gradients, AddTensor))
        return grad_outputs


fn main():
    print("passes")
