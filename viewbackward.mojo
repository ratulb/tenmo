from tensors import Tensor
from shared import TensorLike
from shapes import Shape
from strides import Strides
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList


@fieldwise_init
struct ViewBackward[dtype: DType](Copyable & Movable):
    var shape: Shape
    var strides: Strides
    var offset: Int

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        recipient = output.ancestry().get(0)[]
        flat_grad = Tensor[dtype].zeros(recipient.shape())
        strides = (
            Strides.default(recipient.tensor_address[].shape) if recipient.kind
            == 0 else recipient.view_address[].strides
        )
        for indices in self.shape:
            #flat_index = self.offset + (indices * self.strides.to_list()).sum()
            flat_index = (indices * self.strides.to_list()).sum()
            recipient_index = Self.target_index(flat_index, recipient.shape(), strides)
            flat_grad[recipient_index] = gradients[indices]

        return [
            (
                recipient,
                flat_grad,
                AddTensor,
            )
        ]

    @staticmethod
    fn target_index(flat_index: Int, shape: Shape, strides: Strides) -> IntList:
        indices = IntList.with_capacity(len(shape))
        remaining = flat_index
        for dim in reversed(range(len(shape))):
            indices.append(remaining // strides[dim])
            remaining = remaining % strides[dim]
        indices.reverse()
        return indices


fn main():
    pass
