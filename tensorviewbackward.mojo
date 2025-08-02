from tensors import Tensor
from shared import TensorLike
from shapes import Shape
from strides import Strides
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList


@fieldwise_init
struct TensorViewBackward[dtype: DType](Copyable & Movable & Stringable):
    var shape: Shape
    var strides: Strides

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        parent = output.ancestry().get(0)[]
        parent_shape = parent.shape()

        parent_grad = Tensor[dtype].zeros(parent_shape.num_elements())
        _="""parent_grad = Tensor[dtype].zeros(parent.shape())
        offset_delta = self.offset(???)  - parent.offset()
        for child_indices in self.shape:
            child_flat = (child_indices * self.strides.to_list()).sum()

            parent_flat = child_flat + offset_delta

            parent_indices = IntList.Empty
            remaining = parent_flat
            for stride in parent.strides().to_list():
                dim_idx = remaining // stride
                remaining = remaining % stride
                parent_indices.append(dim_idx)
            parent_grad[parent_indices] += gradients[child_indices]

            return [(parent, parent_grad, AddTensor)]"""

        for child_indices in self.shape:
            child_flat = (child_indices * self.strides.to_list()).sum()
            parent_flat = child_flat
            parent_grad[parent_flat] += gradients[child_indices]
        reshaped = parent_grad.reshape(parent_shape)
        return [(parent, reshaped, AddTensor)]

    fn __str__(self) -> String:
        return "TensorViewBackward"

fn main():
    pass
