from tensors import Tensor
from shapes import Shape
from strides import Strides
from backpropagation import Delegate, BackwardFn
from operators import AddTensor, SubtractTensor
from intlist import IntList
from shared import TensorLite

@fieldwise_init
struct ViewBackward[dtype: DType](Copyable & Movable & Stringable):
    var shape: Shape
    var strides: Strides
    var offset: Int

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        parent = output.ancestry().get(0)[]
        gradients = output.gradients()[]
        offset_delta = self.offset - parent.tensor().offset
        parent_grad = Tensor[dtype].zeros(parent.shape().num_elements())
        _ = """for child_indices in self.shape:
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
        parent_shape = parent.shape()

        for child_indices in self.shape:
            child_flat = (child_indices * self.strides.to_list()).sum()
            parent_flat = child_flat + offset_delta
            parent_grad[parent_flat] += gradients[child_indices]
        reshaped = parent_grad.reshape(parent_shape)

        return [(parent, reshaped, AddTensor), (output, gradients, SubtractTensor)]

    fn __str__(self) -> String:
        return "ViewBackward"

fn main():
    pass
