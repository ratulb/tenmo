from tenmo import Tensor
from shapes import Shape
from strides import Strides
from backpropagationcopy import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList
from sharedcopy import TensorLite

@fieldwise_init
struct ViewBackward[dtype: DType](Copyable & Movable & Stringable):
    var shape: Shape
    var strides: Strides
    var offset: Int

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output_ptr: UnsafePointer[TensorLite[dtype]]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        output = output_ptr[]
        gradients = output.gradients().value()
        print("gradients here2")
        gradients.print()
        parent = output.ancestry().get(0)[]
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
        print("The reshaped******")
        reshaped.print()
        print("The reshaped######")
        return [(parent, reshaped, AddTensor)]

    fn __str__(self) -> String:
        return "ViewBackward"

fn main():
    pass
