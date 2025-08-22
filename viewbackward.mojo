from tensors import Tensor
from shapes import Shape
from strides import Strides
from backpropagation import Delegate, BackwardFn
from operators import AddTensor, SubtractTensor
from intlist import IntList
from shared import TensorLite
from common_utils import LOG_LEVEL


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
        parent_shape = parent.shape()

        for child_indices in self.shape:
            child_flat = (child_indices * self.strides.to_list()).sum()
            parent_flat = child_flat + offset_delta
            parent_grad[parent_flat] += gradients[child_indices]
        reshaped = parent_grad.reshape(parent_shape)

        @parameter
        if LOG_LEVEL == "debug":
            print(
                "\nViewBackward: output owns data? ",
                output.tensor().owns_data.__str__(),
                "parent owns data?",
                parent.tensor().owns_data.__str__(),
                "\n",
            )
            print(
                "\nViewBackward - offset_delta",
                offset_delta.__str__(),
                "parent shape",
                parent_shape.__str__(),
                "\n",
            )
            print("\nViewBackward - gradients\n")
            gradients.print()
            print()
            print("\nreshaped\n")
            print("parent inner_id", parent.inner_id())
            reshaped.print()
        return [
            (parent, reshaped, AddTensor),
            (output, gradients, SubtractTensor),
        ]

    fn __str__(self) -> String:
        return "ViewBackward"


fn main():
    pass
