from tensors import Tensor
from shared import TensorLite
from operators import AddTensor, SubtractTensor
from backpropagation import BackwardFn, Delegate
from common_utils import LOG_LEVEL


@fieldwise_init
struct ReshapeBackward[dtype: DType](Copyable & Movable & Stringable):
    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        reshaped = gradients.reshape(ancestor.shape())

        @parameter
        if LOG_LEVEL == "debug":
            print(
                "\nReshapeBackward: output is view? ",
                output.tensor().owns_data.__str__(),
                "parent is view?",
                ancestor.tensor().owns_data.__str__(),
                "\n",
            )
            print("\nReshapeBackward - gradients\n")
            gradients.print()
            print()
            print("\nreshaped\n")
            print()
            reshaped.print()

        return [
            (ancestor, reshaped, AddTensor),
            (output, gradients, SubtractTensor),
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn __str__(self) -> String:
        return "ReshapeBackward"


fn main():
    print("Yes")
