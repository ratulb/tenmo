from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList
from common_utils import log_debug, LOG_LEVEL


@fieldwise_init
struct PermuteBackward[dtype: DType](Copyable & Movable & Stringable):
    var permutation: IntList  # forward permutation used

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        parent = output.ancestry().get(0)[]
        # Compute inverse permutation
        inverse = IntList.filled(len(self.permutation), 0)
        for i in range(len(self.permutation)):
            inverse[self.permutation[i]] = i

        # Apply inverse permutation to gradients
        parent_grad = gradients.permute(inverse)

        @parameter
        if LOG_LEVEL == "debug":
            print(
                "\nPermuteBackward: output is view? ",
                output.tensor().owns_data.__str__(),
                "parent is view?",
                parent.tensor().owns_data.__str__(),
                "\n",
            )
            print("\nPermuteBackward - gradients\n")
            gradients.print()
            print()
            print("self.permutation", self.permutation)
            print()
            print("\ninverse permutation", inverse)
            print("\nparent_grad\n")
            parent_grad.print()

        return [(parent, parent_grad, AddTensor)]

    fn __str__(self) -> String:
        return "PermuteBackward"


fn main() raises:
    pass
