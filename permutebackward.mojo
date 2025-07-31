from tensors import Tensor
from shared import TensorLike
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList


@fieldwise_init
struct PermuteBackward[dtype: DType](Copyable & Movable & Stringable):
    var permutation: IntList  # forward permutation used

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

        # Compute inverse permutation
        inverse = IntList.filled(len(self.permutation), 0)
        for i in range(len(self.permutation)):
            inverse[self.permutation[i]] = i

        # Apply inverse permutation to gradients
        parent_grad = gradients.permute(inverse).into_tensor()

        return [(parent, parent_grad, AddTensor)]

    fn __str__(self) -> String:
        return "PermuteBackward"

fn main() raises:
    pass
