from tensors import Tensor
from shared import TensorLike
from operators import __tensor_op_tensor__, AddTensor, SubtractTensor
from backpropagation import BackwardFn, Delegate


@fieldwise_init
struct ReshapeBackward[dtype: DType](Copyable & Movable & Stringable):
    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[TensorLike[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        reshaped = gradients.reshape(ancestor.shape())
        # Deduct already contributed portion
        new_contrib = __tensor_op_tensor__[dtype, SubtractTensor](
            reshaped, output.tensor().base[]
        )
        # Update base accumulator
        output.tensor().base.init_pointee_move(reshaped^)
        return [(ancestor, new_contrib, AddTensor)]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn __str__(self) -> String:
        return "ReshapeBackward"

fn main():
    print("Yes")
