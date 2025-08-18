from tenmo import Tensor
from sharedcopy import TensorLite
from operators import AddTensor, SubtractTensor
from backpropagationcopy import BackwardFn, Delegate


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
        # Deduct already contributed portion
        _="""new_contrib = __tensor_op_tensor__[dtype, SubtractTensor](
            reshaped, output.base[]
        )"""

        #new_contrib = reshaped - output.tensor().base[]
        # Update base accumulator
        #output.tensor().base.init_pointee_move(reshaped^)
        #return [(ancestor, new_contrib, AddTensor)]
        return [(ancestor, reshaped, AddTensor), (output, gradients, SubtractTensor)]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn __str__(self) -> String:
        return "ReshapeBackward"

fn main():
    print("Yes")
