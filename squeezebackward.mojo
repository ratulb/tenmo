from tensors import Tensor
from operators import AddTensor
from shared import TensorLite
from intlist import IntList
from backpropagation import Delegate, BackwardFn


@fieldwise_init
@register_passable
struct SqueezeBackward[dtype: DType](Copyable):

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:

        ancestor = output.ancestry().get(0)[]
        gradients = output.gradients()[]

        var original_shape = ancestor.shape()
    
        #Create gradient with the same shape as original tensor
        var final_grad = gradients.reshape(original_shape, requires_grad=False)
        return [(ancestor, final_grad, AddTensor)]


fn main():
    print("passes")
