from tensors import Tensor
from shared import TensorLike

fn main():
    pass

@fieldwise_init
struct BackwardFn[dtype: DType](Movable & Copyable):
    alias Opcode = Int
    alias GradTensor = Tensor[dtype]
    alias Recipient = TensorLike[dtype]
    alias Triple = (Self.Recipient, Self.GradTensor, Self.Opcode)
    alias GradOutputs = List[Self.Triple]
    alias GradFn = fn(gradients: Self.GradTensor) escaping -> Self.GradOutputs

    var grad_fn: Self.GradFn

    fn __call__(self, gradients: Self.GradTensor) -> Self.GradOutputs:
        return self.grad_fn(gradients)
