from tensors import Tensor
from shared import TensorLike


trait Differentiable(Copyable & Movable):
    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        ...


struct BackwardFn[dtype: DType, GradFn: Differentiable](Copyable & Movable):
#struct BackwardFn[dtype: DType](Copyable & Movable):
    var grad_fn: GradFn

    fn __init__(out self, grad_fn: Differentiable):
        self.grad_fn = grad_fn

    fn __moveinit__(out self, owned other: Self):
        self.grad_fn = other.grad_fn

    fn __copyinit__(out self, other: Self):
        self.grad_fn = other.grad_fn

    fn __call__(
        self, out_ptr: UnsafePointer[Tensor[dtype]]
    ) -> List[Tuple[TensorLike[dtype], Tensor[dtype], Int]]:
        return self.grad_fn[](out_ptr)


fn main():
    print("Yes")
