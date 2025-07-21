from tensors import Tensor
from shared import TensorLike
from utils import Variant


trait Differentiable(Copyable & Movable):
    fn backward[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        ...


alias GradFn[T: Differentiable & Copyable & Movable] = Variant[T]


struct BackwardFn[dtype: DType, T: Differentiable & Copyable & Movable](
    Copyable & Movable
):
    var grad_fn: GradFn[T]

    fn __init__(out self, grad_fn: GradFn):
        self.grad_fn = grad_fn

    fn __moveinit__(out self, owned other: Self):
        self.grad_fn = other.grad_fn

    fn __copyinit__(out self, other: Self):
        self.grad_fn = other.grad_fn

    fn __call__(
        self, out_ptr: UnsafePointer[Tensor[dtype]]
    ) -> List[Tuple[TensorLike[dtype], Tensor[dtype], Int]]:
        return self.grad_fn[T].__call__[dtype](out_ptr)


fn main():
    print("Yes")
