from tensors import Tensor
from shared import TensorLike


trait Differentiable(Copyable & Movable):
    fn __call__[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        ...


struct BackwardFn[dtype: DType](Copyable & Movable):
    var facade: BackwardFacade[dtype]

    fn __init__(out self):
        self.facade = BackwardFacade[dtype]()

    fn __moveinit__(out self, owned other: Self):
        self.facade = other.facade

    fn __copyinit__(out self, other: Self):
        self.facade = other.facade

    fn __call__(
        self, out_ptr: UnsafePointer[Tensor[dtype]]
    ) -> List[Tuple[TensorLike[dtype], Tensor[dtype], Int]]:
        return self.facade.__call__[dtype](out_ptr)

    fn replace[
        Target: Differentiable & Copyable & Movable
    ](mut self, target: Target):
        self.facade.replace(target))
        print("Done it successfully")


struct BackwardFacade[
    dtype: DType,
    Target: Differentiable & Copyable & Movable,
](Copyable & Movable):
    var target: Target

    fn __init__(out self, target: Target):
        self.target = target

    fn __moveinit__(out self, owned other: Self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn repalce(mut self, target: Target):
        self.target = target
        print("Installed a new target!")

    fn __call__[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        print("Don't you see? I got called!")
        return self.target(out_ptr)


struct NoopBackward[dtype: DType](Differentiable & Copyable & Movable):
    fn __init__(out self):
        pass

    fn __moveinit__(out self, owned other: Self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn __call__[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        print("Don't you see? I got called too!")
        return []


from operators import __tensor_op_tensor__, AddTensor, SubtractTensor


struct BackwardReshape[dtype: DType](Differentiable & Copyable & Movable):
    fn __init__(out self):
        pass

    fn __moveinit__(out self, owned other: Self):
        pass

    fn __copyinit__(out self, other: Self):
        pass

    fn __call__[
        dtype: DType
    ](self, out_ptr: UnsafePointer[Tensor[dtype]]) -> List[
        Tuple[TensorLike[dtype], Tensor[dtype], Int]
    ]:
        output = out_ptr[]
        gradients = output.grad[]
        ancestor = output.ancestors.get(0)[]
        reshaped = gradients.reshape(ancestor.shape())
        # Deduct already contributed portion
        new_contrib = __tensor_op_tensor__[dtype, SubtractTensor](
            reshaped, output.base[]
        )

        # Update base accumulator
        output.base.init_pointee_move(reshaped^)
        return [(ancestor, new_contrib, AddTensor)]


fn main():
    print("Yes")
    _ = """backward_fn = BackwardFacade[DType.float32]().into_backward_fn()
    backward_fn.replace[BackwardReshape[DType.float32]](
        BackwardReshape[DType.float32]()
    )"""
    d = Differentiator()
    out = d.reshape()
    out.backwardFn.value()((UnsafePointer(to=Tensor.scalar(10))))


@fieldwise_init
struct Differentiator[dtype: DType = DType.float32](Copyable & Movable):
    var backwardFn: Optional[BackwardFn[dtype]]

    fn __init__(out self):
        self.backwardFn = None

    fn reshape(self) -> Differentiator[dtype]:
        out = Differentiator[dtype]()
        backward_fn = BackwardFn[dtype]()
        out.backwardFn = Optional(backward_fn)
        return out
