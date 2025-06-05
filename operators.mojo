from memory import UnsafePointer
from tensors import Tensor
from algorithm import vectorize
from sys import simdwidthof


fn _tensor_op_tensor[
    dtype: DType, op: Int
](this: Tensor[dtype], that: Tensor[dtype]) raises -> Tensor[dtype]:
    if this.shape != that.shape:
        raise Error(
            "_tensor_op__tensor(" + String(op) + ")  -> Dimension mismatch:",
            this.shape,
            that.shape,
        )
    requires_grad = this.requires_grad or that.requires_grad
    out = Tensor[dtype](this.shape, requires_grad)

    @parameter
    fn mul_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            (
                this.data.load[width=simd_width](idx)
                * that.data.load[width=simd_width](idx)
            ),
        )

    @parameter
    fn add_elems[simd_width: Int](idx: Int):
        out.data.store[width=simd_width](
            idx,
            (
                this.data.load[width=simd_width](idx)
                * that.data.load[width=simd_width](idx)
            ),
        )

    if op == MulTensor:
        vectorize[mul_elems, simdwidthof[dtype]()](out.numels())
    elif op == AddTensor:
        vectorize[add_elems, simdwidthof[dtype]()](out.numels())

    return out


fn _Noop():
    print("noop")


fn _AddScalar[
    dtype: DType
](
    left_operand: UnsafePointer[Tensor[dtype]],
    right_operand: UnsafePointer[Tensor[dtype]],
    value: Scalar[dtype],
) raises:
    print("_Addcalar")
    _ = """if (
        left_operand[].grad_tensor_initialized()
        and right_operand[].grad_tensor_initialized()
    ):"""
    left_operand[].grad[] = left_operand[].grad[] + right_operand[].grad[]


alias Noop = 0
alias AddScalar = 1
alias MulTensor = 2
alias AddTensor = 3


@value
struct GradFn[dtype: DType]:
    alias PtrType = UnsafePointer[Tensor[dtype]]
    var func_index: Int
    var left_operand: Self.PtrType
    var right_operand: Self.PtrType
    var factor: Scalar[dtype]

    fn __init__(out self):
        self.func_index = 0  # Noop
        self.left_operand = Self.PtrType()
        self.right_operand = Self.PtrType()
        self.factor = 0

    fn __init__(
        out self,
        _func_index: Int,
        left: Self.PtrType,
        right: Self.PtrType,
        value: Scalar[dtype],
    ):
        self.func_index = _func_index
        self.left_operand = left
        self.right_operand = right
        self.factor = value

    fn __call__(self) raises:
        if self.func_index == Noop:
            _Noop()
        elif self.func_index == AddScalar:
            _AddScalar(self.left_operand, self.right_operand, self.factor)
        else:
            print("What to do?")


fn main():
    h = "howdy"
    hptr = UnsafePointer(to=h)
    _h =hptr[]
    _hptr = UnsafePointer(to=_h)
    print("Howdy?", hptr == _hptr)


