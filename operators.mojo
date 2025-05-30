from memory import UnsafePointer
from tensors import Tensor


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
    if (
        left_operand[].grad_tensor_initialized()
        and right_operand[].grad_tensor_initialized()
    ):
        left_operand[].grad[] = left_operand[].grad[] + right_operand[].grad[]


alias Noop = 0
alias AddScalar = 1


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
