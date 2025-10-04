from tensors import Tensor
from sys import simdwidthof
from os import abort

# from runtime.asyncrt import num_physical_cores
# from sys import num_logical_cores, num_physical_cores

alias Noop = 0
alias MulTensor = 1
alias AddTensor = 2
alias SubtractTensor = 3
alias ZeroGrad = 4
alias Add = 5
alias Subtract = 6
alias Multiply = 7
alias Divide = 8
alias Equal = 9
alias NotEqual = 10
alias LessThan = 11
alias LessThanEqual = 12
alias GreaterThan = 13
alias GreaterThanEqual = 14


@always_inline
fn scalar_ops[
    dtype: DType, op: Int
](lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Scalar[dtype]:
    result = Scalar[dtype](0)
    if op == Add:
        result = lhs + rhs
    elif op == Subtract:
        result = lhs - rhs
    elif op == Multiply:
        result = lhs * rhs
    elif op == Divide:
        result = lhs / rhs
    else:
        abort("operators -> scalar_ops: unsupported operation")
    return result


fn main() raises:
    pass
