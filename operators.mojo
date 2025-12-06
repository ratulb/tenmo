alias Noop = 0
alias MulTensor = 1
alias AddTensor = 2
alias SubtractTensor = 3
alias ZeroGrad = 4
alias Add = 5
alias Subtract = 6
alias ReverseSubtract = 7
alias Multiply = 8
alias Divide = 9
alias ReverseDivide = 10
alias Equal = 11
alias NotEqual = 12
alias LessThan = 13
alias LessThanEqual = 14
alias GreaterThan = 15
alias GreaterThanEqual = 16
alias Overwrite = 17
alias SigmoidOp = 18
alias Log = 19
alias Exp = 20
alias TanhForwardOp = 21
alias TanhBackwardOp = 22
alias ReLUForwardOp = 23
alias ReLUBackwardOp = 24
alias SqrtForwardOp=25
alias SqrtBackwardOp=26
###################
### matul ###########
alias dot = 27  # dot product
alias vm = 28  # vector & tensor matmul
alias mv = 29  # tensor & vector matmul
alias mm = 30  # tensor & tensor matmul
alias invalid = 31  # Invalid case


fn main() raises:
    pass
