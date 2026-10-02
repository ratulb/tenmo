"""Mnemonics for tensor operations - internal constants used by the autograd engine.
"""
from .constants import MAX_RANK

comptime max_rank = MAX_RANK  # Alias: canonical home is shared.constants
comptime Noop = 0
comptime MulTensor = 1
comptime AddTensor = 2
comptime SubtractTensor = 3
comptime ZeroGrad = 4
comptime ScatterAddTensor = 5
comptime Add = 6
comptime Subtract = 7
comptime ReverseSubtract = 8
comptime Multiply = 9
comptime Divide = 10
comptime ReverseDivide = 11
comptime Equal = 12
comptime NotEqual = 13
comptime LessThan = 14
comptime LessThanEqual = 15
comptime GreaterThan = 16
comptime GreaterThanEqual = 17
comptime Overwrite = 18
comptime RELU_FORWARD = 19
comptime SQRT = 20
comptime SQRT_BACKWARD = 21
comptime LOG = 22
comptime dot = 23  # dot product
comptime vm = 24  # vector & tensor matmul
comptime mv = 25  # tensor & vector matmul
comptime mm = 26  # tensor & tensor matmul
comptime invalid = 27  # Invalid case


comptime EXP = 40
comptime NEGATE = 41
comptime ABS = 42
comptime MAX = 43
comptime MIN = 44
comptime POW = 45
comptime TANH_FORWARD = 46
comptime SIGMOID_FORWARD = 47
comptime SIGMOID_BACKWARD = 48
comptime TANH_BACKWARD = 49
comptime LOG_BACKWARD = 50
comptime INVERT = 51
comptime SUM = 52
comptime MEAN = 53
comptime PRODUCT = 54
comptime ABS_BACKWARD = 55
comptime GELU_FORWARD = 58
comptime DEFAULT_INDEX_DTYPE = DType.int64
