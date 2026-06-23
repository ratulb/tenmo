"""Mnemonics for tensor operations - internal constants used by the autograd engine.
"""
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


comptime LINEAR = 28  # Net
comptime LINEAR_BLAS = 29
comptime RELU = 30
comptime SIGMOID = 31
comptime TANH = 32
comptime DROPOUT = 33
comptime CONV2D = 34
comptime FLATTEN = 35
comptime MAXPOOL2D = 36
comptime LAYER_NORM = 37
comptime EMBEDDING = 38

comptime max_rank = 8  # Change this to extend/reduce max supported dimension
comptime EXP = 39
comptime NEGATE = 40
comptime ABS = 41
comptime MAX = 42
comptime MIN = 43
comptime POW = 44
comptime TANH_FORWARD = 45
comptime SIGMOID_FORWARD = 46
comptime SIGMOID_BACKWARD = 47
comptime TANH_BACKWARD = 48
comptime LOG_BACKWARD = 49
comptime INVERT = 50
comptime SUM = 51
comptime MEAN = 52
comptime PRODUCT = 53
comptime ABS_BACKWARD = 54

comptime DEFAULT_INDEX_DTYPE = DType.int64
