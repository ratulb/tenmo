comptime Noop = 0
comptime MulTensor = 1
comptime AddTensor = 2
comptime SubtractTensor = 3
comptime ZeroGrad = 4
comptime Add = 5
comptime Subtract = 6
comptime ReverseSubtract = 7
comptime Multiply = 8
comptime Divide = 9
comptime ReverseDivide = 10
comptime Equal = 11
comptime NotEqual = 12
comptime LessThan = 13
comptime LessThanEqual = 14
comptime GreaterThan = 15
comptime GreaterThanEqual = 16
comptime Overwrite = 17
comptime ReLUForwardOp = 18
comptime ReLUBackwardOp = 19
comptime SqrtForwardOp = 20
comptime SqrtBackwardOp = 21
comptime LOG = 22
###################
### matul ###########
comptime dot = 23  # dot product
comptime vm = 24  # vector & tensor matmul
comptime mv = 25  # tensor & vector matmul
comptime mm = 26  # tensor & tensor matmul
comptime invalid = 27  # Invalid case

########net##############

comptime LINEAR = 28
comptime LINEAR_BLAS = 29
comptime RELU = 30
comptime SIGMOID = 31
comptime TANH = 32
comptime DROPOUT = 33
comptime CONV2D = 34
comptime FLATTEN = 35
comptime MAXPOOL2D = 36

#####################
comptime max_rank = 8  # Change this to extend/reduce max supported dimension
comptime EXP = 37
comptime SQRT = 38
comptime NEGATE = 39
comptime ABS = 40
comptime MAX = 41
comptime MIN = 42
comptime POW = 42
