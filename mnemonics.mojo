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
###################
### matul ###########
comptime dot = 22  # dot product
comptime vm = 23  # vector & tensor matmul
comptime mv = 24  # tensor & vector matmul
comptime mm = 25  # tensor & tensor matmul
comptime invalid = 26  # Invalid case

########net##############

comptime LINEAR = 27
comptime LINEAR_BLAS = 28
comptime RELU = 29
comptime SIGMOID = 30
comptime TANH = 31
comptime DROPOUT = 32
comptime CONV2D = 33
comptime FLATTEN = 34
comptime MAXPOOL2D = 35
#####################
comptime max_rank = 8 # Change this to extend/reduce max supported dimension
