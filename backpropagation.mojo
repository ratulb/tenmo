from tenmo import Tensor
from std.utils import Variant
from walkback import *
from common_utils import panic
from gradbox import Gradbox
from buffers import Buffer
from ndbuffer import NDBuffer
from intarray import IntArray
from strides import Strides
from shapes import Shape
from device_transfer import Flow
from device import GPU
from blashandle import BLASHandleLite
from crossentropy import Reduction
# Centralized backward operation tags

comptime BACKWARD_ADD = 0
comptime BACKWARD_MULTIPLY = 1
comptime BACKWARD_RELU = 2
comptime BACKWARD_MATMUL_ND = 3
comptime BACKWARD_MATMUL_2D = 4
comptime BACKWARD_TRANSPOSE = 5
comptime BACKWARD_PERMUTE = 6
comptime BACKWARD_SIGMOID = 7
comptime BACKWARD_SOFTMAX = 8
comptime BACKWARD_CE_CLASS_INDICES = 9
comptime BACKWARD_CE_PROBABILITIES = 10
comptime BACKWARD_TANH = 11
comptime BACKWARD_SUB = 12
comptime BACKWARD_RESHAPE = 13
comptime BACKWARD_VIEW = 14
comptime BACKWARD_MEAN = 15
comptime BACKWARD_SUM = 16
comptime BACKWARD_LOG_SOFTMAX = 17
comptime BACKWARD_CONTIGUOUS = 18
comptime BACKWARD_DIVIDE = 19
comptime BACKWARD_MATRIX_VECTOR_MUL = 20
comptime BACKWARD_VECTOR_MATMUL = 21
comptime BACKWARD_ADD_SCALAR = 22
comptime BACKWARD_MULTIPLY_SCALAR = 23
comptime BACKWARD_SUB_SCALAR = 24
comptime BACKWARD_DIV_SCALAR = 25
comptime BACKWARD_RIGHT_DIV_SCALAR = 26
comptime BACKWARD_EXPONENTIATION = 27
comptime BACKWARD_DOT = 28
comptime BACKWARD_EXPAND = 29
comptime BACKWARD_FLATTEN = 30
comptime BACKWARD_SQUEEZE = 31
comptime BACKWARD_UNSQUEEZE = 32
comptime BACKWARD_SHUFFLE = 33
comptime BACKWARD_MINMAX = 34
comptime BACKWARD_TILE = 35
comptime BACKWARD_LOG = 36
comptime BACKWARD_SQRT = 37
comptime BACKWARD_CLIP = 38
comptime BACKWARD_VARIANCE = 39
comptime BACKWARD_STD = 40
comptime BLAS_BACKWARD_MATMUL_2D = 41
comptime BACKWARD_CONCAT = 42
comptime BACKWARD_STACK = 43
comptime BACKWARD_PAD = 44
comptime BACKWARD_FUSED_CONV = 45
comptime BACKWARD_MAXPOOL2D = 46
comptime BACKWARD_DROPOUT = 47
comptime BACKWARD_EXPONENTIAL = 48
comptime BACKWARD_DEVICE_TRANSFER = 49
comptime BACKWARD_MAX_SCALAR = 50
comptime BACKWARD_MIN_SCALAR = 51

@fieldwise_init
struct NullArgument(RegisterPassable & ImplicitlyCopyable):
    pass

#comptime ArgumentType[dtype: DType] = Variant[NullArgument, Bool, Int, Scalar[dtype], IntArray, Buffer[dtype], Tuple[Shape, Strides, Int], Tuple[IntArray, Bool], Tuple[Int, List[Int]], Tuple[IntArray, Bool, NDBuffer[dtype]], Tuple[IntArray, NDBuffer[dtype]], Tuple[IntArray, Shape], Tuple[Scalar[dtype], Scalar[dtype]], Tuple[Int, Bool, Bool], Tuple[Int, Bool, Bool, Scalar[dtype]],Tuple[Int, Int], Tuple[List[Tuple[Int, Int]], String], Tuple[Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int], Tuple[Int, Int, Int, Shape, NDBuffer[dtype]], Tuple[Flow, Optional[GPU]], Tuple[Bool, Bool, BLASHandleLite[dtype]], Tuple[NDBuffer[dtype], NDBuffer[DType.int32], Shape, Reduction, Int, Scalar[dtype], Int, Int, Int], Tuple[NDBuffer[dtype], NDBuffer[dtype], Shape, Reduction, Int, Int, Int]]
comptime CEProbabilitiesArg[dtype: DType] = Tuple[NDBuffer[dtype], NDBuffer[dtype], Shape, Reduction, Int, Int, Int]
comptime CEClassIndices[dtype: DType] = Tuple[NDBuffer[dtype], NDBuffer[DType.int32], Shape, Reduction, Int, Scalar[dtype], Int, Int, Int]
comptime ArgShuffle = Tuple[Int, List[Int]]
comptime ArgMinmax[dtype: DType] = Tuple[IntArray, Bool, NDBuffer[dtype]]
comptime ArgSoftmax[dtype: DType] = Tuple[IntArray, NDBuffer[dtype]]
comptime ArgTile = Tuple[IntArray, Shape]
comptime ArgClip[dtype: DType] = Tuple[Scalar[dtype], Scalar[dtype]]
comptime ArgVariance = Tuple[Int, Bool, Bool]
comptime ArgStd[dtype: DType] = Tuple[Int, Bool, Bool, Scalar[dtype]]
comptime ArgStack = Tuple[Int, Int]
comptime ArgPad = Tuple[List[Tuple[Int, Int]], String]
comptime ArgFusedCol2Im = Tuple[Int, Int, Int, Int, Int, Int, Int, Int, Int, Int, Int]
comptime ArgMaxPool2d = Tuple[Int, Int, Int, Shape, NDBuffer[DType.int64]]
comptime ArgDeviceTransfer = Tuple[Flow, Optional[GPU]]
comptime ArgumentType[dtype: DType] = Variant[NullArgument, Bool, Int, Scalar[dtype], IntArray, Buffer[dtype], Tuple[Shape, Strides, Int], Tuple[IntArray, Bool], ArgShuffle, ArgMinmax[dtype], ArgSoftmax[dtype], ArgTile, ArgClip[dtype], ArgVariance, ArgStd[dtype], ArgStack, ArgPad, ArgFusedCol2Im, ArgMaxPool2d, ArgDeviceTransfer, Tuple[Bool, Bool, BLASHandleLite[dtype]], CEClassIndices[dtype], CEProbabilitiesArg[dtype]]

@fieldwise_init
struct BackwardFnArg[dtype: DType](ImplicitlyCopyable & Movable):
    var op_code: Int
    var arg: ArgumentType[Self.dtype]

    @staticmethod
    fn boolean(op_code: Int, is_true: Bool) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, ArgumentType[Self.dtype](is_true))

    @staticmethod
    fn integer(op_code: Int, value: Int) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, ArgumentType[Self.dtype](value))


    @staticmethod
    fn scalar(op_code: Int, scalar: Scalar[Self.dtype]) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, ArgumentType[Self.dtype](scalar))

    @staticmethod
    fn from_intarray(op_code: Int, array: IntArray) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, ArgumentType[Self.dtype](array))

    @staticmethod
    fn null_arg(op_code: Int) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, ArgumentType[Self.dtype](NullArgument()))


@fieldwise_init
struct Backward[dtype: DType](RegisterPassable & ImplicitlyCopyable):

    @staticmethod
    fn invoke(
            output: Tensor[Self.dtype]
    ) -> List[
        Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        """O(1) dispatch using integer tag comparison.
        Order: Most common operations first for branch prediction.
        """

        # ========== TIER 1: MOST COMMON ==========
        ref arg = output.bwd_fn_arg()
        var op_code = arg.op_code
        if op_code == BACKWARD_ADD:
            return AddBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_ADD_SCALAR:
            return AddBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_MULTIPLY_SCALAR:
            return MultiplyBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_MULTIPLY:
            return MultiplyBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SUB:
            return SubBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SUB_SCALAR:
            return SubLeftRightBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_DIV_SCALAR:
            return TrueDivBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_RIGHT_DIV_SCALAR:
            return RightTrueDivBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_DIVIDE:
            return DivideBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SUM:
            return SumBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MEAN:
            return MeanBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_RESHAPE:
            return ReshapeBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MATMUL_2D:
            return Matmul2dBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MATMUL_ND:
            return MatmulNdBackward[Self.dtype].backward(output)
        elif op_code == BLAS_BACKWARD_MATMUL_2D:
            return BLASMatmul2dBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_RELU:
            return ReLUBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_VIEW:
            return ViewBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_TRANSPOSE:
            return TransposeBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_CE_CLASS_INDICES:
            return CEClassIndicesBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_CE_PROBABILITIES:
            return CEProbabilitiesBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_CONTIGUOUS:
            return ContiguousBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SIGMOID:
            return SigmoidBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_EXPONENTIATION:
            return ExponentiationBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_EXPAND:
            return ExpandBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_FLATTEN:
            return FlattenBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SQUEEZE:
            return SqueezeBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_UNSQUEEZE:
            return UnsqueezeBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_PERMUTE:
            return PermuteBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SHUFFLE:
            return ShuffleBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MINMAX:
            return MinMaxBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SOFTMAX:
            return SoftmaxBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_LOG_SOFTMAX:
            return LogSoftmaxBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_TILE:
            return TileBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_TANH:
            return TanhBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_LOG:
            return LogBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_CLIP:
            return ClipBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SQRT:
            return SqrtBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_VARIANCE:
            return VarianceBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_STD:
            return StdBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_CONCAT:
            return ConcatBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_STACK:
            return StackBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_PAD:
            return PadBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_FUSED_CONV:
            return FusedCol2ImBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MAXPOOL2D:
            return MaxPool2dBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_DROPOUT:
            return DropoutBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_EXPONENTIAL:
            return ExponentialBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_DEVICE_TRANSFER:
            return DeviceTransferBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MAX_SCALAR:
            return MaxBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_MIN_SCALAR:
            return MinBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_VECTOR_MATMUL:
            return VectorMatmulNdBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MATRIX_VECTOR_MUL:
            return MatrixVectorMulNdBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_DOT:
            return DotBackward[Self.dtype].backward(output)

        else: return []


fn main() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3([[[1, 0, 3]]], requires_grad=True)
    var b = Tensor[dtype].d2([[1], [2], [3]], requires_grad=True)
    #var r =  a.reshape(3,1)
    #print("r requires_grad? ", r.requires_grad)
    var v = a.into_view()
    var m = v.sigmoid()
    m.backward()
    a.grad().print()
    b.grad().print()
    m.print()

    var A = Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    var B = A.transpose(-1, 0)
    var C = B * 89
    C.backward()
    A.grad().print()
