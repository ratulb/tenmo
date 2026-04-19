from ancestry import Ancestor
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
from device import Device
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
comptime BACKWARD_ADD_BROADCAST = 23
comptime BACKWARD_MULTIPLY_SCALAR = 24
comptime BACKWARD_SUB_SCALAR = 25
comptime BACKWARD_SUBTRACT_BROADCAST = 26
comptime BACKWARD_DIV_SCALAR = 27
comptime BACKWARD_RIGHT_DIV_SCALAR = 28
comptime BACKWARD_EXPONENTIATION = 29
comptime BACKWARD_DOT = 30
comptime BACKWARD_EXPAND = 31
comptime BACKWARD_FLATTEN = 32
comptime BACKWARD_SQUEEZE = 33
comptime BACKWARD_UNSQUEEZE = 34
comptime BACKWARD_SHUFFLE = 35
comptime BACKWARD_MINMAX = 36
comptime BACKWARD_TILE = 37
comptime BACKWARD_LOG = 38
comptime BACKWARD_SQRT = 39
comptime BACKWARD_CLIP = 40
comptime BACKWARD_VARIANCE = 41
comptime BACKWARD_STD = 42
comptime BLAS_BACKWARD_MATMUL_2D = 43
comptime BACKWARD_CONCAT = 44
comptime BACKWARD_STACK = 45
comptime BACKWARD_PAD = 46
comptime BACKWARD_FUSED_CONV = 47
comptime BACKWARD_MAXPOOL2D = 48
comptime BACKWARD_DROPOUT = 49
comptime BACKWARD_EXPONENTIAL = 50
comptime BACKWARD_DEVICE_TRANSFER = 51
comptime BACKWARD_MAX_SCALAR = 52
comptime BACKWARD_MIN_SCALAR = 53
comptime BACKWARD_MULTIPLY_BROADCAST = 54


trait ArgumentType(ImplicitlyCopyable & Movable):
    pass


comptime DestroyerFn = def(UnsafePointer[UInt8, MutAnyOrigin]) -> None


def make_destroyer[T: ArgumentType]() -> DestroyerFn:
    def destroy(p: UnsafePointer[UInt8, MutAnyOrigin]) -> None:
        p.bitcast[T]().destroy_pointee()
        p.bitcast[T]().free()

    return destroy


comptime CopyFn = fn(UnsafePointer[UInt8, MutAnyOrigin]) -> UnsafePointer[
    UInt8, MutAnyOrigin
]


def make_copier[T: ArgumentType]() -> CopyFn:
    def copy_it(
        src: UnsafePointer[UInt8, MutAnyOrigin]
    ) -> UnsafePointer[UInt8, MutAnyOrigin]:
        var dst = alloc[T](1)
        dst.init_pointee_copy(src.bitcast[T]()[])
        return dst.bitcast[UInt8]()

    return copy_it


@fieldwise_init
struct BackwardFnArg[dtype: DType](ImplicitlyCopyable & Movable):
    var op_code: Int
    var ptr: UnsafePointer[UInt8, MutAnyOrigin]  # type-erased arg
    var destroy: DestroyerFn
    var copy_fn: CopyFn

    fn __init__[T: ArgumentType, //](out self, op_code: Int, var arg: T):
        var p = alloc[T](1)
        p.init_pointee_move(arg^)
        self.op_code = op_code
        self.ptr = p.bitcast[UInt8]()
        self.destroy = make_destroyer[T]()
        self.copy_fn = make_copier[T]()

    fn __del__(deinit self):
        self.destroy(self.ptr)  # calls T.__del__

    fn __moveinit__(out self, deinit take: Self):
        self.op_code = take.op_code
        self.ptr = take.ptr
        self.destroy = take.destroy
        self.copy_fn = take.copy_fn

    fn __copyinit__(out self, copy: Self):
        self.op_code = copy.op_code
        self.destroy = copy.destroy
        self.copy_fn = copy.copy_fn
        self.ptr = self.copy_fn(copy.ptr)  # deep copy via T.__copyinit__

    fn get[T: ArgumentType](ref self) -> ref[self.ptr] T:
        return self.ptr.bitcast[T]()[]

    @staticmethod
    fn null_arg(op_code: Int) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, NullArg(0))

    @staticmethod
    fn boolean_arg(op_code: Int, is_true: Bool) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, Boolean(is_true))

    @staticmethod
    fn scalar_arg(
        op_code: Int, value: Scalar[Self.dtype]
    ) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, ScalarArg[Self.dtype](value))

    @staticmethod
    fn integer_arg(op_code: Int, value: Int) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, Integer(value))

    @staticmethod
    fn from_intarray(
        op_code: Int, array: IntArray
    ) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, IntArrayArg(array))

    @staticmethod
    fn from_buffer(
        op_code: Int, buffer: Buffer[Self.dtype]
    ) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, BufferArg[Self.dtype](buffer))

    @staticmethod
    fn from_ndbuffer(
        op_code: Int, ndb: NDBuffer[Self.dtype]
    ) -> BackwardFnArg[Self.dtype]:
        return BackwardFnArg[Self.dtype](op_code, NDBufferArg[Self.dtype](ndb))


@fieldwise_init
struct NullArg(ArgumentType):
    var zero: UInt8


@fieldwise_init
struct Boolean(ArgumentType):
    var is_true: Bool


@fieldwise_init
struct ScalarArg[dtype: DType](ArgumentType):
    var value: Scalar[Self.dtype]


@fieldwise_init
struct Integer(ArgumentType):
    var value: Int


@fieldwise_init
struct IntArrayArg(ArgumentType):
    var array: IntArray


@fieldwise_init
struct ReductionArg(ArgumentType):
    var axes: IntArray
    var keepdims: Bool


@fieldwise_init
struct BlasArg[dtype: DType](ArgumentType):
    var transpose_A: Bool
    var transpose_B: Bool
    var blas: BLASHandleLite[Self.dtype]


@fieldwise_init
struct StackArg(ArgumentType):
    var axis: Int
    var num_tensors: Int


@fieldwise_init
struct BufferArg[dtype: DType](ArgumentType):
    var buffer: Buffer[Self.dtype]


@fieldwise_init
struct NDBufferArg[dtype: DType](ArgumentType):
    var ndb: NDBuffer[Self.dtype]


@fieldwise_init
struct ViewArg(ArgumentType):
    var shape: Shape
    var strides: Strides
    var offset: Int


@fieldwise_init
struct ShuffleArg(ArgumentType):
    var axis: Int
    var permutation: List[Int]

    fn __copyinit__(out self, copy: Self):
        self.axis = copy.axis
        self.permutation = copy.permutation.copy()


@fieldwise_init
struct PadArg(ArgumentType):
    var pad: List[Tuple[Int, Int]]
    var mode: String

    fn __copyinit__(out self, copy: Self):
        self.pad = copy.pad.copy()
        self.mode = copy.mode.copy()


@fieldwise_init
struct MinMaxArg[dtype: DType](ArgumentType):
    var axes: IntArray
    var keepdims: Bool
    var mask: NDBuffer[Self.dtype]


@fieldwise_init
struct SoftmaxArg[dtype: DType](ArgumentType):
    var axes: IntArray
    var softmax_out: NDBuffer[Self.dtype]


@fieldwise_init
struct ClipArg[dtype: DType](ArgumentType):
    var min_val: Scalar[Self.dtype]
    var max_val: Scalar[Self.dtype]


@fieldwise_init
struct TilesArg(ArgumentType):
    var repeat: IntArray
    var orig_shape: Shape


@fieldwise_init
struct StdArg[dtype: DType](ArgumentType):
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool
    var epsilon: Scalar[Self.dtype]


@fieldwise_init
struct Backward[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    @staticmethod
    fn invoke(
        output: Ancestor[Self.dtype],
    ) -> List[
        Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
    ] where Self.dtype.is_floating_point():
        if not output.has_ancestry():  # guard!
            print("Inside Backward invoke: output ancestry is not set")
            return []
        ref arg = output.ancestry().backward_fn_arg()
        var op_code = arg.op_code
        if op_code == BACKWARD_ADD_SCALAR:
            return AddBackwardScalar[Self.dtype].backward(output)
        if op_code == BACKWARD_ADD:
            return AddBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_ADD_BROADCAST:
            return AddBroadcastBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SUB:
            return SubBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_SUB_SCALAR:
            return SubLeftRightBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_SUBTRACT_BROADCAST:
            return SubtractBroadcastBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MULTIPLY_SCALAR:
            return MultiplyBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_MULTIPLY:
            return MultiplyBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MULTIPLY_BROADCAST:
            return MultiplyBroadcastBackward[Self.dtype].backward(output)
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
        elif op_code == BACKWARD_TRANSPOSE:
            return TransposeBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_PERMUTE:
            return PermuteBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_RELU:
            return ReLUBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_VIEW:
            return ViewBackward[Self.dtype].backward(output)
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
        elif op_code == BACKWARD_DEVICE_TRANSFER:
            return DeviceTransferBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MAX_SCALAR:
            return MaxBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_MIN_SCALAR:
            return MinBackwardScalar[Self.dtype].backward(output)
        elif op_code == BACKWARD_EXPONENTIAL:
            return ExponentialBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_VECTOR_MATMUL:
            return VectorMatmulNdBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_MATRIX_VECTOR_MUL:
            return MatrixVectorMulNdBackward[Self.dtype].backward(output)
        elif op_code == BACKWARD_DOT:
            return DotBackward[Self.dtype].backward(output)
        else:
            return []


fn main() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].rand(3, 5, 3, requires_grad=True)
    var B = Tensor[dtype].rand(3, 3, requires_grad=True)
    C = A.matmul(B) + A * 42 + 100
    C.backward()
    A.grad().print()
    B.grad().print()
    print("passes")
