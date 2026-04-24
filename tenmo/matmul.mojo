from .tensor import Tensor
from std.sys import simd_width_of
from .matrixshapevalidator import MatrixShapeValidator
from .backpropagation import (
    BackwardFnArg,
    BACKWARD_MATMUL_ND,
    BACKWARD_MATMUL_2D,
)
from .mnemonics import AddTensor, mm, vm, mv, dot, invalid
from .gradbox import Gradbox
from .shapes import Shape
from .common_utils import panic
from .vectormatrix import VectorMatmulNd
from .matrixvector import MatrixVectorMulNd
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from .intarray import IntArray
from .ancestry import Ancestor


@fieldwise_init
struct Matmul2dBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref grad_out = output.gradients()[]
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)

        var result = List[
            Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        # GRADIENT FOR A: dL/dA = grad_out × B^T
        if A.requires_grad:
            var B_buffer = B.buffer()
            var ndb = grad_out.buffer.matmul_2d(
                B_buffer.transpose(IntArray(-1, -2))
            )
            var grad_A = Gradbox[Self.dtype](ndb^, share=False)

            result.append((A, grad_A^, AddTensor))

        # GRADIENT FOR B: dL/dB = A^T × grad_out
        if B.requires_grad:
            var A_buffer = A.buffer()
            var A_buffer_transposed = A_buffer.transpose(IntArray(-1, -2))
            var ndb = A_buffer_transposed.matmul_2d(grad_out.buffer)
            var grad_B = Gradbox[Self.dtype](ndb^, share=False)

            result.append((B^, grad_B^, AddTensor))

        return result^


@fieldwise_init
struct Matmul2d[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    @always_inline
    fn forward[
        track_grad: Bool = True,
    ](A: Tensor[Self.dtype], B: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        var ndb = A.buffer.matmul_2d(B.buffer)
        var C = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var requires_grad = A.requires_grad or B.requires_grad
            if requires_grad:
                C.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_MATMUL_2D
                )
                C.add_ancestry(backwardFnArg^, A, B)

        return C^

    @staticmethod
    @always_inline
    fn forward(
        A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        var ndb = A.buffer.matmul_2d(B.buffer)
        var C = Gradbox[Self.dtype](ndb^, share=False)
        return C^

    @staticmethod
    @always_inline
    fn forward(
        A: Gradbox[Self.dtype], B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        var ndb = A.buffer.matmul_2d(B.buffer)
        var C = Gradbox[Self.dtype](ndb^, share=False)
        return C^


@fieldwise_init
struct MatmulNdBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref grad_out = output.gradients()[]
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)
        var A_buffer = A.buffer()
        var B_buffer = B.buffer()

        ref A_shape = A_buffer.shape
        ref B_shape = B_buffer.shape
        var results = List[
            Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        if A.requires_grad:
            var B_transposed = B_buffer.transpose(axes=IntArray(-1, -2))

            var A_batch_grad = MatmulNd[Self.dtype].forward(
                grad_out, Tensor[Self.dtype](B_transposed^, requires_grad=False)
            )
            var final_grad_A = A_batch_grad^.sum_over_broadcasted_axes(A_shape)

            results.append((A, final_grad_A^, AddTensor))

        if B.requires_grad:
            var A_transposed = A_buffer.transpose(axes=IntArray(-1, -2))
            var B_batch_grad = MatmulNd[Self.dtype].forward(
                Tensor[Self.dtype](A_transposed^, requires_grad=False), grad_out
            )

            var final_grad_B = B_batch_grad^.sum_over_broadcasted_axes(B_shape)

            results.append((B^, final_grad_B^, AddTensor))
        return results^


@fieldwise_init
struct MatmulNd[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](A: Tensor[Self.dtype], B: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        ref A_shape = A.shape()
        ref B_shape = B.shape()

        # Short-circuit for pure 2D case
        if A_shape.rank() == 2 and B_shape.rank() == 2:
            return Matmul2d[Self.dtype].forward[track_grad](A, B)

        var ndb = A.buffer.matmul_nd(B.buffer)
        var C = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var requires_grad = A.requires_grad or B.requires_grad
            if requires_grad:
                C.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_MATMUL_ND
                )
                C.add_ancestry(backwardFnArg^, A, B)

        return C^

    @always_inline
    @staticmethod
    fn forward(
        A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        ref A_shape = A.shape()
        ref B_shape = B.shape()

        if A_shape.rank() == 2 and B_shape.rank() == 2:
            return Matmul2d[Self.dtype].forward(A, B)
        var ndb = A.buffer.matmul_nd(B.buffer)
        var C = Gradbox[Self.dtype](ndb^, share=False)

        return C^

    @always_inline
    @staticmethod
    fn forward(
        A: Gradbox[Self.dtype], B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        ref A_shape = A.shape()
        ref B_shape = B.shape()

        if A_shape.rank() == 2 and B_shape.rank() == 2:
            return Matmul2d[Self.dtype].forward(A, B)

        var ndb = A.buffer.matmul_nd(B.buffer)
        var C = Gradbox[Self.dtype](ndb^, share=False)

        return C^


@fieldwise_init
struct Matmul[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True, mode: Int = mm
    ](A: Tensor[Self.dtype], B: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        comptime if mode == mm:
            # Step 1: Pure analysis - get the opcode
            var opcode = classify_matmul(A.shape(), B.shape())

            # Step 2: Simple dispatch based on opcode

            if dot == opcode:
                return A.dot[track_grad](B)

            if vm == opcode:
                return VectorMatmulNd[Self.dtype].forward[track_grad](A, B)

            if mv == opcode:
                return MatrixVectorMulNd[Self.dtype].forward[track_grad](A, B)

            if mm == opcode:
                return MatmulNd[Self.dtype].forward[track_grad](A, B)

            # Invalid case
            panic("Matmul: incompatible shapes")
            return Tensor[Self.dtype].scalar(0)

        elif mode == dot:
            return A.dot[track_grad](B)

        elif mode == vm:
            return VectorMatmulNd[Self.dtype].forward[track_grad](A, B)

        elif mode == mv:
            return MatrixVectorMulNd[Self.dtype].forward[track_grad](A, B)
        else:
            # Invalid case
            panic("Matmul: incompatible shapes")
            return Tensor[Self.dtype].scalar(0)

    @always_inline
    @staticmethod
    fn forward(
        A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        return MatmulNd[Self.dtype].forward(A, B)

    @always_inline
    @staticmethod
    fn forward(
        A: Gradbox[Self.dtype], B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        return MatmulNd[Self.dtype].forward(A, B)


fn classify_matmul(a: Shape, b: Shape) -> Int:
    var rank_a = a.rank()
    var rank_b = b.rank()

    if rank_a <= 1 and rank_b <= 1:
        return dot
    elif rank_a == 1 and rank_b >= 2:
        return vm
    elif rank_a >= 2 and rank_b == 1:
        return mv
    else:  # rank_a >= 2 and rank_b >= 2
        if a[-1] == b[-2]:
            return mm
        else:
            return invalid

