from .tensor import Tensor
from .backpropagation import (
    BackwardFnType,
    BackwardFn,
)
from .shared.mnemonics import AddTensor, mm, vm, mv, dot, invalid
from .gradbox import Gradbox
from .shared.shapes import Shape
from .shared.panic import panic
from .vectormatrix import VectorMatmulNd
from .matrixvector import MatrixVectorMulNd
from .multiplication import Multiplicator
from .shared.intarray import IntArray
from .ancestry import Ancestor
from .ndbuffer import NDBuffer
from .blas_ndbuffer import blas_matmul, BLASCache


def _blas_eligible[dtype: DType](
    A: NDBuffer[dtype], B: NDBuffer[dtype]
) -> Bool:
    """True iff the 2D operands can use the forward-only BLAS GEMM.

    Requires: BLAS opt-in (-D BLAS) + library loaded, dtype float32/float64,
    both on CPU, contiguous, and zero-offset (BLAS reads the base buffer
    pointer).
    """
    if not BLASCache.is_enabled():
        return False
    if not BLASCache.is_available():
        return False
    var ok_dtype = False
    comptime if dtype == DType.float32:
        ok_dtype = True
    elif dtype == DType.float64:
        ok_dtype = True
    if not ok_dtype:
        return False
    var a = A
    var b = B
    return (
        not a.is_on_gpu()
        and not b.is_on_gpu()
        and a.is_contiguous()
        and b.is_contiguous()
        and a.offset == 0
        and b.offset == 0
    )



def _blas_or_native[dtype: DType](
    A: NDBuffer[dtype],
    B: NDBuffer[dtype],
    transpose_A: Bool = False,
    transpose_B: Bool = False,
    sync: Bool = False,
) -> NDBuffer[dtype]:
    """CPU-only BLAS path guarded so non-float dtypes never instantiate
    blas_matmul (which is float-only). Falls back to native matmul_2d."""
    comptime if dtype == DType.float32 or dtype == DType.float64:
        if _blas_eligible(A, B):
            return blas_matmul[dtype](
                A, B, transpose_A=transpose_A, transpose_B=transpose_B, sync=sync
            )
    var a = A
    var b = B
    if transpose_A:
        a = a.transpose(IntArray(-1, -2))
    if transpose_B:
        b = b.transpose(IntArray(-1, -2))
    return a.matmul_2d(b, sync=sync)


@fieldwise_init
struct Matmul2dBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref grad_out = output.gradients()
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)

        # GRADIENT FOR A: dL/dA = grad_out × B^T (computed only if
        # needed; the id is always appended — parent_ids is the engine's
        # fanin-completion signal and must cover every ancestry parent).
        if A.requires_grad:
            var B_buffer = B.buffer()
            var ndb: NDBuffer[Self.dtype]
            ndb = _blas_or_native[Self.dtype](
                grad_out.buffer(), B_buffer, transpose_B=True
            )
            var grad_A = Gradbox[Self.dtype](ndb^)

            A.update_grad(grad_A^, AddTensor, None)
        parent_ids.append(A._id)

        # GRADIENT FOR B: dL/dB = A^T × grad_out (same contract).
        if B.requires_grad:
            var A_buffer = A.buffer()
            var ndb: NDBuffer[Self.dtype]
            ndb = _blas_or_native[Self.dtype](
                A_buffer, grad_out.buffer(), transpose_A=True
            )
            var grad_B = Gradbox[Self.dtype](ndb^)

            B.update_grad(grad_B^, AddTensor, None)
        parent_ids.append(B._id)
        grad_out.zero_grad()


@fieldwise_init
struct Matmul2d[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    @always_inline
    def forward[
        track_grad: Bool = True,
    ](
        A: Tensor[Self.dtype], B: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        var ndb: NDBuffer[Self.dtype]
        ndb = _blas_or_native[Self.dtype](A.buffer, B.buffer, sync=sync)
        var C = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var requires_grad = A.requires_grad or B.requires_grad
            if requires_grad:
                C.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    Matmul2dBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                C.add_ancestry(backwardFn^, A, B)

        return C^

    @staticmethod
    @always_inline
    def forward(
        A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        var ndb: NDBuffer[Self.dtype]
        ndb = _blas_or_native[Self.dtype](A.buffer, B.buffer())
        var C = Gradbox[Self.dtype](ndb^)
        return C^

    @staticmethod
    @always_inline
    def forward(
        A: Gradbox[Self.dtype], B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        var ndb: NDBuffer[Self.dtype]
        ndb = _blas_or_native[Self.dtype](A.buffer(), B.buffer)
        var C = Gradbox[Self.dtype](ndb^)
        return C^


@fieldwise_init
struct MatmulNdBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref grad_out = output.gradients()
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)
        var A_buffer = A.buffer()
        var B_buffer = B.buffer()

        ref A_shape = A_buffer.shape
        ref B_shape = B_buffer.shape

        if A.requires_grad:
            var B_transposed = B_buffer.transpose(axes=IntArray(-1, -2))

            var A_batch_grad = MatmulNd[Self.dtype].forward(
                grad_out, Tensor[Self.dtype](B_transposed^, requires_grad=False)
            )
            var final_grad_A = A_batch_grad^.sum_over_broadcasted_axes(A_shape)

            A.update_grad(final_grad_A^, AddTensor, None)
        parent_ids.append(A._id)

        if B.requires_grad:
            var A_transposed = A_buffer.transpose(axes=IntArray(-1, -2))
            var B_batch_grad = MatmulNd[Self.dtype].forward(
                Tensor[Self.dtype](A_transposed^, requires_grad=False), grad_out
            )

            var final_grad_B = B_batch_grad^.sum_over_broadcasted_axes(B_shape)

            B.update_grad(final_grad_B^, AddTensor, None)
        parent_ids.append(B._id)
        grad_out.zero_grad()


@fieldwise_init
struct MatmulNd[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        A: Tensor[Self.dtype], B: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        ref A_shape = A.shape()
        ref B_shape = B.shape()

        # Short-circuit for pure 2D case
        if A_shape.rank() == 2 and B_shape.rank() == 2:
            return Matmul2d[Self.dtype].forward[track_grad](A, B, sync=sync)

        var ndb = A.buffer.matmul_nd(B.buffer, sync=sync)
        var C = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var requires_grad = A.requires_grad or B.requires_grad
            if requires_grad:
                C.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    MatmulNdBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                C.add_ancestry(backwardFn^, A, B)

        return C^

    @always_inline
    @staticmethod
    def forward(
        A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        ref A_shape = A.shape()
        var B_shape = B.shape()

        if A_shape.rank() == 2 and B_shape.rank() == 2:
            return Matmul2d[Self.dtype].forward(A, B)
        var ndb = A.buffer.matmul_nd(B.buffer())
        var C = Gradbox[Self.dtype](ndb^)

        return C^

    @always_inline
    @staticmethod
    def forward(
        A: Gradbox[Self.dtype], B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        var A_shape = A.shape()
        ref B_shape = B.shape()

        if A_shape.rank() == 2 and B_shape.rank() == 2:
            return Matmul2d[Self.dtype].forward(A, B)

        var ndb = A.buffer().matmul_nd(B.buffer)
        var C = Gradbox[Self.dtype](ndb^)

        return C^


@fieldwise_init
struct Matmul[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    def forward[
        track_grad: Bool = True, mode: Int = mm
    ](
        A: Tensor[Self.dtype], B: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        comptime if mode == mm:
            # Step 1: Pure analysis - get the opcode
            # Scalar generosity (mirrors Dot.forward, same numels()==1
            # predicate): a lone scalar operand scales the other side
            # instead of panicking in classify_matmul. Routes through
            # elementwise * (full autograd + GPU + broadcast support), so
            # dL/dM = s·upstream and dL/ds = sum(M·upstream). Both-scalar
            # still falls through to dot below; (1,1)x(1,1) still takes
            # the mm path.
            if A.numels() == 1 and B.numels() > 1:
                return Multiplicator[Self.dtype].forward[track_grad](
                    A, B, sync=sync
                )
            if B.numels() == 1 and A.numels() > 1:
                return Multiplicator[Self.dtype].forward[track_grad](
                    A, B, sync=sync
                )
            var opcode = classify_matmul(A.shape(), B.shape())

            # Step 2: Simple dispatch based on opcode

            if dot == opcode:
                return A.dot[track_grad](B, sync=sync)

            if vm == opcode:
                return VectorMatmulNd[Self.dtype].forward[track_grad](
                    A, B, sync=sync
                )

            if mv == opcode:
                return MatrixVectorMulNd[Self.dtype].forward[track_grad](
                    A, B, sync=sync
                )

            if mm == opcode:
                return MatmulNd[Self.dtype].forward[track_grad](A, B, sync=sync)

            # Invalid case
            panic("Matmul: incompatible shapes")
            return Tensor[Self.dtype].scalar(0)

        elif mode == dot:
            return A.dot[track_grad](B, sync=sync)

        elif mode == vm:
            return VectorMatmulNd[Self.dtype].forward[track_grad](
                A, B, sync=sync
            )

        elif mode == mv:
            return MatrixVectorMulNd[Self.dtype].forward[track_grad](
                A, B, sync=sync
            )
        else:
            # Invalid case
            panic("Matmul: incompatible shapes")
            return Tensor[Self.dtype].scalar(0)

    @always_inline
    @staticmethod
    def forward(
        A: Tensor[Self.dtype], B: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        return MatmulNd[Self.dtype].forward(A, B)

    @always_inline
    @staticmethod
    def forward(
        A: Gradbox[Self.dtype], B: Tensor[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        return MatmulNd[Self.dtype].forward(A, B)


def classify_matmul(a: Shape, b: Shape) -> Int:
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
