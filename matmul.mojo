from tenmo import Tensor
from std.sys import simd_width_of
from matrixshapevalidator import MatrixShapeValidator
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_MATMUL_ND,
    BACKWARD_MATMUL_2D,
)
from mnemonics import AddTensor, mm, vm, mv, dot, invalid
from gradbox import Gradbox
from shapes import Shape
from common_utils import panic
from vectormatrix import VectorMatmulNd
from matrixvector import MatrixVectorMulNd
from std.algorithm import parallelize
from std.sys.info import num_physical_cores
from intarray import IntArray


@fieldwise_init
struct Matmul2dBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    comptime TAG = BACKWARD_MATMUL_2D

    @always_inline
    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref grad_out = output.gradients()[]
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)

        var result = List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]()

        # ===== GRADIENT FOR A: dL/dA = grad_out × B^T =====
        if A.requires_grad:
            var ndb = grad_out.buffer.matmul_2d(
                B.buffer.transpose(IntArray(-1, -2))
            )
            var grad_A = Gradbox[Self.dtype](ndb^, share=False)

            result.append((A, grad_A^, AddTensor))

        # ===== GRADIENT FOR B: dL/dB = A^T × grad_out =====
        if B.requires_grad:
            var A_buffer_transposed = A.buffer.transpose(IntArray(-1, -2))
            var ndb = A_buffer_transposed.matmul_2d(grad_out.buffer)
            var grad_B = Gradbox[Self.dtype](ndb^, share=False)

            result.append((B^, grad_B^, AddTensor))

        return result^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
struct Matmul2d[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                var backward_fn = Matmul2dBackward[
                    Self.dtype
                ]().into_backward_fn()
                C.backwardFn = Optional(backward_fn^)
                C.add_ancestry(A)
                C.add_ancestry(B)

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
struct MatmulNdBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    comptime TAG = BACKWARD_MATMUL_ND

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref grad_out = output.gradients()[]
        var A = output.ancestry().get(0)
        var B = output.ancestry().get(1)

        ref A_shape = A.shape()
        ref B_shape = B.shape()
        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        if A.requires_grad:
            var B_transposed = B.transpose[track_grad=False](
                axes=[-1, -2]
            )

            var A_batch_grad = MatmulNd[Self.dtype].forward(
                grad_out, B_transposed
            )
            var final_grad_A = A_batch_grad^.sum_over_broadcasted_axes(A_shape)

            results.append((A, final_grad_A^, AddTensor))

        if B.requires_grad:
            var A_transposed = A.transpose[track_grad=False](axes=[-1, -2])
            var B_batch_grad = MatmulNd[Self.dtype].forward(
                A_transposed, grad_out
            )

            var final_grad_B = B_batch_grad^.sum_over_broadcasted_axes(B_shape)

            results.append((B^, final_grad_B^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
struct MatmulNd[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                var backward_fn = MatmulNdBackward[
                    Self.dtype
                ]().into_backward_fn()
                C.backwardFn = Optional(backward_fn^)
                C.add_ancestry(A)
                C.add_ancestry(B)

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
struct Matmul[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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


from std.time import perf_counter_ns as now
from std.sys import argv
from blashandle import BLASHandle


fn test_gflops_blas() raises:
    comptime dtype = DType.float32
    var M = 128
    var K = 512
    var O = 512
    var N = 256

    var A_shape = Shape(M, K)
    var B_shape = Shape(O, N)

    var shape_dims = argv()
    try:
        var length = len(shape_dims)
        if length > 1:
            if length == 2:
                var dim = Int(shape_dims[1])
                A_shape = Shape(dim, dim)
                B_shape = Shape(dim, dim)
            elif length == 3:
                var dim1, dim2 = Int(shape_dims[1]), Int(shape_dims[2])
                A_shape = Shape(dim1, dim2)
                B_shape = Shape(dim2, dim2)
            elif length == 4:
                var dim1, dim2, dim3, dim4 = (
                    Int(shape_dims[1]),
                    Int(shape_dims[2]),
                    Int(shape_dims[3]),
                    Int(shape_dims[4]),
                )
                A_shape = Shape(dim1, dim2)
                B_shape = Shape(dim3, dim4)

    except e:
        print("Could not parse shape dims")

    if A_shape[1] != B_shape[0]:
        panic(
            "Matrix dimension mismatch: ", String(A_shape), String(B_shape)
        )
    # Setup
    var A = Tensor[dtype].randn(A_shape)
    var B = Tensor[dtype].randn(B_shape)

    print("Benchmarking Matmul2d...")
    var blas = BLASHandle[dtype]()
    # Warmup
    for _ in range(10):
        var _ = blas.matmul(A, B)

    # Benchmark
    var start = now()
    for _ in range(100):
        var _ = blas.matmul(A, B)
    var end = now()

    # Calculations
    var avg_time_ms = Float64(end - start) / (100.0 * 1_000_000.0)

    var gflops = (2.0 * Float64(M * K * N)) / (avg_time_ms * 1_000_000.0)

    M = A_shape[0]
    K = A_shape[1]
    N = B_shape[1]

    # Formatted output
    var shape_str = String("(") + String(M) + "×" + String(K) + ") @ ("
    shape_str += String(K) + "×" + String(N) + ")"

    print("Blas matmul Results:")
    print("  Matrix dimensions: " + shape_str)
    print("  Average time:      " + String(avg_time_ms) + " ms")
    print("  Performance:       " + String(gflops) + " GFLOPS")
    print("  Operations:        " + String(2 * M * K * N) + " FLOP")


fn test_gflops() raises:
    comptime dtype = DType.float32
    var M = 128
    var K = 512
    var O = 512
    var N = 256

    var A_shape = Shape(M, K)
    var B_shape = Shape(O, N)

    var shape_dims = argv()
    try:
        var length = len(shape_dims)
        if length > 1:
            if length == 2:
                var dim = Int(shape_dims[1])
                A_shape = Shape(dim, dim)
                B_shape = Shape(dim, dim)
            elif length == 3:
                var dim1, dim2 = Int(shape_dims[1]), Int(shape_dims[2])
                A_shape = Shape(dim1, dim2)
                B_shape = Shape(dim2, dim2)
            elif length == 4:
                var dim1, dim2, dim3, dim4 = (
                    Int(shape_dims[1]),
                    Int(shape_dims[2]),
                    Int(shape_dims[3]),
                    Int(shape_dims[4]),
                )
                A_shape = Shape(dim1, dim2)
                B_shape = Shape(dim3, dim4)

    except e:
        print("Could not parse shape dims")

    if A_shape[1] != B_shape[0]:
        panic(
            "Matrix dimension mismatch: ", String(A_shape), String(B_shape)
        )
    # Setup
    var A = Tensor[dtype].randn(A_shape)
    var B = Tensor[dtype].randn(B_shape)

    print("Benchmarking Matmul2d...")

    # Warmup
    for _ in range(10):
        var _ = Matmul2d[dtype].forward(A, B)

    # Benchmark
    var start = now()
    for _ in range(100):
        var _ = Matmul2d[dtype].forward(A, B)
    var end = now()

    # Calculations
    var avg_time_ms = Float64(end - start) / (100.0 * 1_000_000.0)

    var gflops = (2.0 * Float64(M * K * N)) / (avg_time_ms * 1_000_000.0)

    M = A_shape[0]
    K = A_shape[1]
    N = B_shape[1]

    # Formatted output
    var shape_str = String("(") + String(M) + "×" + String(K) + ") @ ("
    shape_str += String(K) + "×" + String(N) + ")"

    print("Results:")
    print("  Matrix dimensions: " + shape_str)
    print("  Average time:      " + String(avg_time_ms) + " ms")
    print("  Performance:       " + String(gflops) + " GFLOPS")
    print("  Operations:        " + String(2 * M * K * N) + " FLOP")


from std.testing import assert_true


fn main() raises:
    test_gflops()
    test_gflops_blas()
    comptime dtype = DType.float64
    var A = Tensor[dtype].randn(128, 128)
    var B = Tensor[dtype].randn(128, 128)

    var R_mojo = Matmul2d[dtype].forward(A, B)
    var blas = BLASHandle[dtype]()
    var R_blas = blas.matmul(A, B)
    assert_true(R_mojo.all_close(R_blas))

    var A_T = A.transpose()
    R_mojo = Matmul2d[dtype].forward(A_T, B)
    R_blas = blas.matmul(A_T, B, transpose_A=True)
    assert_true(R_mojo.all_close(R_blas))

    var B_T = A.transpose()
    R_mojo = Matmul2d[dtype].forward(A_T, B_T)
    R_blas = blas.matmul(A_T, B_T, transpose_A=True, transpose_B=True)
    assert_true(R_mojo.all_close(R_blas))

    R_mojo = Matmul2d[dtype].forward(A, B_T)
    R_blas = blas.matmul(A, B_T, transpose_B=True)
    assert_true(R_mojo.all_close(R_blas))

    print("BLAS and Mojo computes same result")
