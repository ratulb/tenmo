from time import perf_counter_ns
from tensors import Tensor
from intlist import IntList
from testing import assert_true
from algorithm import vectorize
from sys import simdwidthof


fn main() raises:
    A_rows, A_cols = 256, 64
    B_rows, B_cols = 64, 256

    A = Tensor.rand(A_rows, A_cols)
    B = Tensor.rand(B_rows, B_cols)

    expected, t1 = mm_naive(A, B)
    result, t2 = bench_tensor_tensor(A, B)
    assert_true(expected.all_close(result))
    print("Naive vs tensor matmul without backward pass: ", t1/t2)

    # backward
    A.requires_grad_()
    B.requires_grad_()
    result, _, _, t3 = bench_tensor_tensor_backward(A, B)

    assert_true(expected.all_close(result))
    print("Tensor matmul with/without backward pass: ", t3/t2)


fn bench_tensor_tensor[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (Tensor[dtype], UInt):
    start = perf_counter_ns()
    C = A.matmul(B)
    end = perf_counter_ns()
    print("bench_tensor_tensor -> Total:            ", end - start)

    return C, (end - start)



fn bench_tensor_tensor_backward[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (
    Tensor[dtype],
    Tensor[dtype],
    Tensor[dtype], UInt
):
    start = perf_counter_ns()
    C = A.matmul(B)
    end = perf_counter_ns()
    print("Tensor.matmul(Tensor) -> Total:                  ", end - start)
    start = perf_counter_ns()
    C.backward()
    end = perf_counter_ns()
    print("Tensor.matmul(Tensor) backward -> Total:         ", end - start)

    return C, A.gradbox[], B.gradbox[], (end - start)


fn mm_naive[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (Tensor[dtype], UInt):
    start = perf_counter_ns()
    M = A.rows()
    K = A.cols()
    N = B.cols()
    C = Tensor[dtype].zeros(M, N, requires_grad=False)
    for m in range(M):
        for n in range(N):
            for k in range(K):
                C[m, n] += A[m, k] * B[k, n]
    end = perf_counter_ns()
    print("mm_naive -> Total:                               ", end - start)
    return C, (end - start)


fn mm_vectorize[
    dtype: DType, //, simd_width: Int = simdwidthof[dtype]()
](A: Tensor[dtype], B: Tensor[dtype]) -> (Tensor[dtype], UInt):
    rows_a = A.rows()
    cols_a = A.cols()
    cols_b = B.cols()
    C = Tensor[dtype].zeros(rows_a, cols_b, requires_grad=False)
    start = perf_counter_ns()
    for i in range(rows_a):
        for j in range(cols_a):
            A_value = A.buffer.load[simdwidth=1](i * cols_a + j)

            @parameter
            fn dot[simdwidth: Int](k: Int):
                C.buffer.store[simdwidth=simdwidth](
                    i * cols_b + k,
                    C.buffer.load[simdwidth=simdwidth](i * cols_b + k)
                    + A_value * B.buffer.load[simdwidth=simdwidth](j * cols_b + k),
                )

            vectorize[dot, simd_width](cols_b)

    end = perf_counter_ns()
    print("mm_vectorize -> Total:                           ", end - start)
    return C, (end - start)


fn mm_simd_tiled[
    dtype: DType,
    simd_width: Int = simdwidthof[dtype](),
    TILE_I: Int = 16,
    TILE_J: Int = 16,
    TILE_K: Int = 2 * simd_width,
](A: Tensor[dtype], B: Tensor[dtype]) -> (Tensor[dtype], UInt):
    rows = A.rows()
    cols = A.cols()
    cols_b = B.cols()
    C = Tensor[dtype].zeros(rows, cols_b)

    start = perf_counter_ns()
    # Tile the loops
    for i_tile in range(0, rows, TILE_I):
        for j_tile in range(0, cols, TILE_J):
            for k_tile in range(0, cols_b, TILE_K):
                # Process a tile of C[i:i+TILE_I, k:k+TILE_K]
                for i in range(i_tile, min(i_tile + TILE_I, rows)):
                    for j in range(j_tile, min(j_tile + TILE_J, cols)):
                        # Load A[i,j] once (scalar)
                        a_val = A.buffer.load[simdwidth=1](i * cols + j)

                        # Vectorize over k-tile
                        @parameter
                        fn dot[simdwidth: Int](k: Int):
                            k_start = k_tile + k
                            B_vec = B.buffer.load[simdwidth=simdwidth](
                                j * cols_b + k_start
                            )
                            C_vec = C.buffer.load[simdwidth=simdwidth](
                                i * cols_b + k_start
                            )
                            C.buffer.store[simdwidth=simdwidth](
                                i * cols_b + k_start, C_vec + a_val * B_vec
                            )

                        vectorize[dot, simd_width](min(TILE_K, cols_b - k_tile))

    end = perf_counter_ns()
    print("mm_simd_tiled -> Total:                          ", end - start)

    return C, (end - start)



