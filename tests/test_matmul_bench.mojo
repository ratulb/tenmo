from time import perf_counter_ns
from tensors import Tensor
from views import TensorView
from intlist import IntList
from testing import assert_true
from algorithm import vectorize
from sys import simdwidthof


fn main() raises:
    A_rows, A_cols = 64, 128
    B_rows, B_cols = 128, 128

    A = Tensor.rand(A_rows, A_cols, requires_grad=True)
    B = Tensor.rand(B_rows, B_cols, requires_grad=True)
    C0 = mm_vectorize(A, B)
    C = mm_naive(A, B)
    CT = mm_simd_tiled(A, B)
    CHAT = mm_vectorize_A(A, B)
    A_no_grad = A
    B_no_grad = B
    A_no_grad.requires_grad = False
    B_no_grad.requires_grad = False
    C_NO_GRAD = bench_tensor_tensor_no_grad(A_no_grad, B_no_grad)
    assert_true(C_NO_GRAD.all_close(C))

    assert_true(CHAT.all_close(C))
    assert_true(C0.all_close(C))
    assert_true(CT.all_close(C))

    C1, A_grad_1, B_grad_1 = bench_tensor_tensor(A, B)
    A.zero_grad()
    B.zero_grad()

    C2, A_grad_2, B_grad_2 = bench_view_view(A, B)

    assert_true(C.all_close(C1))
    assert_true(C1.all_close(C2))
    assert_true(A_grad_1.all_close(A_grad_2))
    assert_true(B_grad_1.all_close(B_grad_2))

    A.zero_grad()
    B.zero_grad()

    C3, A_grad_3, B_grad_3 = bench_tensor_view(A, B)

    assert_true(C.all_close(C2))
    assert_true(C2.all_close(C3))
    assert_true(A_grad_2.all_close(A_grad_3))
    assert_true(B_grad_2.all_close(B_grad_3))

    A.zero_grad()
    B.zero_grad()

    C4, A_grad_4, B_grad_4 = bench_view_tensor(A, B)

    assert_true(C.all_close(C3))
    assert_true(C4.all_close(C3))
    assert_true(A_grad_4.all_close(A_grad_3))
    assert_true(B_grad_4.all_close(B_grad_3))

fn bench_tensor_tensor_no_grad[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    start = perf_counter_ns()
    C = A.matmul(B)
    end = perf_counter_ns()
    print("bench_tensor_tensor_no_grad -> Total:            ", end - start)

    return C



fn bench_tensor_tensor[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (
    Tensor[dtype],
    Tensor[dtype],
    Tensor[dtype],
):
    start = perf_counter_ns()
    C = A.matmul(B)
    end = perf_counter_ns()
    print("Tensor.matmul(Tensor) -> Total:                  ", end - start)
    start = perf_counter_ns()
    C.backward()
    end = perf_counter_ns()
    print("Tensor.matmul(Tensor) backward -> Total:         ", end - start)

    return C, A.grad[], B.grad[]


fn bench_view_view[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (
    Tensor[dtype],
    Tensor[dtype],
    Tensor[dtype],
):
    AV = A[:, :]
    BV = B[:, :]
    start = perf_counter_ns()
    C = AV.matmul(BV)
    end = perf_counter_ns()
    print("TensorView.matmul(TensorView) -> Total:          ", end - start)
    start = perf_counter_ns()
    C.backward()
    end = perf_counter_ns()
    print("TensorView.matmul(TensorView) backward -> Total: ", end - start)

    return C, A.grad[], B.grad[]


fn bench_tensor_view[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (
    Tensor[dtype],
    Tensor[dtype],
    Tensor[dtype],
):
    BV = B[:, :]
    start = perf_counter_ns()
    C = A.matmul(BV)
    end = perf_counter_ns()
    print("Tensor.matmul(TensorView) -> Total:              ", end - start)
    start = perf_counter_ns()
    C.backward()
    end = perf_counter_ns()
    print("Tensor.matmul(TensorView) backward -> Total:     ", end - start)
    return C, A.grad[], B.grad[]


fn bench_view_tensor[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> (
    Tensor[dtype],
    Tensor[dtype],
    Tensor[dtype],
):
    AV = A[:, :]
    start = perf_counter_ns()
    C = AV.matmul(B)
    end = perf_counter_ns()
    print("TensorView.matmul(Tensor) -> Total:              ", end - start)
    start = perf_counter_ns()
    C.backward()
    end = perf_counter_ns()
    print("TensorView.matmul(Tensor) backward -> Total:     ", end - start)
    return C, A.grad[], B.grad[]


fn mm_naive[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    start = perf_counter_ns()
    M = A.rows()
    K = A.cols()
    # L = B.rows()
    N = B.cols()
    C = Tensor[dtype].zeros(M, N, requires_grad=False)
    for m in range(M):
        for n in range(N):
            for k in range(K):
                C[m, n] += A[m, k] * B[k, n]
    end = perf_counter_ns()
    print("mm_naive -> Total:                               ", end - start)
    return C


fn mm_vectorize[
    dtype: DType, //, simd_width: Int = simdwidthof[dtype]()
](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    rows_a = A.rows()
    cols_a = A.cols()
    cols_b = B.cols()
    C = Tensor[dtype].zeros(rows_a, cols_b, requires_grad=False)
    start = perf_counter_ns()
    for i in range(rows_a):
        for j in range(cols_a):
            A_value = A.data.load[width=1](i * cols_a + j)

            @parameter
            fn dot[simdwidth: Int](k: Int):
                C.data.store[width=simdwidth](
                    i * cols_b + k,
                    C.data.load[width=simdwidth](i * cols_b + k)
                    + A_value * B.data.load[width=simdwidth](j * cols_b + k),
                )

            vectorize[dot, simd_width](cols_b)

    end = perf_counter_ns()
    print("mm_vectorize -> Total:                           ", end - start)
    return C


fn mm_simd_tiled[
    dtype: DType,
    simd_width: Int = simdwidthof[dtype](),
    TILE_I: Int = 16,
    TILE_J: Int = 16,
    TILE_K: Int = 2 * simd_width,
](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
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
                        a_val = A.data.load[width=1](i * cols + j)

                        # Vectorize over k-tile
                        @parameter
                        fn dot[simdwidth: Int](k: Int):
                            k_start = k_tile + k
                            B_vec = B.data.load[width=simdwidth](
                                j * cols_b + k_start
                            )
                            C_vec = C.data.load[width=simdwidth](
                                i * cols_b + k_start
                            )
                            C.data.store[width=simdwidth](
                                i * cols_b + k_start, C_vec + a_val * B_vec
                            )

                        vectorize[dot, simd_width](min(TILE_K, cols_b - k_tile))

    end = perf_counter_ns()
    print("mm_simd_tiled -> Total:                          ", end - start)

    return C


fn mm_vectorize_A[
    dtype: DType, simdwidth: Int = simdwidthof[dtype]()
](A: Tensor[dtype], B: Tensor[dtype]) -> Tensor[dtype]:
    rows = A.rows()
    cols = A.cols()
    cols_b = B.cols()

    # Output matrix
    C = Tensor[dtype].zeros(rows, cols_b)

    start = perf_counter_ns()
    for i in range(rows):
        for k in range(cols_b):
            var sum: Scalar[dtype] = 0

            @parameter
            fn dot[width: Int](j: Int):
                a_vec = (A.data + i * cols + j).load[width=width](0)
                b_vec = (B.data + j * cols_b + k).strided_load[width=width](
                    cols_b
                )
                prod = a_vec * b_vec
                sum += prod.reduce_add()

            # Vectorized over j
            vectorize[dot, simdwidth](cols)

            # Store final scalar into C[i, k]
            C.data[i * cols_b + k] = sum
    end = perf_counter_ns()
    print("mm_vectorize_A -> Total:                         ", end - start)

    return C
