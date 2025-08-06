from tensors import Tensor
from views import TensorView
from shapes import Shape
from os import abort
from sys import simdwidthof

# ---------------------------------------
# 🧠 Matrix Multiplication: A × B = C
# A: Tensor[dtype] (MxK)
# B: TensorView[dtype] (KxN)
# C: Tensor[dtype] (MxN)
# ---------------------------------------


fn matmul[
    dtype: DType, //, simd_width: Int = 2 * simdwidthof[dtype]()
](A: Tensor[dtype], TV: UnsafePointer[TensorView[dtype]], C: Tensor[dtype]):
    B = TV[]
    rows_a = A.shape[0]
    cols_a = A.shape[1]
    cols_b = B.shape[1]
    packed = B.strides[1] == 1
    if cols_a != B.shape[0] or C.shape != Shape([rows_a, cols_b]):
        abort("Shape mismatch in matmul")
    for i in range(0, rows_a):
        for j in range(0, cols_b, simd_width):
            mbatch = min(simd_width, cols_b - j)
            var accum = SIMD[dtype, simd_width](0)

            for k in range(0, cols_a):
                scalar_a = A.load(i, k)

                if packed and mbatch == simd_width:
                    simd_vector = B.load[simd_width](k, j)
                    accum += simd_vector * scalar_a
                else:
                    # mbatch < simd_width or scattered B cols
                    for step in range(0, mbatch):
                        scalar_b = B.load(k, j + step)
                        accum[step] += scalar_a * scalar_b

            if mbatch == simd_width:
                C.store[simd_width](i, j, accum)
            else:
                for step in range(0, mbatch):
                    C.store(i, j + step, accum[step])


# ---------------------------------------
# 🧪 Main Test
# ---------------------------------------
from time import perf_counter_ns
from testing import assert_true


fn main() raises:
    A = Tensor.d2([[1, 2, 3], [4, 5, 6]])
    B = Tensor.d2([[7, 8], [9, 10], [11, 12]])
    AV = A[:, :]
    BV = B[:, :]
    _expected = Tensor.d2([[58, 64], [139, 154]])
    C = Tensor.zeros(2, 2)
    matmul(A, UnsafePointer(to=BV), C)
    C.print()
    A = Tensor.rand(1024, 513)
    B = Tensor.rand(513, 1024)
    BV = B[:, :]
    C = Tensor.zeros(1024, 1024)
    start = perf_counter_ns()
    matmul(A, UnsafePointer(to=BV), C)
    end = perf_counter_ns()
    print("matmul1 took: ", end - start)

    start = perf_counter_ns()
    R = A.matmul(B)
    end = perf_counter_ns()
    print("matmul2 took: ", end - start)
    assert_true(C.all_close(R))

    _ = """proof = C == R
    C.print()
    proof.print()
    R.print()"""
