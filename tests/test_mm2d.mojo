from tenmo import Tensor
from shapes import Shape
from common_utils import panic
from sys import simd_width_of

fn matmul_naive[
    dtype: DType, //
](A: Tensor[dtype], B: Tensor[dtype]) raises -> Tensor[dtype]:
    """Naive matrix multiplication for verification purposes.

    Uses simple triple loops without any optimizations.
    Perfect for comparing against optimized matmul results.
    """
    var A_shape = A.shape()
    var B_shape = B.shape()
    var m = A_shape[0]
    var n = A_shape[1]
    var p = B_shape[1]

    # Validate inner dimensions match
    assert_true(n == B_shape[0], "matmul_naive: Inner dimensions must match")

    var C = Tensor[dtype].zeros(m, p)

    # Naive triple loops - no optimizations
    for i in range(m):
        for j in range(p):
            var sum: Scalar[dtype] = 0.0
            for k in range(n):
                sum += A[i, k] * B[k, j]
            C[i, j] = sum

    return C^

fn validate_matmul_2d_grads[dtype: DType, //](
    A: Tensor[dtype],
    B: Tensor[dtype],
    C: Tensor[dtype]
) raises:
    print("validate_matmul_2d_grads")

    # --- Early exit if no gradients were tracked ---
    if not A.requires_grad and not B.requires_grad and not C.requires_grad:
        print(
            "ℹ️ validate_matmul_2d_grads → No gradients to validate "
            + "(requires_grad == False for all tensors)."
        )
        return

    var gradC = C.grad().copy()  # Guaranteed to exist if C.requires_grad == True

    var B_T = B.transpose[track_grad=False](1, 0)
    var A_T = A.transpose[track_grad=False](1, 0)

    var expected_grad_A = gradC.matmul(B_T)
    var expected_grad_B = A_T.matmul(gradC)

    # --- Validate A.grad ---
    if A.requires_grad:
        var auto_grad_A = A.grad().copy()
        if expected_grad_A.shape() != auto_grad_A.shape():
            panic(
                "Shape mismatch for grad(A). Expected "
                + expected_grad_A.shape().__str__()
                + ", got "
                + auto_grad_A.shape().__str__()
            )

        if not auto_grad_A.all_close(expected_grad_A):
            print("❌ Gradient mismatch for A")
            print("Expected gradA:", expected_grad_A)
            print("Actual gradA:", auto_grad_A)
            panic("validate_matmul_2d_grads → Gradient mismatch for A.")
    else:
        print("⚠️ Skipping A.grad validation (requires_grad == False)")

    # --- Validate B.grad ---
    if B.requires_grad:
        var auto_grad_B = B.grad().copy()
        if expected_grad_B.shape() != auto_grad_B.shape():
            panic(
                "Shape mismatch for grad(B). Expected "
                + expected_grad_B.shape().__str__()
                + ", got "
                + auto_grad_B.shape().__str__()
            )

        if not auto_grad_B.all_close(expected_grad_B):
            print("❌ Gradient mismatch for B")
            print("Expected gradB:", expected_grad_B)
            print("Actual gradB:", auto_grad_B)
            panic("validate_matmul_2d_grads → Gradient mismatch for B.")
    else:
        print("⚠️ Skipping B.grad validation (requires_grad == False)")

    print("✅ Matmul_2d gradient validation passed for all applicable tensors")


fn main() raises:
    run_all_matmul_2d_forward_tests()

fn run_all_matmul_2d_forward_tests() raises:
    print("Running matmul_2d tests")
    test_matmul_2d_basic()
    test_matmul_2d_rectangular()
    test_matmul_2d_identity()
    test_matmul_2d_non_contiguous_both_views()
    test_matmul_2d_single_row_col()
    test_matmul_2d_high_values()
    test_matmul_2d_mixed_stride_views()

    test_matmul_2d_1x1_identity()
    test_matmul_2d_1x1_scalar_multiplication()
    test_matmul_2d_2x2_square_matrices()
    test_matmul_2d_2x3_times_3x2()
    test_matmul_2d_1x3_times_3x1()
    test_matmul_2d_3x1_times_1x3()
    test_matmul_2d_identity_matrix()
    test_matmul_2d_zero_matrix()
    test_matmul_2d_single_element_vectors()
    test_matmul_2d_rectangular_tall_a()
    test_matmul_2d_rectangular_wide_b()

    test_matmul_2d_non_contiguous_B_view()

    test_matmul_2d_contiguous_b_fast_path()
    test_matmul_2d_transposed_b_slow_path()
    test_matmul_2d_view_slice_of_a()
    test_matmul_2d_view_slice_of_b()
    test_matmul_2d_both_views()
    test_matmul_2d_strided_view_columns()
    test_matmul_2d_view_with_offset()
    test_matmul_2d_contiguous_after_transpose()

    test_matmul_2d_non_contiguous_A_view()

    test_all_matmul_2d_large_simd()

    print("✓ All matmul_2d tests passed!")


fn test_all_matmul_2d_large_simd() raises:
    print("Running large matrix matmul_2d SIMD tests...")

    test_matmul_2d_large_contiguous_simd_path()
    test_matmul_2d_very_large_matrices()
    test_matmul_2d_simd_width_boundary()
    test_matmul_2d_non_multiple_simd_width()
    test_matmul_2d_large_views_simd_path()
    test_matmul_2d_large_transposed_slow_path()
    test_matmul_2d_very_very_large_matrices()
    print("✓ All large matmul_2d SIMD tests passed!")


# ===== LARGE MATRICES FOR SIMD PATH TESTING =====


fn test_matmul_2d_large_contiguous_simd_path() raises:
    print("test_matmul_2d_large_contiguous_simd_path")
    # Create matrices larger than SIMD width to ensure vectorization
    var m: Int = 8
    var n: Int = 12  # Multiple of SIMD width (4)
    var p: Int = 16  # Multiple of SIMD width (4)

    var A = Tensor.rand(m, n)
    var B = Tensor.rand(n, p)  # Contiguous by default

    # Verify B is contiguous and suitable for SIMD
    var b_strides = B.strides()
    assert_true(b_strides[1] == 1)  # Columns are contiguous
    alias dtype = DType.float32
    assert_true(p >= simd_width_of[dtype]())  # Ensure SIMD can be used

    var C = A.matmul(B)
    assert_true(C.shape() == Shape(m, p))
    assert_true(
        matmul_naive(A, B).all_close(C),
        "matmul_2d result did not match naive matmul result",
    )
    # Just verify computation completes with correct shape


fn test_matmul_2d_very_large_matrices() raises:
    print("test_matmul_2d_very_large_matrices")
    var m: Int = 32
    var n: Int = 64  # Large multiple of SIMD width
    var p: Int = 48  # Large multiple of SIMD width

    var A = Tensor.rand(m, n, requires_grad=True)
    var B = Tensor.rand(n, p, requires_grad=True)

    var C = A.matmul(B)
    assert_true(C.shape() == Shape(m, p))
    assert_true(
        matmul_naive(A, B).all_close(C),
        "matmul_2d result did not match naive matmul result",
    )
    # Test that large matrices don't break the implementation

    validate_matmul_2d_grads(A, B, C)

fn test_matmul_2d_very_very_large_matrices() raises:
    print("test_matmul_2d_very_very_large_matrices")
    var m: Int = 512
    var n: Int = 128  # Large multiple of SIMD width
    var p: Int = 1024  # Large multiple of SIMD width

    var A = Tensor.rand(m, n, requires_grad=True)
    var B = Tensor.rand(n, p, requires_grad=True)

    var C = A.matmul(B)
    assert_true(C.shape() == Shape(m, p))
    assert_true(
        matmul_naive(A, B).all_close(C),
        "matmul_2d result did not match naive matmul result",
    )
    # Test that large matrices don't break the implementation

    validate_matmul_2d_grads(A, B, C)


fn test_matmul_2d_simd_width_boundary() raises:
    print("test_matmul_2d_simd_width_boundary")
    # Test right at SIMD width boundary

    alias dtype = DType.float32
    var simd_width = simd_width_of[dtype]()
    var m: Int = simd_width  # 4 for float32
    var n: Int = simd_width * 2  # 8 for float32
    var p: Int = simd_width * 3  # 12 for float32

    var A = Tensor.rand(m, n)
    var B = Tensor.rand(n, p)

    var C = A.matmul(B)
    assert_true(C.shape() == Shape(m, p))
    assert_true(
        matmul_naive(A, B).all_close(C),
        "matmul_2d result did not match naive matmul result",
    )


fn test_matmul_2d_non_multiple_simd_width() raises:
    print("test_matmul_2d_non_multiple_simd_width")
    # Test with dimensions not multiples of SIMD width
    # This tests the vectorize[] function's handling of remainder
    var m: Int = 7  # Not multiple of 4
    var n: Int = 11  # Not multiple of 4
    var p: Int = 13  # Not multiple of 4

    var A = Tensor.rand(m, n, init_seed=42, requires_grad=True)
    var B = Tensor.rand(n, p, init_seed=42, requires_grad=True)
    var C = A.matmul(B)
    assert_true(C.shape() == Shape(m, p))
    assert_true(
        matmul_naive(A, B).all_close(C),
        "matmul_2d result did not match naive matmul result",
    )

    validate_matmul_2d_grads(A, B, C)

fn test_matmul_2d_large_views_simd_path() raises:
    print("test_matmul_2d_large_views_simd_path")
    # Create large base tensors and take contiguous views from them
    var A_base = Tensor.rand(16, 24)
    var B_base = Tensor.rand(24, 32)

    # Create contiguous views (should use SIMD path)
    var A = A_base.view(shape=Shape(8, 12), strides=Strides(24, 1), offset=0)
    var B = B_base.view(shape=Shape(12, 16), strides=Strides(32, 1), offset=0)

    alias dtype = DType.float32
    # Verify views are SIMD-friendly
    assert_true(B.strides()[1] == 1)
    assert_true(B.shape()[1] >= simd_width_of[dtype]())

    var C = A.matmul(B)
    assert_true(C.shape() == Shape(8, 16))
    assert_true(
        matmul_naive(A, B).all_close(C),
        "matmul_2d result did not match naive matmul result",
    )


fn test_matmul_2d_large_transposed_slow_path() raises:
    print("test_matmul_2d_large_transposed_slow_path")
    # Large matrices but B is transposed (non-contiguous -> slow path)
    var m: Int = 8
    var n: Int = 12
    var p: Int = 16

    var A = Tensor.rand(m, n)
    var B_orig = Tensor.rand(p, n)  # Note: dimensions swapped
    var B = B_orig.transpose()  # Makes B non-contiguous

    # Verify B is NOT SIMD-friendly
    var b_strides = B.strides()
    assert_true(b_strides[1] != 1)  # Columns are not contiguous

    var C = A.matmul(B)
    assert_true(C.shape() == Shape(m, p))
    assert_true(
        matmul_naive(A, B).all_close(C),
        "matmul_2d result did not match naive matmul result",
    )


fn test_matmul_2d_basic() raises:
    print("test_matmul_2d_basic")
    var A = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()
    var B = Tensor.d2([[5.0, 6.0], [7.0, 8.0]]).float()
    var C = A.matmul(B)
    var expected = Tensor.d2([[19.0, 22.0], [43.0, 50.0]]).float()
    assert_true(C.all_close(expected))
    validate_matmul_2d_grads(A, B, C)

fn test_matmul_2d_rectangular() raises:
    print("test_matmul_2d_rectangular")
    var A = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).float()
    var B = Tensor.d2([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]).float()
    var C = A.matmul(B)
    var expected = Tensor.d2([[58.0, 64.0], [139.0, 154.0]]).float()
    assert_true(C.all_close(expected))
    validate_matmul_2d_grads(A, B, C)


fn test_matmul_2d_non_contiguous_A_view() raises:
    # fn main() raises:
    print("test_matmul_2d_non_contiguous_A_view")
    var A_base = Tensor.d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True
    ).float()
    # View skips first row: start offset = 3, shape = (2×3)
    var A1 = A_base.view(shape=Shape(2, 3), strides=Strides(3, 1), offset=3)
    var B = Tensor.d2([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True).float()
    var C = A1.matmul(B)
    var expected = Tensor.d2([[10.0, 11.0], [16.0, 17.0]]).float()
    assert_true(C.all_close(expected))

    validate_matmul_2d_grads(A1, B, C)

fn test_matmul_2d_identity() raises:
    print("test_matmul_2d_identity")
    var I = Tensor.d2([[1.0, 0.0], [0.0, 1.0]]).float()
    var A = Tensor.d2([[3.0, 4.0], [5.0, 6.0]]).float()
    var C = A.matmul(I)
    assert_true(C.all_close(A))

    validate_matmul_2d_grads(A, I, C)

fn test_matmul_2d_non_contiguous_B_view() raises:
    print("test_matmul_2d_non_contiguous_B_view")
    var BB_base = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).float()
    # Permute shape (3×2), stride (1, 3)
    var AA = BB_base.view(shape=Shape(3, 2), strides=Strides(1, 3), offset=0)
    var BB = Tensor.d2([[1.0, 2.0], [3.0, 4.0]]).float()

    var C = AA.matmul(BB)
    var expected = Tensor.d2([[13.0, 18.0], [17, 24.0], [21.0, 30.0]]).float()
    assert_true(C.all_close(expected))

    validate_matmul_2d_grads(AA, BB, C)

fn test_matmul_2d_non_contiguous_both_views() raises:
    print("test_matmul_2d_non_contiguous_both_views")
    var base = Tensor.d2(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        requires_grad=True,
    )

    var AA = base.view(shape=Shape(2, 2), strides=Strides(8, 1), offset=0)
    var BB = base.view(shape=Shape(2, 2), strides=Strides(4, 1), offset=10)

    var C = AA.matmul(BB)
    # A = [[1,2],[9,10]], B = [[11,12],[15,16]]
    var expected = Tensor.d2([[41.0, 44.0], [249.0, 268.0]]).float()
    assert_true(C.all_close(expected))

    validate_matmul_2d_grads(AA, BB, C)

fn test_matmul_2d_single_row_col() raises:
    print("test_matmul_2d_single_row_col")
    var A = Tensor.d2([[1.0, 2.0, 3.0]]).float()  # 1×3
    var B = Tensor.d2([[4.0], [5.0], [6.0]]).float()  # 3×1
    var C = A.matmul(B)
    var expected = Tensor.d2([[32.0]]).float()  # 1×1
    assert_true(C.all_close(expected))

    validate_matmul_2d_grads(A, B, C)

fn test_matmul_2d_high_values() raises:
    print("test_matmul_2d_high_values")
    var A = Tensor.d2([[1000.0, 2000.0], [3000.0, 4000.0]]).float()
    var B = Tensor.d2([[5.0, 6.0], [7.0, 8.0]]).float()
    var C = A.matmul(B)
    var expected = Tensor.d2([[19000.0, 22000.0], [43000.0, 50000.0]]).float()
    assert_true(C.all_close(expected))
    assert_true(
        matmul_naive(A, B).all_close(C),
        "matmul_2d result did not match naive matmul result",
    )

    validate_matmul_2d_grads(A, B, C)

fn test_matmul_2d_mixed_stride_views() raises:
    print("test_matmul_2d_mixed_stride_views")
    var A_base = Tensor.d2([[1, 2, 3, 4], [5, 6, 7, 8]])
    var A = A_base.view(
        shape=Shape(2, 2), strides=Strides(2, 1), offset=1
    )  # picks [[2,3],[4,5]]
    var B_base = Tensor.d2([[1, 2], [3, 4], [5, 6], [7, 8]])
    var B = B_base.view(
        shape=Shape(2, 2), strides=Strides(4, 1), offset=2
    )  # picks [[3,4],[7,8]] (assuming reshape)
    var expected = Tensor.d2([[27.0, 32.0], [47.0, 56.0]]).float()
    var C = A.matmul(B)
    assert_true(C.all_close(expected))

    validate_matmul_2d_grads(A, B, C)

# ===== BASIC MATRIX MULTIPLICATION =====

from testing import assert_true
from shapes import Shape
from strides import Strides


fn test_matmul_2d_1x1_identity() raises:
    print("test_matmul_2d_1x1_identity")
    var a = Tensor.d2([[2.0]])
    var b = Tensor.d2([[1.0]])
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[2.0]])))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_1x1_scalar_multiplication() raises:
    print("test_matmul_2d_1x1_scalar_multiplication")
    var a = Tensor.d2([[3.0]])
    var b = Tensor.d2([[4.0]])
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[12.0]])))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_2x2_square_matrices() raises:
    print("test_matmul_2d_2x2_square_matrices")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var b = Tensor.d2([[5.0, 6.0], [7.0, 8.0]])
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[19.0, 22.0], [43.0, 50.0]])))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_2x3_times_3x2() raises:
    print("test_matmul_2d_2x3_times_3x2")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var b = Tensor.d2([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[58.0, 64.0], [139.0, 154.0]])))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_1x3_times_3x1() raises:
    print("test_matmul_2d_1x3_times_3x1")
    var a = Tensor.d2([[1.0, 2.0, 3.0]])
    var b = Tensor.d2([[4.0], [5.0], [6.0]])
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[32.0]])))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_3x1_times_1x3() raises:
    print("test_matmul_2d_3x1_times_1x3")
    var a = Tensor.d2([[1.0], [2.0], [3.0]])
    var b = Tensor.d2([[4.0, 5.0, 6.0]])
    var c = a.matmul(b)
    assert_true(
        c.all_close(
            Tensor.d2([[4.0, 5.0, 6.0], [8.0, 10.0, 12.0], [12.0, 15.0, 18.0]])
        )
    )

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_identity_matrix() raises:
    print("test_matmul_2d_identity_matrix")
    var a = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var identity = Tensor.d2(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    var c = a.matmul(identity)
    assert_true(c.all_close(a))

    validate_matmul_2d_grads(a, identity, c)

fn test_matmul_2d_zero_matrix() raises:
    print("test_matmul_2d_zero_matrix")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var zeros = Tensor.d2([[0.0, 0.0], [0.0, 0.0]])
    var c = a.matmul(zeros)
    assert_true(c.all_close(zeros))

    validate_matmul_2d_grads(a, zeros, c)

# ===== EDGE CASES =====


fn test_matmul_2d_single_element_vectors() raises:
    print("test_matmul_2d_single_element_vectors")
    var a = Tensor.d2([[5.0]])
    var b = Tensor.d2([[3.0]])
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[15.0]])))


fn test_matmul_2d_rectangular_tall_a() raises:
    print("test_matmul_2d_rectangular_tall_a")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var b = Tensor.d2([[7.0, 8.0], [9.0, 10.0]])
    var c = a.matmul(b)
    assert_true(c.shape() == Shape(3, 2))


fn test_matmul_2d_rectangular_wide_b() raises:
    print("test_matmul_2d_rectangular_wide_b")
    var a = Tensor.d2([[1.0, 2.0, 3.0]])
    var b = Tensor.d2(
        [[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]
    )
    var c = a.matmul(b)
    assert_true(c.shape() == Shape(1, 4))


# ===== VIEWS AND NON-CONTIGUOUS MATRICES =====


fn test_matmul_2d_contiguous_b_fast_path() raises:
    print("test_matmul_2d_contiguous_b_fast_path")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor.d2([[5.0, 6.0], [7.0, 8.0]])  # Contiguous by default
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[19.0, 22.0], [43.0, 50.0]])))


fn test_matmul_2d_transposed_b_slow_path() raises:
    print("test_matmul_2d_transposed_b_slow_path")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var b_orig = Tensor.d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    var b = b_orig.transpose()  # Creates non-contiguous view
    var c = a.matmul(b)
    # A × B^T = different result!
    assert_true(c.all_close(Tensor.d2([[17.0, 23.0], [39.0, 53.0]])))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_view_slice_of_a() raises:
    print("test_matmul_2d_view_slice_of_a")
    var a_orig = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)
    var a = a_orig.view(
        shape=Shape(2, 2), strides=Strides(3, 1), offset=0
    )  # First 2x2 block
    var b = Tensor.d2([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[2.0, 4.0], [8.0, 10.0]])))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_view_slice_of_b() raises:
    print("test_matmul_2d_view_slice_of_b")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b_orig = Tensor.d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True)
    var b = b_orig.view(
        shape=Shape(2, 2), strides=Strides(4, 1), offset=2
    )  # Middle 2x2 columns
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[17.0, 20], [37.0, 44.0]])))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_both_views() raises:
    print("test_matmul_2d_both_views")
    var a_orig = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)
    var b_orig = Tensor.d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], requires_grad=True)

    var a = a_orig.view(
        shape=Shape(2, 2), strides=Strides(3, 1), offset=1
    )  # Skip first element
    var b = b_orig.view(
        shape=Shape(2, 2), strides=Strides(3, 1), offset=3
    )  # Skip first row

    var c = a.matmul(b)
    # Should compute with sub-matrices
    assert_true(c.shape() == Shape(2, 2))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_strided_view_columns() raises:
    print("test_matmul_2d_strided_view_columns")
    var a_orig = Tensor.d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], requires_grad=True)
    var a = a_orig.view(
        shape=Shape(2, 2), strides=Strides(4, 2), offset=0
    )  # Take every other column
    var b = Tensor.d2([[2.0, 0.0], [0.0, 2.0]], requires_grad=True)
    var c = a.matmul(b)
    assert_true(c.all_close(Tensor.d2([[2.0, 6.0], [10.0, 14.0]])))

    validate_matmul_2d_grads(a, b, c)

fn test_matmul_2d_view_with_offset() raises:
    print("test_matmul_2d_view_with_offset")
    var a_orig = Tensor.d2(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
        ]
    )
    # Create view that skips first 2 elements, takes 2x4 block
    var a = a_orig.view(shape=Shape(2, 4), strides=Strides(6, 1), offset=2)
    var b = Tensor.d2([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    var c = a.matmul(b)
    # Should extract first 2 columns of the view
    assert_true(c.all_close(Tensor.d2([[3.0, 4.0], [9.0, 10.0]])))


fn test_matmul_2d_contiguous_after_transpose() raises:
    print("test_matmul_2d_contiguous_after_transpose")
    var a = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])
    var b_orig = Tensor.d2([[5.0, 6.0], [7.0, 8.0]])
    var b_transposed = b_orig.transpose()  # Non-contiguous
    var b_contiguous = b_transposed.contiguous()  # Make it contiguous again
    var c = a.matmul(b_contiguous)
    # Should use fast path and give correct result
    assert_true(c.shape() == Shape(2, 2))
