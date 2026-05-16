from tenmo.tensor import Tensor
from tenmo.shapes import Shape
from tenmo.common_utils import panic
from std.sys import simd_width_of
from std.testing import assert_true, TestSuite
from tenmo.strides import Strides


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


fn validate_matmul_2d_grads[
    dtype: DType, //
](mut A: Tensor[dtype], mut B: Tensor[dtype], C: Tensor[dtype]) raises:
    print("validate_matmul_2d_grads")

    # --- Early exit if no gradients were tracked ---
    if not A.requires_grad and not B.requires_grad and not C.requires_grad:
        print(
            "validate_matmul_2d_grads → No gradients to validate "
            + "(requires_grad == False for all tensors)."
        )
        return

    var gradC = (
        C.gradients()[].copy()
    )  # Guaranteed to exist if C.requires_grad == True

    var B_T = B.transpose[track_grad=False](1, 0)
    var A_T = A.transpose[track_grad=False](1, 0)

    var expected_grad_A = gradC.matmul(B_T)
    var expected_grad_B = A_T.matmul(gradC)

    # --- Validate A.grad ---
    if A.requires_grad:
        var auto_grad_A = A.grad()
        if expected_grad_A.shape() != auto_grad_A.shape():
            panic(
                "Shape mismatch for grad(A). Expected "
                + expected_grad_A.shape().__str__()
                + ", got "
                + auto_grad_A.shape().__str__()
            )

        if not auto_grad_A.all_close(expected_grad_A):
            print("Gradient mismatch for A")
            print("Expected gradA:", expected_grad_A)
            print("Actual gradA:", auto_grad_A)
            panic("validate_matmul_2d_grads → Gradient mismatch for A.")
    else:
        print("gSkipping A.grad validation (requires_grad == False)")

    # --- Validate B.grad ---
    if B.requires_grad:
        var auto_grad_B = B.grad()
        if expected_grad_B.shape() != auto_grad_B.shape():
            panic(
                "Shape mismatch for grad(B). Expected "
                + expected_grad_B.shape().__str__()
                + ", got "
                + auto_grad_B.shape().__str__()
            )

        if not auto_grad_B.all_close(expected_grad_B):
            print("Gradient mismatch for B")
            print("Expected gradB:", expected_grad_B)
            print("Actual gradB:", auto_grad_B)
            panic("validate_matmul_2d_grads → Gradient mismatch for B.")
    else:
        print("Skipping B.grad validation (requires_grad == False)")

    print("Matmul_2d gradient validation passed for all applicable tensors")


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


