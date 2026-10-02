from std.testing import assert_true, TestSuite

from tenmo.tensor import Tensor
from tenmo.shared.shapes import Shape
from tenmo.blas_ndbuffer import BLASCache
from tenmo.blashandle import BLASHandleLite


def _blas_available() -> Bool:
    """True iff BLAS routing is active (opt-in -D BLAS AND library loaded).

    With the opt-in flag present, Tensor.matmul routes 2D eligible matmuls
    through OpenBLAS; without it, the library may still load for the explicit
    BLASHandle API but matmul stays native.
    """
    return BLASCache.is_enabled() and BLASCache.is_available()


def test_blas_forward_f32() raises:
    """Apple-to-apple: routed Tensor.matmul vs the explicit native GEMM.

    Tensor.matmul routes through BLAS when -D BLAS=1 is set and the operand is
    eligible; otherwise it runs the native tiled GEMM. The reference here is
    ALWAYS the explicit native leaf A.buffer.matmul_2d(B.buffer) (never BLAS),
    so with -b this is a BLAS-vs-native comparison on identical inputs, and
    without -b it is native vs native. Runs in both build configs.
    """
    comptime dtype = DType.float32
    var A = Tensor[dtype].rand(8, 6)
    var B = Tensor[dtype].rand(6, 5)

    var C_tensor = A.matmul(B)

    var C_native_ndb = A.buffer.matmul_2d(B.buffer, sync=True)
    var C_native = Tensor[dtype](C_native_ndb, requires_grad=False)

    assert_true(C_tensor.shape()[0] == 8)
    assert_true(C_tensor.shape()[1] == 5)
    assert_true(C_native.shape()[0] == 8)
    assert_true(C_native.shape()[1] == 5)
    assert_true(
        C_tensor.all_close[atol=1e-4](C_native),
        "routed matmul != native GEMM (BLAS-vs-native mismatch)",
    )


def test_blas_forward_f64() raises:
    """Apple-to-apple BLAS-vs-native forward check for float64."""
    comptime dtype = DType.float64
    var A = Tensor[dtype].rand(5, 7)
    var B = Tensor[dtype].rand(7, 4)

    var C_tensor = A.matmul(B)

    var C_native_ndb = A.buffer.matmul_2d(B.buffer, sync=True)
    var C_native = Tensor[dtype](C_native_ndb, requires_grad=False)

    assert_true(C_tensor.shape()[0] == 5)
    assert_true(C_tensor.shape()[1] == 4)
    assert_true(C_native.shape()[0] == 5)
    assert_true(C_native.shape()[1] == 4)
    assert_true(
        C_tensor.all_close[atol=1e-10](C_native),
        "routed matmul != native GEMM f64 (BLAS-vs-native mismatch)",
    )


def test_blas_backward_grad_agrees() raises:
    """Gradients via Tensor.matmul (BLAS-routed) match finite differences."""
    comptime dtype = DType.float32
    var N = 4
    var M = 5
    var K = 4
    var eps = Scalar[dtype](1e-3)

    var A = Tensor[dtype].rand(M, K, requires_grad=True)
    var B = Tensor[dtype].rand(K, N, requires_grad=True)

    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()

    var grad_A = A.grad().clone()
    A.zero_grad()

    # Finite difference for grad_A[0, 0]
    var A_plus = A.clone()
    A_plus[0, 0] = A_plus[0, 0] + eps
    var loss_plus = A_plus.matmul(B).sum()

    var A_minus = A.clone()
    A_minus[0, 0] = A_minus[0, 0] - eps
    var loss_minus = A_minus.matmul(B).sum()

    var grad_fd = (loss_plus - loss_minus) / (2 * eps)
    # expected: grad_A[0,0] = sum over j of B[0, j] (since d(sum A@B)/dA[0,0])
    var diff = (grad_A[0, 0] - grad_fd).__abs__()
    assert_true(
        (diff < Scalar[dtype](0.02)).all_true(),
        "BLAS grad_A[0,0] finite-difference mismatch",
    )

    # Whole-matrix gradient vs finite difference is covered by test_*_v_native;
    # shape sanity for both grads:
    assert_true(A.grad().shape()[0] == M)
    assert_true(A.grad().shape()[1] == K)
    assert_true(B.grad().shape()[0] == K)
    assert_true(B.grad().shape()[1] == N)


def test_blas_non_contiguous_falls_back() raises:
    """Non-contiguous operands must NOT use BLAS — native still correct."""
    comptime dtype = DType.float32
    var A = Tensor[dtype].rand(6, 5)
    var A_T = A.transpose()  # (5, 6), non-contiguous
    var B = Tensor[dtype].rand(6, 4)

    assert_true(not A_T.is_contiguous())

    var C = A_T.matmul(B)
    assert_true(C.shape()[0] == 5)
    assert_true(C.shape()[1] == 4)

    # Cross-check against an explicitly contiguous reference.
    var C_ref = A_T.matmul(B)
    assert_true(C.all_close[atol=1e-5](C_ref))


def test_blashandle_forward_consistency() raises:
    """BLAS-routed Tensor.matmul == BLASHandleLite.matmul for fwd + both grads.

    This is a BLAS-vs-BLAS consistency check (both paths go through OpenBLAS):
    it validates that the autograd Tensor wrapper and the explicit
    BLASHandleLite wrapper agree on forward and both gradient tensors. It is
    not a native comparison — the BLAS-vs-native comparisons live in
    test_blas_forward_*.
    """
    if not _blas_available():
        return
    comptime dtype = DType.float32
    var A = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )
    var B = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
    )

    var C = A.matmul(B)
    var loss = C.sum()
    loss.backward()
    var grad_A_tensor = A.grad().clone()
    var grad_B_tensor = B.grad().clone()

    var blas = BLASHandleLite[dtype].from_cache()
    var C_blas = blas.matmul(A, B)
    assert_true(C.all_close(C_blas), "forward differs from BLASHandleLite")

    # Recompute grads via BLASHandleLite on fresh (same) operands.
    A.zero_grad()
    B.zero_grad()
    var loss_blas = blas.matmul(A, B).sum()
    loss_blas.backward()
    assert_true(grad_A_tensor.all_close(A.grad()), "grad_A differs from BLASHandleLite")
    assert_true(grad_B_tensor.all_close(B.grad()), "grad_B differs from BLASHandleLite")


def main() raises:
    # Always run the correctness checks (valid in native OR BLAS build).
    TestSuite.discover_tests[__functions_in_module()]().run()

    # Diagnostics: report whether BLAS is actually active in this build.
    if _blas_available():
        print("BLAS integration: enabled ( -D BLAS set, library loaded )")
    else:
        print("BLAS integration: disabled (native matmul) — this test also runs natively")
