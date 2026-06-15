from tenmo.tensor import Tensor
from tenmo.common_utils import i
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

# ===----------------------------------------------------------------------=== #
# Where tests — prefix: where_
# Covers: forward, backward, scalar/tensor variants, broadcasting, GPU parity
# ===----------------------------------------------------------------------=== #

comptime F32 = DType.float32

# ===----------------------------------------------------------------------=== #
# CPU – Forward
# ===----------------------------------------------------------------------=== #


def test_where_cpu_both_tensor() raises:
    var cond = Tensor[DType.bool].d2(
        [[True, False],
         [False, True]]
    )
    var a = Tensor[F32].d2(
        [[1, 2],
         [3, 4]]
    )
    var b = Tensor[F32].d2(
        [[10, 20],
         [30, 40]]
    )
    var y = Tensor[F32].where[track_grad=False](cond, a, b)
    var expected = Tensor[F32].d2(
        [[1, 20],
         [30, 4]]
    )
    assert_true(y.all_close(expected))


def test_where_cpu_a_scalar() raises:
    var cond = Tensor[DType.bool].d2(
        [[True, False],
         [False, True]]
    )
    var a: Scalar[F32] = 99.0
    var b = Tensor[F32].d2(
        [[1, 2],
         [3, 4]]
    )
    var y = Tensor[F32].where[track_grad=False](cond, a, b)
    var expected = Tensor[F32].d2(
        [[99, 2],
         [3, 99]]
    )
    assert_true(y.all_close(expected))


def test_where_cpu_b_scalar() raises:
    var cond = Tensor[DType.bool].d2(
        [[True, False],
         [False, True]]
    )
    var a = Tensor[F32].d2(
        [[1, 2],
         [3, 4]]
    )
    var b: Scalar[F32] = 99.0
    var y = Tensor[F32].where[track_grad=False](cond, a, b)
    var expected = Tensor[F32].d2(
        [[1, 99],
         [99, 4]]
    )
    assert_true(y.all_close(expected))


def test_where_cpu_both_scalar() raises:
    var cond = Tensor[DType.bool].d2(
        [[True, False],
         [False, True]]
    )
    var a: Scalar[F32] = 10.0
    var b: Scalar[F32] = 20.0
    var y = Tensor[F32].where[track_grad=False](cond, a, b)
    var expected = Tensor[F32].d2(
        [[10, 20],
         [20, 10]]
    )
    assert_true(y.all_close(expected))


def test_where_cpu_cond_broadcast_row() raises:
    var cond = Tensor[DType.bool].d1([True, False, True])
    var a = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    var b = Tensor[F32].d2(
        [[10, 20, 30],
         [40, 50, 60]]
    )
    var y = Tensor[F32].where[track_grad=False](cond, a, b)
    var expected = Tensor[F32].d2(
        [[1, 20, 3],
         [4, 50, 6]]
    )
    assert_true(y.all_close(expected))


def test_where_cpu_cond_broadcast_col() raises:
    var cond = Tensor[DType.bool].d2(
        [[True],
         [False]]
    )
    var a = Tensor[F32].d2(
        [[1, 2],
         [3, 4]]
    )
    var b = Tensor[F32].d2(
        [[10, 20],
         [30, 40]]
    )
    var y = Tensor[F32].where[track_grad=False](cond, a, b)
    var expected = Tensor[F32].d2(
        [[1, 2],
         [30, 40]]
    )
    assert_true(y.all_close(expected))


def test_where_cpu_no_requires_grad() raises:
    var cond = Tensor[DType.bool].d1([True, False])
    var a = Tensor[F32].d1([1, 2], requires_grad=False)
    var b = Tensor[F32].d1([10, 20], requires_grad=False)
    var y = Tensor[F32].where[track_grad=True](cond, a, b)
    assert_true(not y.requires_grad)


# ===----------------------------------------------------------------------=== #
# CPU – Backward
# ===----------------------------------------------------------------------=== #


def test_where_cpu_backward_both_require_grad() raises:
    var cond = Tensor[DType.bool].d2(
        [[True, False],
         [False, True]]
    )
    var a = Tensor[F32].d2(
        [[1, 2],
         [3, 4]],
        requires_grad=True,
    )
    var b = Tensor[F32].d2(
        [[10, 20],
         [30, 40]],
        requires_grad=True,
    )
    var y = Tensor[F32].where(cond, a, b)
    var loss = y.sum()
    loss.backward()
    var expected_grad_a = Tensor[F32].d2(
        [[1, 0],
         [0, 1]]
    )
    var expected_grad_b = Tensor[F32].d2(
        [[0, 1],
         [1, 0]]
    )
    assert_true(a.grad().all_close(expected_grad_a))
    assert_true(b.grad().all_close(expected_grad_b))


def test_where_cpu_backward_only_a() raises:
    var cond = Tensor[DType.bool].d1([True, False, True])
    var a = Tensor[F32].d1([1, 2, 3], requires_grad=True)
    var b = Tensor[F32].d1([10, 20, 30], requires_grad=False)
    var y = Tensor[F32].where(cond, a, b)
    var loss = y.sum()
    loss.backward()
    var expected_grad_a = Tensor[F32].d1([1, 0, 1])
    assert_true(a.grad().all_close(expected_grad_a))


def test_where_cpu_backward_only_b() raises:
    var cond = Tensor[DType.bool].d1([True, False, True])
    var a = Tensor[F32].d1([1, 2, 3], requires_grad=False)
    var b = Tensor[F32].d1([10, 20, 30], requires_grad=True)
    var y = Tensor[F32].where(cond, a, b)
    var loss = y.sum()
    loss.backward()
    var expected_grad_b = Tensor[F32].d1([0, 1, 0])
    assert_true(b.grad().all_close(expected_grad_b))


def test_where_cpu_backward_a_scalar() raises:
    var cond = Tensor[DType.bool].d1([True, False])
    var a: Scalar[F32] = 99.0
    var b = Tensor[F32].d1([1, 2], requires_grad=True)
    var y = Tensor[F32].where(cond, a, b)
    var loss = y.sum()
    loss.backward()
    var expected_grad_b = Tensor[F32].d1([0, 1])
    assert_true(b.grad().all_close(expected_grad_b))


def test_where_cpu_backward_b_scalar() raises:
    var cond = Tensor[DType.bool].d1([True, False])
    var a = Tensor[F32].d1([1, 2], requires_grad=True)
    var b: Scalar[F32] = 99.0
    var y = Tensor[F32].where(cond, a, b)
    var loss = y.sum()
    loss.backward()
    var expected_grad_a = Tensor[F32].d1([1, 0])
    assert_true(a.grad().all_close(expected_grad_a))


def test_where_cpu_backward_cond_broadcast() raises:
    var cond = Tensor[DType.bool].d2(
        [[True],
         [False]]
    )
    var a = Tensor[F32].d2(
        [[1, 2],
         [3, 4]],
        requires_grad=True,
    )
    var b = Tensor[F32].d2(
        [[10, 20],
         [30, 40]],
        requires_grad=True,
    )
    var y = Tensor[F32].where(cond, a, b)
    var loss = y.sum()
    loss.backward()
    var expected_grad_a = Tensor[F32].d2(
        [[1, 1],
         [0, 0]]
    )
    var expected_grad_b = Tensor[F32].d2(
        [[0, 0],
         [1, 1]]
    )
    assert_true(a.grad().all_close(expected_grad_a))
    assert_true(b.grad().all_close(expected_grad_b))


# ===----------------------------------------------------------------------=== #
# GPU – Forward and backward parity
# ===----------------------------------------------------------------------=== #


def test_where_gpu_forward() raises:
    comptime if has_accelerator():
        var cond = Tensor[DType.bool].d2(
            [[True, False],
             [False, True]]
        )
        var a = Tensor[F32].d2(
            [[1, 2],
             [3, 4]]
        )
        var b = Tensor[F32].d2(
            [[10, 20],
             [30, 40]]
        )
        var y_cpu = Tensor[F32].where[track_grad=False](cond, a, b)
        var cond_gpu = cond.to_gpu()
        var a_gpu = a.to_gpu()
        var b_gpu = b.to_gpu()
        var y_gpu = Tensor[F32].where[track_grad=False](cond_gpu, a_gpu, b_gpu).to_cpu()
        assert_true(y_cpu.all_close[atol=1e-6](y_gpu))


def test_where_gpu_backward() raises:
    comptime if has_accelerator():
        var cond = Tensor[DType.bool].d2(
            [[True, False],
             [False, True]]
        )
        var a = Tensor[F32].d2(
            [[1, 2],
             [3, 4]],
            requires_grad=True,
        )
        var b = Tensor[F32].d2(
            [[10, 20],
             [30, 40]],
            requires_grad=True,
        )
        var y_cpu = Tensor[F32].where(cond, a, b)
        var loss_cpu = y_cpu.sum()
        loss_cpu.backward()

        var cond_gpu = cond.to_gpu()
        var a_gpu = Tensor[F32].d2(
            [[1, 2],
             [3, 4]],
            requires_grad=True,
        ).to_gpu()
        var b_gpu = Tensor[F32].d2(
            [[10, 20],
             [30, 40]],
            requires_grad=True,
        ).to_gpu()
        var y_gpu = Tensor[F32].where(cond_gpu, a_gpu, b_gpu)
        var loss_gpu = y_gpu.sum()
        loss_gpu.backward()

        assert_true(a.grad().all_close[atol=1e-6](a_gpu.grad()))
        assert_true(b.grad().all_close[atol=1e-6](b_gpu.grad()))


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
