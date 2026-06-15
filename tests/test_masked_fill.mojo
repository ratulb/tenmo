from tenmo.tensor import Tensor
from tenmo.common_utils import i
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

# ===----------------------------------------------------------------------=== #
# Masked fill tests — prefix: masked_fill_
# Covers: forward, backward, broadcasting, GPU parity
# ===----------------------------------------------------------------------=== #

comptime F32 = DType.float32

# ===----------------------------------------------------------------------=== #
# CPU – Forward
# ===----------------------------------------------------------------------=== #


def test_masked_fill_cpu_forward() raises:
    var mask = Tensor[DType.bool].d2(
        [[True, False],
         [False, True]]
    )
    var x = Tensor[F32].d2(
        [[1, 2],
         [3, 4]]
    )
    var y = x.masked_fill[track_grad=False](mask, 99.0)
    var expected = Tensor[F32].d2(
        [[99, 2],
         [3, 99]]
    )
    assert_true(y.all_close(expected))


def test_masked_fill_cpu_all_true() raises:
    var mask = Tensor[DType.bool].full([2, 3], True)
    var x = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    var y = x.masked_fill[track_grad=False](mask, 42.0)
    var expected = Tensor[F32].full([2, 3], 42.0)
    assert_true(y.all_close(expected))


def test_masked_fill_cpu_all_false() raises:
    var mask = Tensor[DType.bool].full([2, 3], False)
    var x = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    var y = x.masked_fill[track_grad=False](mask, 42.0)
    assert_true(y.all_close(x))


def test_masked_fill_cpu_mask_broadcast_row() raises:
    var mask = Tensor[DType.bool].d1([True, False, True])
    var x = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6]]
    )
    var y = x.masked_fill[track_grad=False](mask, 99.0)
    var expected = Tensor[F32].d2(
        [[99, 2, 99],
         [99, 5, 99]]
    )
    assert_true(y.all_close(expected))


def test_masked_fill_cpu_no_requires_grad() raises:
    var mask = Tensor[DType.bool].d1([True, False])
    var x = Tensor[F32].d1([1, 2], requires_grad=False)
    var y = x.masked_fill[track_grad=True](mask, 99.0)
    assert_true(not y.requires_grad)


# ===----------------------------------------------------------------------=== #
# CPU – Backward
# ===----------------------------------------------------------------------=== #


def test_masked_fill_cpu_backward() raises:
    var mask = Tensor[DType.bool].d2(
        [[True, False],
         [False, True]]
    )
    var x = Tensor[F32].d2(
        [[1, 2],
         [3, 4]],
        requires_grad=True,
    )
    var y = x.masked_fill(mask, 99.0)
    var loss = y.sum()
    loss.backward()
    # gradient flows to x where mask is False (original values kept)
    var expected_grad = Tensor[F32].d2(
        [[0, 1],
         [1, 0]]
    )
    assert_true(x.grad().all_close(expected_grad))


def test_masked_fill_cpu_backward_all_true() raises:
    var mask = Tensor[DType.bool].full([2, 3], True)
    var x = Tensor[F32].rand([2, 3], requires_grad=True)
    var y = x.masked_fill(mask, 99.0)
    var loss = y.sum()
    loss.backward()
    # no gradient flows to x (all replaced by fill value)
    var expected_grad = Tensor[F32].zeros([2, 3])
    assert_true(x.grad().all_close(expected_grad))


def test_masked_fill_cpu_backward_all_false() raises:
    var mask = Tensor[DType.bool].full([2, 3], False)
    var x = Tensor[F32].rand([2, 3], requires_grad=True)
    var y = x.masked_fill(mask, 99.0)
    var loss = y.sum()
    loss.backward()
    # full gradient flows to x (nothing replaced)
    var expected_grad = Tensor[F32].ones(2, 3)
    assert_true(x.grad().all_close(expected_grad))


def test_masked_fill_cpu_backward_mask_broadcast() raises:
    var mask = Tensor[DType.bool].d1([True, False])
    var x = Tensor[F32].d2(
        [[1, 2],
         [3, 4]],
        requires_grad=True,
    )
    var y = x.masked_fill(mask, 99.0)
    var loss = y.sum()
    loss.backward()
    # mask broadcasts to (2,2): [[True, False], [True, False]]
    var expected_grad = Tensor[F32].d2(
        [[0, 1],
         [0, 1]]
    )
    assert_true(x.grad().all_close(expected_grad))


# ===----------------------------------------------------------------------=== #
# GPU – Forward and backward parity
# ===----------------------------------------------------------------------=== #


def test_masked_fill_gpu_forward() raises:
    comptime if has_accelerator():
        var mask = Tensor[DType.bool].d2(
            [[True, False],
             [False, True]]
        )
        var x = Tensor[F32].d2(
            [[1, 2],
             [3, 4]]
        )
        var y_cpu = x.masked_fill[track_grad=False](mask, 99.0)
        var mask_gpu = mask.to_gpu()
        var x_gpu = x.to_gpu()
        var y_gpu = x_gpu.masked_fill[track_grad=False](mask_gpu, 99.0).to_cpu()
        assert_true(y_cpu.all_close[atol=1e-6](y_gpu))


def test_masked_fill_gpu_backward() raises:
    comptime if has_accelerator():
        var mask = Tensor[DType.bool].d2(
            [[True, False],
             [False, True]]
        )
        var x_cpu = Tensor[F32].d2(
            [[1, 2],
             [3, 4]],
            requires_grad=True,
        )
        var y_cpu = x_cpu.masked_fill(mask, 99.0)
        var loss_cpu = y_cpu.sum()
        loss_cpu.backward()

        var x_gpu = Tensor[F32].d2(
            [[1, 2],
             [3, 4]],
            requires_grad=True,
        ).to_gpu()
        var mask_gpu = mask.to_gpu()
        var y_gpu = x_gpu.masked_fill(mask_gpu, 99.0)
        var loss_gpu = y_gpu.sum()
        loss_gpu.backward()

        assert_true(x_cpu.grad().all_close[atol=1e-6](x_gpu.grad()))


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
