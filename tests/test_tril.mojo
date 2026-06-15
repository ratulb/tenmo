from tenmo.tensor import Tensor
from tenmo.common_utils import i
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

# ===----------------------------------------------------------------------=== #
# Tril tests — prefix: tril_
# Covers: forward 2D, batched, diagonal offset, backward, GPU parity
# ===----------------------------------------------------------------------=== #

comptime F32 = DType.float32

# ===----------------------------------------------------------------------=== #
# CPU – Forward
# ===----------------------------------------------------------------------=== #


def test_tril_cpu_2d_forward() raises:
    var x = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    )
    var y = x.tril[track_grad=False]()
    var expected = Tensor[F32].d2(
        [[1, 0, 0],
         [4, 5, 0],
         [7, 8, 9]]
    )
    assert_true(y.all_close(expected))


def test_tril_cpu_with_diagonal_positive() raises:
    var x = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    )
    var y = x.tril[track_grad=False](diagonal=1)
    var expected = Tensor[F32].d2(
        [[1, 2, 0],
         [4, 5, 6],
         [7, 8, 9]]
    )
    assert_true(y.all_close(expected))


def test_tril_cpu_with_diagonal_negative() raises:
    var x = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    )
    var y = x.tril[track_grad=False](diagonal=-1)
    var expected = Tensor[F32].d2(
        [[0, 0, 0],
         [4, 0, 0],
         [7, 8, 0]]
    )
    assert_true(y.all_close(expected))


def test_tril_cpu_rectangular() raises:
    var x = Tensor[F32].d2(
        [[1, 2, 3, 4],
         [5, 6, 7, 8]]
    )
    var y = x.tril[track_grad=False]()
    var expected = Tensor[F32].d2(
        [[1, 0, 0, 0],
         [5, 6, 0, 0]]
    )
    assert_true(y.all_close(expected))


def test_tril_cpu_batched_3d() raises:
    var x = Tensor[F32].d3(
        [
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            [[9, 8, 7],
             [6, 5, 4],
             [3, 2, 1]],
        ]
    )
    var y = x.tril[track_grad=False]()
    var expected = Tensor[F32].d3(
        [
            [[1, 0, 0],
             [4, 5, 0],
             [7, 8, 9]],
            [[9, 0, 0],
             [6, 5, 0],
             [3, 2, 1]],
        ]
    )
    assert_true(y.all_close(expected))


def test_tril_cpu_batched_4d() raises:
    var x = Tensor[F32].full([2, 2, 3, 3], 1.0)
    var y = x.tril[track_grad=False]()
    for b0 in range(2):
        for b1 in range(2):
            for r in range(3):
                for c in range(3):
                    var val = y[i(b0), i(b1), i(r), i(c)].item()
                    if c <= r:
                        assert_true(val == 1.0)
                    else:
                        assert_true(val == 0.0)


def test_tril_cpu_no_requires_grad() raises:
    var x = Tensor[F32].d2([[1, 2], [3, 4]], requires_grad=False)
    var y = x.tril[track_grad=False]()
    assert_true(not y.requires_grad)


# ===----------------------------------------------------------------------=== #
# CPU – Backward
# ===----------------------------------------------------------------------=== #


def test_tril_cpu_backward() raises:
    var x = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        requires_grad=True,
    )
    var y = x.tril()
    var loss = y.sum()
    loss.backward()
    var expected_grad = Tensor[F32].d2(
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    )
    assert_true(x.grad().all_close(expected_grad))


def test_tril_cpu_backward_diagonal_positive() raises:
    var x = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        requires_grad=True,
    )
    var y = x.tril(diagonal=1)
    var loss = y.sum()
    loss.backward()
    var expected_grad = Tensor[F32].d2(
        [[1, 1, 0],
         [1, 1, 1],
         [1, 1, 1]]
    )
    assert_true(x.grad().all_close(expected_grad))


def test_tril_cpu_backward_diagonal_negative() raises:
    var x = Tensor[F32].d2(
        [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]],
        requires_grad=True,
    )
    var y = x.tril(diagonal=-1)
    var loss = y.sum()
    loss.backward()
    var expected_grad = Tensor[F32].d2(
        [[0, 0, 0],
         [1, 0, 0],
         [1, 1, 0]]
    )
    assert_true(x.grad().all_close(expected_grad))


def test_tril_cpu_backward_batched() raises:
    var x = Tensor[F32].d3(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        requires_grad=True,
    )
    var y = x.tril()
    var loss = y.sum()
    loss.backward()
    var expected_grad = Tensor[F32].d3(
        [
            [[1, 0], [1, 1]],
            [[1, 0], [1, 1]],
        ]
    )
    assert_true(x.grad().all_close(expected_grad))


# ===----------------------------------------------------------------------=== #
# GPU – Forward and backward parity
# ===----------------------------------------------------------------------=== #


def test_tril_gpu_forward() raises:
    comptime if has_accelerator():
        var x = Tensor[F32].d2(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        var y_cpu = x.tril[track_grad=False]()
        var x_gpu = x.to_gpu()
        var y_gpu = x_gpu.tril[track_grad=False]().to_cpu()
        assert_true(y_cpu.all_close[atol=1e-6](y_gpu))


def test_tril_gpu_backward() raises:
    comptime if has_accelerator():
        var x = Tensor[F32].d2(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            requires_grad=True,
        )
        var y_cpu = x.tril()
        var loss_cpu = y_cpu.sum()
        loss_cpu.backward()

        var x_gpu = Tensor[F32].d2(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]],
            requires_grad=True,
        ).to_gpu()
        var y_gpu = x_gpu.tril()
        var loss_gpu = y_gpu.sum()
        loss_gpu.backward()

        assert_true(x.grad().all_close[atol=1e-6](x_gpu.grad()))


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
