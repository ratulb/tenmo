from tenmo.tensor import Tensor
from tenmo.common_utils import i
from std.testing import assert_true, TestSuite
from std.sys import has_accelerator

comptime F32 = DType.float32

# ===----------------------------------------------------------------------=== #
# CPU — Forward
# ===----------------------------------------------------------------------=== #


def test_cumsum_1d() raises:
    var x = Tensor[F32].d1([1.0, 2.0, 3.0, 4.0])
    var y = x.cumsum[track_grad=False]()
    var expected = Tensor[F32].d1([1.0, 3.0, 6.0, 10.0])
    assert_true(y.all_close(expected))


def test_cumsum_2d_axis0() raises:
    var x = Tensor[F32].d2([[1.0, 2.0], [3.0, 4.0]])
    var y = x.cumsum[track_grad=False](axis=0)
    var expected = Tensor[F32].d2([[1.0, 2.0], [4.0, 6.0]])
    assert_true(y.all_close(expected))


def test_cumsum_2d_axis1() raises:
    var x = Tensor[F32].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var y = x.cumsum[track_grad=False](axis=1)
    var expected = Tensor[F32].d2([[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]])
    assert_true(y.all_close(expected))


def test_cumsum_2d_axis_neg1() raises:
    var x = Tensor[F32].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var y = x.cumsum[track_grad=False](axis=-1)
    var expected = Tensor[F32].d2([[1.0, 3.0, 6.0], [4.0, 9.0, 15.0]])
    assert_true(y.all_close(expected))


def test_cumsum_3d_axis1() raises:
    var x = Tensor[F32].d3(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )
    var y = x.cumsum[track_grad=False](axis=1)
    var expected = Tensor[F32].d3(
        [
            [[1.0, 2.0], [4.0, 6.0]],
            [[5.0, 6.0], [12.0, 14.0]],
        ]
    )
    assert_true(y.all_close(expected))


def test_cumsum_1d_single_element() raises:
    var x = Tensor[F32].d1([5.0])
    var y = x.cumsum[track_grad=False]()
    assert_true(y.all_close(Tensor[F32].d1([5.0])))


def test_cumsum_no_requires_grad() raises:
    var x = Tensor[F32].d1([1.0, 2.0])
    var y = x.cumsum[track_grad=False]()
    assert_true(not y.requires_grad)


# ===----------------------------------------------------------------------=== #
# CPU — Backward
# ===----------------------------------------------------------------------=== #


def test_cumsum_backward_1d() raises:
    var x = Tensor[F32].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var y = x.cumsum()
    var loss = y.sum()
    loss.backward()
    # dy = [1,1,1,1], backward of cumsum = suffix sum → [4,3,2,1]
    var expected_grad = Tensor[F32].d1([4.0, 3.0, 2.0, 1.0])
    assert_true(x.grad().all_close(expected_grad))


def test_cumsum_backward_2d_axis0() raises:
    var x = Tensor[F32].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.cumsum(axis=0)
    var loss = y.sum()
    loss.backward()
    # dy = [[1,1],[1,1]], suffix sum along rows → [[2,2],[1,1]]
    # row 0: dx[0,:] = dy[0,:] + dy[1,:] = [2,2]
    # row 1: dx[1,:] = dy[1,:] = [1,1]
    var expected_grad = Tensor[F32].d2([[2.0, 2.0], [1.0, 1.0]])
    assert_true(x.grad().all_close(expected_grad))


def test_cumsum_backward_2d_axis1() raises:
    var x = Tensor[F32].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var y = x.cumsum(axis=1)
    var loss = y.sum()
    loss.backward()
    # dy = [[1,1,1],[1,1,1]], suffix sum along cols → [[3,2,1],[3,2,1]]
    var expected_grad = Tensor[F32].d2([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    assert_true(x.grad().all_close(expected_grad))


def test_cumsum_backward_3d_axis1() raises:
    var x = Tensor[F32].d3(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        requires_grad=True,
    )
    var y = x.cumsum(axis=1)
    var loss = y.sum()
    loss.backward()
    var expected_grad = Tensor[F32].d3(
        [
            [[2.0, 2.0], [1.0, 1.0]],
            [[2.0, 2.0], [1.0, 1.0]],
        ]
    )
    assert_true(x.grad().all_close(expected_grad))


# ===----------------------------------------------------------------------=== #
# GPU — Forward and backward parity
# ===----------------------------------------------------------------------=== #


def test_cumsum_gpu_forward() raises:
    comptime if has_accelerator():
        var x = Tensor[F32].d1([1.0, 2.0, 3.0, 4.0])
        var y_cpu = x.cumsum[track_grad=False]()
        var x_gpu = x.to_gpu()
        var y_gpu = x_gpu.cumsum[track_grad=False]().to_cpu()
        assert_true(y_cpu.all_close[atol=1e-6](y_gpu))


def test_cumsum_gpu_backward() raises:
    comptime if has_accelerator():
        var x = Tensor[F32].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        var y_cpu = x.cumsum()
        var loss_cpu = y_cpu.sum()
        loss_cpu.backward()

        var x_gpu = (
            Tensor[F32].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True).to_gpu()
        )
        var y_gpu = x_gpu.cumsum()
        var loss_gpu = y_gpu.sum()
        loss_gpu.backward()

        assert_true(x.grad().all_close[atol=1e-6](x_gpu.grad()))


# ===----------------------------------------------------------------------=== #
# Entry point
# ===----------------------------------------------------------------------=== #


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
