from std.testing import assert_true, TestSuite
from tenmo import Tensor, Shape
from std.sys import has_accelerator


# ─────────────────────────────────────────────
#  CPU — Tensor / Tensor  (same shape)
# ─────────────────────────────────────────────


def test_div_cpu_tt_1d_forward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([6.0, 9.0, 12.0], requires_grad=True)
    var y = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var z = x / y
    assert_true(z.all_close(Tensor[dtype].d1([3.0, 3.0, 3.0])))


def test_div_cpu_tt_1d_backward_x_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([6.0, 9.0, 12.0], requires_grad=True)
    var y = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dx = 1/y
    assert_true(
        x.grad().all_close[atol=1e-5](Tensor[dtype].d1([0.5, 0.333333, 0.25]))
    )


def test_div_cpu_tt_1d_backward_y_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([6.0, 9.0, 12.0], requires_grad=True)
    var y = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dy = -x/y^2
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].d1([-1.5, -1.0, -0.75]))
    )


def test_div_cpu_tt_2d_forward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[4.0, 9.0], [16.0, 25.0]], requires_grad=True)
    var y = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    var z = x / y
    assert_true(z.all_close(Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]])))


def test_div_cpu_tt_2d_backward_x_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[4.0, 9.0], [16.0, 25.0]], requires_grad=True)
    var y = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    assert_true(
        x.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[0.5, 0.333333], [0.25, 0.2]])
        )
    )


def test_div_cpu_tt_2d_backward_y_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[4.0, 9.0], [16.0, 25.0]], requires_grad=True)
    var y = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dy = -x/y^2 = [[-1.0, -1.0], [-1.0, -1.0]]
    assert_true(
        y.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[-1.0, -1.0], [-1.0, -1.0]])
        )
    )


def test_div_cpu_tt_3d_forward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[8.0, 6.0], [4.0, 2.0]], [[9.0, 12.0], [15.0, 18.0]]],
        requires_grad=True,
    )
    var y = Tensor[dtype].d3(
        [[[2.0, 3.0], [2.0, 1.0]], [[3.0, 4.0], [5.0, 6.0]]], requires_grad=True
    )
    var z = x / y
    assert_true(
        z.all_close(
            Tensor[dtype].d3(
                [[[4.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]
            )
        )
    )


def test_div_cpu_tt_3d_backward_x_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[8.0, 6.0], [4.0, 2.0]], [[9.0, 12.0], [15.0, 18.0]]],
        requires_grad=True,
    )
    var y = Tensor[dtype].d3(
        [[[2.0, 3.0], [2.0, 1.0]], [[3.0, 4.0], [5.0, 6.0]]], requires_grad=True
    )
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dx = 1/y
    assert_true(
        x.grad().all_close[atol=1e-5](
            Tensor[dtype].d3(
                [
                    [[0.5, 0.333333], [0.5, 1.0]],
                    [[0.333333, 0.25], [0.2, 0.166667]],
                ]
            )
        )
    )


def test_div_cpu_tt_3d_backward_y_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[8.0, 6.0], [4.0, 2.0]], [[9.0, 12.0], [15.0, 18.0]]],
        requires_grad=True,
    )
    var y = Tensor[dtype].d3(
        [[[2.0, 3.0], [2.0, 1.0]], [[3.0, 4.0], [5.0, 6.0]]], requires_grad=True
    )
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dy = -x/y^2
    assert_true(
        y.grad().all_close[atol=1e-5](
            Tensor[dtype].d3(
                [
                    [[-2.0, -0.666667], [-1.0, -2.0]],
                    [[-1.0, -0.75], [-0.6, -0.5]],
                ]
            )
        )
    )


def test_div_cpu_tt_4d_forward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(2, 2, 2, 2), 8.0, requires_grad=True)
    var y = Tensor[dtype].full(Shape(2, 2, 2, 2), 2.0, requires_grad=True)
    var z = x / y
    assert_true(z.all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 4.0)))


def test_div_cpu_tt_4d_backward_x_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(2, 2, 2, 2), 8.0, requires_grad=True)
    var y = Tensor[dtype].full(Shape(2, 2, 2, 2), 2.0, requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    assert_true(x.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 0.5)))


def test_div_cpu_tt_4d_backward_y_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(2, 2, 2, 2), 8.0, requires_grad=True)
    var y = Tensor[dtype].full(Shape(2, 2, 2, 2), 2.0, requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dy = -x/y^2 = -8/4 = -2.0
    assert_true(y.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), -2.0)))


def test_div_cpu_tt_scalar_tensor_forward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(), 10.0, requires_grad=True)
    var y = Tensor[dtype].full(Shape(), 2.0, requires_grad=True)
    var z = x / y
    assert_true(z.all_close(Tensor[dtype].full(Shape(), 5.0)))


def test_div_cpu_tt_scalar_tensor_backward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(), 10.0, requires_grad=True)
    var y = Tensor[dtype].full(Shape(), 2.0, requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    assert_true(x.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(), 0.5)))
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(), -2.5))
    )


# ─────────────────────────────────────────────
#  CPU — Tensor / Scalar
# ─────────────────────────────────────────────


def test_div_cpu_ts_1d_forward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([2.0, 4.0, 8.0], requires_grad=True)
    var z = x / Scalar[dtype](2.0)
    assert_true(z.all_close(Tensor[dtype].d1([1.0, 2.0, 4.0])))


def test_div_cpu_ts_1d_backward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d1([2.0, 4.0, 8.0], requires_grad=True)
    var z = x / Scalar[dtype](2.0)
    var loss = z.sum()
    loss.backward()
    # dx = 1/scalar = 0.5 everywhere
    assert_true(x.grad().all_close(Tensor[dtype].d1([0.5, 0.5, 0.5])))


def test_div_cpu_ts_2d_forward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[4.0, 8.0], [12.0, 16.0]], requires_grad=True)
    var z = x / Scalar[dtype](4.0)
    assert_true(z.all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])))


def test_div_cpu_ts_2d_backward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[4.0, 8.0], [12.0, 16.0]], requires_grad=True)
    var z = x / Scalar[dtype](4.0)
    var loss = z.sum()
    loss.backward()
    assert_true(
        x.grad().all_close(Tensor[dtype].d2([[0.25, 0.25], [0.25, 0.25]]))
    )


def test_div_cpu_ts_3d_forward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[3.0, 6.0], [9.0, 12.0]], [[15.0, 18.0], [21.0, 24.0]]],
        requires_grad=True,
    )
    var z = x / Scalar[dtype](3.0)
    assert_true(
        z.all_close(
            Tensor[dtype].d3(
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
            )
        )
    )


def test_div_cpu_ts_3d_backward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[3.0, 6.0], [9.0, 12.0]], [[15.0, 18.0], [21.0, 24.0]]],
        requires_grad=True,
    )
    var z = x / Scalar[dtype](3.0)
    var loss = z.sum()
    loss.backward()
    assert_true(
        x.grad().all_close[atol=1e-5](
            Tensor[dtype].d3(
                [
                    [[0.333333, 0.333333], [0.333333, 0.333333]],
                    [[0.333333, 0.333333], [0.333333, 0.333333]],
                ]
            )
        )
    )


def test_div_cpu_ts_4d_forward_backward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(2, 2, 2, 2), 10.0, requires_grad=True)
    var z = x / Scalar[dtype](5.0)
    assert_true(z.all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 2.0)))
    var loss = z.sum()
    loss.backward()
    assert_true(x.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 0.2)))


# ─────────────────────────────────────────────
#  CPU — Scalar / Tensor
# ─────────────────────────────────────────────


def test_div_cpu_st_1d_forward() raises:
    comptime dtype = DType.float32
    var y = Tensor[dtype].d1([2.0, 4.0, 8.0], requires_grad=True)
    var z = Scalar[dtype](8.0) / y
    assert_true(z.all_close(Tensor[dtype].d1([4.0, 2.0, 1.0])))


def test_div_cpu_st_1d_backward() raises:
    comptime dtype = DType.float32
    var y = Tensor[dtype].d1([2.0, 4.0, 8.0], requires_grad=True)
    var z = Scalar[dtype](8.0) / y
    var loss = z.sum()
    loss.backward()
    # dy = -scalar/y^2
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].d1([-2.0, -0.5, -0.125]))
    )


def test_div_cpu_st_2d_forward() raises:
    comptime dtype = DType.float32
    var y = Tensor[dtype].d2([[2.0, 4.0], [5.0, 10.0]], requires_grad=True)
    var z = Scalar[dtype](20.0) / y
    assert_true(z.all_close(Tensor[dtype].d2([[10.0, 5.0], [4.0, 2.0]])))


def test_div_cpu_st_2d_backward() raises:
    comptime dtype = DType.float32
    var y = Tensor[dtype].d2([[2.0, 4.0], [5.0, 10.0]], requires_grad=True)
    var z = Scalar[dtype](20.0) / y
    var loss = z.sum()
    loss.backward()
    # dy = -20/y^2
    assert_true(
        y.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[-5.0, -1.25], [-0.8, -0.2]])
        )
    )


def test_div_cpu_st_3d_forward() raises:
    comptime dtype = DType.float32
    var y = Tensor[dtype].d3(
        [[[1.0, 2.0], [4.0, 5.0]], [[10.0, 20.0], [25.0, 50.0]]],
        requires_grad=True,
    )
    var z = Scalar[dtype](100.0) / y
    assert_true(
        z.all_close(
            Tensor[dtype].d3(
                [[[100.0, 50.0], [25.0, 20.0]], [[10.0, 5.0], [4.0, 2.0]]]
            )
        )
    )


def test_div_cpu_st_3d_backward() raises:
    comptime dtype = DType.float32
    var y = Tensor[dtype].d3(
        [[[1.0, 2.0], [4.0, 5.0]], [[10.0, 20.0], [25.0, 50.0]]],
        requires_grad=True,
    )
    var z = Scalar[dtype](100.0) / y
    var loss = z.sum()
    loss.backward()
    # dy = -100/y^2
    assert_true(
        y.grad().all_close[atol=1e-5](
            Tensor[dtype].d3(
                [
                    [[-100.0, -25.0], [-6.25, -4.0]],
                    [[-1.0, -0.25], [-0.16, -0.04]],
                ]
            )
        )
    )


def test_div_cpu_st_4d_forward_backward() raises:
    comptime dtype = DType.float32
    var y = Tensor[dtype].full(Shape(2, 2, 2, 2), 2.0, requires_grad=True)
    var z = Scalar[dtype](8.0) / y
    assert_true(z.all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 4.0)))
    var loss = z.sum()
    loss.backward()
    # dy = -8/4 = -2.0
    assert_true(y.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), -2.0)))


# ─────────────────────────────────────────────
#  CPU — Scalar Tensor / Tensor
# ─────────────────────────────────────────────


def test_div_cpu_stt_1d_forward() raises:
    comptime dtype = DType.float32
    var s = Tensor[dtype].scalar(6.0)
    var y = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var z = s / y
    assert_true(z.all_close(Tensor[dtype].d1([6.0, 3.0, 2.0])))


def test_div_cpu_stt_1d_backward() raises:
    comptime dtype = DType.float32
    var s = Tensor[dtype].scalar(6.0)
    var y = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var z = s / y
    var loss = z.sum()
    loss.backward()
    # dy = -6/y^2
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].d1([-6.0, -1.5, -0.666667]))
    )


def test_div_cpu_stt_2d_forward() raises:
    comptime dtype = DType.float32
    var s = Tensor[dtype].scalar(12.0)
    var y = Tensor[dtype].d2([[2.0, 3.0], [4.0, 6.0]], requires_grad=True)
    var z = s / y
    assert_true(z.all_close(Tensor[dtype].d2([[6.0, 4.0], [3.0, 2.0]])))


def test_div_cpu_stt_2d_backward() raises:
    comptime dtype = DType.float32
    var s = Tensor[dtype].scalar(12.0)
    var y = Tensor[dtype].d2([[2.0, 3.0], [4.0, 6.0]], requires_grad=True)
    var z = s / y
    var loss = z.sum()
    loss.backward()
    assert_true(
        y.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[-3.0, -1.333333], [-0.75, -0.333333]])
        )
    )


def test_div_cpu_stt_3d_forward() raises:
    comptime dtype = DType.float32
    var s = Tensor[dtype].scalar(24.0)
    var y = Tensor[dtype].d3(
        [[[2.0, 4.0], [6.0, 8.0]], [[3.0, 6.0], [12.0, 24.0]]],
        requires_grad=True,
    )
    var z = s / y
    assert_true(
        z.all_close(
            Tensor[dtype].d3(
                [[[12.0, 6.0], [4.0, 3.0]], [[8.0, 4.0], [2.0, 1.0]]]
            )
        )
    )


def test_div_cpu_stt_3d_backward() raises:
    comptime dtype = DType.float32
    var s = Tensor[dtype].scalar(24.0)
    var y = Tensor[dtype].d3(
        [[[2.0, 4.0], [6.0, 8.0]], [[3.0, 6.0], [12.0, 24.0]]],
        requires_grad=True,
    )
    var z = s / y
    var loss = z.sum()
    loss.backward()
    assert_true(
        y.grad().all_close[atol=1e-5](
            Tensor[dtype].d3(
                [
                    [[-6.0, -1.5], [-0.666667, -0.375]],
                    [[-2.666667, -0.666667], [-0.166667, -0.041667]],
                ]
            )
        )
    )


def test_div_cpu_stt_4d_forward_backward() raises:
    comptime dtype = DType.float32
    var s = Tensor[dtype].scalar(16.0)
    var y = Tensor[dtype].full(Shape(2, 2, 2, 2), 4.0, requires_grad=True)
    var z = s / y
    assert_true(z.all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 4.0)))
    var loss = z.sum()
    loss.backward()
    # dy = -16/16 = -1.0
    assert_true(y.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), -1.0)))


# ─────────────────────────────────────────────
#  CPU — Broadcasting
# ─────────────────────────────────────────────


def test_div_cpu_broadcast_2d_1d_forward() raises:
    comptime dtype = DType.float32
    # [2,3] / [3]  →  row-wise division
    var x = Tensor[dtype].d2(
        [[2.0, 6.0, 9.0], [4.0, 12.0, 18.0]], requires_grad=True
    )
    var y = Tensor[dtype].d1([2.0, 3.0, 3.0], requires_grad=True)
    var z = x / y
    assert_true(
        z.all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]))
    )


def test_div_cpu_broadcast_2d_1d_backward_x_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[2.0, 6.0, 9.0], [4.0, 12.0, 18.0]], requires_grad=True
    )
    var y = Tensor[dtype].d1([2.0, 3.0, 3.0], requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dx = 1/y broadcast to [2,3]
    assert_true(
        x.grad().all_close[atol=1e-5](
            Tensor[dtype].d2(
                [[0.5, 0.333333, 0.333333], [0.5, 0.333333, 0.333333]]
            )
        )
    )


def test_div_cpu_broadcast_2d_1d_backward_y_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[2.0, 6.0, 9.0], [4.0, 12.0, 18.0]], requires_grad=True
    )
    var y = Tensor[dtype].d1([2.0, 3.0, 3.0], requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dy = sum over broadcast dim of -x/y^2
    # col0: -(2+4)/4=-1.5, col1: -(6+12)/9=-2.0, col2: -(9+18)/9=-3.0
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].d1([-1.5, -2.0, -3.0]))
    )


def test_div_cpu_broadcast_3d_1d_forward() raises:
    comptime dtype = DType.float32
    # [2,2,2] / [2]  →  last-dim broadcast
    var x = Tensor[dtype].d3(
        [[[4.0, 6.0], [8.0, 9.0]], [[2.0, 12.0], [6.0, 3.0]]],
        requires_grad=True,
    )
    var y = Tensor[dtype].d1([2.0, 3.0], requires_grad=True)
    var z = x / y
    assert_true(
        z.all_close(
            Tensor[dtype].d3(
                [[[2.0, 2.0], [4.0, 3.0]], [[1.0, 4.0], [3.0, 1.0]]]
            )
        )
    )


def test_div_cpu_broadcast_3d_1d_backward_x_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d3(
        [[[4.0, 6.0], [8.0, 9.0]], [[2.0, 12.0], [6.0, 3.0]]],
        requires_grad=True,
    )
    var y = Tensor[dtype].d1([2.0, 3.0], requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    assert_true(
        x.grad().all_close[atol=1e-5](
            Tensor[dtype].d3(
                [
                    [[0.5, 0.333333], [0.5, 0.333333]],
                    [[0.5, 0.333333], [0.5, 0.333333]],
                ]
            )
        )
    )


def test_div_cpu_broadcast_col_forward() raises:
    comptime dtype = DType.float32
    # [2,3] / [2,1]
    var x = Tensor[dtype].d2(
        [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], requires_grad=True
    )
    var y = Tensor[dtype].d2([[2.0], [3.0]], requires_grad=True)
    var z = x / y
    assert_true(
        z.all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    )


def test_div_cpu_broadcast_col_backward_x_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], requires_grad=True
    )
    var y = Tensor[dtype].d2([[2.0], [3.0]], requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    assert_true(
        x.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[0.5, 0.5, 0.5], [0.333333, 0.333333, 0.333333]])
        )
    )


def test_div_cpu_broadcast_col_backward_y_grad() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], requires_grad=True
    )
    var y = Tensor[dtype].d2([[2.0], [3.0]], requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dy = sum over broadcast cols of -x/y^2
    # row0: -(2+4+6)/4=-3.0, row1: -(3+6+9)/9=-2.0
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].d2([[-3.0], [-2.0]]))
    )


def test_div_cpu_broadcast_non_scalar_backward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].d2([[4.0, 8.0], [6.0, 12.0]], requires_grad=True)
    var y = Tensor[dtype].d1([2.0, 4.0], requires_grad=True)
    var z = x / y
    # backward on non-scalar directly
    z.backward()
    assert_true(
        x.grad().all_close[atol=1e-5](
            Tensor[dtype].d2([[0.5, 0.25], [0.5, 0.25]])
        )
    )


# ─────────────────────────────────────────────
#  CPU — float64
# ─────────────────────────────────────────────


def test_div_cpu_float64_tt_forward_backward() raises:
    comptime dtype = DType.float64
    var x = Tensor[dtype].d1([6.0, 9.0, 12.0], requires_grad=True)
    var y = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var z = x / y
    assert_true(z.all_close(Tensor[dtype].d1([3.0, 3.0, 3.0])))
    var loss = z.sum()
    loss.backward()
    assert_true(
        x.grad().all_close[atol=1e-9](Tensor[dtype].d1([0.5, 0.333333, 0.25]))
    )
    assert_true(
        y.grad().all_close[atol=1e-9](Tensor[dtype].d1([-1.5, -1.0, -0.75]))
    )


# ─────────────────────────────────────────────
#  GPU — Tensor / Tensor
# ─────────────────────────────────────────────


def test_div_gpu_tt_1d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([6.0, 9.0, 12.0], requires_grad=True)
        var y = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(z.to_cpu().all_close(Tensor[dtype].d1([3.0, 3.0, 3.0])))


def test_div_gpu_tt_1d_backward_x_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([6.0, 9.0, 12.0], requires_grad=True)
        var y = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([0.5, 0.333333, 0.25])
            )
        )


def test_div_gpu_tt_1d_backward_y_grad() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([6.0, 9.0, 12.0], requires_grad=True)
        var y = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close[atol=1e-5](Tensor[dtype].d1([-1.5, -1.0, -0.75]))
        )


def test_div_gpu_tt_2d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[4.0, 9.0], [16.0, 25.0]], requires_grad=True)
        var y = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(
            z.to_cpu().all_close(Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]]))
        )


def test_div_gpu_tt_2d_backward_both_grads() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[4.0, 9.0], [16.0, 25.0]], requires_grad=True)
        var y = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[0.5, 0.333333], [0.25, 0.2]])
            )
        )
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[-1.0, -1.0], [-1.0, -1.0]])
            )
        )


def test_div_gpu_tt_3d_forward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d3(
            [[[8.0, 6.0], [4.0, 2.0]], [[9.0, 12.0], [15.0, 18.0]]],
            requires_grad=True,
        )
        var y = Tensor[dtype].d3(
            [[[2.0, 3.0], [2.0, 1.0]], [[3.0, 4.0], [5.0, 6.0]]],
            requires_grad=True,
        )
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(
            z.to_cpu().all_close(
                Tensor[dtype].d3(
                    [[[4.0, 2.0], [2.0, 2.0]], [[3.0, 3.0], [3.0, 3.0]]]
                )
            )
        )


def test_div_gpu_tt_3d_backward_both_grads() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d3(
            [[[8.0, 6.0], [4.0, 2.0]], [[9.0, 12.0], [15.0, 18.0]]],
            requires_grad=True,
        )
        var y = Tensor[dtype].d3(
            [[[2.0, 3.0], [2.0, 1.0]], [[3.0, 4.0], [5.0, 6.0]]],
            requires_grad=True,
        )
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](
                Tensor[dtype].d3(
                    [
                        [[0.5, 0.333333], [0.5, 1.0]],
                        [[0.333333, 0.25], [0.2, 0.166667]],
                    ]
                )
            )
        )
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d3(
                    [
                        [[-2.0, -0.666667], [-1.0, -2.0]],
                        [[-1.0, -0.75], [-0.6, -0.5]],
                    ]
                )
            )
        )


def test_div_gpu_tt_4d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].full(Shape(2, 2, 2, 2), 8.0, requires_grad=True)
        var y = Tensor[dtype].full(Shape(2, 2, 2, 2), 2.0, requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(
            z.to_cpu().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 4.0))
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 0.5))
        )
        assert_true(
            y.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), -2.0))
        )


# ─────────────────────────────────────────────
#  GPU — Tensor / Scalar
# ─────────────────────────────────────────────


def test_div_gpu_ts_1d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d1([2.0, 4.0, 8.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var z = x_gpu / Scalar[dtype](2.0)
        assert_true(z.to_cpu().all_close(Tensor[dtype].d1([1.0, 2.0, 4.0])))
        var loss = z.sum()
        loss.backward()
        assert_true(x.grad().all_close(Tensor[dtype].d1([0.5, 0.5, 0.5])))


def test_div_gpu_ts_2d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[4.0, 8.0], [12.0, 16.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var z = x_gpu / Scalar[dtype](4.0)
        assert_true(
            z.to_cpu().all_close(Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]))
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close(Tensor[dtype].d2([[0.25, 0.25], [0.25, 0.25]]))
        )


def test_div_gpu_ts_3d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d3(
            [[[3.0, 6.0], [9.0, 12.0]], [[15.0, 18.0], [21.0, 24.0]]],
            requires_grad=True,
        )
        var x_gpu = x.to_gpu()
        var z = x_gpu / Scalar[dtype](3.0)
        assert_true(
            z.to_cpu().all_close(
                Tensor[dtype].d3(
                    [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
                )
            )
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](
                Tensor[dtype].d3(
                    [
                        [[0.333333, 0.333333], [0.333333, 0.333333]],
                        [[0.333333, 0.333333], [0.333333, 0.333333]],
                    ]
                )
            )
        )


def test_div_gpu_ts_4d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].full(Shape(2, 2, 2, 2), 10.0, requires_grad=True)
        var x_gpu = x.to_gpu()
        var z = x_gpu / Scalar[dtype](5.0)
        assert_true(
            z.to_cpu().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 2.0))
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 0.2))
        )


# ─────────────────────────────────────────────
#  GPU — Scalar / Tensor
# ─────────────────────────────────────────────


def test_div_gpu_st_1d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var y = Tensor[dtype].d1([2.0, 4.0, 8.0], requires_grad=True)
        var y_gpu = y.to_gpu()
        var z = Scalar[dtype](8.0) / y_gpu
        assert_true(z.to_cpu().all_close(Tensor[dtype].d1([4.0, 2.0, 1.0])))
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([-2.0, -0.5, -0.125])
            )
        )


def test_div_gpu_st_2d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var y = Tensor[dtype].d2([[2.0, 4.0], [5.0, 10.0]], requires_grad=True)
        var y_gpu = y.to_gpu()
        var z = Scalar[dtype](20.0) / y_gpu
        assert_true(
            z.to_cpu().all_close(Tensor[dtype].d2([[10.0, 5.0], [4.0, 2.0]]))
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[-5.0, -1.25], [-0.8, -0.2]])
            )
        )


def test_div_gpu_st_3d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var y = Tensor[dtype].d3(
            [[[1.0, 2.0], [4.0, 5.0]], [[10.0, 20.0], [25.0, 50.0]]],
            requires_grad=True,
        )
        var y_gpu = y.to_gpu()
        var z = Scalar[dtype](100.0) / y_gpu
        assert_true(
            z.to_cpu().all_close(
                Tensor[dtype].d3(
                    [[[100.0, 50.0], [25.0, 20.0]], [[10.0, 5.0], [4.0, 2.0]]]
                )
            )
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d3(
                    [
                        [[-100.0, -25.0], [-6.25, -4.0]],
                        [[-1.0, -0.25], [-0.16, -0.04]],
                    ]
                )
            )
        )


def test_div_gpu_st_4d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var y = Tensor[dtype].full(Shape(2, 2, 2, 2), 2.0, requires_grad=True)
        var y_gpu = y.to_gpu()
        var z = Scalar[dtype](8.0) / y_gpu
        assert_true(
            z.to_cpu().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 4.0))
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), -2.0))
        )


# ─────────────────────────────────────────────
#  GPU — Scalar Tensor / Tensor
# ─────────────────────────────────────────────


def test_div_gpu_stt_1d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var s = Tensor[dtype].scalar(6.0)
        var y = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var s_gpu = s.to_gpu()
        var y_gpu = y.to_gpu()
        var z = s_gpu / y_gpu
        assert_true(z.to_cpu().all_close(Tensor[dtype].d1([6.0, 3.0, 2.0])))
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d1([-6.0, -1.5, -0.666667])
            )
        )


def test_div_gpu_stt_2d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var s = Tensor[dtype].scalar(12.0)
        var y = Tensor[dtype].d2([[2.0, 3.0], [4.0, 6.0]], requires_grad=True)
        var s_gpu = s.to_gpu()
        var y_gpu = y.to_gpu()
        var z = s_gpu / y_gpu
        assert_true(
            z.to_cpu().all_close(Tensor[dtype].d2([[6.0, 4.0], [3.0, 2.0]]))
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[-3.0, -1.333333], [-0.75, -0.333333]])
            )
        )


def test_div_gpu_stt_3d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var s = Tensor[dtype].scalar(24.0)
        var y = Tensor[dtype].d3(
            [[[2.0, 4.0], [6.0, 8.0]], [[3.0, 6.0], [12.0, 24.0]]],
            requires_grad=True,
        )
        var s_gpu = s.to_gpu()
        var y_gpu = y.to_gpu()
        var z = s_gpu / y_gpu
        assert_true(
            z.to_cpu().all_close(
                Tensor[dtype].d3(
                    [[[12.0, 6.0], [4.0, 3.0]], [[8.0, 4.0], [2.0, 1.0]]]
                )
            )
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d3(
                    [
                        [[-6.0, -1.5], [-0.666667, -0.375]],
                        [[-2.666667, -0.666667], [-0.166667, -0.041667]],
                    ]
                )
            )
        )


def test_div_gpu_stt_4d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var s = Tensor[dtype].scalar(16.0)
        var y = Tensor[dtype].full(Shape(2, 2, 2, 2), 4.0, requires_grad=True)
        var s_gpu = s.to_gpu()
        var y_gpu = y.to_gpu()
        var z = s_gpu / y_gpu
        assert_true(
            z.to_cpu().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), 4.0))
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close(Tensor[dtype].full(Shape(2, 2, 2, 2), -1.0))
        )


# ─────────────────────────────────────────────
#  GPU — Broadcasting
# ─────────────────────────────────────────────


def test_div_gpu_broadcast_2d_1d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2(
            [[2.0, 6.0, 9.0], [4.0, 12.0, 18.0]], requires_grad=True
        )
        var y = Tensor[dtype].d1([2.0, 3.0, 3.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(
            z.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
            )
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](
                Tensor[dtype].d2(
                    [[0.5, 0.333333, 0.333333], [0.5, 0.333333, 0.333333]]
                )
            )
        )
        assert_true(
            y.grad().all_close[atol=1e-5](Tensor[dtype].d1([-1.5, -2.0, -3.0]))
        )


def test_div_gpu_broadcast_col_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2(
            [[2.0, 4.0, 6.0], [3.0, 6.0, 9.0]], requires_grad=True
        )
        var y = Tensor[dtype].d2([[2.0], [3.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(
            z.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            )
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](
                Tensor[dtype].d2(
                    [[0.5, 0.5, 0.5], [0.333333, 0.333333, 0.333333]]
                )
            )
        )
        assert_true(
            y.grad().all_close[atol=1e-5](Tensor[dtype].d2([[-3.0], [-2.0]]))
        )


def test_div_gpu_broadcast_3d_1d_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d3(
            [[[4.0, 6.0], [8.0, 9.0]], [[2.0, 12.0], [6.0, 3.0]]],
            requires_grad=True,
        )
        var y = Tensor[dtype].d1([2.0, 3.0], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(
            z.to_cpu().all_close(
                Tensor[dtype].d3(
                    [[[2.0, 2.0], [4.0, 3.0]], [[1.0, 4.0], [3.0, 1.0]]]
                )
            )
        )
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](
                Tensor[dtype].d3(
                    [
                        [[0.5, 0.333333], [0.5, 0.333333]],
                        [[0.5, 0.333333], [0.5, 0.333333]],
                    ]
                )
            )
        )


# ─────────────────────────────────────────────
#  GPU — non-scalar backward
# ─────────────────────────────────────────────


def test_div_gpu_non_scalar_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[4.0, 8.0], [6.0, 12.0]], requires_grad=True)
        var y = Tensor[dtype].d2([[2.0, 4.0], [3.0, 6.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        z.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[0.5, 0.25], [0.333333, 0.166667]])
            )
        )
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[-1.0, -0.5], [-0.666667, -0.333333]])
            )
        )


# ─────────────────────────────────────────────
#  GPU — grad stays on CPU (originating device)
# ─────────────────────────────────────────────


def test_div_gpu_grad_stays_on_cpu() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].d2([[4.0, 8.0], [2.0, 6.0]], requires_grad=True)
        var y = Tensor[dtype].d2([[2.0, 4.0], [1.0, 3.0]], requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        var loss = z.sum()
        loss.backward()
        # x and y live on CPU — grads must be directly accessible without to_cpu()
        assert_true(
            x.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[0.5, 0.25], [1.0, 0.333333]])
            )
        )
        assert_true(
            y.grad().all_close[atol=1e-5](
                Tensor[dtype].d2([[-1.0, -0.5], [-2.0, -0.666667]])
            )
        )


# ─────────────────────────────────────────────
#  GPU — SIMD path (tensors >= 32 elements for f32 SIMD width 16)
# ─────────────────────────────────────────────


def test_div_gpu_tt_large_simd_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].full(Shape(32), 4.0, requires_grad=True)
        var y = Tensor[dtype].full(Shape(32), 2.0, requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(z.to_cpu().all_close(Tensor[dtype].full(Shape(32), 2.0)))
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(32), 0.5))
        )
        assert_true(
            y.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(32), -1.0))
        )


def test_div_gpu_tt_large_simd_scalar_tail_forward_backward() raises:
    """20 elements: 1 SIMD chunk of 16 + 4 scalar tail for f32."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].full(Shape(20), 4.0, requires_grad=True)
        var y = Tensor[dtype].full(Shape(20), 2.0, requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(z.to_cpu().all_close(Tensor[dtype].full(Shape(20), 2.0)))
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(20), 0.5))
        )
        assert_true(
            y.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(20), -1.0))
        )


def test_div_gpu_ts_large_simd_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var x = Tensor[dtype].full(Shape(32), 8.0, requires_grad=True)
        var x_gpu = x.to_gpu()
        var z = x_gpu / Scalar[dtype](2.0)
        assert_true(z.to_cpu().all_close(Tensor[dtype].full(Shape(32), 4.0)))
        var loss = z.sum()
        loss.backward()
        assert_true(x.grad().all_close(Tensor[dtype].full(Shape(32), 0.5)))


def test_div_gpu_st_large_simd_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var y = Tensor[dtype].full(Shape(32), 2.0, requires_grad=True)
        var y_gpu = y.to_gpu()
        var z = Scalar[dtype](8.0) / y_gpu
        assert_true(z.to_cpu().all_close(Tensor[dtype].full(Shape(32), 4.0)))
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(32), -2.0))
        )


def test_div_gpu_st_large_simd_tail_forward_backward() raises:
    """20 elements: 1 SIMD chunk of 16 + 4 scalar tail."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var y = Tensor[dtype].full(Shape(20), 2.0, requires_grad=True)
        var y_gpu = y.to_gpu()
        var z = Scalar[dtype](8.0) / y_gpu
        assert_true(z.to_cpu().all_close(Tensor[dtype].full(Shape(20), 4.0)))
        var loss = z.sum()
        loss.backward()
        assert_true(
            y.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(20), -2.0))
        )


def test_div_gpu_tt_large_float64_simd_forward_backward() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float64
        var x = Tensor[dtype].full(Shape(32), 4.0, requires_grad=True)
        var y = Tensor[dtype].full(Shape(32), 2.0, requires_grad=True)
        var x_gpu = x.to_gpu()
        var y_gpu = y.to_gpu()
        var z = x_gpu / y_gpu
        assert_true(z.to_cpu().all_close(Tensor[dtype].full(Shape(32), 2.0)))
        var loss = z.sum()
        loss.backward()
        assert_true(
            x.grad().all_close[atol=1e-9](Tensor[dtype].full(Shape(32), 0.5))
        )
        assert_true(
            y.grad().all_close[atol=1e-9](Tensor[dtype].full(Shape(32), -1.0))
        )


# ─────────────────────────────────────────────
#  CPU — SIMD path (tensors >= 32 elements for f32 SIMD width 16)
# ─────────────────────────────────────────────


def test_div_cpu_tt_large_simd_forward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(32), 4.0, requires_grad=True)
    var y = Tensor[dtype].full(Shape(32), 2.0, requires_grad=True)
    var z = x / y
    assert_true(z.all_close(Tensor[dtype].full(Shape(32), 2.0)))


def test_div_cpu_tt_large_simd_backward_both() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(32), 4.0, requires_grad=True)
    var y = Tensor[dtype].full(Shape(32), 2.0, requires_grad=True)
    var z = x / y
    var loss = z.sum()
    loss.backward()
    # dx = 1/y = 0.5 everywhere
    assert_true(
        x.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(32), 0.5))
    )
    # dy = -x/y^2 = -4/4 = -1.0 everywhere
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(32), -1.0))
    )


def test_div_cpu_tt_large_simd_scalar_tail_forward_backward() raises:
    """20 elements: 1 SIMD chunk of 16 + 4 scalar tail for f32."""
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(20), 4.0, requires_grad=True)
    var y = Tensor[dtype].full(Shape(20), 2.0, requires_grad=True)
    var z = x / y
    assert_true(z.all_close(Tensor[dtype].full(Shape(20), 2.0)))
    var loss = z.sum()
    loss.backward()
    assert_true(
        x.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(20), 0.5))
    )
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(20), -1.0))
    )


def test_div_cpu_ts_large_simd_forward_backward() raises:
    comptime dtype = DType.float32
    var x = Tensor[dtype].full(Shape(32), 8.0, requires_grad=True)
    var z = x / Scalar[dtype](2.0)
    assert_true(z.all_close(Tensor[dtype].full(Shape(32), 4.0)))
    var loss = z.sum()
    loss.backward()
    assert_true(x.grad().all_close(Tensor[dtype].full(Shape(32), 0.5)))


def test_div_cpu_st_large_simd_forward_backward() raises:
    comptime dtype = DType.float32
    var y = Tensor[dtype].full(Shape(32), 2.0, requires_grad=True)
    var z = Scalar[dtype](8.0) / y
    assert_true(z.all_close(Tensor[dtype].full(Shape(32), 4.0)))
    var loss = z.sum()
    loss.backward()
    # dy = -8/y^2 = -8/4 = -2.0 everywhere
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(32), -2.0))
    )


def test_div_cpu_st_large_simd_tail_forward_backward() raises:
    """20 elements: 1 SIMD chunk of 16 + 4 scalar tail."""
    comptime dtype = DType.float32
    var y = Tensor[dtype].full(Shape(20), 2.0, requires_grad=True)
    var z = Scalar[dtype](8.0) / y
    assert_true(z.all_close(Tensor[dtype].full(Shape(20), 4.0)))
    var loss = z.sum()
    loss.backward()
    assert_true(
        y.grad().all_close[atol=1e-5](Tensor[dtype].full(Shape(20), -2.0))
    )


def test_div_cpu_tt_large_float64_simd_forward_backward() raises:
    comptime dtype = DType.float64
    var x = Tensor[dtype].full(Shape(32), 4.0, requires_grad=True)
    var y = Tensor[dtype].full(Shape(32), 2.0, requires_grad=True)
    var z = x / y
    assert_true(z.all_close(Tensor[dtype].full(Shape(32), 2.0)))
    var loss = z.sum()
    loss.backward()
    assert_true(
        x.grad().all_close[atol=1e-9](Tensor[dtype].full(Shape(32), 0.5))
    )
    assert_true(
        y.grad().all_close[atol=1e-9](Tensor[dtype].full(Shape(32), -1.0))
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
