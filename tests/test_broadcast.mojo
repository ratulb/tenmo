from std.sys import has_accelerator
from tenmo.shapes import Shape
from std.testing import assert_true, assert_false, TestSuite
from tenmo.tensor import Tensor


comptime dtype = DType.float32


# Old tests
def test_gpu_add_broadcast_scalar_tensor_result() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # scalar + scalar → Shape() result, broadcast backward gets Shape() grad
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = a + b  # Shape()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = a_gpu + b_gpu  # Shape()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_scalar_times_matrix_result_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # scalar * matrix → matrix, but scalar backward gets Shape() grad from upstream
    # scalar has Shape(), matrix has (2,3) → broadcasts to (2,3)
    # then .mean() → Shape() incoming to BroadcastBackward
    var a = Tensor[dtype].scalar(2.0, requires_grad=True)
    var b = Tensor[dtype].rand(2, 3, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).mean()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).mean()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_add_scalar_plus_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # scalar + (3,4) → (3,4), sum → Shape()
    var a = Tensor[dtype].scalar(1.0, requires_grad=True)
    var b = Tensor[dtype].rand(3, 4, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a + b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu + b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_sub_scalar_minus_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # scalar - (3,4) → (3,4), mean → Shape()
    var a = Tensor[dtype].scalar(5.0, requires_grad=True)
    var b = Tensor[dtype].rand(3, 4, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a - b).mean()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu - b_gpu).mean()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_scalar_times_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # Shape() * Shape() → Shape() — result AND grad are Shape()
    var a = Tensor[dtype].scalar(3.0, requires_grad=True)
    var b = Tensor[dtype].scalar(4.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = a * b  # Shape()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = a_gpu * b_gpu  # Shape()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_chained_scalar_broadcast() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # scalar * matrix + scalar → matrix, mean → Shape()
    # tests multiple broadcast backward handlers with Shape() incoming grad
    var a = Tensor[dtype].scalar(2.0, requires_grad=True)
    var b = Tensor[dtype].rand(2, 3, requires_grad=True)
    var c = Tensor[dtype].scalar(1.0, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var c_gpu = c.to_gpu()
    var cpu_result = (a * b + c).mean()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    var c_cpu_grad = c.grad().copy()
    a.zero_grad()
    b.zero_grad()
    c.zero_grad()
    var gpu_result = (a_gpu * b_gpu + c_gpu).mean()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))
    assert_true(c.grad().all_close(c_cpu_grad))


def test_gpu_mul_broadcast_incoming_grad_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,4) * (3,4) → (3,4), then mean over all → Shape()
    # grad flowing into MultiplyBroadcastBackward is Shape()
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    var b = Tensor[dtype].rand(3, 4, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).mean()  # Shape()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).mean()  # Shape()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_add_broadcast_incoming_grad_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,3) + (4,3) → (4,3), mean → Shape()
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
    var b = Tensor[dtype].rand(4, 3, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a + b).mean()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu + b_gpu).mean()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_sub_broadcast_incoming_grad_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (3,1) - (3,4) → (3,4), mean → Shape()
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var b = Tensor[dtype].rand(3, 4, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a - b).mean()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu - b_gpu).mean()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_broadcast_3d_incoming_grad_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,1,4) * (2,3,4) → (2,3,4), mean → Shape()
    var a = Tensor[dtype].rand(1, 1, 4, requires_grad=True)
    var b = Tensor[dtype].rand(2, 3, 4, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).mean()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).mean()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_broadcast_scalar_row_times_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,4) * (3,4) → broadcast along axis 0
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
        requires_grad=True,
    )
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_broadcast_col_times_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (3,1) * (3,4) → broadcast along axis 1
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        requires_grad=True,
    )
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_broadcast_1d_times_2d() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (4,) * (3,4) → broadcast 1d across rows
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
        requires_grad=True,
    )
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_broadcast_scalar_times_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,) * (3,4) → full broadcast
    var a = Tensor[dtype].d1([2.0], requires_grad=True)
    var b = Tensor[dtype].rand(3, 4, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_broadcast_3d_batch() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,3,4) * (2,3,4) → broadcast batch dim
    var a = Tensor[dtype].rand(1, 3, 4, requires_grad=True)
    var b = Tensor[dtype].rand(2, 3, 4, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_broadcast_only_lhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # Only a requires grad — only grad_a computed
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor[dtype].d2([[1.0], [1.0]])  # (2,1) broadcast to (2,2)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    a.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))


def test_gpu_mul_broadcast_only_rhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # Only b requires grad — only grad_b computed
    var a = Tensor[dtype].d2([[1.0], [2.0]])  # (2,1) broadcast to (2,3)
    var b = Tensor[dtype].rand(2, 3, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).sum()
    cpu_result.backward()
    var b_cpu_grad = b.grad().copy()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_mul_broadcast_large() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,64) * (32,64) — larger broadcast
    var a = Tensor[dtype].rand(1, 64, requires_grad=True)
    var b = Tensor[dtype].rand(32, 64, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a * b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


# End of old tests


# ═════════════════════════════════════════════════════════════════════════════
# CPU broadcast_to Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_bcast_cpu_1d_to_2d() raises:
    comptime dtype = DType.float32
    # (3,) → (2, 3)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.broadcast_to(Shape(2, 3))
    assert_true(result.shape() == Shape(2, 3))
    assert_true(
        result.all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    )


def test_bcast_cpu_scalar_to_1d() raises:
    comptime dtype = DType.float32
    # (1,) → (4,)
    var a = Tensor[dtype].d1([5.0])
    var result = a.broadcast_to(Shape(4))
    assert_true(result.shape() == Shape(4))
    assert_true(result.all_close(Tensor[dtype].full(Shape(4), 5.0)))


def test_bcast_cpu_1d_to_3d() raises:
    comptime dtype = DType.float32
    # (3,) → (2, 4, 3)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.broadcast_to(Shape(2, 4, 3))
    assert_true(result.shape() == Shape(2, 4, 3))
    assert_true(result.numels() == 24)
    # Every row should be [1, 2, 3]
    for i in range(2):
        for j in range(4):
            assert_true(result[[i, j, 0]] == Scalar[dtype](1.0))
            assert_true(result[[i, j, 1]] == Scalar[dtype](2.0))
            assert_true(result[[i, j, 2]] == Scalar[dtype](3.0))


def test_bcast_cpu_2d_col_to_2d() raises:
    comptime dtype = DType.float32
    # (3, 1) → (3, 4)
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]])
    var result = a.broadcast_to(Shape(3, 4))
    assert_true(result.shape() == Shape(3, 4))
    for j in range(4):
        assert_true(result[[0, j]] == Scalar[dtype](1.0))
        assert_true(result[[1, j]] == Scalar[dtype](2.0))
        assert_true(result[[2, j]] == Scalar[dtype](3.0))


def test_bcast_cpu_2d_row_to_2d() raises:
    comptime dtype = DType.float32
    # (1, 3) → (4, 3)
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]])
    var result = a.broadcast_to(Shape(4, 3))
    assert_true(result.shape() == Shape(4, 3))
    for i in range(4):
        assert_true(result[[i, 0]] == Scalar[dtype](1.0))
        assert_true(result[[i, 1]] == Scalar[dtype](2.0))
        assert_true(result[[i, 2]] == Scalar[dtype](3.0))


def test_bcast_cpu_identity() raises:
    comptime dtype = DType.float32
    # Same shape — no change
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.broadcast_to(Shape(2, 2))
    assert_true(result.shape() == Shape(2, 2))
    assert_true(result.all_close(a))


def test_bcast_cpu_3d_to_3d() raises:
    comptime dtype = DType.float32
    # (1, 2, 1) → (3, 2, 4)
    var a = Tensor[dtype].d3([[[1.0], [2.0]]])
    var result = a.broadcast_to(Shape(3, 2, 4))
    assert_true(result.shape() == Shape(3, 2, 4))
    for i in range(3):
        for k in range(4):
            assert_true(result[[i, 0, k]] == Scalar[dtype](1.0))
            assert_true(result[[i, 1, k]] == Scalar[dtype](2.0))


def test_bcast_cpu_values_correct() raises:
    comptime dtype = DType.float32
    # (2, 1) → (2, 3) — verify each value
    var a = Tensor[dtype].d2([[10.0], [20.0]])
    var result = a.broadcast_to(Shape(2, 3))
    for j in range(3):
        assert_true(result[[0, j]] == Scalar[dtype](10.0))
        assert_true(result[[1, j]] == Scalar[dtype](20.0))


def test_bcast_cpu_no_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=False)
    var result = a.broadcast_to(Shape(3, 2))
    assert_true(not result.requires_grad)


def test_bcast_cpu_requires_grad() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var result = a.broadcast_to(Shape(3, 2))
    assert_true(result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GPU broadcast_to Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_bcast_gpu_1d_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.broadcast_to(Shape(2, 3))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 3))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
            )
        )


def test_bcast_gpu_scalar_to_1d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([5.0]).to_gpu()
        var result = a.broadcast_to(Shape(4))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].full(Shape(4), 5.0))
        )


def test_bcast_gpu_1d_to_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.broadcast_to(Shape(2, 4, 3))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(2, 4, 3))
        assert_true(result.numels() == 24)
        var result_cpu = result.to_cpu()
        for i in range(2):
            for j in range(4):
                assert_true(result_cpu[[i, j, 0]] == Scalar[dtype](1.0))
                assert_true(result_cpu[[i, j, 1]] == Scalar[dtype](2.0))
                assert_true(result_cpu[[i, j, 2]] == Scalar[dtype](3.0))


def test_bcast_gpu_2d_col_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]]).to_gpu()
        var result = a.broadcast_to(Shape(3, 4))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3, 4))
        var result_cpu = result.to_cpu()
        for j in range(4):
            assert_true(result_cpu[[0, j]] == Scalar[dtype](1.0))
            assert_true(result_cpu[[1, j]] == Scalar[dtype](2.0))
            assert_true(result_cpu[[2, j]] == Scalar[dtype](3.0))


def test_bcast_gpu_2d_row_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]]).to_gpu()
        var result = a.broadcast_to(Shape(4, 3))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4, 3))
        var result_cpu = result.to_cpu()
        for i in range(4):
            assert_true(result_cpu[[i, 0]] == Scalar[dtype](1.0))
            assert_true(result_cpu[[i, 1]] == Scalar[dtype](2.0))
            assert_true(result_cpu[[i, 2]] == Scalar[dtype](3.0))


def test_bcast_gpu_identity() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var result = a.broadcast_to(Shape(2, 2))
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
            )
        )


def test_bcast_gpu_3d_to_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0], [2.0]]]).to_gpu()
        var result = a.broadcast_to(Shape(3, 2, 4))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(3, 2, 4))
        var result_cpu = result.to_cpu()
        for i in range(3):
            for k in range(4):
                assert_true(result_cpu[[i, 0, k]] == Scalar[dtype](1.0))
                assert_true(result_cpu[[i, 1, k]] == Scalar[dtype](2.0))


# ═════════════════════════════════════════════════════════════════════════════
# CPU/GPU Parity Tests
# ═════════════════════════════════════════════════════════════════════════════


def test_bcast_parity_1d_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.broadcast_to(Shape(2, 3)).all_close(
                a_gpu.broadcast_to(Shape(2, 3)).to_cpu()
            )
        )


def test_bcast_parity_2d_col() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0], [2.0], [3.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.broadcast_to(Shape(3, 4)).all_close(
                a_gpu.broadcast_to(Shape(3, 4)).to_cpu()
            )
        )


def test_bcast_parity_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d3([[[1.0], [2.0]]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.broadcast_to(Shape(3, 2, 4)).all_close(
                a_gpu.broadcast_to(Shape(3, 2, 4)).to_cpu()
            )
        )


# =============================================================================
# Exhaustive tests for Tensor.broadcast_to()
# Prefix: test_bcast_ on all test names to avoid collision with existing tests
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — CPU · Forward correctness · no grad
# ─────────────────────────────────────────────────────────────────────────────


def test_bcast_cpu_fwd_scalar_to_1d() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([5.0])
    var result = a.broadcast_to(Shape(4))
    assert_true(result.all_close(Tensor[dtype].d1([5.0, 5.0, 5.0, 5.0])))


def test_bcast_cpu_fwd_1d_to_2d() raises:
    comptime dtype = DType.float32
    # shape (3,) → (2,3)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.broadcast_to(Shape(2, 3))
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [
                    [1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0],
                ]
            )
        )
    )


def test_bcast_cpu_fwd_2d_row_to_2d() raises:
    comptime dtype = DType.float32
    # shape (1,3) → (4,3)
    var a = Tensor[dtype].d2([[10.0, 20.0, 30.0]])
    var result = a.broadcast_to(Shape(4, 3))
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [
                    [10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0],
                    [10.0, 20.0, 30.0],
                ]
            )
        )
    )


def test_bcast_cpu_fwd_2d_col_to_2d() raises:
    comptime dtype = DType.float32
    # shape (3,1) → (3,4)
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]])
    var result = a.broadcast_to(Shape(3, 4))
    assert_true(
        result.all_close(
            Tensor[dtype].d2(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0, 2.0],
                    [3.0, 3.0, 3.0, 3.0],
                ]
            )
        )
    )


def test_bcast_cpu_fwd_1d_to_3d() raises:
    comptime dtype = DType.float32
    # shape (3,) → (2,4,3)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.broadcast_to(Shape(2, 4, 3))
    # Every row of every slab must equal [1,2,3]
    for s in range(2):
        for r in range(4):
            var tmp_slice = result.slice(start=s, end=s + 1, axis=0)
            var sliced = tmp_slice.slice(start=r, end=r + 1, axis=1)
            assert_true(
                sliced.squeeze().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0]))
            )


def test_bcast_cpu_fwd_2d_to_3d() raises:
    comptime dtype = DType.float32
    # shape (1,3) → (2,1,3) → (2,4,3)  but broadcast_to goes directly
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]])
    var result = a.broadcast_to(Shape(5, 3))
    assert_true(result.shape() == Shape(5, 3))
    var sliced = result.slice(start=3, end=4, axis=0)
    assert_true(sliced.squeeze().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


def test_bcast_cpu_fwd_same_shape_noop() raises:
    comptime dtype = DType.float32
    # Broadcasting to the same shape is a valid no-op
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.broadcast_to(Shape(2, 2))
    assert_true(result.all_close(a))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · Backward · 1-D cases
# ─────────────────────────────────────────────────────────────────────────────


def test_bcast_cpu_bwd_1d_to_2d_sum_rows() raises:
    comptime dtype = DType.float32
    # a.shape=(3,) broadcast to (2,3); grad must sum over axis 0
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.broadcast_to(Shape(2, 3))
    var loss = b.sum()
    loss.backward()
    # Each element of a contributes to 2 rows → grad = [2, 2, 2]
    assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


def test_bcast_cpu_bwd_1d_to_3d() raises:
    comptime dtype = DType.float32
    # a.shape=(3,) broadcast to (2,4,3); grad sums over axes 0 and 1
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.broadcast_to(Shape(2, 4, 3))
    var loss = b.sum()
    loss.backward()
    # Each element contributes to 2*4=8 positions → grad = [8, 8, 8]
    assert_true(a.grad().all_close(Tensor[dtype].d1([8.0, 8.0, 8.0])))


def test_bcast_cpu_bwd_scalar_to_1d() raises:
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([3.0], requires_grad=True)
    var b = a.broadcast_to(Shape(5))
    var loss = b.sum()
    loss.backward()
    # Single element broadcasts to 5 positions → grad = 5
    assert_true(a.grad().all_close(Tensor[dtype].d1([5.0])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · Backward · 2-D cases
# ─────────────────────────────────────────────────────────────────────────────


def test_bcast_cpu_bwd_2d_row_to_2d() raises:
    comptime dtype = DType.float32
    # a.shape=(1,3) broadcast to (4,3); grad sums over axis 0
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
    var b = a.broadcast_to(Shape(4, 3))
    var loss = b.sum()
    loss.backward()
    # Each element used 4 times → grad = [[4, 4, 4]]
    assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0]])))


def test_bcast_cpu_bwd_2d_col_to_2d() raises:
    comptime dtype = DType.float32
    # a.shape=(3,1) broadcast to (3,4); grad sums over axis 1
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var b = a.broadcast_to(Shape(3, 4))
    var loss = b.sum()
    loss.backward()
    # Each element used 4 times → grad = [[4],[4],[4]]
    assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0], [4.0], [4.0]])))


def test_bcast_cpu_bwd_2d_to_3d() raises:
    comptime dtype = DType.float32
    # a.shape=(1,3) broadcast to (2,4,3)
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
    var b = a.broadcast_to(Shape(2, 4, 3))
    var loss = b.sum()
    loss.backward()
    # Each element used 2*4=8 times → grad = [[8,8,8]]
    assert_true(a.grad().all_close(Tensor[dtype].d2([[8.0, 8.0, 8.0]])))


def test_bcast_cpu_bwd_same_shape_noop() raises:
    comptime dtype = DType.float32
    # No axes broadcast — grad passes through unchanged
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = a.broadcast_to(Shape(2, 2))
    var loss = b.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor.ones_like(a)))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · Backward · chained ops after broadcast_to
# ─────────────────────────────────────────────────────────────────────────────


def test_bcast_cpu_bwd_chained_add() raises:
    comptime dtype = DType.float32
    # broadcast then add then sum — verify chain rule
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.broadcast_to(Shape(3, 3))
    var c = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    )
    var d = b + c
    var loss = d.sum()
    loss.backward()
    # grad of sum w.r.t b is all-ones (3,3)
    # grad of b w.r.t a sums over axis 0 → [3,3,3]
    assert_true(a.grad().all_close(Tensor[dtype].d1([3.0, 3.0, 3.0])))


def test_bcast_cpu_bwd_chained_multiply() raises:
    comptime dtype = DType.float32
    # a.shape=(1,3) broadcast to (2,3) then element-wise multiply
    var a = Tensor[dtype].d2([[2.0, 3.0, 4.0]], requires_grad=True)
    var b = a.broadcast_to(Shape(2, 3))
    var scale = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var c = b * scale
    var loss = c.sum()
    loss.backward()
    # upstream grad = scale, sum over axis 0 per element:
    # col0: 1+4=5, col1: 2+5=7, col2: 3+6=9
    assert_true(a.grad().all_close(Tensor[dtype].d2([[5.0, 7.0, 9.0]])))


def test_bcast_cpu_bwd_chained_double_broadcast() raises:
    comptime dtype = DType.float32
    # Two separate tensors broadcast then added
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)  # (1,3)
    var b = Tensor[dtype].d2(
        [[10.0], [20.0], [30.0]], requires_grad=True
    )  # (3,1)
    var ab = a.broadcast_to(Shape(3, 3)) + b.broadcast_to(Shape(3, 3))
    var loss = ab.sum()
    loss.backward()
    # a: each col used 3 times → [[3,3,3]]
    assert_true(a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0, 3.0]])))
    # b: each row used 3 times → [[3],[3],[3]]
    assert_true(b.grad().all_close(Tensor[dtype].d2([[3.0], [3.0], [3.0]])))


def test_bcast_cpu_bwd_3d_partial_broadcast() raises:
    comptime dtype = DType.float32
    # a.shape=(1,1,4) broadcast to (3,2,4)
    var a = Tensor[dtype].d3([[[1.0, 2.0, 3.0, 4.0]]], requires_grad=True)
    var b = a.broadcast_to(Shape(3, 2, 4))
    var loss = b.sum()
    loss.backward()
    # Each element used 3*2=6 times → grad = [[[6,6,6,6]]]
    assert_true(a.grad().all_close(Tensor[dtype].d3([[[6.0, 6.0, 6.0, 6.0]]])))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — GPU · Forward correctness
# ─────────────────────────────────────────────────────────────────────────────


def test_bcast_gpu_fwd_1d_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(2, 3))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [1.0, 2.0, 3.0],
                        [1.0, 2.0, 3.0],
                    ]
                )
            )
        )


def test_bcast_gpu_fwd_2d_row_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[10.0, 20.0, 30.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(4, 3))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [10.0, 20.0, 30.0],
                        [10.0, 20.0, 30.0],
                        [10.0, 20.0, 30.0],
                        [10.0, 20.0, 30.0],
                    ]
                )
            )
        )


def test_bcast_gpu_fwd_2d_col_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(3, 4))
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0, 3.0],
                    ]
                )
            )
        )


def test_bcast_gpu_fwd_1d_to_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(2, 4, 3))
        assert_true(result.shape() == Shape(2, 4, 3))
        # Spot check: every position along last axis should match a
        var cpu = result.to_cpu()
        var cpu_sliced = cpu.slice(start=0, end=1, axis=0)
        var sliced = cpu_sliced.slice(start=2, end=3, axis=1)
        assert_true(
            sliced.squeeze().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0]))
        )


def test_bcast_gpu_fwd_same_shape_noop() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(2, 2))
        assert_true(result.to_cpu().all_close(a))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GPU · Backward · grad flows correctly
# ─────────────────────────────────────────────────────────────────────────────


def test_bcast_gpu_bwd_1d_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(2, 3))
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


def test_bcast_gpu_bwd_2d_row_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(4, 3))
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0]])))


def test_bcast_gpu_bwd_2d_col_to_2d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(3, 4))
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0], [4.0], [4.0]])))


def test_bcast_gpu_bwd_1d_to_3d() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(2, 4, 3))
        var loss = b.sum()
        loss.backward()
        # 2*4=8 uses per element
        assert_true(a.grad().all_close(Tensor[dtype].d1([8.0, 8.0, 8.0])))


def test_bcast_gpu_bwd_same_shape_noop() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(2, 2))
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


def test_bcast_gpu_bwd_chained_multiply() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[2.0, 3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(2, 3))
        var scale = (
            Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        )
        var c = b * scale
        var loss = c.sum()
        loss.backward()
        # col0: 1+4=5, col1: 2+5=7, col2: 3+6=9
        assert_true(a.grad().all_close(Tensor[dtype].d2([[5.0, 7.0, 9.0]])))


def test_bcast_gpu_bwd_3d_partial_broadcast() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0, 3.0, 4.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(3, 2, 4))
        var loss = b.sum()
        loss.backward()
        # 3*2=6 uses per element
        assert_true(
            a.grad().all_close(Tensor[dtype].d3([[[6.0, 6.0, 6.0, 6.0]]]))
        )


def test_bcast_gpu_bwd_chained_add() raises:
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(3, 3))
        var c = (
            Tensor[dtype]
            .d2([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
            .to_gpu()
        )
        var d = b + c
        var loss = d.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([3.0, 3.0, 3.0])))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — Scalar broadcast tests
# ─────────────────────────────────────────────────────────────────────────────


def _scalar(v: Float32, rg: Bool = False) -> Tensor[dtype]:
    return Tensor[dtype].scalar(v, requires_grad=rg)


def _scalar1(v: Float32, rg: Bool = False) -> Tensor[dtype]:
    return Tensor[dtype].d1([v], requires_grad=rg)


def _mat(rows: List[List[Float32]], rg: Bool = False) -> Tensor[dtype]:
    return Tensor[dtype].d2(rows, requires_grad=rg)


# ─────────────────────────────────────────────────────────────────────────────
# A — Shape equivalence: () and (1,) behave identically in all 4 ops
# ─────────────────────────────────────────────────────────────────────────────


def test_A1_shape_eq_add() raises:
    var a = _scalar(5.0)
    var b = _scalar1(5.0)
    var c = _scalar(7.0)
    var d = _scalar1(7.0)

    assert_true((a + c).all_close(Tensor[dtype].scalar(12.0)))
    assert_true((b + c).all_close(Tensor[dtype].d1([12.0])))
    assert_true((a + d).all_close(Tensor[dtype].d1([12.0])))
    assert_true((b + d).all_close(Tensor[dtype].d1([12.0])))
    assert_true((b + d).all_close(c + b))
    print("  A1 add: ✓")


def test_A2_shape_eq_sub() raises:
    var a = _scalar(5.0)
    var b = _scalar1(5.0)
    var c = _scalar(7.0)
    var d = _scalar1(7.0)

    assert_true((a - c).all_close(Tensor[dtype].scalar(-2.0)))
    assert_true((b - c).all_close(Tensor[dtype].d1([-2.0])))
    assert_true((a - d).all_close(Tensor[dtype].d1([-2.0])))
    assert_true((b - d).all_close(Tensor[dtype].d1([-2.0])))
    print("  A2 sub: ✓")


def test_A3_shape_eq_mul() raises:
    var a = _scalar(5.0)
    var b = _scalar1(5.0)
    var c = _scalar(7.0)
    var d = _scalar1(7.0)

    assert_true((a * c).all_close(Tensor[dtype].scalar(35.0)))
    assert_true((b * c).all_close(Tensor[dtype].d1([35.0])))
    assert_true((a * d).all_close(Tensor[dtype].d1([35.0])))
    assert_true((b * d).all_close(Tensor[dtype].d1([35.0])))
    assert_true((b * d).all_close(c * b))
    print("  A3 mul: ✓")


def test_A4_shape_eq_div() raises:
    var a = _scalar(10.0)
    var b = _scalar1(10.0)
    var c = _scalar(2.0)
    var d = _scalar1(2.0)

    assert_true((a / c).all_close(Tensor[dtype].scalar(5.0)))
    assert_true((b / c).all_close(Tensor[dtype].d1([5.0])))
    assert_true((a / d).all_close(Tensor[dtype].d1([5.0])))
    assert_true((b / d).all_close(Tensor[dtype].d1([5.0])))
    print("  A4 div: ✓")


def test_A5_scalar_vs_vector() raises:
    var s_ = _scalar(2.0)
    var s1 = _scalar1(2.0)
    var v = Tensor[dtype].d1([1.0, 2.0, 3.0])

    assert_true((s_ * v).all_close(Tensor[dtype].d1([2.0, 4.0, 6.0])))
    assert_true((s1 * v).all_close(Tensor[dtype].d1([2.0, 4.0, 6.0])))
    assert_true((v * s_).all_close(Tensor[dtype].d1([2.0, 4.0, 6.0])))
    assert_true((v * s1).all_close(Tensor[dtype].d1([2.0, 4.0, 6.0])))
    print("  A5 vec: ✓")


def test_A6_not_scalar() raises:
    var m = Tensor[dtype].full(Shape(1, 1), 5.0)
    var s_ = _scalar(7.0)
    var r = m * s_
    assert_true(r.shape() == Shape(1, 1))
    assert_true(r.all_close(Tensor[dtype].full(Shape(1, 1), 35.0)))
    print("  A6 (1,1) not scalar: ✓")


# ─────────────────────────────────────────────────────────────────────────────
# B — Op-code flip: Subtract→ReverseSubtract, Divide→ReverseDivide
# ─────────────────────────────────────────────────────────────────────────────


def test_B1_flip_sub_scalar_left() raises:
    assert_true(
        (_scalar(5.0) - Tensor[dtype].d1([10.0, 20.0])).all_close(
            Tensor[dtype].d1([-5.0, -15.0])
        )
    )
    print("  B1 flip sub ()-vec: ✓")


def test_B2_flip_sub_scalar1_left() raises:
    assert_true(
        (_scalar1(5.0) - Tensor[dtype].d1([10.0, 20.0])).all_close(
            Tensor[dtype].d1([-5.0, -15.0])
        )
    )
    print("  B2 flip sub (1,)-vec: ✓")


def test_B3_no_flip_sub_right() raises:
    assert_true(
        (Tensor[dtype].d1([10.0, 20.0]) - _scalar(5.0)).all_close(
            Tensor[dtype].d1([5.0, 15.0])
        )
    )
    print("  B3 no-flip sub: ✓")


def test_B4_no_flip_sub_scalar1_right() raises:
    assert_true(
        (Tensor[dtype].d1([10.0, 20.0]) - _scalar1(5.0)).all_close(
            Tensor[dtype].d1([5.0, 15.0])
        )
    )
    print("  B4 no-flip sub scalar1: ✓")


def test_B5_flip_div_scalar_left() raises:
    assert_true(
        (_scalar(10.0) / Tensor[dtype].d1([2.0, 4.0])).all_close(
            Tensor[dtype].d1([5.0, 2.5])
        )
    )
    print("  B5 flip div ()-vec: ✓")


def test_B6_flip_div_scalar1_left() raises:
    assert_true(
        (_scalar1(10.0) / Tensor[dtype].d1([2.0, 4.0])).all_close(
            Tensor[dtype].d1([5.0, 2.5])
        )
    )
    print("  B6 flip div (1,)-vec: ✓")


def test_B7_no_flip_div_right() raises:
    assert_true(
        (Tensor[dtype].d1([10.0, 20.0]) / _scalar(2.0)).all_close(
            Tensor[dtype].d1([5.0, 10.0])
        )
    )
    print("  B7 no-flip div: ✓")


def test_B8_no_flip_div_scalar1_right() raises:
    assert_true(
        (Tensor[dtype].d1([10.0, 20.0]) / _scalar1(2.0)).all_close(
            Tensor[dtype].d1([5.0, 10.0])
        )
    )
    print("  B8 no-flip div scalar1: ✓")


def test_B9_commutative_same() raises:
    var s_ = _scalar(3.0)
    var s1 = _scalar1(3.0)
    var v = Tensor[dtype].d1([1.0, 2.0, 4.0])

    assert_true((s_ + v).all_close(s1 + v))
    assert_true((v + s_).all_close(v + s1))
    assert_true((s_ * v).all_close(s1 * v))
    assert_true((v * s_).all_close(v * s1))
    assert_true((s_ * v).all_close(s1 * v))
    assert_true((v * s_).all_close(v * s1))
    print("  B9 commutative: ✓")


# ─────────────────────────────────────────────────────────────────────────────
# C — Non-contiguous path (index_iterator + item order)
# ─────────────────────────────────────────────────────────────────────────────
# Transposing a (3,2) tensor gives a (2,3) view with non-default strides.
# This forces broadcast_scalar_buffer to the index_iterator path (non-SIMD).


def test_C1_noncontig_scalar_left_add() raises:
    var A = _mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var T = A.transpose()
    var s = _scalar(10.0)
    var R = s + T
    var expected = _mat([[11.0, 13.0, 15.0], [12.0, 14.0, 16.0]])
    assert_true(R.all_close(expected))
    print("  C1 noncontig ()+transpose: ✓")


def test_C2_noncontig_scalar1_left_mul() raises:
    var A = _mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var T = A.transpose()
    var s1 = _scalar1(3.0)
    var R = s1 * T
    var expected = _mat([[3.0, 9.0, 15.0], [6.0, 12.0, 18.0]])
    assert_true(R.all_close(expected))
    print("  C2 noncontig (1,)*transpose: ✓")


def test_C3_noncontig_scalar_right_sub() raises:
    var A = _mat([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    var T = A.transpose()
    var s = _scalar(5.0)
    var R = T - s
    var expected = _mat([[5.0, 25.0, 45.0], [15.0, 35.0, 55.0]])
    assert_true(R.all_close(expected))
    print("  C3 noncontig transpose-(): ✓")


def test_C4_noncontig_scalar1_right_div() raises:
    var A = _mat([[2.0, 4.0], [8.0, 10.0], [14.0, 16.0]])
    var T = A.transpose()
    var s1 = _scalar1(2.0)
    var R = T / s1
    var expected = _mat([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0]])
    assert_true(R.all_close(expected))
    print("  C4 noncontig transpose/(1,): ✓")


def test_C5_noncontig_scalar_left_sub_flip() raises:
    """() - transpose: scalar left SUBTRACT → ReverseSubtract, index_iterator path.
    """
    var A = _mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    var T = A.transpose()
    var s = _scalar(100.0)
    var R = s - T
    var expected = _mat([[99.0, 97.0, 95.0], [98.0, 96.0, 94.0]])
    assert_true(R.all_close(expected))
    print("  C5 noncontig ()-transpose flip: ✓")


# ─────────────────────────────────────────────────────────────────────────────
# D — Backward pass (gradient through broadcast)
# ─────────────────────────────────────────────────────────────────────────────


def test_D1_backward_scalar_left() raises:
    var a = _scalar(2.0, rg=True)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var c = a * b
    c.backward()
    assert_true(a.grad().all_close(Tensor[dtype].scalar(6.0)))
    var gb = b.grad().detach()
    assert_true(gb.all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))
    print("  D1 backward scalar-left: ✓")


def test_D2_backward_scalar_right() raises:
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = _scalar(2.0, rg=True)
    var c = a * b
    c.backward()
    var ga = a.grad().detach()
    assert_true(ga.all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))
    assert_true(b.grad().all_close(Tensor[dtype].scalar(6.0)))
    print("  D2 backward scalar-right: ✓")


def test_D3_backward_sub_scalar_left() raises:
    var a = _scalar(10.0, rg=True)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var c = a - b
    c.backward()
    assert_true(a.grad().all_close(Tensor[dtype].scalar(3.0)))
    var gb = b.grad().detach()
    assert_true(gb.all_close(Tensor[dtype].d1([-1.0, -1.0, -1.0])))
    print("  D3 backward sub scalar-left: ✓")


def test_D4_backward_div_scalar_left() raises:
    var a = _scalar(12.0, rg=True)
    var b = Tensor[dtype].d1([2.0, 3.0, 4.0], requires_grad=True)
    var c = a / b
    c.backward()
    var exp_ga = (
        Float32(1.0) / Float32(2.0)
        + Float32(1.0) / Float32(3.0)
        + Float32(1.0) / Float32(4.0)
    )
    assert_true(a.grad().all_close(Tensor[dtype].scalar(exp_ga)))
    var gb = b.grad().detach()
    assert_true(
        gb.all_close(
            Tensor[dtype].d1(
                [
                    Float32(-12.0 / 4.0),
                    Float32(-12.0 / 9.0),
                    Float32(-12.0 / 16.0),
                ]
            ),
        )
    )
    print("  D4 backward div scalar-left: ✓")


def test_D5_backward_scalar1_left() raises:
    var a = _scalar1(2.0, rg=True)
    var b = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var c = a * b
    c.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([6.0])))
    var gb = b.grad().detach()
    assert_true(gb.all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))
    print("  D5 backward scalar1-left: ✓")


def test_D6_backward_through_reshape() raises:
    var a = _scalar(42.0, rg=True)
    var b = a.reshape(Shape.of(1))
    var c = _scalar(2.0, rg=True)
    var d = b * c
    d.backward()
    assert_true(a.grad().all_close(Tensor[dtype].scalar(2.0)))
    print("  D6 backward through reshape: ✓")


def test_D7_backward_2d_broadcast() raises:
    var a = _scalar(5.0, rg=True)
    var b = _mat([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], rg=True)
    var c = a * b
    c.backward()
    assert_true(a.grad().all_close(Tensor[dtype].scalar(21.0)))
    var gb = b.grad().detach()
    assert_true(gb.all_close(Tensor[dtype].full(Shape(2, 3), 5.0)))
    print("  D7 backward 2d broadcast: ✓")


def test_D8_backward_scalar1_2d() raises:
    var a = _scalar1(5.0, rg=True)
    var b = _mat([[1.0, 2.0], [3.0, 4.0]], rg=True)
    var c = a + b
    c.backward()
    assert_true(a.grad().all_close(Tensor[dtype].d1([4.0])))
    var gb = b.grad().detach()
    assert_true(gb.all_close(Tensor[dtype].full(Shape(2, 2), 1.0)))
    print("  D8 backward scalar1+2d: ✓")


def test_D9_backward_noncontig() raises:
    var a = _scalar(2.0, rg=True)
    var A = _mat([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], rg=True)
    var T = A.transpose()
    var c = a * T
    var loss = c.sum()
    loss.backward()
    assert_true(a.grad().all_close(Tensor[dtype].scalar(21.0)))
    var ga = A.grad().detach()
    assert_true(ga.all_close(Tensor[dtype].full(Shape(3, 2), 2.0)))
    print("  D9 backward noncontig: ✓")


# ─────────────────────────────────────────────────────────────────────────────
# E — Boundary cases
# ─────────────────────────────────────────────────────────────────────────────


def test_E1_same_shape_scalar1_ok() raises:
    var a = _scalar1(5.0)
    var b = _scalar1(7.0)
    var r = a + b
    assert_true(r.all_close(Tensor[dtype].d1([12.0])))
    print("  E1 (1,)+(1,) OK: ✓")


# =============================================================================
# Main
# =============================================================================


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll [broadcast_to] tests passed!")
