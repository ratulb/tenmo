from std.sys import has_accelerator
from tenmo.shapes import Shape
from std.testing import assert_true, assert_false, TestSuite
from tenmo.tensor import Tensor



comptime dtype = DType.float32


# Old tests
fn test_gpu_add_broadcast_scalar_tensor_result() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_add_broadcast_scalar_tensor_result")
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
    print("passed")


fn test_gpu_mul_scalar_times_matrix_result_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_scalar_times_matrix_result_scalar")
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
    print("passed")


fn test_gpu_add_scalar_plus_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_add_scalar_plus_matrix")
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
    print("passed")


fn test_gpu_sub_scalar_minus_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_sub_scalar_minus_matrix")
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
    print("passed")


fn test_gpu_mul_scalar_times_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_scalar_times_scalar")
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
    print("passed")


fn test_gpu_chained_scalar_broadcast() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_chained_scalar_broadcast")
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
    print("passed")


fn test_gpu_mul_broadcast_incoming_grad_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_incoming_grad_scalar")
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
    print("passed")


fn test_gpu_add_broadcast_incoming_grad_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_add_broadcast_incoming_grad_scalar")
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
    print("passed")


fn test_gpu_sub_broadcast_incoming_grad_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_sub_broadcast_incoming_grad_scalar")
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
    print("passed")


fn test_gpu_mul_broadcast_3d_incoming_grad_scalar() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_3d_incoming_grad_scalar")
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
    print("passed")


fn test_gpu_mul_broadcast_scalar_row_times_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_scalar_row_times_matrix")
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
    print("passed")


fn test_gpu_mul_broadcast_col_times_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_col_times_matrix")
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
    print("passed")


fn test_gpu_mul_broadcast_1d_times_2d() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_1d_times_2d")
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
    print("passed")


fn test_gpu_mul_broadcast_scalar_times_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_scalar_times_matrix")
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
    print("passed")


fn test_gpu_mul_broadcast_3d_batch() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_3d_batch")
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
    print("passed")


fn test_gpu_mul_broadcast_only_lhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_only_lhs_requires_grad")
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
    print("passed")


fn test_gpu_mul_broadcast_only_rhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_only_rhs_requires_grad")
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
    print("passed")


fn test_gpu_mul_broadcast_large() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return


    print("test_gpu_mul_broadcast_large")
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
    print("passed")


# End of old tests


# ═════════════════════════════════════════════════════════════════════════════
# CPU broadcast_to Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_bcast_cpu_1d_to_2d() raises:
    print("test_bcast_cpu_1d_to_2d")
    comptime dtype = DType.float32
    # (3,) → (2, 3)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.broadcast_to(Shape(2, 3))
    assert_true(result.shape() == Shape(2, 3))
    assert_true(
        result.all_close(Tensor[dtype].d2([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]))
    )


fn test_bcast_cpu_scalar_to_1d() raises:
    print("test_bcast_cpu_scalar_to_1d")
    comptime dtype = DType.float32
    # (1,) → (4,)
    var a = Tensor[dtype].d1([5.0])
    var result = a.broadcast_to(Shape(4))
    assert_true(result.shape() == Shape(4))
    assert_true(result.all_close(Tensor[dtype].full(Shape(4), 5.0)))


fn test_bcast_cpu_1d_to_3d() raises:
    print("test_bcast_cpu_1d_to_3d")
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


fn test_bcast_cpu_2d_col_to_2d() raises:
    print("test_bcast_cpu_2d_col_to_2d")
    comptime dtype = DType.float32
    # (3, 1) → (3, 4)
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]])
    var result = a.broadcast_to(Shape(3, 4))
    assert_true(result.shape() == Shape(3, 4))
    for j in range(4):
        assert_true(result[[0, j]] == Scalar[dtype](1.0))
        assert_true(result[[1, j]] == Scalar[dtype](2.0))
        assert_true(result[[2, j]] == Scalar[dtype](3.0))


fn test_bcast_cpu_2d_row_to_2d() raises:
    print("test_bcast_cpu_2d_row_to_2d")
    comptime dtype = DType.float32
    # (1, 3) → (4, 3)
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]])
    var result = a.broadcast_to(Shape(4, 3))
    assert_true(result.shape() == Shape(4, 3))
    for i in range(4):
        assert_true(result[[i, 0]] == Scalar[dtype](1.0))
        assert_true(result[[i, 1]] == Scalar[dtype](2.0))
        assert_true(result[[i, 2]] == Scalar[dtype](3.0))


fn test_bcast_cpu_identity() raises:
    print("test_bcast_cpu_identity")
    comptime dtype = DType.float32
    # Same shape — no change
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.broadcast_to(Shape(2, 2))
    assert_true(result.shape() == Shape(2, 2))
    assert_true(result.all_close(a))


fn test_bcast_cpu_3d_to_3d() raises:
    print("test_bcast_cpu_3d_to_3d")
    comptime dtype = DType.float32
    # (1, 2, 1) → (3, 2, 4)
    var a = Tensor[dtype].d3([[[1.0], [2.0]]])
    var result = a.broadcast_to(Shape(3, 2, 4))
    assert_true(result.shape() == Shape(3, 2, 4))
    for i in range(3):
        for k in range(4):
            assert_true(result[[i, 0, k]] == Scalar[dtype](1.0))
            assert_true(result[[i, 1, k]] == Scalar[dtype](2.0))


fn test_bcast_cpu_values_correct() raises:
    print("test_bcast_cpu_values_correct")
    comptime dtype = DType.float32
    # (2, 1) → (2, 3) — verify each value
    var a = Tensor[dtype].d2([[10.0], [20.0]])
    var result = a.broadcast_to(Shape(2, 3))
    for j in range(3):
        assert_true(result[[0, j]] == Scalar[dtype](10.0))
        assert_true(result[[1, j]] == Scalar[dtype](20.0))


fn test_bcast_cpu_no_grad() raises:
    print("test_bcast_cpu_no_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=False)
    var result = a.broadcast_to(Shape(3, 2))
    assert_true(not result.requires_grad)


fn test_bcast_cpu_requires_grad() raises:
    print("test_bcast_cpu_requires_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var result = a.broadcast_to(Shape(3, 2))
    assert_true(result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GPU broadcast_to Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


fn test_bcast_gpu_1d_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_1d_to_2d")
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


fn test_bcast_gpu_scalar_to_1d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_scalar_to_1d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([5.0]).to_gpu()
        var result = a.broadcast_to(Shape(4))
        assert_true(result.is_on_gpu())
        assert_true(result.shape() == Shape(4))
        assert_true(
            result.to_cpu().all_close(Tensor[dtype].full(Shape(4), 5.0))
        )


fn test_bcast_gpu_1d_to_3d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_1d_to_3d")
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


fn test_bcast_gpu_2d_col_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_2d_col_to_2d")
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


fn test_bcast_gpu_2d_row_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_2d_row_to_2d")
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


fn test_bcast_gpu_identity() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_identity")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]]).to_gpu()
        var result = a.broadcast_to(Shape(2, 2))
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close(
                Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
            )
        )


fn test_bcast_gpu_3d_to_3d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_3d_to_3d")
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


fn test_bcast_parity_1d_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_parity_1d_to_2d")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.broadcast_to(Shape(2, 3)).all_close(
                a_gpu.broadcast_to(Shape(2, 3)).to_cpu()
            )
        )


fn test_bcast_parity_2d_col() raises:
    comptime if has_accelerator():
        print("test_bcast_parity_2d_col")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0], [2.0], [3.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.broadcast_to(Shape(3, 4)).all_close(
                a_gpu.broadcast_to(Shape(3, 4)).to_cpu()
            )
        )


fn test_bcast_parity_3d() raises:
    comptime if has_accelerator():
        print("test_bcast_parity_3d")
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

fn test_bcast_cpu_fwd_scalar_to_1d() raises:
    print("test_bcast_cpu_fwd_scalar_to_1d")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([5.0])
    var result = a.broadcast_to(Shape(4))
    assert_true(result.all_close(Tensor[dtype].d1([5.0, 5.0, 5.0, 5.0])))


fn test_bcast_cpu_fwd_1d_to_2d() raises:
    print("test_bcast_cpu_fwd_1d_to_2d")
    comptime dtype = DType.float32
    # shape (3,) → (2,3)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.broadcast_to(Shape(2, 3))
    assert_true(result.all_close(Tensor[dtype].d2([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    ])))


fn test_bcast_cpu_fwd_2d_row_to_2d() raises:
    print("test_bcast_cpu_fwd_2d_row_to_2d")
    comptime dtype = DType.float32
    # shape (1,3) → (4,3)
    var a = Tensor[dtype].d2([[10.0, 20.0, 30.0]])
    var result = a.broadcast_to(Shape(4, 3))
    assert_true(result.all_close(Tensor[dtype].d2([
        [10.0, 20.0, 30.0],
        [10.0, 20.0, 30.0],
        [10.0, 20.0, 30.0],
        [10.0, 20.0, 30.0],
    ])))


fn test_bcast_cpu_fwd_2d_col_to_2d() raises:
    print("test_bcast_cpu_fwd_2d_col_to_2d")
    comptime dtype = DType.float32
    # shape (3,1) → (3,4)
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]])
    var result = a.broadcast_to(Shape(3, 4))
    assert_true(result.all_close(Tensor[dtype].d2([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0, 3.0],
    ])))


fn test_bcast_cpu_fwd_1d_to_3d() raises:
    print("test_bcast_cpu_fwd_1d_to_3d")
    comptime dtype = DType.float32
    # shape (3,) → (2,4,3)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.broadcast_to(Shape(2, 4, 3))
    # Every row of every slab must equal [1,2,3]
    for s in range(2):
        for r in range(4):
            var tmp_slice = result.slice(start=s, end=s+1, axis=0)
            var sliced = tmp_slice.slice(start=r, end=r+1, axis=1)
            assert_true(
                      sliced.squeeze()
                      .all_close(Tensor[dtype].d1([1.0, 2.0, 3.0]))
            )


fn test_bcast_cpu_fwd_2d_to_3d() raises:
    print("test_bcast_cpu_fwd_2d_to_3d")
    comptime dtype = DType.float32
    # shape (1,3) → (2,1,3) → (2,4,3)  but broadcast_to goes directly
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]])
    var result = a.broadcast_to(Shape(5, 3))
    assert_true(result.shape() == Shape(5, 3))
    var sliced = result.slice(start=3, end=4, axis=0)
    assert_true(sliced.squeeze().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


fn test_bcast_cpu_fwd_same_shape_noop() raises:
    print("test_bcast_cpu_fwd_same_shape_noop")
    comptime dtype = DType.float32
    # Broadcasting to the same shape is a valid no-op
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.broadcast_to(Shape(2, 2))
    assert_true(result.all_close(a))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · Backward · 1-D cases
# ─────────────────────────────────────────────────────────────────────────────

fn test_bcast_cpu_bwd_1d_to_2d_sum_rows() raises:
    print("test_bcast_cpu_bwd_1d_to_2d_sum_rows")
    comptime dtype = DType.float32
    # a.shape=(3,) broadcast to (2,3); grad must sum over axis 0
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.broadcast_to(Shape(2, 3))
    var loss = b.sum()
    loss.backward()
    # Each element of a contributes to 2 rows → grad = [2, 2, 2]
    assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


fn test_bcast_cpu_bwd_1d_to_3d() raises:
    print("test_bcast_cpu_bwd_1d_to_3d")
    comptime dtype = DType.float32
    # a.shape=(3,) broadcast to (2,4,3); grad sums over axes 0 and 1
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.broadcast_to(Shape(2, 4, 3))
    var loss = b.sum()
    loss.backward()
    # Each element contributes to 2*4=8 positions → grad = [8, 8, 8]
    assert_true(a.grad().all_close(Tensor[dtype].d1([8.0, 8.0, 8.0])))


fn test_bcast_cpu_bwd_scalar_to_1d() raises:
    print("test_bcast_cpu_bwd_scalar_to_1d")
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

fn test_bcast_cpu_bwd_2d_row_to_2d() raises:
    print("test_bcast_cpu_bwd_2d_row_to_2d")
    comptime dtype = DType.float32
    # a.shape=(1,3) broadcast to (4,3); grad sums over axis 0
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
    var b = a.broadcast_to(Shape(4, 3))
    var loss = b.sum()
    loss.backward()
    # Each element used 4 times → grad = [[4, 4, 4]]
    assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0]])))


fn test_bcast_cpu_bwd_2d_col_to_2d() raises:
    print("test_bcast_cpu_bwd_2d_col_to_2d")
    comptime dtype = DType.float32
    # a.shape=(3,1) broadcast to (3,4); grad sums over axis 1
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var b = a.broadcast_to(Shape(3, 4))
    var loss = b.sum()
    loss.backward()
    # Each element used 4 times → grad = [[4],[4],[4]]
    assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0], [4.0], [4.0]])))


fn test_bcast_cpu_bwd_2d_to_3d() raises:
    print("test_bcast_cpu_bwd_2d_to_3d")
    comptime dtype = DType.float32
    # a.shape=(1,3) broadcast to (2,4,3)
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
    var b = a.broadcast_to(Shape(2, 4, 3))
    var loss = b.sum()
    loss.backward()
    # Each element used 2*4=8 times → grad = [[8,8,8]]
    assert_true(a.grad().all_close(Tensor[dtype].d2([[8.0, 8.0, 8.0]])))


fn test_bcast_cpu_bwd_same_shape_noop() raises:
    print("test_bcast_cpu_bwd_same_shape_noop")
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

fn test_bcast_cpu_bwd_chained_add() raises:
    print("test_bcast_cpu_bwd_chained_add")
    comptime dtype = DType.float32
    # broadcast then add then sum — verify chain rule
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = a.broadcast_to(Shape(3, 3))
    var c = Tensor[dtype].d2([[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0]])
    var d = b + c
    var loss = d.sum()
    loss.backward()
    # grad of sum w.r.t b is all-ones (3,3)
    # grad of b w.r.t a sums over axis 0 → [3,3,3]
    assert_true(a.grad().all_close(Tensor[dtype].d1([3.0, 3.0, 3.0])))


fn test_bcast_cpu_bwd_chained_multiply() raises:
    print("test_bcast_cpu_bwd_chained_multiply")
    comptime dtype = DType.float32
    # a.shape=(1,3) broadcast to (2,3) then element-wise multiply
    var a = Tensor[dtype].d2([[2.0, 3.0, 4.0]], requires_grad=True)
    var b = a.broadcast_to(Shape(2, 3))
    var scale = Tensor[dtype].d2([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
    var c = b * scale
    var loss = c.sum()
    loss.backward()
    # upstream grad = scale, sum over axis 0 per element:
    # col0: 1+4=5, col1: 2+5=7, col2: 3+6=9
    assert_true(a.grad().all_close(Tensor[dtype].d2([[5.0, 7.0, 9.0]])))


fn test_bcast_cpu_bwd_chained_double_broadcast() raises:
    print("test_bcast_cpu_bwd_chained_double_broadcast")
    comptime dtype = DType.float32
    # Two separate tensors broadcast then added
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)  # (1,3)
    var b = Tensor[dtype].d2([[10.0],[20.0],[30.0]], requires_grad=True)  # (3,1)
    var ab = a.broadcast_to(Shape(3, 3)) + b.broadcast_to(Shape(3, 3))
    var loss = ab.sum()
    loss.backward()
    # a: each col used 3 times → [[3,3,3]]
    assert_true(a.grad().all_close(Tensor[dtype].d2([[3.0, 3.0, 3.0]])))
    # b: each row used 3 times → [[3],[3],[3]]
    assert_true(b.grad().all_close(Tensor[dtype].d2([[3.0],[3.0],[3.0]])))


fn test_bcast_cpu_bwd_3d_partial_broadcast() raises:
    print("test_bcast_cpu_bwd_3d_partial_broadcast")
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

fn test_bcast_gpu_fwd_1d_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_fwd_1d_to_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(2, 3))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
        ])))


fn test_bcast_gpu_fwd_2d_row_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_fwd_2d_row_to_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[10.0, 20.0, 30.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(4, 3))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
            [10.0, 20.0, 30.0],
        ])))


fn test_bcast_gpu_fwd_2d_col_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_fwd_2d_col_to_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0],[2.0],[3.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(3, 4))
        assert_true(result.to_cpu().all_close(Tensor[dtype].d2([
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0],
        ])))


fn test_bcast_gpu_fwd_1d_to_3d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_fwd_1d_to_3d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(2, 4, 3))
        assert_true(result.shape() == Shape(2, 4, 3))
        # Spot check: every position along last axis should match a
        var cpu = result.to_cpu()
        var cpu_sliced = cpu.slice(start=0, end=1, axis=0)
        var sliced = cpu_sliced.slice(start=2, end=3, axis=1)
        assert_true(sliced.squeeze().all_close(Tensor[dtype].d1([1.0, 2.0, 3.0])))


fn test_bcast_gpu_fwd_same_shape_noop() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_fwd_same_shape_noop")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0],[3.0, 4.0]])
        var a_gpu = a.to_gpu()
        var result = a_gpu.broadcast_to(Shape(2, 2))
        assert_true(result.to_cpu().all_close(a))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — GPU · Backward · grad flows correctly
# ─────────────────────────────────────────────────────────────────────────────

fn test_bcast_gpu_bwd_1d_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_bwd_1d_to_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(2, 3))
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([2.0, 2.0, 2.0])))


fn test_bcast_gpu_bwd_2d_row_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_bwd_2d_row_to_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(4, 3))
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0, 4.0, 4.0]])))


fn test_bcast_gpu_bwd_2d_col_to_2d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_bwd_2d_col_to_2d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0],[2.0],[3.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(3, 4))
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d2([[4.0],[4.0],[4.0]])))


fn test_bcast_gpu_bwd_1d_to_3d() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_bwd_1d_to_3d")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(2, 4, 3))
        var loss = b.sum()
        loss.backward()
        # 2*4=8 uses per element
        assert_true(a.grad().all_close(Tensor[dtype].d1([8.0, 8.0, 8.0])))


fn test_bcast_gpu_bwd_same_shape_noop() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_bwd_same_shape_noop")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0],[3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(2, 2))
        var loss = b.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor.ones_like(a)))


fn test_bcast_gpu_bwd_chained_multiply() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_bwd_chained_multiply")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[2.0, 3.0, 4.0]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(2, 3))
        var scale = Tensor[dtype].d2([[1.0,2.0,3.0],[4.0,5.0,6.0]]).to_gpu()
        var c = b * scale
        var loss = c.sum()
        loss.backward()
        # col0: 1+4=5, col1: 2+5=7, col2: 3+6=9
        assert_true(a.grad().all_close(Tensor[dtype].d2([[5.0, 7.0, 9.0]])))


fn test_bcast_gpu_bwd_3d_partial_broadcast() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_bwd_3d_partial_broadcast")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d3([[[1.0, 2.0, 3.0, 4.0]]], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(3, 2, 4))
        var loss = b.sum()
        loss.backward()
        # 3*2=6 uses per element
        assert_true(a.grad().all_close(Tensor[dtype].d3([[[6.0, 6.0, 6.0, 6.0]]])))


fn test_bcast_gpu_bwd_chained_add() raises:
    comptime if has_accelerator():
        print("test_bcast_gpu_bwd_chained_add")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var b = a_gpu.broadcast_to(Shape(3, 3))
        var c = Tensor[dtype].d2([[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0]]).to_gpu()
        var d = b + c
        var loss = d.sum()
        loss.backward()
        assert_true(a.grad().all_close(Tensor[dtype].d1([3.0, 3.0, 3.0])))


# =============================================================================
# Main
# =============================================================================

def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll [broadcast_to] tests passed!")
