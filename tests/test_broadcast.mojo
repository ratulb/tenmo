from tenmo import Tensor
from testing import assert_true
from sys import has_accelerator
from shapes import Shape

comptime dtype = DType.float32


fn test_gpu_mul_broadcast_scalar_row_times_matrix() raises:
    print("test_gpu_mul_broadcast_scalar_row_times_matrix")
    # (1,4) * (3,4) → broadcast along axis 0
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0, 2.0],
         [3.0, 3.0, 3.0, 3.0]], requires_grad=True
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
    print("test_gpu_mul_broadcast_col_times_matrix")
    # (3,1) * (3,4) → broadcast along axis 1
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0],
         [1.0, 2.0, 3.0, 4.0],
         [1.0, 2.0, 3.0, 4.0]], requires_grad=True
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
    print("test_gpu_mul_broadcast_1d_times_2d")
    # (4,) * (3,4) → broadcast 1d across rows
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0, 1.0],
         [2.0, 2.0, 2.0, 2.0],
         [3.0, 3.0, 3.0, 3.0]], requires_grad=True
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


fn main() raises:
    @parameter
    if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    test_gpu_mul_broadcast_scalar_row_times_matrix()
    test_gpu_mul_broadcast_col_times_matrix()
    test_gpu_mul_broadcast_1d_times_2d()
    test_gpu_mul_broadcast_scalar_times_matrix()
    test_gpu_mul_broadcast_3d_batch()
    test_gpu_mul_broadcast_only_lhs_requires_grad()
    test_gpu_mul_broadcast_only_rhs_requires_grad()
    test_gpu_mul_broadcast_large()

    print("\n=== ALL MULTIPLY BROADCAST GPU TESTS PASSED ===")
