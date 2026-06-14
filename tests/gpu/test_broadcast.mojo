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
    print("printing the copied grads")
    a_cpu_grad.print()
    b_cpu_grad.print()
    a.zero_grad()
    b.zero_grad()
    print("a and b's grad after zeroing them\n")
    a.grad().print()
    b.grad().print()
    var gpu_result = (a_gpu * b_gpu).mean()
    gpu_result.backward()
    print("after gpu result backward a and b's grads\n")
    a.grad().print()
    print()
    b.grad().print()
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
    print("Copied grads\n")
    a_cpu_grad.print()
    b_cpu_grad.print()
    c_cpu_grad.print()
    a.zero_grad()
    b.zero_grad()
    c.zero_grad()
    var gpu_result = (a_gpu * b_gpu + c_gpu).mean()
    gpu_result.backward()
    print("Now a, b and c's grads\n")
    a.grad().print()
    b.grad().print()
    c.grad().print()
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
    print("Copied grads\n")
    a_cpu_grad.print()
    b_cpu_grad.print()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    print("And here a and b's grads\n")
    a.grad().print()
    b.grad().print()
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
    print("b_cpu_grad\n")
    b_cpu_grad.print()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    print("And now b's grad\n")
    b.grad().print()
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
    print("The copied gradients are\n")
    a_cpu_grad.print()
    b_cpu_grad.print()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu * b_gpu).sum()
    gpu_result.backward()
    print("post gpu backward a and b's grads are")
    a.grad().print()
    b.grad().print()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


# ═════════════════════════════════════════════════════════════════════════════
# GPU Add/Sub/Div Broadcast Backward Tests
# ═════════════════════════════════════════════════════════════════════════════

def test_gpu_add_broadcast_col_plus_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (3,1) + (3,4) → broadcast along axis 1
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        requires_grad=True,
    )
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


def test_gpu_add_broadcast_1d_plus_2d() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (4,) + (3,4) → broadcast 1d across rows
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
        requires_grad=True,
    )
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


def test_gpu_add_broadcast_3d_batch() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,3,4) + (2,3,4) → broadcast batch dim
    var a = Tensor[dtype].rand(1, 3, 4, requires_grad=True)
    var b = Tensor[dtype].rand(2, 3, 4, requires_grad=True)
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


def test_gpu_add_broadcast_only_lhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor[dtype].d2([[1.0], [1.0]])
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a + b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    a.zero_grad()
    var gpu_result = (a_gpu + b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))


def test_gpu_add_broadcast_only_rhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    var a = Tensor[dtype].d2([[1.0], [2.0]])
    var b = Tensor[dtype].rand(2, 3, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a + b).sum()
    cpu_result.backward()
    var b_cpu_grad = b.grad().copy()
    b.zero_grad()
    var gpu_result = (a_gpu + b_gpu).sum()
    gpu_result.backward()
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_sub_broadcast_col_minus_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (3,1) - (3,4) → broadcast along axis 1
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        requires_grad=True,
    )
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a - b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu - b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_sub_broadcast_1d_minus_2d() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (4,) - (3,4) → broadcast 1d across rows
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
        requires_grad=True,
    )
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a - b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu - b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_sub_broadcast_only_lhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor[dtype].d2([[1.0], [1.0]])
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a - b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    a.zero_grad()
    var gpu_result = (a_gpu - b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))


def test_gpu_sub_broadcast_only_rhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    var a = Tensor[dtype].d2([[1.0], [2.0]])
    var b = Tensor[dtype].rand(2, 3, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a - b).sum()
    cpu_result.backward()
    var b_cpu_grad = b.grad().copy()
    b.zero_grad()
    var gpu_result = (a_gpu - b_gpu).sum()
    gpu_result.backward()
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_div_broadcast_col_divided_by_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (3,1) / (3,4) → broadcast along axis 1
    var a = Tensor[dtype].d2([[1.0], [2.0], [3.0]], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        requires_grad=True,
    )
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a / b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu / b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_div_broadcast_row_divided_by_matrix() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,4) / (3,4) → broadcast along axis 0
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    var b = Tensor[dtype].d2(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
        requires_grad=True,
    )
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a / b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu / b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_div_broadcast_3d_batch() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    # (1,3,4) / (2,3,4) → broadcast batch dim
    var a = Tensor[dtype].rand(1, 3, 4, requires_grad=True)
    var b = Tensor[dtype].rand(2, 3, 4, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a / b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    var b_cpu_grad = b.grad().copy()
    a.zero_grad()
    b.zero_grad()
    var gpu_result = (a_gpu / b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))
    assert_true(b.grad().all_close(b_cpu_grad))


def test_gpu_div_broadcast_only_lhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var b = Tensor[dtype].d2([[1.0], [1.0]])
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a / b).sum()
    cpu_result.backward()
    var a_cpu_grad = a.grad().copy()
    a.zero_grad()
    var gpu_result = (a_gpu / b_gpu).sum()
    gpu_result.backward()
    assert_true(a.grad().all_close(a_cpu_grad))


def test_gpu_div_broadcast_only_rhs_requires_grad() raises:
    comptime if not has_accelerator():
        print("No GPU available — skipping tests")
        return

    var a = Tensor[dtype].d2([[1.0], [2.0]])
    var b = Tensor[dtype].rand(2, 3, requires_grad=True)
    var a_gpu = a.to_gpu()
    var b_gpu = b.to_gpu()
    var cpu_result = (a / b).sum()
    cpu_result.backward()
    var b_cpu_grad = b.grad().copy()
    b.zero_grad()
    var gpu_result = (a_gpu / b_gpu).sum()
    gpu_result.backward()
    assert_true(b.grad().all_close(b_cpu_grad))


# ═════════════════════════════════════════════════════════════════════════════
# CPU broadcast_to Forward Tests
# ═════════════════════════════════════════════════════════════════════════════


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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CPU · Backward · 1-D cases
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CPU · Backward · 2-D cases
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — CPU · Backward · chained ops after broadcast_to
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# B — Op-code flip: Subtract→ReverseSubtract, Divide→ReverseDivide
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# C — Non-contiguous path (index_iterator + item order)
# ─────────────────────────────────────────────────────────────────────────────
# Transposing a (3,2) tensor gives a (2,3) view with non-default strides.
# This forces broadcast_scalar_buffer to the index_iterator path (non-SIMD).


# ─────────────────────────────────────────────────────────────────────────────
# D — Backward pass (gradient through broadcast)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# E — Boundary cases
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# Main
# =============================================================================


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll [broadcast_to] tests passed!")
