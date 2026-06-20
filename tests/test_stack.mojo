from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite
from tenmo.shapes import Shape
from std.sys import has_accelerator

# ============================================================================
# FORWARD PASS TESTS - STACK OPERATIONS
# ============================================================================


def test_stack_axis0_2d() raises:
    """Test stack along axis 0 for 2D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (2, 2, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)

    # Expected:
    # [[[1, 2, 3], [4, 5, 6]],
    #  [[7, 8, 9], [10, 11, 12]]]
    var expected = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


def test_stack_axis1_2d() raises:
    """Test stack along axis 1 for 2D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)

    # Expected shape: (2, 2, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)

    # Expected:
    # [[[1, 2, 3], [7, 8, 9]],
    #  [[4, 5, 6], [10, 11, 12]]]
    var expected = Tensor[dtype].d3(
        [
            [[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
            [[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


def test_stack_axis2_2d() raises:
    """Test stack along axis 2 for 2D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=2)

    # Expected shape: (2, 3, 2)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 3)
    assert_true(result.shape()[2] == 2)

    # Expected:
    # [[[1, 7], [2, 8], [3, 9]],
    #  [[4, 10], [5, 11], [6, 12]]]
    var expected = Tensor[dtype].d3(
        [
            [[1.0, 7.0], [2.0, 8.0], [3.0, 9.0]],
            [[4.0, 10.0], [5.0, 11.0], [6.0, 12.0]],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


def test_stack_axis_negative() raises:
    """Test stack with negative axis."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    # axis=-1 should be same as axis=2 for 2D input
    var result1 = Tensor[dtype].stack(tensors, axis=2)
    var result2 = Tensor[dtype].stack(tensors, axis=-1)

    assert_true(result1.all_close[atol=1e-6](result2))


def test_stack_1d_tensors() raises:
    """Test stack of 1D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])
    var C = Tensor[dtype].d1([7.0, 8.0, 9.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (3, 3)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 3)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


def test_stack_1d_axis1() raises:
    """Test stack of 1D tensors along axis 1."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)

    # Expected shape: (3, 2)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 2)

    var expected = Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    assert_true(result.all_close[atol=1e-6](expected))


def test_stack_3d_tensors() raises:
    """Test stack of 3D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d3([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)
    var B = Tensor[dtype].d3([[[5.0, 6.0], [7.0, 8.0]]])  # (1, 2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (2, 1, 2, 2)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 1)
    assert_true(result.shape()[2] == 2)
    assert_true(result.shape()[3] == 2)


def test_stack_many_tensors() raises:
    """Test stack of many tensors."""

    comptime dtype = DType.float32
    var tensors = List[Tensor[dtype]]()

    # Create 5 tensors
    for i in range(5):
        var t = Tensor[dtype].ones(2, 3) * Float32((i + 1))
        tensors.append(t)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (5, 2, 3)
    assert_true(result.shape()[0] == 5)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)


def test_stack_single_tensor() raises:
    """Test stack of single tensor (edge case)."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)

    var result = Tensor[dtype].stack(tensors, axis=0)

    # Expected shape: (1, 2, 2)
    assert_true(result.shape()[0] == 1)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 2)


# ============================================================================
# VSTACK TESTS
# ============================================================================
def test_vstack_2d() raises:
    """Test vstack with 2D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])  # (3,)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])  # (3,)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)
    # Expected shape: (2, 3)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 3)

    var expected = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert_true(result.all_close[atol=1e-6](expected))


def test_vstack_1d() raises:
    """Test vstack with 1D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])
    var C = Tensor[dtype].d1([7.0, 8.0, 9.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].vstack(tensors)

    # Expected shape: (3, 3)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 3)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


def test_vstack_mixed_rows() raises:
    """Test vstack with different number of rows."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0, 6.0]])  # (1, 2)
    var C = Tensor[dtype].d2([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])  # (3, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].vstack(tensors)
    # Expected shape: (6, 2)
    assert_true(result.shape()[0] == 6)
    assert_true(result.shape()[1] == 2)

    var expected = Tensor[dtype].d2(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )

    assert_true(result.all_close[atol=1e-6](expected))


# ============================================================================
# HSTACK TESTS
# ============================================================================


def test_hstack_2d() raises:
    """Test hstack with 2D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0], [2.0], [3.0]])  # (3, 1)
    var B = Tensor[dtype].d2([[4.0], [5.0], [6.0]])  # (3, 1)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)

    # Expected shape: (3, 2)
    assert_true(result.shape()[0] == 3)
    assert_true(result.shape()[1] == 2)

    var expected = Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    assert_true(result.all_close[atol=1e-6](expected))


def test_hstack_1d() raises:
    """Test hstack with 1D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0])
    var C = Tensor[dtype].d1([6.0, 7.0, 8.0, 9.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].hstack(tensors)

    # Expected shape: (9,)
    assert_true(result.shape()[0] == 9)

    var expected = Tensor[dtype].d1(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    )

    assert_true(result.all_close[atol=1e-6](expected))


def test_hstack_mixed_cols() raises:
    """Test hstack with different number of columns."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0], [6.0]])  # (2, 1)
    var C = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # (2, 3)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].hstack(tensors)

    # Expected shape: (2, 6)
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 6)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 5.0, 7.0, 8.0, 9.0], [3.0, 4.0, 6.0, 10.0, 11.0, 12.0]]
    )

    assert_true(result.all_close[atol=1e-6](expected))


# ============================================================================
# BACKWARD PASS TESTS - STACK
# ============================================================================


def test_stack_backward_axis0() raises:
    """Test gradient flow through stack along axis 0."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].ones(2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


def test_stack_backward_axis1() raises:
    """Test gradient flow through stack along axis 1."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].ones(2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


def test_stack_backward_weighted() raises:
    """Test gradient flow with weighted loss."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)
    # Apply weights
    var weights = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()
    # Gradients should be the weights
    var expected_grad_A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var expected_grad_B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


def test_stack_backward_three_tensors() raises:
    """Test gradient flow through stack of three tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var C = Tensor[dtype].d1([5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 2)
    var loss = result.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].ones(2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))
    assert_true(C.grad().all_close[atol=1e-6](expected_grad))


def test_stack_backward_chain() raises:
    """Test gradient flow through chained operations."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var stacked = Tensor[dtype].stack(tensors, axis=0)  # (2, 2)
    var squared = stacked * stacked  # Element-wise square
    var loss = squared.sum()
    loss.backward()

    # d(loss)/d(A) = 2 * A
    var expected_grad_A = Tensor[dtype].d1([2.0, 4.0])
    var expected_grad_B = Tensor[dtype].d1([6.0, 8.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


# ============================================================================
# BACKWARD PASS TESTS - VSTACK
# ============================================================================


def test_vstack_backward() raises:
    """Test gradient flow through vstack."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0]], requires_grad=True)  # (1, 2)
    var B = Tensor[dtype].d2(
        [[3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (3, 2)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(1, 2)
    var expected_grad_B = Tensor[dtype].ones(2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


def test_vstack_backward_1d() raises:
    """Test gradient flow through vstack with 1D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (2, 3)
    var loss = result.sum()
    loss.backward()

    var expected_grad = Tensor[dtype].ones(3)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


def test_vstack_backward_weighted() raises:
    """Test gradient flow through vstack with weighted loss."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var C = Tensor[dtype].d1([5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].vstack(tensors)  # (3, 2)

    # Row-wise weights
    var weights = Tensor[dtype].d2([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d1([1.0, 1.0])
    var expected_grad_B = Tensor[dtype].d1([2.0, 2.0])
    var expected_grad_C = Tensor[dtype].d1([3.0, 3.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))
    assert_true(C.grad().all_close[atol=1e-6](expected_grad_C))


# ============================================================================
# BACKWARD PASS TESTS - HSTACK
# ============================================================================


def test_hstack_backward() raises:
    """Test gradient flow through hstack."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0], [2.0]], requires_grad=True)  # (2, 1)
    var B = Tensor[dtype].d2(
        [[3.0, 4.0], [5.0, 6.0]], requires_grad=True
    )  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (2, 3)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(2, 1)
    var expected_grad_B = Tensor[dtype].ones(2, 2)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


def test_hstack_backward_1d() raises:
    """Test gradient flow through hstack with 1D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0, 5.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (5,)
    var loss = result.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].ones(2)
    var expected_grad_B = Tensor[dtype].ones(3)

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


# ============================================================================
# GPU CONCATENATION TESTS
# ============================================================================


def test_concat_gpu_axis0_2d() raises:
    """Concatenate 2D tensors along axis 0 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())

        var result = Tensor[dtype].concat(tensors, axis=0)
        assert_true(result.is_on_gpu())

        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(A)
        tensors_cpu.append(B)
        var expected = Tensor[dtype].concat(tensors_cpu, axis=0)
        assert_true(result.to_cpu().all_close[atol=1e-6](expected))


def test_concat_gpu_axis1_2d() raises:
    """Concatenate 2D tensors along axis 1 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())

        var result = Tensor[dtype].concat(tensors, axis=1)
        assert_true(result.is_on_gpu())

        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(A)
        tensors_cpu.append(B)
        var expected = Tensor[dtype].concat(tensors_cpu, axis=1)
        assert_true(result.to_cpu().all_close[atol=1e-6](expected))


def test_concat_gpu_axis0_3d() raises:
    """Concatenate 3D tensors along axis 0 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d3([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        var B = Tensor[dtype].d3([
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())

        var result = Tensor[dtype].concat(tensors, axis=0)
        assert_true(result.is_on_gpu())

        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(A)
        tensors_cpu.append(B)
        var expected = Tensor[dtype].concat(tensors_cpu, axis=0)
        assert_true(result.to_cpu().all_close[atol=1e-6](expected))


def test_concat_gpu_axis1_3d() raises:
    """Concatenate 3D tensors along axis 1 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d3([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        var B = Tensor[dtype].d3([
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())

        var result = Tensor[dtype].concat(tensors, axis=1)
        assert_true(result.is_on_gpu())

        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(A)
        tensors_cpu.append(B)
        var expected = Tensor[dtype].concat(tensors_cpu, axis=1)
        assert_true(result.to_cpu().all_close[atol=1e-6](expected))


def test_concat_gpu_axis2_3d() raises:
    """Concatenate 3D tensors along axis 2 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d3([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ])
        var B = Tensor[dtype].d3([
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())

        var result = Tensor[dtype].concat(tensors, axis=2)
        assert_true(result.is_on_gpu())

        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(A)
        tensors_cpu.append(B)
        var expected = Tensor[dtype].concat(tensors_cpu, axis=2)
        assert_true(result.to_cpu().all_close[atol=1e-6](expected))


def test_concat_gpu_three_inputs() raises:
    """Concatenate three tensors on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])
        var C = Tensor[dtype].d2([[9.0, 10.0], [11.0, 12.0]])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())
        tensors.append(C.to_gpu())

        var result = Tensor[dtype].concat(tensors, axis=0)
        assert_true(result.is_on_gpu())

        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(A)
        tensors_cpu.append(B)
        tensors_cpu.append(C)
        var expected = Tensor[dtype].concat(tensors_cpu, axis=0)
        assert_true(result.to_cpu().all_close[atol=1e-6](expected))


def test_concat_gpu_backward_axis0() raises:
    """Gradient flows correctly through GPU concat axis=0."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # CPU backward
        var cpu_A = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        )
        var cpu_B = Tensor[dtype].d2(
            [[5.0, 6.0], [7.0, 8.0]], requires_grad=True
        )
        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(cpu_A)
        tensors_cpu.append(cpu_B)
        var result_cpu = Tensor[dtype].concat(tensors_cpu, axis=0)
        var loss_cpu = result_cpu.sum()
        loss_cpu.backward()

        # GPU backward
        var gpu_A = Tensor[dtype].d2(
            [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
        ).to_gpu()
        var gpu_B = Tensor[dtype].d2(
            [[5.0, 6.0], [7.0, 8.0]], requires_grad=True
        ).to_gpu()
        var tensors_gpu = List[Tensor[dtype]]()
        tensors_gpu.append(gpu_A)
        tensors_gpu.append(gpu_B)
        var result_gpu = Tensor[dtype].concat(tensors_gpu, axis=0)
        var loss_gpu = result_gpu.sum()
        loss_gpu.backward()

        assert_true(
            gpu_A.grad().to_cpu().all_close[atol=1e-6](cpu_A.grad())
        )
        assert_true(
            gpu_B.grad().to_cpu().all_close[atol=1e-6](cpu_B.grad())
        )


def test_concat_gpu_backward_axis1() raises:
    """Gradient flows correctly through GPU concat axis=1."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        # CPU backward
        var cpu_A = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var cpu_B = Tensor[dtype].d2(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
        )
        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(cpu_A)
        tensors_cpu.append(cpu_B)
        var result_cpu = Tensor[dtype].concat(tensors_cpu, axis=1)
        var loss_cpu = result_cpu.sum()
        loss_cpu.backward()

        # GPU backward
        var gpu_A = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        ).to_gpu()
        var gpu_B = Tensor[dtype].d2(
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
        ).to_gpu()
        var tensors_gpu = List[Tensor[dtype]]()
        tensors_gpu.append(gpu_A)
        tensors_gpu.append(gpu_B)
        var result_gpu = Tensor[dtype].concat(tensors_gpu, axis=1)
        var loss_gpu = result_gpu.sum()
        loss_gpu.backward()

        assert_true(
            gpu_A.grad().to_cpu().all_close[atol=1e-6](cpu_A.grad())
        )
        assert_true(
            gpu_B.grad().to_cpu().all_close[atol=1e-6](cpu_B.grad())
        )


def test_concat_gpu_backward_axis2_3d() raises:
    """Gradient flows correctly through GPU concat axis=2 on 3D tensors."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var cpu_A = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        )
        var cpu_B = Tensor[dtype].d3(
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
            requires_grad=True,
        )
        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(cpu_A)
        tensors_cpu.append(cpu_B)
        var result_cpu = Tensor[dtype].concat(tensors_cpu, axis=2)
        var loss_cpu = result_cpu.sum()
        loss_cpu.backward()

        var gpu_A = Tensor[dtype].d3(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            requires_grad=True,
        ).to_gpu()
        var gpu_B = Tensor[dtype].d3(
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
            requires_grad=True,
        ).to_gpu()
        var tensors_gpu = List[Tensor[dtype]]()
        tensors_gpu.append(gpu_A)
        tensors_gpu.append(gpu_B)
        var result_gpu = Tensor[dtype].concat(tensors_gpu, axis=2)
        var loss_gpu = result_gpu.sum()
        loss_gpu.backward()

        assert_true(
            gpu_A.grad().to_cpu().all_close[atol=1e-6](cpu_A.grad())
        )
        assert_true(
            gpu_B.grad().to_cpu().all_close[atol=1e-6](cpu_B.grad())
        )


def test_stack_gpu_axis0() raises:
    """Stack tensors along axis 0 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())

        var result = Tensor[dtype].stack(tensors, axis=0)
        assert_true(result.is_on_gpu())

        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(A)
        tensors_cpu.append(B)
        var expected = Tensor[dtype].stack(tensors_cpu, axis=0)
        assert_true(result.to_cpu().all_close[atol=1e-6](expected))


def test_stack_gpu_axis1() raises:
    """Stack tensors along axis 1 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())

        var result = Tensor[dtype].stack(tensors, axis=1)
        assert_true(result.is_on_gpu())

        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(A)
        tensors_cpu.append(B)
        var expected = Tensor[dtype].stack(tensors_cpu, axis=1)
        assert_true(result.to_cpu().all_close[atol=1e-6](expected))


def test_stack_gpu_axis2() raises:
    """Stack tensors along axis 2 on GPU."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var B = Tensor[dtype].d2([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())

        var result = Tensor[dtype].stack(tensors, axis=2)
        assert_true(result.is_on_gpu())

        var tensors_cpu = List[Tensor[dtype]]()
        tensors_cpu.append(A)
        tensors_cpu.append(B)
        var expected = Tensor[dtype].stack(tensors_cpu, axis=2)
        assert_true(result.to_cpu().all_close[atol=1e-6](expected))


def test_concat_gpu_no_track_grad() raises:
    """GPU concat with track_grad=False produces no gradbox."""
    comptime if has_accelerator():
        comptime dtype = DType.float32
        var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
        var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])

        var tensors = List[Tensor[dtype]]()
        tensors.append(A.to_gpu())
        tensors.append(B.to_gpu())

        var result = Tensor[dtype].concat(tensors, axis=0)
        assert_true(result.is_on_gpu())
        assert_true(not result.requires_grad)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


# """
# Comprehensive test suite for Tensor stack/vstack/hstack operations.
# Test prefix: stk_
# """

# ============================================================================
# STACK TESTS - Forward Pass
# ============================================================================


def test_stk_basic_2_tensors_axis0() raises:
    """Test stacking 2 tensors along axis 0."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)

    var expected = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_basic_3_tensors_axis0() raises:
    """Test stacking 3 tensors along axis 0."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0])
    var B = Tensor[dtype].d1([3.0, 4.0])
    var C = Tensor[dtype].d1([5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 2)

    var expected = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_axis1_2d() raises:
    """Test stacking along axis 1 for 2D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)  # (2, 2, 2)

    # Expected shape: (2, 2, 2)
    # result[0] = [[1, 2], [5, 6]]
    # result[1] = [[3, 4], [7, 8]]
    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 2)


def test_stk_axis_negative() raises:
    """Test stacking with negative axis."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    # axis=-1 should be equivalent to axis=1 for 1D tensors
    var result = Tensor[dtype].stack(tensors, axis=-1)  # (3, 2)

    var expected = Tensor[dtype].d2([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_single_element_tensors() raises:
    """Test stacking single-element tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0])
    var B = Tensor[dtype].d1([2.0])
    var C = Tensor[dtype].d1([3.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 1)

    var expected = Tensor[dtype].d2([[1.0], [2.0], [3.0]])
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_3d_tensors() raises:
    """Test stacking 3D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].zeros(2, 3, 4)
    var B = Tensor[dtype].ones(2, 3, 4)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 2, 3, 4)

    assert_true(result.shape()[0] == 2)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)
    assert_true(result.shape()[3] == 4)


# ============================================================================
# STACK TESTS - Backward Pass
# ============================================================================


def test_stk_backward_simple_axis0() raises:
    """Test gradient flow through stack with axis=0."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].d1([1.0, 1.0, 1.0])
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


def test_stk_backward_weighted() raises:
    """Test gradient flow with weighted loss."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)

    # Apply weights
    var weights = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    # Gradients should be the weights
    var expected_grad_A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var expected_grad_B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


def test_stk_backward_axis1() raises:
    """Test gradient flow through stack with axis=1."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=1)  # (2, 2, 2)
    var loss = result.sum()
    loss.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].d2([[1.0, 1.0], [1.0, 1.0]])
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


def test_stk_backward_three_tensors() raises:
    """Test gradient flow with three input tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)
    var C = Tensor[dtype].d1([5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 2)

    # Multiply by different weights for each stacked tensor
    var weights = Tensor[dtype].d2([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d1([2.0, 3.0])
    var expected_grad_B = Tensor[dtype].d1([4.0, 5.0])
    var expected_grad_C = Tensor[dtype].d1([6.0, 7.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))
    assert_true(C.grad().all_close[atol=1e-6](expected_grad_C))


def test_stk_backward_selective_grad() raises:
    """Test gradient flow when only some tensors require grad."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=False)
    var C = Tensor[dtype].d1([7.0, 8.0, 9.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 3)
    var loss = result.sum()
    loss.backward()

    # A and C should have gradients, B should not
    var expected_grad = Tensor[dtype].d1([1.0, 1.0, 1.0])
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(C.grad().all_close[atol=1e-6](expected_grad))
    assert_true(not B.has_grad())


# ============================================================================
# VSTACK TESTS
# ============================================================================


def test_stk_vstack_1d_tensors() raises:
    """Test vstack with 1D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (2, 3)

    var expected = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_vstack_2d_tensors() raises:
    """Test vstack with 2D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (4, 2)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_vstack_different_rows() raises:
    """Test vstack with tensors having different number of rows."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0, 3.0]])  # (1, 3)
    var B = Tensor[dtype].d2([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # (2, 3)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (3, 3)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_vstack_backward() raises:
    """Test gradient flow through vstack."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].vstack(tensors)  # (2, 3)

    # Apply different weights to each row
    var weights = Tensor[dtype].d2([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d1([1.0, 1.0, 1.0])
    var expected_grad_B = Tensor[dtype].d1([2.0, 2.0, 2.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


# ============================================================================
# HSTACK TESTS
# ============================================================================


def test_stk_hstack_1d_tensors() raises:
    """Test hstack with 1D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0])

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (6,)

    var expected = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_hstack_2d_tensors() raises:
    """Test hstack with 2D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (2, 4)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_hstack_different_columns() raises:
    """Test hstack with tensors having different number of columns."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0], [2.0]])  # (2, 1)
    var B = Tensor[dtype].d2([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])  # (2, 3)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (2, 4)

    var expected = Tensor[dtype].d2(
        [[1.0, 3.0, 4.0, 5.0], [2.0, 6.0, 7.0, 8.0]]
    )
    assert_true(result.all_close[atol=1e-6](expected))


def test_stk_hstack_backward() raises:
    """Test gradient flow through hstack."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0], requires_grad=True)
    var B = Tensor[dtype].d1([3.0, 4.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (4,)

    # Apply different weights
    var weights = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d1([1.0, 2.0])
    var expected_grad_B = Tensor[dtype].d1([3.0, 4.0])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


def test_stk_hstack_2d_backward() raises:
    """Test gradient flow through hstack with 2D tensors."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var B = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].hstack(tensors)  # (2, 4)

    # Apply different weights
    var weights = Tensor[dtype].d2([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    var weighted = result * weights
    var loss = weighted.sum()
    loss.backward()

    var expected_grad_A = Tensor[dtype].d2([[1.0, 2.0], [5.0, 6.0]])
    var expected_grad_B = Tensor[dtype].d2([[3.0, 4.0], [7.0, 8.0]])

    assert_true(A.grad().all_close[atol=1e-6](expected_grad_A))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad_B))


# ============================================================================
# EDGE CASES AND STRESS TESTS
# ============================================================================


def test_stk_large_number_tensors() raises:
    """Test stacking a large number of tensors."""

    comptime dtype = DType.float32
    var tensors = List[Tensor[dtype]]()

    for i in range(10):
        var tensor = Tensor[dtype].full(Shape(2, 3), Scalar[dtype](i))
        tensors.append(tensor^)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (10, 2, 3)

    assert_true(result.shape()[0] == 10)
    assert_true(result.shape()[1] == 2)
    assert_true(result.shape()[2] == 3)


def test_stk_zeros_and_ones() raises:
    """Test stacking tensors with zeros and ones."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].zeros(3, 4)
    var B = Tensor[dtype].ones(3, 4)
    var C = Tensor[dtype].zeros(3, 4)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)
    tensors.append(C)

    var result = Tensor[dtype].stack(tensors, axis=0)  # (3, 3, 4)

    # Check middle slice is all ones
    var sum_middle = Float32(0.0)
    for i in range(3):
        for j in range(4):
            sum_middle += result[1, i, j]

    assert_true(abs(sum_middle - 12.0) < 1e-6)


def test_stk_chain_operations() raises:
    """Test stacking followed by other operations."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var B = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var stacked = Tensor[dtype].stack(tensors, axis=0)  # (2, 3)
    var transposed = stacked.transpose(0, 1)  # (3, 2)
    var result = transposed.sum()
    result.backward()

    # All gradients should be 1.0
    var expected_grad = Tensor[dtype].d1([1.0, 1.0, 1.0])
    assert_true(A.grad().all_close[atol=1e-6](expected_grad))
    assert_true(B.grad().all_close[atol=1e-6](expected_grad))


def test_stk_mixed_operations() raises:
    """Test mixing stack, vstack, and hstack."""

    comptime dtype = DType.float32
    var A = Tensor[dtype].d1([1.0, 2.0])
    var B = Tensor[dtype].d1([3.0, 4.0])

    var tensors1 = List[Tensor[dtype]]()
    tensors1.append(A)
    tensors1.append(B)

    var vstacked = Tensor[dtype].vstack(tensors1)  # (2, 2)

    var C = Tensor[dtype].d2([[5.0, 6.0], [7.0, 8.0]])

    var tensors2 = List[Tensor[dtype]]()
    tensors2.append(vstacked)
    tensors2.append(C)

    var hstacked = Tensor[dtype].hstack(tensors2)  # (2, 4)

    var expected = Tensor[dtype].d2(
        [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
    )
    assert_true(hstacked.all_close[atol=1e-6](expected))


def test_stack_operations() raises:
    comptime dtype = DType.float32

    var A = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var B = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
    )

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Tensor[dtype].stack(tensors, axis=0, requires_grad=True)
    _ = """assert_true(result == Tensor[dtype].d3([[[1, 2, 3],[4, 5, 6]],[[7, 8, 9],[10, 11, 12]]])

    # Expected: (2, 2, 3)
    # [[[1, 2, 3],
    #   [4, 5, 6]],
    #  [[7, 8, 9],
    #   [10, 11, 12]]]


    var result2 = Tensor[dtype].stack(tensors, axis=1, requires_grad=True)
    # Expected: (2, 2, 3)
    # [[[1, 2, 3],
    #   [7, 8, 9]],
    #  [[4, 5, 6],
    #   [10, 11, 12]]]


    var result3 = Tensor[dtype].stack(tensors, axis=2, requires_grad=True)
    # Expected: (2, 3, 2)
    # [[[1, 7],
    #   [2, 8],
    #   [3, 9]],
    #  [[4, 10],
    #   [5, 11],
    #   [6, 12]]]"""

    assert_true(
        result
        == Tensor[dtype].d3([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    )

    var result2 = Tensor[dtype].stack(tensors, axis=1, requires_grad=True)
    assert_true(
        result2
        == Tensor[dtype].d3([[[1, 2, 3], [7, 8, 9]], [[4, 5, 6], [10, 11, 12]]])
    )

    var result3 = Tensor[dtype].stack(tensors, axis=2, requires_grad=True)
    assert_true(
        result3
        == Tensor[dtype].d3(
            [[[1, 7], [2, 8], [3, 9]], [[4, 10], [5, 11], [6, 12]]]
        )
    )

    var loss = result.sum()
    loss.backward()

    # Expected: all ones (2x3)

    # Expected: all ones (2x3)

    var C = Tensor[dtype].ones(1, 3, requires_grad=True)
    var D = Tensor[dtype].ones(1, 3, requires_grad=True) * 2.0

    var vstack_list = List[Tensor[dtype]]()
    vstack_list.append(C)
    vstack_list.append(D)

    var vstacked = Tensor[dtype].vstack(vstack_list, requires_grad=True)
    # Expected: (2, 3)
    # [[1, 1, 1],
    #  [2, 2, 2]]
    assert_true(vstacked == Tensor[dtype].d2([[1, 1, 1], [2, 2, 2]]))

    var E = Tensor[dtype].ones(2, 1, requires_grad=True)
    var F = Tensor[dtype].ones(2, 2, requires_grad=True) * 3.0

    var hstack_list = List[Tensor[dtype]]()
    hstack_list.append(E)
    hstack_list.append(F)

    var hstacked = Tensor[dtype].hstack(hstack_list, requires_grad=True)
    # Expected: (2, 3)
    # [[1, 3, 3],
    #  [1, 3, 3]]

    assert_true(hstacked == Tensor[dtype].d2([[1, 3, 3], [1, 3, 3]]))

    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var b = Tensor[dtype].d1([4.0, 5.0, 6.0], requires_grad=True)

    var vstack_1d = List[Tensor[dtype]]()
    vstack_1d.append(a)
    vstack_1d.append(b)

    var vstacked_1d = Tensor[dtype].vstack(vstack_1d, requires_grad=True)
    # Expected: (2, 3)
    # [[1, 2, 3],
    #  [4, 5, 6]]

    assert_true(vstacked_1d == Tensor[dtype].d2([[1, 2, 3], [4, 5, 6]]))
    var hstacked_1d = Tensor[dtype].hstack(vstack_1d, requires_grad=True)
    # Expected: (6,)
    # [1, 2, 3, 4, 5, 6]

    assert_true(hstacked_1d == Tensor[dtype].d1([1, 2, 3, 4, 5, 6]))


# ```

# ---


### **1. Forward Pass Strategy:**
# ```
# stack = unsqueeze_each + concat
# ```

### **2. Backward Pass Strategy:**
# ```
# Split gradient along stacked axis → Squeeze each split
# ```

### **3. Edge Cases Handled:**
# - Single tensor: just unsqueeze
# - Empty list: raises error
# - Shape mismatch: validates before stacking
# - 1D tensors in vstack: reshapes to 2D
# - 1D tensors in hstack: uses concat

### **4. Gradient Flow:**
# ```
# Forward:  A(2,3) → unsqueeze(0) → A'(1,2,3) ──┐
#          B(2,3) → unsqueeze(0) → B'(1,2,3) ──┤→ concat → Result(2,2,3)
#          C(2,3) → unsqueeze(0) → C'(1,2,3) ──┘

# Backward: grad_Result(2,2,3) → split → [grad_A'(1,2,3),
#                                         grad_B'(1,2,3),
#                                         grad_C'(1,2,3)]
#                                    ↓ squeeze(0)
#                                [grad_A(2,3), grad_B(2,3), grad_C(2,3)]
