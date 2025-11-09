from tenmo import Tensor
from shapes import Shape
from testing import assert_true

fn test_repeat_scalar_to_1d() raises:
    print("test_repeat_scalar_to_1d")
    var x = Tensor.scalar(2.0, requires_grad=True)
    var y = x.repeat(5)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(5))
    assert_true(y.all_close(Tensor.d1([2.0, 2.0, 2.0, 2.0, 2.0])))
    assert_true(x.grad().item() == 5.0)

fn test_repeat_scalar_to_2d() raises:
    print("test_repeat_scalar_to_2d")
    var x = Tensor.scalar(3.0, requires_grad=True)
    var y = x.repeat(2, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 3))
    assert_true(y.all_close(Tensor.d2([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0]])))
    assert_true(x.grad().item() == 6.0)

fn test_repeat_scalar_to_3d() raises:
    print("test_repeat_scalar_to_3d")
    var x = Tensor.scalar(4.0, requires_grad=True)
    var y = x.repeat(2, 3, 1)  # Note: must have at least 1 dimension for scalar
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 3, 1))
    assert_true(x.grad().item() == 6.0)

fn test_repeat_1d_to_longer_1d() raises:
    print("test_repeat_1d_to_longer_1d")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var y = x.repeat(4)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(12))
    var expected_data = Tensor.d1([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    assert_true(y.all_close(expected_data))
    assert_true(x.grad().all_close(Tensor.d1([4.0, 4.0, 4.0])))

fn test_repeat_1d_to_2d() raises:
    print("test_repeat_1d_to_2d")
    var x = Tensor.d1([1.0, 2.0], requires_grad=True)
    var y = x.repeat(3, 2)  # Must provide exactly 1 repeat dimension for 1D tensor
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3, 4))
    var expected = Tensor.d2([
        [1.0, 2.0, 1.0, 2.0],
        [1.0, 2.0, 1.0, 2.0],
        [1.0, 2.0, 1.0, 2.0]
    ])
    assert_true(y.all_close(expected))
    assert_true(x.grad().all_close(Tensor.d1([6.0, 6.0])))

fn test_repeat_1d_to_3d() raises:
    print("test_repeat_1d_to_3d")
    var x = Tensor.d1([1.0, 2.0], requires_grad=True)
    var y = x.repeat(2, 3, 1)  # Must provide exactly 1 repeat dimension for 1D tensor
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 3, 2))
    assert_true(x.grad().all_close(Tensor.d1([6.0, 6.0])))

fn test_repeat_2d_same_rank() raises:
    print("test_repeat_2d_same_rank")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.repeat(2, 3)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(4, 6))
    var expected = Tensor.d2([
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        [3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        [3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    ])
    assert_true(y.all_close(expected))
    assert_true(x.grad().all_close(Tensor.d2([[6.0, 6.0], [6.0, 6.0]])))

fn test_repeat_2d_to_3d() raises:
    print("test_repeat_2d_to_3d")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.repeat(2, 1, 1)  # Add batch dimension
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2, 2))
    assert_true(x.grad().all_close(Tensor.d2([[2.0, 2.0], [2.0, 2.0]])))

fn test_repeat_3d_same_rank() raises:
    print("test_repeat_3d_same_rank")
    var x = Tensor.d3([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    var y = x.repeat(2, 1, 1)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2, 2))
    assert_true(x.grad().all_close(Tensor.d3([[[2.0, 2.0], [2.0, 2.0]]])))

fn test_repeat_3d_all_dims() raises:
    print("test_repeat_3d_all_dims")
    var x = Tensor.d3([[[1.0], [2.0]]], requires_grad=True)
    var y = x.repeat(2, 3, 4)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 6, 4))
    assert_true(x.grad().all_close(Tensor.d3([[[24.0], [24.0]]])))

fn test_repeat_identity_operation() raises:
    print("test_repeat_identity_operation")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var y = x.repeat(1, 1)  # Repeat once along each dimension = no change
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(2, 2))
    assert_true(y.all_close(x))
    assert_true(x.grad().all_close(Tensor.d2([[1.0, 1.0], [1.0, 1.0]])))

fn test_repeat_complex_pattern() raises:
    print("test_repeat_complex_pattern")
    var x = Tensor.d2([[1.0, 2.0]], requires_grad=True)
    var y = x.repeat(3, 2, 1)  # Shape: (1, 2) → (3, 2, 2)
    var loss = y.sum()
    loss.backward()

    assert_true(y.shape() == Shape(3, 2, 2))
    assert_true(x.grad().all_close(Tensor.d2([[6.0, 6.0]])))

fn test_repeat_gradient_accumulation() raises:
    print("test_repeat_gradient_accumulation")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var y1 = x.repeat(2)
    var y2 = x.repeat(3)
    var loss = y1.sum() + y2.sum()
    loss.backward()

    # Each element appears 2 times in y1 and 3 times in y2 = 5 times total
    assert_true(x.grad().all_close(Tensor.d1([5.0, 5.0, 5.0])))

fn test_repeat_in_computational_graph() raises:
    print("test_repeat_in_computational_graph")
    var a = Tensor.d1([1.0, 2.0], requires_grad=True)
    var b = Tensor.d1([3.0, 4.0], requires_grad=True)

    var c = a * b  # [3.0, 8.0]
    var d = c.repeat(2, 2)  # Shape: (2,) → (2, 4)
    var loss = d.sum()
    loss.backward()

    # d contains each element of c 4 times (2×2)
    assert_true(a.grad().all_close(Tensor.d1([12.0, 16.0])))  # 4*3, 4*4
    assert_true(b.grad().all_close(Tensor.d1([4.0, 8.0])))    # 4*1, 4*2

fn test_repeat_strict_validation() raises:
    print("test_repeat_strict_validation")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]])

    # These should all PANIC due to strict PyTorch rules:
    # x.repeat()           # Empty repeat list for 2D tensor
    # x.repeat(3)          # Too few dimensions (1 vs rank 2)

    # These should work:
    var y1 = x.repeat(2, 3)     # Exact match: 2 repeat dims for rank 2
    var y2 = x.repeat(1, 1, 2)  # More dims: 3 repeat dims for rank 2
    assert_true(y1.shape() == Shape(4, 6))
    assert_true(y2.shape() == Shape(1, 2, 4))

# Consolidated test function
fn main() raises:
    print("Running comprehensive PyTorch-compatible repeat functionality tests...")

    test_repeat_scalar_to_1d()
    test_repeat_scalar_to_2d()
    test_repeat_scalar_to_3d()
    test_repeat_1d_to_longer_1d()
    test_repeat_1d_to_2d()
    test_repeat_1d_to_3d()
    test_repeat_2d_same_rank()
    test_repeat_2d_to_3d()
    test_repeat_3d_same_rank()
    test_repeat_3d_all_dims()
    test_repeat_identity_operation()
    test_repeat_complex_pattern()
    test_repeat_gradient_accumulation()
    test_repeat_in_computational_graph()
    test_repeat_strict_validation()

    print("✓ All PyTorch-compatible repeat functionality tests passed!")
