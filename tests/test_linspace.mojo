from testing import assert_true
from tenmo import Tensor

fn main() raises:
    test_tensor_linspace_basic()
    test_tensor_linspace_edge_cases()
    test_tensor_linspace_precision()
    test_tensor_linspace_with_gradients()


fn test_tensor_linspace_basic() raises:
    print("test_tensor_linspace_basic")

    # Basic linspace: 5 points from 0 to 1
    var x = Tensor.linspace(0.0, 1.0, 5)
    var expected = Tensor.d1([0.0, 0.25, 0.5, 0.75, 1.0])
    assert_true(x.all_close(expected))

    # Negative to positive range
    var y = Tensor.linspace(-2.0, 2.0, 5)
    var expected_y = Tensor.d1([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert_true(y.all_close(expected_y))

    print("✓ Passed linspace basic test")

fn test_tensor_linspace_edge_cases() raises:
    print("test_tensor_linspace_edge_cases")

    # Single point
    var single = Tensor.linspace(3.0, 7.0, 1)
    assert_true(single == Tensor.d1([3.0]))

    # Two points
    var two_points = Tensor.linspace(0.0, 1.0, 2)
    assert_true(two_points == Tensor.d1([0.0, 1.0]))

    # Same start and end
    var same = Tensor.linspace(5.0, 5.0, 4)
    var expected_same = Tensor.d1([5.0, 5.0, 5.0, 5.0])
    assert_true(same.all_close(expected_same))

    print("✓ Passed linspace edge cases test")

fn test_tensor_linspace_precision() raises:
    print("test_tensor_linspace_precision")
    alias dtype = DType.float32
    # Test with many points for precision
    var many_points = Tensor.linspace(0.0, 1.0, 11).float()
    # Should be exactly [0.0, 0.1, 0.2, ..., 1.0]
    for i in range(11):
        var expected_val = Scalar[dtype](i) / Scalar[dtype](10)
        assert_true(abs(many_points.element_at(i) - expected_val) < 1e-6)

    print("✓ Passed linspace precision test")

fn test_tensor_linspace_with_gradients() raises:
    print("test_tensor_linspace_with_gradients")

    # Linspace with requires_grad = True
    var x = Tensor.linspace(0.0, 2.0, 3, requires_grad=True)
    # x = [0.0, 1.0, 2.0]

    var y = x.sum()
    y.backward()

    # Gradient should be [1.0, 1.0, 1.0] for sum()
    var expected_grad = Tensor.d1([1.0, 1.0, 1.0])
    assert_true(x.grad().all_close(expected_grad))

    print("✓ Passed linspace with gradients test")
