from std.testing import assert_true, TestSuite
from tenmo.tensor import Tensor


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_tensor_linspace_basic() raises:
    print("test_tensor_linspace_basic")

    # Basic linspace: 5 points from 0 to 1
    var x = Tensor[DType.float32].linspace(0.0, 1.0, 5)
    var expected = Tensor[DType.float32].d1([0.0, 0.25, 0.5, 0.75, 1.0])
    assert_true(x.all_close(expected))

    # Negative to positive range
    var y = Tensor[DType.float32].linspace(-2.0, 2.0, 5)
    var expected_y = Tensor[DType.float32].d1([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert_true(y.all_close(expected_y))

    print("Passed linspace basic test")


def test_tensor_linspace_edge_cases() raises:
    print("test_tensor_linspace_edge_cases")

    # Single point
    var single = Tensor[DType.float32].linspace(3.0, 7.0, 1)
    assert_true(single == Tensor[DType.float32].d1([3.0]))

    # Two points
    var two_points = Tensor[DType.float32].linspace(0.0, 1.0, 2)
    assert_true(two_points == Tensor[DType.float32].d1([0.0, 1.0]))

    # Same start and end
    var same = Tensor[DType.float32].linspace(5.0, 5.0, 4)
    var expected_same = Tensor[DType.float32].d1([5.0, 5.0, 5.0, 5.0])
    assert_true(same.all_close(expected_same))

    print("Passed linspace edge cases test")


def test_tensor_linspace_precision() raises:
    print("test_tensor_linspace_precision")
    comptime dtype = DType.float32
    # Test with many points for precision
    var many_points = Tensor[DType.float32].linspace(0.0, 1.0, 11).float()
    # Should be exactly [0.0, 0.1, 0.2, ..., 1.0]
    for i in range(11):
        var expected_val = Scalar[dtype](i) / Scalar[dtype](10)
        assert_true(abs(many_points.get(i) - expected_val) < 1e-6)

    print("Passed linspace precision test")


def test_tensor_linspace_with_gradients() raises:
    print("test_tensor_linspace_with_gradients")

    # Linspace with requires_grad = True
    var x = Tensor[DType.float32].linspace(0.0, 2.0, 3, requires_grad=True)
    # x = [0.0, 1.0, 2.0]

    var y = x.sum()
    y.backward()

    # Gradient should be [1.0, 1.0, 1.0] for sum()
    var expected_grad = Tensor[DType.float32].d1([1.0, 1.0, 1.0])
    assert_true(x.grad().all_close(expected_grad))

    print("Passed linspace with gradients test")
