from tenmo import Tensor
from testing import assert_true
from math import exp
from shapes import Shape

fn main() raises:
    test_tensor_softmax_backward_1d()
    test_softmax_1d_basic()
    test_softmax_1d_with_grad_validation()
    test_softmax_2d_axis_0()
    test_softmax_2d_axis_1()
    test_softmax_2d_multiple_axes()
    test_softmax_3d_axis_2()
    test_softmax_gradient_validation_2d()
    test_softmax_numerical_stability()
    test_softmax_negative_values()
    print("passes")




fn test_tensor_softmax_backward_1d() raises:
    print("test_tensor_softmax_backward_1d")
    alias dtype = DType.float32

    var t = Tensor[dtype].d1([1.0, 2.0, 3.0])
    t.requires_grad_(True)
    var s = t.softmax(axes=[0])
    var loss = s.sum()
    loss.backward()
    assert_true(t.grad() == Tensor[dtype].zeros(Shape(3)))
    print("✓ Passed 1D softmax backward test")


fn test_softmax_1d_basic() raises:
    print("test_softmax_1d_basic")
    # Test basic 1D softmax
    var input_data = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # Expected softmax output
    var exp_sum = exp(1.0) + exp(2.0) + exp(3.0)
    var expected = Tensor.d1(
        [exp(1.0) / exp_sum, exp(2.0) / exp_sum, exp(3.0) / exp_sum]
    )
    assert_true(output.all_close(expected))

    # Test backward pass
    s = output.sum()
    s.backward()
    assert_true(input_data.grad().all_close(Tensor.d1([0.0, 0.0, 0.0])))

fn test_softmax_1d_with_grad_validation() raises:
    print("test_softmax_1d_with_grad_validation")
    var input_data = Tensor.d1([0.1, 0.2, 0.3], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # Create a loss and compute gradients
    var target = Tensor.d1([0.0, 1.0, 0.0])
    mse_loss = output.mse(target)
    mse_loss.backward()

    assert_true(
        input_data.grad().all_close(
            Tensor.d1(
                [0.05957773470833636, -0.1486374678358427, 0.08905973312750633]
            )
        )
    )


fn test_softmax_2d_axis_0() raises:
    print("test_softmax_2d_axis_0")
    var input_data = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # Softmax along axis 0 (rows)
    var col1_exp_sum = exp(1.0) + exp(3.0)
    var col2_exp_sum = exp(2.0) + exp(4.0)
    var expected = Tensor.d2(
        [
            [exp(1.0) / col1_exp_sum, exp(2.0) / col2_exp_sum],
            [exp(3.0) / col1_exp_sum, exp(4.0) / col2_exp_sum],
        ]
    )
    assert_true(output.all_close(expected))

    target = Tensor.d2([[0, 1.5], [2.5, 0.0]]).float64()
    # Test backward
    loss = output.mse(target)
    loss.backward()

    assert_true(
        input_data.grad().all_close(
            Tensor.d2(
                [
                    [0.09126073122623839, -0.11872643958063916],
                    [-0.09126073122623841, 0.11872643958063919],
                ]
            )
        )
    )


fn test_softmax_2d_axis_1() raises:
    print("test_softmax_2d_axis_1")
    var input_data = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var output = input_data.softmax(axes=[1])

    # Softmax along axis 1 (columns)
    var row1_exp_sum = exp(1.0) + exp(2.0)
    var row2_exp_sum = exp(3.0) + exp(4.0)
    var expected = Tensor.d2(
        [
            [exp(1.0) / row1_exp_sum, exp(2.0) / row1_exp_sum],
            [exp(3.0) / row2_exp_sum, exp(4.0) / row2_exp_sum],
        ]
    )
    assert_true(output.all_close(expected))

    var target = Tensor.d2([[0.0, 1.9], [2.9, 0.0]])
    loss = output.mse(target)
    loss.backward()
    assert_true(
        input_data.grad().all_close(
            Tensor.d2(
                [
                    [0.14135246274290422, -0.1413524627429042],
                    [-0.3305161770365907, 0.3305161770365907],
                ]
            ).float64()
        )
    )


fn test_softmax_2d_multiple_axes() raises:
    print("test_softmax_2d_multiple_axes")
    var input_data = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var output = input_data.softmax(axes=[0, 1])  # Softmax over entire matrix

    var total_exp_sum = exp(1.0) + exp(2.0) + exp(3.0) + exp(4.0)
    var expected = Tensor.d2(
        [
            [exp(1.0) / total_exp_sum, exp(2.0) / total_exp_sum],
            [exp(3.0) / total_exp_sum, exp(4.0) / total_exp_sum],
        ]
    )
    assert_true(output.all_close(expected))
    var target = Tensor.d2([[0.0, 1.9], [2.9, 0.0]])
    loss = output.mse(target)
    loss.backward()

    assert_true(
        input_data.grad().all_close(
            Tensor.d2(
                [
                    [0.0064955867864787, -0.06273006370483511],
                    [-0.2712241624609286, 0.3274586393792851],
                ]
            ).float64()
        )
    )


fn test_softmax_3d_axis_2() raises:
    print("test_softmax_3d_axis_2")
    var input_data = Tensor.d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    var output = input_data.softmax(axes=[2])  # Softmax along last dimension

    # For each 2-element vector along axis 2
    var slice1_exp_sum = exp(1.0) + exp(2.0)
    var slice2_exp_sum = exp(3.0) + exp(4.0)
    var slice3_exp_sum = exp(5.0) + exp(6.0)
    var slice4_exp_sum = exp(7.0) + exp(8.0)

    var expected = Tensor.d3(
        [
            [
                [exp(1.0) / slice1_exp_sum, exp(2.0) / slice1_exp_sum],
                [exp(3.0) / slice2_exp_sum, exp(4.0) / slice2_exp_sum],
            ],
            [
                [exp(5.0) / slice3_exp_sum, exp(6.0) / slice3_exp_sum],
                [exp(7.0) / slice4_exp_sum, exp(8.0) / slice4_exp_sum],
            ],
        ]
    )
    assert_true(output.all_close(expected))
    var target = Tensor.d3([[[1.0, 1.0], [3.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]])
    loss = output.mse(target)
    loss.backward()

    expected = Tensor.d3(
        [
            [
                [-0.022714436918239593, 0.0227144369182396],
                [-0.07186742022860364, 0.07186742022860365],
            ],
            [
                [0.026438546392124458, -0.026438546392124486],
                [0.02643854639212449, -0.026438546392124406],
            ],
        ]
    ).float64()

    assert_true(input_data.grad() == expected)


fn test_softmax_gradient_validation_1d() raises:
    print("test_softmax_gradient_validation_1d")
    # This test validates gradients against known mathematical properties
    var input_data = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # For softmax, the gradient should satisfy: sum(grad) = 0 when output is used in loss
    output.sum().backward()
    var grad_sum = input_data.grad().sum().item()
    assert_true(abs(grad_sum) < 1e-6)  # Should be very close to 0


fn test_softmax_gradient_validation_2d() raises:
    print("test_softmax_gradient_validation_2d")
    var input_data = Tensor.d2([[0.5, 1.5], [2.5, 0.5]], requires_grad=True)
    var output = input_data.softmax(axes=[1])

    # Use cross entropy-like loss to get meaningful gradients
    var target = Tensor.d2([[1.0, 0.0], [0.0, 1.0]])
    var loss = (output * target).sum()
    loss.backward()
    expected = Tensor.d2(
        [
            [0.1966119332414562, -0.19661193324145623],
            [-0.10499358540343878, 0.10499358540343878],
        ]
    ).float64()

    assert_true(input_data.grad() == expected)


fn test_softmax_numerical_stability() raises:
    print("test_softmax_numerical_stability")
    # Test with large values to ensure numerical stability
    var input_data = Tensor.d1([1000.0, 1001.0, 1002.0], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # Output should be valid probabilities (sum to 1, all between 0 and 1)
    var output_sum = output.sum().item()
    assert_true(abs(output_sum - 1.0) < 1e-6)

    # All values should be between 0 and 1
    for i in range(output.numels()):
        var val = output.element_at(i)
        assert_true(val >= 0.0 and val <= 1.0)


fn test_softmax_negative_values() raises:
    print("test_softmax_negative_values")
    var input_data = Tensor.d1([-2.0, -1.0, 0.0], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # Should still produce valid probabilities
    var output_sum = output.sum().item()
    assert_true(abs(output_sum - 1.0) < 1e-6)

    output.sum().backward()

