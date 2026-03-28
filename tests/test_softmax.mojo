from tenmo import Tensor
from testing import assert_true
from shapes import Shape
from sys import has_accelerator
from shapes import Shape
from math import log, exp


# Old tests
fn test_tensor_softmax_backward_1d() raises:
    print("test_tensor_softmax_backward_1d")
    comptime dtype = DType.float64

    var t = Tensor[dtype].d1([1.0, 2.0, 3.0])
    t.requires_grad_(True)
    var s = t.softmax(axes=[0])
    var loss = s.sum()
    loss.backward()
    assert_true(t.grad().all_close(Tensor[dtype].zeros(Shape(3))))
    print("Passed 1D softmax backward test")


fn test_softmax_1d_basic() raises:
    print("test_softmax_1d_basic")
    comptime dtype = DType.float64
    # Test basic 1D softmax
    var input_data = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # Expected softmax output
    var exp_sum = exp(1.0) + exp(2.0) + exp(3.0)
    var expected = Tensor[dtype].d1(
        [exp(1.0) / exp_sum, exp(2.0) / exp_sum, exp(3.0) / exp_sum]
    )
    assert_true(output.all_close(expected))

    # Test backward pass
    s = output.sum()
    s.backward()
    assert_true(input_data.grad().all_close(Tensor[dtype].d1([0.0, 0.0, 0.0])))


fn test_softmax_1d_with_grad_validation() raises:
    print("test_softmax_1d_with_grad_validation")
    comptime dtype = DType.float64
    var input_data = Tensor[dtype].d1([0.1, 0.2, 0.3], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # Create a loss and compute gradients
    var target = Tensor[dtype].d1([0.0, 1.0, 0.0])
    mse_loss = output.mse(target)
    mse_loss.backward()

    assert_true(
        input_data.grad().all_close(
            Tensor[dtype].d1(
                [0.05957773470833636, -0.1486374678358427, 0.08905973312750633]
            )
        )
    )


fn test_softmax_2d_axis_0() raises:
    print("test_softmax_2d_axis_0")
    comptime dtype = DType.float64
    var input_data = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var output = input_data.softmax(axes=[0])

    # Softmax along axis 0 (rows)
    var col1_exp_sum = exp(1.0) + exp(3.0)
    var col2_exp_sum = exp(2.0) + exp(4.0)
    var expected = Tensor[dtype].d2(
        [
            [exp(1.0) / col1_exp_sum, exp(2.0) / col2_exp_sum],
            [exp(3.0) / col1_exp_sum, exp(4.0) / col2_exp_sum],
        ]
    )
    assert_true(output.all_close(expected))

    target = Tensor[dtype].d2([[0, 1.5], [2.5, 0.0]])
    # Test backward
    loss = output.mse(target)
    loss.backward()

    assert_true(
        input_data.grad().all_close(
            Tensor[dtype].d2(
                [
                    [0.09126073122623839, -0.11872643958063916],
                    [-0.09126073122623841, 0.11872643958063919],
                ]
            )
        )
    )


fn test_softmax_2d_axis_1() raises:
    print("test_softmax_2d_axis_1")
    comptime dtype = DType.float64
    var input_data = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var output = input_data.softmax(axes=[1])

    # Softmax along axis 1 (columns)
    var row1_exp_sum = exp(1.0) + exp(2.0)
    var row2_exp_sum = exp(3.0) + exp(4.0)
    var expected = Tensor[dtype].d2(
        [
            [exp(1.0) / row1_exp_sum, exp(2.0) / row1_exp_sum],
            [exp(3.0) / row2_exp_sum, exp(4.0) / row2_exp_sum],
        ]
    )
    assert_true(output.all_close(expected))

    var target = Tensor[dtype].d2([[0.0, 1.9], [2.9, 0.0]])
    loss = output.mse(target)
    loss.backward()
    assert_true(
        input_data.grad().all_close(
            Tensor[dtype]
            .d2(
                [
                    [0.14135246274290422, -0.1413524627429042],
                    [-0.3305161770365907, 0.3305161770365907],
                ]
            )
            .float64()
        )
    )


fn test_softmax_2d_multiple_axes() raises:
    print("test_softmax_2d_multiple_axes")
    comptime dtype = DType.float64
    var input_data = Tensor[dtype].d2(
        [[1.0, 2.0], [3.0, 4.0]], requires_grad=True
    )
    var output = input_data.softmax(axes=[0, 1])  # Softmax over entire matrix

    var total_exp_sum = exp(1.0) + exp(2.0) + exp(3.0) + exp(4.0)
    var expected = Tensor[dtype].d2(
        [
            [exp(1.0) / total_exp_sum, exp(2.0) / total_exp_sum],
            [exp(3.0) / total_exp_sum, exp(4.0) / total_exp_sum],
        ]
    )
    assert_true(output.all_close(expected))
    var target = Tensor[dtype].d2([[0.0, 1.9], [2.9, 0.0]])
    loss = output.mse(target)
    loss.backward()

    assert_true(
        input_data.grad().all_close(
            Tensor[dtype]
            .d2(
                [
                    [0.0064955867864787, -0.06273006370483511],
                    [-0.2712241624609286, 0.3274586393792851],
                ]
            )
            .float64()
        )
    )


fn test_softmax_3d_axis_2() raises:
    print("test_softmax_3d_axis_2")
    comptime dtype = DType.float64
    var input_data = Tensor[dtype].d3(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True
    )

    var output = input_data.softmax(axes=[2])  # Softmax along last dimension

    # For each 2-element vector along axis 2
    var slice1_exp_sum = exp(1.0) + exp(2.0)
    var slice2_exp_sum = exp(3.0) + exp(4.0)
    var slice3_exp_sum = exp(5.0) + exp(6.0)
    var slice4_exp_sum = exp(7.0) + exp(8.0)

    var expected = Tensor[dtype].d3(
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
    var target = Tensor[dtype].d3(
        [[[1.0, 1.0], [3.0, 2.0]], [[3.0, 4.0], [5.0, 6.0]]]
    )
    loss = output.mse(target)
    loss.backward()

    expected = (
        Tensor[dtype]
        .d3(
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
        )
        .float64()
    )

    assert_true(input_data.grad() == expected)


fn test_softmax_gradient_validation_1d() raises:
    print("test_softmax_gradient_validation_1d")
    comptime dtype = DType.float64
    # This test validates gradients against known mathematical properties
    var input_data = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # For softmax, the gradient should satisfy: sum(grad) = 0 when output is used in loss
    s = output.sum()
    s.backward()
    var grad_sum = input_data.grad().sum().item()
    assert_true(abs(grad_sum) < 1e-6)  # Should be very close to 0


fn test_softmax_gradient_validation_2d() raises:
    print("test_softmax_gradient_validation_2d")
    comptime dtype = DType.float64
    var input_data = Tensor[dtype].d2(
        [[0.5, 1.5], [2.5, 0.5]], requires_grad=True
    )
    var output = input_data.softmax(axes=[1])

    # Use cross entropy-like loss to get meaningful gradients
    var target = Tensor[dtype].d2([[1.0, 0.0], [0.0, 1.0]])
    var loss = (output * target).sum()
    loss.backward()
    expected = (
        Tensor[dtype]
        .d2(
            [
                [0.1966119332414562, -0.19661193324145623],
                [-0.10499358540343878, 0.10499358540343878],
            ]
        )
        .float64()
    )

    assert_true(input_data.grad() == expected)


fn test_softmax_numerical_stability() raises:
    print("test_softmax_numerical_stability")
    comptime dtype = DType.float64
    # Test with large values to ensure numerical stability
    var input_data = Tensor[dtype].d1(
        [1000.0, 1001.0, 1002.0], requires_grad=True
    )
    var output = input_data.softmax(axes=[0])

    # Output should be valid probabilities (sum to 1, all between 0 and 1)
    var output_sum = output.sum().item()
    assert_true(abs(output_sum - 1.0) < 1e-6)

    # All values should be between 0 and 1
    for i in range(output.numels()):
        var val = output.get(i)
        assert_true(val >= 0.0 and val <= 1.0)


fn test_softmax_negative_values() raises:
    print("test_softmax_negative_values")
    comptime dtype = DType.float64
    var input_data = Tensor[dtype].d1([-2.0, -1.0, 0.0], requires_grad=True)
    var output = input_data.softmax(axes=[0])

    # Should still produce valid probabilities
    var output_sum = output.sum().item()
    assert_true(abs(output_sum - 1.0) < 1e-6)

    s = output.sum()
    s.backward()


# End of old tests


# ── Helper: verify softmax properties ────────────────────────────────────────


fn verify_softmax_properties_1d(result: Tensor[DType.float32]) raises:
    """Verify 1D softmax: all in [0,1] and sum == 1."""
    comptime dtype = DType.float32
    var total = Scalar[dtype](0)
    for i in range(result.shape()[0]):
        var val = result[[i]]
        assert_true(val >= Scalar[dtype](0) and val <= Scalar[dtype](1))
        total += val
    assert_true(abs(total - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


fn verify_softmax_properties_2d_axis1(
    result: Tensor[DType.float32],
) raises:
    """Verify 2D softmax along axis=1: each row sums to 1."""
    comptime dtype = DType.float32
    for i in range(result.shape()[0]):
        var row_sum = Scalar[dtype](0)
        for j in range(result.shape()[1]):
            var val = result[[i, j]]
            assert_true(val >= Scalar[dtype](0) and val <= Scalar[dtype](1))
            row_sum += val
        assert_true(abs(row_sum - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP A: CPU Softmax Forward (11 tests)
# ═════════════════════════════════════════════════════════════════════════════


fn test_softmax_cpu_1d_basic() raises:
    print("test_softmax_cpu_1d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.softmax()
    verify_softmax_properties_1d(result)
    assert_true(
        result.all_close[atol=1e-5](
            Tensor[dtype].d1([0.09003057, 0.24472848, 0.66524094]),
        )
    )


fn test_softmax_cpu_1d_uniform() raises:
    print("test_softmax_cpu_1d_uniform")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0])
    var result = a.softmax()
    assert_true(result.all_close[atol=1e-5](Tensor[dtype].full(Shape(4), 0.25)))


fn test_softmax_cpu_1d_large_values() raises:
    print("test_softmax_cpu_1d_large_values")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1000.0, 1001.0, 1002.0])
    var result = a.softmax()
    verify_softmax_properties_1d(result)


fn test_softmax_cpu_1d_negative_values() raises:
    print("test_softmax_cpu_1d_negative_values")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([-1.0, -2.0, -3.0])
    var result = a.softmax()
    verify_softmax_properties_1d(result)
    assert_true(
        result.all_close[atol=1e-5](
            Tensor[dtype].d1([0.66524094, 0.24472848, 0.09003057]),
        )
    )


fn test_softmax_cpu_2d_axis0() raises:
    print("test_softmax_cpu_2d_axis0")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.softmax(axes=[0])
    for j in range(2):
        var col_sum = result[[0, j]] + result[[1, j]]
        assert_true(abs(col_sum - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


fn test_softmax_cpu_2d_axis1() raises:
    print("test_softmax_cpu_2d_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.softmax(axes=[1])
    verify_softmax_properties_2d_axis1(result)


fn test_softmax_cpu_2d_default_axis() raises:
    print("test_softmax_cpu_2d_default_axis")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0], [3.0, 4.0]])
    var result = a.softmax()
    var total = Scalar[dtype](0)
    for i in range(2):
        for j in range(2):
            total += result[[i, j]]
    assert_true(abs(total - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


fn test_softmax_cpu_3d_axis2() raises:
    print("test_softmax_cpu_3d_axis2")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d3(
        [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]]
    )
    var result = a.softmax(axes=[2])
    for i in range(2):
        for j in range(2):
            var slice_sum = Scalar[dtype](0)
            for k in range(3):
                slice_sum += result[[i, j, k]]
            assert_true(
                abs(slice_sum - Scalar[dtype](1.0)) < Scalar[dtype](1e-5)
            )


fn test_softmax_cpu_no_grad() raises:
    print("test_softmax_cpu_no_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=False)
    var result = a.softmax()
    assert_true(not result.requires_grad)


fn test_softmax_cpu_requires_grad_propagates() raises:
    print("test_softmax_cpu_requires_grad_propagates")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var result = a.softmax()
    assert_true(result.requires_grad)


fn test_softmax_cpu_suppress_grad() raises:
    print("test_softmax_cpu_suppress_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var result = a.softmax(requires_grad=False)
    assert_true(not result.requires_grad)


# ═════════════════════════════════════════════════════════════════════════════
# GROUP B: CPU Softmax Backward (5 tests)
# ═════════════════════════════════════════════════════════════════════════════


fn test_softmax_cpu_1d_backward_sum_grad() raises:
    print("test_softmax_cpu_1d_backward_sum_grad")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var result = a.softmax()
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close[atol=1e-5](Tensor[dtype].zeros(Shape(3))))


fn test_softmax_cpu_1d_backward_single_output() raises:
    print("test_softmax_cpu_1d_backward_single_output")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var s = a.softmax()
    var loss = (s * Tensor[dtype].d1([1.0, 0.0, 0.0])).sum()
    loss.backward()
    var s0 = s[[0]]
    var s1 = s[[1]]
    var s2 = s[[2]]
    var expected = Tensor[dtype].d1([s0 * (1.0 - s0), -s0 * s1, -s0 * s2])
    assert_true(a.grad().all_close[atol=1e-5](expected))


fn test_softmax_cpu_2d_backward_axis1() raises:
    print("test_softmax_cpu_2d_backward_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var result = a.softmax(axes=[1])
    var loss = result.sum()
    loss.backward()
    assert_true(a.grad().all_close[atol=1e-5](Tensor[dtype].zeros(Shape(2, 3))))


fn test_softmax_cpu_backward_chain() raises:
    print("test_softmax_cpu_backward_chain")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var s = a.softmax()
    var loss = (s * 2.0).sum()
    loss.backward()
    assert_true(a.grad().all_close[atol=1e-5](Tensor[dtype].zeros(Shape(3))))


fn test_softmax_cpu_backward_finite_difference() raises:
    print("test_softmax_cpu_backward_finite_difference")
    comptime dtype = DType.float32
    var eps = Scalar[dtype](1e-3)
    var x_plus = Tensor[dtype].d1([1.0 + eps, 2.0, 3.0])
    var x_minus = Tensor[dtype].d1([1.0 - eps, 2.0, 3.0])
    var numerical_grad = (x_plus.softmax()[[0]] - x_minus.softmax()[[0]]) / (
        2.0 * eps
    )
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var loss = (a.softmax() * Tensor[dtype].d1([1.0, 0.0, 0.0])).sum()
    loss.backward()
    assert_true(abs(a.grad()[[0]] - numerical_grad) < Scalar[dtype](1e-4))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP C: CPU LogSoftmax Forward (4 tests)
# ═════════════════════════════════════════════════════════════════════════════


fn test_log_softmax_cpu_1d_basic() raises:
    print("test_log_softmax_cpu_1d_basic")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0])
    var result = a.softmax[log=True]()
    var softmax_result = a.softmax()
    for i in range(3):
        assert_true(
            abs(result[[i]] - log(softmax_result[[i]])) < Scalar[dtype](1e-5)
        )


fn test_log_softmax_cpu_1d_uniform() raises:
    print("test_log_softmax_cpu_1d_uniform")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([0.0, 0.0, 0.0, 0.0])
    var result = a.softmax[log=True]()
    var expected = log(Float32(0.25))
    for i in range(4):
        assert_true(abs(result[[i]] - expected) < Scalar[dtype](1e-5))


fn test_log_softmax_cpu_1d_numerical_stability() raises:
    print("test_log_softmax_cpu_1d_numerical_stability")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1000.0, 1001.0, 1002.0])
    var result = a.softmax[log=True]()
    for i in range(3):
        assert_true(result[[i]] < Scalar[dtype](0))


fn test_log_softmax_cpu_2d_axis1() raises:
    print("test_log_softmax_cpu_2d_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    var result = a.softmax[log=True](axes=[1])
    for i in range(2):
        var row_sum = Scalar[dtype](0)
        for j in range(3):
            row_sum += exp(result[[i, j]])
        assert_true(abs(row_sum - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP D: CPU LogSoftmax Backward (3 tests)
# ═════════════════════════════════════════════════════════════════════════════


fn test_log_softmax_cpu_1d_backward_sum() raises:
    print("test_log_softmax_cpu_1d_backward_sum")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var result = a.softmax[log=True]()
    var loss = result.sum()
    loss.backward()
    # grad_i = 1 - n * softmax_i
    var s = Tensor[dtype].d1([1.0, 2.0, 3.0]).softmax()
    var expected_grad = Tensor[dtype].d1(
        [
            1.0 - s[[0]] * 3.0,
            1.0 - s[[1]] * 3.0,
            1.0 - s[[2]] * 3.0,
        ]
    )
    assert_true(a.grad().all_close[atol=1e-5](expected_grad))


fn test_log_softmax_cpu_2d_backward_axis1() raises:
    print("test_log_softmax_cpu_2d_backward_axis1")
    comptime dtype = DType.float32
    var a = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var result = a.softmax[log=True](axes=[1])
    var loss = result.sum()
    loss.backward()
    var s = (
        Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).softmax(axes=[1])
    )
    var n = Scalar[dtype](3.0)
    var expected = Tensor[dtype].d2(
        [
            [1.0 - s[[0, 0]] * n, 1.0 - s[[0, 1]] * n, 1.0 - s[[0, 2]] * n],
            [1.0 - s[[1, 0]] * n, 1.0 - s[[1, 1]] * n, 1.0 - s[[1, 2]] * n],
        ]
    )
    assert_true(a.grad().all_close[atol=1e-5](expected))


fn test_log_softmax_cpu_backward_finite_difference() raises:
    print("test_log_softmax_cpu_backward_finite_difference")
    comptime dtype = DType.float32
    var eps = Scalar[dtype](1e-3)
    var x_plus = Tensor[dtype].d1([1.0 + eps, 2.0, 3.0])
    var x_minus = Tensor[dtype].d1([1.0 - eps, 2.0, 3.0])
    var numerical_grad = (
        x_plus.softmax[log=True]().sum() - x_minus.softmax[log=True]().sum()
    ) / (2.0 * eps)
    var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
    var loss = a.softmax[log=True]().sum()
    loss.backward()
    assert_true(abs(a.grad()[[0]] - numerical_grad.item()) < Float32(1e-3))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP E: GPU Softmax Forward (5 tests)
# ═════════════════════════════════════════════════════════════════════════════


fn test_softmax_gpu_1d_basic() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_gpu_1d_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.softmax()
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close[atol=1e-5](
                Tensor[dtype].d1([0.09003057, 0.24472848, 0.66524094]),
            )
        )


fn test_softmax_gpu_1d_uniform() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_gpu_1d_uniform")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 1.0, 1.0, 1.0]).to_gpu()
        var result = a.softmax()
        assert_true(result.is_on_gpu())
        assert_true(
            result.to_cpu().all_close[atol=1e-5](
                Tensor[dtype].full(Shape(4), 0.25)
            )
        )


fn test_softmax_gpu_1d_numerical_stability() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_gpu_1d_numerical_stability")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1000.0, 1001.0, 1002.0]).to_gpu()
        var result = a.softmax()
        assert_true(result.is_on_gpu())
        var result_cpu = result.to_cpu()
        var total = result_cpu[[0]] + result_cpu[[1]] + result_cpu[[2]]
        assert_true(abs(total - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


fn test_softmax_gpu_2d_axis1() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_gpu_2d_axis1")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.softmax(axes=[1])
        assert_true(result.is_on_gpu())
        verify_softmax_properties_2d_axis1(result.to_cpu())


fn test_softmax_gpu_3d_axis2() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_gpu_3d_axis2")
        comptime dtype = DType.float32
        var a = (
            Tensor[dtype]
            .d3(
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]],
                ]
            )
            .to_gpu()
        )
        var result = a.softmax(axes=[2])
        assert_true(result.is_on_gpu())
        var result_cpu = result.to_cpu()
        for i in range(2):
            for j in range(2):
                var slice_sum = Scalar[dtype](0)
                for k in range(3):
                    slice_sum += result_cpu[[i, j, k]]
                assert_true(
                    abs(slice_sum - Scalar[dtype](1.0)) < Scalar[dtype](1e-5)
                )


# ═════════════════════════════════════════════════════════════════════════════
# GROUP F: GPU Softmax Backward (3 tests)
# ═════════════════════════════════════════════════════════════════════════════


fn test_softmax_gpu_1d_backward_sum_grad() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_gpu_1d_backward_sum_grad")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var result = a_gpu.softmax()
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close[atol=1e-5](Tensor[dtype].zeros(Shape(3)))
        )


fn test_softmax_gpu_2d_backward_axis1() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_gpu_2d_backward_axis1")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = a.to_gpu()
        var result = a_gpu.softmax(axes=[1])
        var loss = result.sum()
        loss.backward()
        assert_true(
            a.grad().all_close[atol=1e-5](Tensor[dtype].zeros(Shape(2, 3)))
        )


fn test_softmax_gpu_backward_chain() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_gpu_backward_chain")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a.to_gpu()
        var loss = (a_gpu.softmax() * 2.0).sum()
        loss.backward()
        assert_true(
            a.grad().all_close[atol=1e-5](Tensor[dtype].zeros(Shape(3)))
        )


# ═════════════════════════════════════════════════════════════════════════════
# GROUP G: GPU LogSoftmax Forward (3 tests)
# ═════════════════════════════════════════════════════════════════════════════


fn test_log_softmax_gpu_1d_basic() raises:
    @parameter
    if has_accelerator():
        print("test_log_softmax_gpu_1d_basic")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1.0, 2.0, 3.0]).to_gpu()
        var result = a.softmax[log=True]()
        assert_true(result.is_on_gpu())
        var cpu_result = Tensor[dtype].d1([1.0, 2.0, 3.0]).softmax[log=True]()
        assert_true(result.to_cpu().all_close[atol=1e-5](cpu_result))


fn test_log_softmax_gpu_1d_numerical_stability() raises:
    @parameter
    if has_accelerator():
        print("test_log_softmax_gpu_1d_numerical_stability")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d1([1000.0, 1001.0, 1002.0]).to_gpu()
        var result = a.softmax[log=True]()
        assert_true(result.is_on_gpu())
        var result_cpu = result.to_cpu()
        for i in range(3):
            assert_true(result_cpu[[i]] < Scalar[dtype](0))


fn test_log_softmax_gpu_2d_axis1() raises:
    @parameter
    if has_accelerator():
        print("test_log_softmax_gpu_2d_axis1")
        comptime dtype = DType.float32
        var a = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).to_gpu()
        var result = a.softmax[log=True](axes=[1])
        assert_true(result.is_on_gpu())
        var result_cpu = result.to_cpu()
        for i in range(2):
            var row_sum = Scalar[dtype](0)
            for j in range(3):
                row_sum += exp(result_cpu[[i, j]])
            assert_true(abs(row_sum - Scalar[dtype](1.0)) < Scalar[dtype](1e-5))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP H: GPU LogSoftmax Backward (2 tests)
# ═════════════════════════════════════════════════════════════════════════════


fn test_log_softmax_gpu_1d_backward() raises:
    @parameter
    if has_accelerator():
        print("test_log_softmax_gpu_1d_backward")
        comptime dtype = DType.float32
        var a_gpu = (
            Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True).to_gpu()
        )
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var loss_gpu = a_gpu.softmax[log=True]().sum()
        loss_gpu.backward()
        var loss_cpu = a_cpu.softmax[log=True]().sum()
        loss_cpu.backward()
        assert_true(a_gpu.grad().to_cpu().all_close[atol=1e-5](a_cpu.grad()))


fn test_log_softmax_gpu_2d_backward_axis1() raises:
    @parameter
    if has_accelerator():
        print("test_log_softmax_gpu_2d_backward_axis1")
        comptime dtype = DType.float32
        var a_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var loss_gpu = a_gpu.softmax[log=True](axes=[1]).sum()
        loss_gpu.backward()
        var loss_cpu = a_cpu.softmax[log=True](axes=[1]).sum()
        loss_cpu.backward()
        assert_true(a_gpu.grad().to_cpu().all_close[atol=1e-5](a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# GROUP I: Parity Tests (8 tests)
# ═════════════════════════════════════════════════════════════════════════════


fn test_softmax_parity_1d_forward() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_parity_1d_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.softmax().all_close[atol=1e-5](a_gpu.softmax().to_cpu())
        )


fn test_softmax_parity_2d_axis1_forward() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_parity_2d_axis1_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.softmax(axes=[1]).all_close[atol=1e-5](
                a_gpu.softmax(axes=[1]).to_cpu()
            )
        )


fn test_softmax_parity_1d_backward() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_parity_1d_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True).to_gpu()
        )
        var loss_cpu = (
            a_cpu.softmax() * Tensor[dtype].d1([1.0, 0.0, 0.0])
        ).sum()
        loss_cpu.backward()
        var loss_gpu = (
            a_gpu.softmax() * Tensor[dtype].d1([1.0, 0.0, 0.0]).to_gpu()
        ).sum()
        loss_gpu.backward()
        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu.grad().to_cpu()))


fn test_softmax_parity_2d_backward() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_parity_2d_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
        )
        var a_gpu = (
            Tensor[dtype]
            .d2([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
            .to_gpu()
        )
        var loss_cpu = a_cpu.softmax(axes=[1]).sum()
        loss_cpu.backward()
        var loss_gpu = a_gpu.softmax(axes=[1]).sum()
        loss_gpu.backward()
        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu.grad().to_cpu()))


fn test_log_softmax_parity_1d_forward() raises:
    @parameter
    if has_accelerator():
        print("test_log_softmax_parity_1d_forward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0, 4.0])
        var a_gpu = a_cpu.to_gpu()
        assert_true(
            a_cpu.softmax[log=True]().all_close[atol=1e-5](
                a_gpu.softmax[log=True]().to_cpu()
            )
        )


fn test_log_softmax_parity_1d_backward() raises:
    @parameter
    if has_accelerator():
        print("test_log_softmax_parity_1d_backward")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = (
            Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True).to_gpu()
        )
        var loss_cpu = a_cpu.softmax[log=True]().sum()
        loss_cpu.backward()
        var loss_gpu = a_gpu.softmax[log=True]().sum()
        loss_gpu.backward()
        assert_true(a_cpu.grad().all_close[atol=1e-5](a_gpu.grad().to_cpu()))


fn test_softmax_parity_using_zero_grad() raises:
    @parameter
    if has_accelerator():
        print("test_softmax_parity_using_zero_grad")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss_cpu = a_cpu.softmax().sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()
        a_cpu.zero_grad()
        var loss_gpu = a_gpu.softmax().sum()
        loss_gpu.backward()
        assert_true(cpu_grad.all_close[atol=1e-5](a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close[atol=1e-5](a_cpu.grad()))


fn test_log_softmax_parity_using_zero_grad() raises:
    @parameter
    if has_accelerator():
        print("test_log_softmax_parity_using_zero_grad")
        comptime dtype = DType.float32
        var a_cpu = Tensor[dtype].d1([1.0, 2.0, 3.0], requires_grad=True)
        var a_gpu = a_cpu.to_gpu()
        var loss_cpu = a_cpu.softmax[log=True]().sum()
        loss_cpu.backward()
        var cpu_grad = a_cpu.grad().copy()
        a_cpu.zero_grad()
        var loss_gpu = a_gpu.softmax[log=True]().sum()
        loss_gpu.backward()
        assert_true(cpu_grad.all_close[atol=1e-5](a_gpu.grad().to_cpu()))
        assert_true(cpu_grad.all_close[atol=1e-5](a_cpu.grad()))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — uncomment groups one at a time to binary search LLVM lowering issue
# Start with Group A only, then add B, then C, D, E, F, G, H, I
# ═════════════════════════════════════════════════════════════════════════════


fn main() raises:
    # ── Group A: CPU Softmax Forward ──────────────────────────────────────────
    test_softmax_cpu_1d_basic()
    test_softmax_cpu_1d_uniform()
    test_softmax_cpu_1d_large_values()
    test_softmax_cpu_1d_negative_values()
    test_softmax_cpu_2d_axis0()
    test_softmax_cpu_2d_axis1()
    test_softmax_cpu_2d_default_axis()
    test_softmax_cpu_3d_axis2()
    test_softmax_cpu_no_grad()
    test_softmax_cpu_requires_grad_propagates()
    test_softmax_cpu_suppress_grad()
    print("Group A passed!")

    # ── Group B: CPU Softmax Backward ─────────────────────────────────────────
    test_softmax_cpu_1d_backward_sum_grad()
    test_softmax_cpu_1d_backward_single_output()
    test_softmax_cpu_2d_backward_axis1()
    test_softmax_cpu_backward_chain()
    test_softmax_cpu_backward_finite_difference()
    print("Group B passed!")

    # ── Group C: CPU LogSoftmax Forward ───────────────────────────────────────
    test_log_softmax_cpu_1d_basic()
    test_log_softmax_cpu_1d_uniform()
    test_log_softmax_cpu_1d_numerical_stability()
    test_log_softmax_cpu_2d_axis1()
    print("Group C passed!")

    # ── Group D: CPU LogSoftmax Backward ──────────────────────────────────────
    test_log_softmax_cpu_1d_backward_sum()
    test_log_softmax_cpu_2d_backward_axis1()
    test_log_softmax_cpu_backward_finite_difference()
    print("Group D passed!")

    # ── Group E: GPU Softmax Forward ──────────────────────────────────────────
    test_softmax_gpu_1d_basic()
    test_softmax_gpu_1d_uniform()
    test_softmax_gpu_1d_numerical_stability()
    test_softmax_gpu_2d_axis1()
    test_softmax_gpu_3d_axis2()
    print("Group E passed!")

    # ── Group F: GPU Softmax Backward ─────────────────────────────────────────
    test_softmax_gpu_1d_backward_sum_grad()
    test_softmax_gpu_2d_backward_axis1()
    test_softmax_gpu_backward_chain()
    print("Group F passed!")

    # ── Group G: GPU LogSoftmax Forward ───────────────────────────────────────
    test_log_softmax_gpu_1d_basic()
    test_log_softmax_gpu_1d_numerical_stability()
    test_log_softmax_gpu_2d_axis1()
    print("Group G passed!")

    # ── Group H: GPU LogSoftmax Backward ──────────────────────────────────────
    test_log_softmax_gpu_1d_backward()
    test_log_softmax_gpu_2d_backward_axis1()
    print("Group H passed!")

    # ── Group I: Parity Tests ─────────────────────────────────────────────────
    test_softmax_parity_1d_forward()
    test_softmax_parity_2d_axis1_forward()
    test_softmax_parity_1d_backward()
    test_softmax_parity_2d_backward()
    test_log_softmax_parity_1d_forward()
    test_log_softmax_parity_1d_backward()
    test_softmax_parity_using_zero_grad()
    test_log_softmax_parity_using_zero_grad()
    print("Group I passed!")

    # Old tests

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

    # End of old tests

    print("All softmax tests passed!")
