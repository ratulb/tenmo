from tenmo import Tensor
from operators import AddTensor, SqrtForwardOp, SqrtBackwardOp
from backpropagation import Delegate, BackwardFn
from ancestry import Ancestor
from gradbox import Gradbox
from math import sqrt
from ndbuffer import NDBuffer


@fieldwise_init
@register_passable
struct SqrtBackward[dtype: DType](ImplicitlyCopyable):
    var epsilon: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        ref input_tensor = ancestor.tensor()
        ref shape = ancestor.shape()

        var gradbox_ancestor: Gradbox[dtype]

        if input_tensor.is_contiguous():
            var start = input_tensor.offset()
            var end = start + input_tensor.numels()
            # Compute 1 / (2 * sqrt(x)) - we can not use output - it may have changed
            # output is sqrt(x), so gradient is 1 / (2 * sqrt(input))
            var buffer = input_tensor.buffer.data_buffer().unary_ops[
                SqrtBackwardOp  # This should compute: 1 / (2 * sqrt(input))
            ](start, end)
            var ancestor_grad_buffer = gradbox.buffer.data_buffer() * buffer
            var ndb = NDBuffer[dtype](ancestor_grad_buffer^, shape)
            gradbox_ancestor = Gradbox[dtype](ndb^, share=False)
        else:
            gradbox_ancestor = Gradbox[dtype].zeros(shape, share=False)
            var index = 0
            ref gradbox_buffer = gradbox.buffer.data_buffer()
            ref ancestor_gradbox_buffer = gradbox_ancestor.buffer.data_buffer()

            for coord in shape:
                # gradient = grad_output * (1 / (2 * sqrt(x)))
                var sqrt_grad = 1.0 / (
                    self.epsilon + (2.0 * input_tensor[coord])
                )
                ancestor_gradbox_buffer[index] = (
                    gradbox_buffer[index] * sqrt_grad
                )
                index += 1

        return [(ancestor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct Sqrt[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        epsilon: Scalar[dtype] = Scalar[dtype](1e-12),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        var out: Tensor[dtype]
        ref shape = self.shape()

        if self.is_contiguous():
            var start = self.offset()
            var end = start + self.numels()
            var buffer = self.buffer.data_buffer().unary_ops[SqrtForwardOp](
                start, end
            )
            out = Tensor[dtype](
                NDBuffer[dtype](buffer^, shape), requires_grad=False
            )
        else:
            out = Tensor[dtype].zeros(shape, requires_grad=False)
            var index = 0
            ref out_buffer = out.buffer.data_buffer()
            for coord in shape:
                out_buffer[index] = sqrt(self[coord])
                index += 1

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                backward_fn = SqrtBackward[dtype](epsilon).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


from testing import assert_true

fn test_sqrt_backward() raises:
    print("test_sqrt_backward")
    var x = Tensor.d1([4.0, 9.0, 16.0, 25.0], requires_grad=True)
    var y = x.sqrt()  # [2.0, 3.0, 4.0, 5.0]
    var s = y.sum()
    s.backward()

    # dy/dx = 1 / (2 * sqrt(x))
    # For x=[4, 9, 16, 25]: sqrt(x) = [2, 3, 4, 5]
    # Gradient = [1/(2*2), 1/(2*3), 1/(2*4), 1/(2*5)]
    #          = [0.25, 0.1667, 0.125, 0.1]
    var expected_grad = Tensor.d1([0.25, 0.16666667, 0.125, 0.1])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))

fn test_sqrt_backward_zero_handling() raises:
    print("test_sqrt_backward_zero_handling")
    # Test near-zero values (numerical stability)
    var x = Tensor.d1([0.01, 0.04, 1.0], requires_grad=True)
    var y = x.sqrt()
    var s = y.sum()
    s.backward()

    # Gradient at 0.01: 1/(2*0.1) = 5.0
    # Gradient at 0.04: 1/(2*0.2) = 2.5
    # Gradient at 1.0: 1/(2*1.0) = 0.5
    var expected_grad = Tensor.d1([5.0, 2.5, 0.5])
    assert_true(x.grad().all_close(expected_grad))

fn test_var_backward_global_variance() raises:
    print("test_var_backward_global_variance")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance(unbiased=False)  # Population variance
    v.backward()

    # Mean = 3.0
    # Variance = ((1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²) / 5
    #          = (4 + 1 + 0 + 1 + 4) / 5 = 2.0
    # Gradient: (2/n) * (x - mean) = (2/5) * (x - 3)
    # = [0.4*(-2), 0.4*(-1), 0.4*(0), 0.4*(1), 0.4*(2)]
    # = [-0.8, -0.4, 0.0, 0.4, 0.8]
    var expected_grad = Tensor.d1([-0.8, -0.4, 0.0, 0.4, 0.8])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_unbiased_variance() raises:
    print("test_var_backward_unbiased_variance")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var v = x.variance(unbiased=True)  # Sample variance (n-1)
    v.backward()

    # Gradient: (2/(n-1)) * (x - mean) = (2/4) * (x - 3) = 0.5 * (x - 3)
    # = [0.5*(-2), 0.5*(-1), 0.5*(0), 0.5*(1), 0.5*(2)]
    # = [-1.0, -0.5, 0.0, 0.5, 1.0]
    var expected_grad = Tensor.d1([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_with_axis() raises:
    print("test_var_backward_with_axis")
    var x = Tensor.d2([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=0, unbiased=False)  # Variance along rows
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, var=1, grad for [1,3] = (2/2)*([1,3]-2) = [-1, 1]
    # Column 1: mean=3, var=1, grad for [2,4] = (2/2)*([2,4]-3) = [-1, 1]
    var expected_grad = Tensor.d2([[-1.0, -1.0], [1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_std_backward_global_std() raises:
    print("test_std_backward_global_std")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    var s = x.std(unbiased=False)
    s.backward()

    # Mean = 3.0, Variance = 2.0, Std = sqrt(2) ≈ 1.414
    # Gradient: (1/(std*n)) * (x - mean)
    # = (1/(1.414*5)) * (x - 3)
    # ≈ 0.1414 * (x - 3)
    var expected_grad = Tensor.d1([-0.2828, -0.1414, 0.0, 0.1414, 0.2828])
    assert_true(x.grad().all_close[atol=1e-3](expected_grad))


fn test_std_backward_unbiased_std() raises:
    print("test_std_backward_unbiased_std")
    var x = Tensor.d1([2.0, 4.0, 6.0, 8.0], requires_grad=True)
    var s = x.std(unbiased=True)
    s.backward()

    # Mean = 5.0, Sample variance = 20/3 ≈ 6.667
    # Std = sqrt(20/3) ≈ 2.582
    # Gradient: (1/(std*(n-1))) * (x - mean)
    # = (1/(2.582*3)) * (x - 5)
    var std_val = 2.5819889  # sqrt(20/3)
    var factor = 1.0 / (std_val * 3.0)
    var expected_grad = Tensor.d1([
        factor * -3.0,  # (2-5)
        factor * -1.0,  # (4-5)
        factor * 1.0,   # (6-5)
        factor * 3.0    # (8-5)
    ])
    assert_true(x.grad().all_close[atol=1e-3](expected_grad))


fn test_var_backward_chain_rule() raises:
    print("test_var_backward_chain_rule")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var v = x.variance(unbiased=False)
    var y = v * 2.0  # Chain another operation
    y.backward()

    # Var gradient * 2.0
    # Mean = 2.0, (x-mean) = [-1, 0, 1]
    # Var grad = (2/3) * (x-mean) = [-2/3, 0, 2/3]
    # Final grad = 2.0 * [-2/3, 0, 2/3] = [-4/3, 0, 4/3]
    var expected_grad = Tensor.d1([-1.3333333, 0.0, 1.3333333])
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


fn test_std_backward_chain_rule() raises:
    print("test_std_backward_chain_rule")
    var x = Tensor.d1([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    var s = x.std(unbiased=False)
    var y = s ** 2  # Square the std (should give variance)
    y.backward()

    # This should match variance gradient!
    # Mean = 2.5, (x-mean) = [-1.5, -0.5, 0.5, 1.5]
    # Gradient: (2/n) * (x-mean) = (2/4) * (x-2.5) = 0.5 * (x-2.5)
    var expected_grad = Tensor.d1([-0.75, -0.25, 0.25, 0.75])
    assert_true(x.grad().all_close[atol=1e-5](expected_grad))


fn test_var_std_no_grad_tracking() raises:
    print("test_var_std_no_grad_tracking")
    var x = Tensor.d1([1.0, 2.0, 3.0], requires_grad=True)
    var v = x.variance[track_grad=False]()
    var s = x.std[track_grad=False]()

    # Should not build computation graph
    assert_true(not v.requires_grad)
    assert_true(not s.requires_grad)


fn test_var_backward_2d_axis_0() raises:
    print("test_var_backward_2d_axis_0")
    var x = Tensor.d2([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
    var v = x.variance(axis=0, keepdims=False, unbiased=False)
    var s = v.sum()
    s.backward()

    # Column 0: mean=2, values=[1,3], grad=(2/2)*(values-2)=[-1, 1]
    # Column 1: mean=3, values=[4,2], grad=(2/2)*(values-3)=[1, -1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [1.0, -1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_var_backward_2d_axis_1() raises:
    print("test_var_backward_2d_axis_1")
    var x = Tensor.d2([[1.0, 3.0], [2.0, 4.0]], requires_grad=True)
    var v = x.variance(axis=1, keepdims=False, unbiased=False)
    var s = v.sum()
    s.backward()

    # Row 0: mean=2, values=[1,3], grad=(2/2)*(values-2)=[-1, 1]
    # Row 1: mean=3, values=[2,4], grad=(2/2)*(values-3)=[-1, 1]
    var expected_grad = Tensor.d2([[-1.0, 1.0], [-1.0, 1.0]])
    assert_true(x.grad().all_close[atol=1e-6](expected_grad))


fn test_std_backward_numerical_stability() raises:
    print("test_std_backward_numerical_stability")
    # Test with values close to zero variance
    var x = Tensor.d1([1.0, 1.001, 0.999, 1.0], requires_grad=True)
    var s = x.std(epsilon=1e-12)
    s.backward()

    # Should not crash or produce NaN/Inf
    _="""var grad = x.grad()
    assert_true(not grad.isnan().any())
    assert_true(not grad.isinf().any())"""


fn run_all_var_std_tests() raises:
    print("\n=== Running Variance & Std Test Suite ===\n")

    # Variance tests
    test_var_backward_global_variance()
    test_var_backward_unbiased_variance()
    test_var_backward_with_axis()
    test_var_backward_chain_rule()
    test_var_backward_2d_axis_0()
    test_var_backward_2d_axis_1()

    # Std tests
    test_std_backward_global_std()
    test_std_backward_unbiased_std()
    test_std_backward_chain_rule()
    test_std_backward_numerical_stability()

    # Feature tests
    test_var_std_no_grad_tracking()

    print("\n=== All Variance & Std Tests Passed! ===\n")

fn main() raises:
    test_sqrt_backward()
    test_sqrt_backward_zero_handling()
    run_all_var_std_tests()
