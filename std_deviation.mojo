from tenmo import Tensor
from operators import AddTensor
from backpropagation import Delegate, BackwardFn, BACKWARD_STD
from gradbox import Gradbox
from common_utils import panic


@fieldwise_init
@register_passable
struct StdBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_STD
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool
    var epsilon: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad()  # Copy
        var input_tensor = output.ancestry().get(0)
        ref input_shape = input_tensor.shape()

        var dim = List[Int]()
        if self.axis != -100:
            dim.append(self.axis)

        # Always use keepdims=True internally
        var mean_val = input_tensor.mean[track_grad=False](dim, keepdims=True)
        var diff = input_tensor.__sub__[track_grad=False](mean_val)

        var n: Scalar[dtype]
        if self.axis != -100:
            n = Scalar[dtype](input_shape[self.axis])
        else:
            n = Scalar[dtype](input_shape.num_elements())

        var divisor = n
        if self.unbiased and n > 1:
            divisor = n - 1

        # Compute std (recompute from input for numerical stability)
        var var_val = (
            diff.__mul__[track_grad=False](diff).sum[track_grad=False](
                dim, keepdims=True
            )
        ).__truediv__[track_grad=False](divisor)
        var std_val = var_val.sqrt[track_grad=False](epsilon=self.epsilon)

        # Gradient: (1 / (std * divisor)) * (x - mean)
        # var local_grad = diff / ((std_val + self.epsilon) * divisor)
        var local_grad = diff.__truediv__[track_grad=False](
            (
                (std_val.__add__[track_grad=False](self.epsilon)).__mul__[
                    track_grad=False
                ](divisor)
            )
        )

        # Handle keepdims
        var gradbox_ancestor: Gradbox[dtype]
        if not self.keepdims:
            if self.axis != -100:
                gradbox_ancestor = gradbox.unsqueeze([self.axis])
            else:
                var scalar_grad = gradbox.item()
                gradbox_ancestor = Gradbox[dtype].full(
                    input_shape, scalar_grad, share=False
                )
        else:
            gradbox_ancestor = gradbox^

        # Broadcasting handles the rest
        gradbox_ancestor = local_grad.__mul__(gradbox_ancestor^)

        return [(input_tensor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct StdDev[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        epsilon: Scalar[dtype] = Scalar[dtype](1e-12),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        if axis != -100 and (axis < 0 or axis >= self.rank()):
            panic("axis is not valid for standard deviation")
        var var_result = self.variance[track_grad=False](
            axis, keepdims=True, unbiased=unbiased
        )
        var result = var_result.sqrt[track_grad=False](epsilon=epsilon)

        if not keepdims:
            if axis != -100:
                result = result.squeeze[track_grad=False]([axis])
            else:
                result = result.squeeze[track_grad=False]()

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                backward_fn = StdBackward[dtype](
                    axis=axis,
                    unbiased=unbiased,
                    keepdims=keepdims,
                    epsilon=epsilon,
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(self)

        return result^


fn main() raises:
    pass
