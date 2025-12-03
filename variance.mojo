from tenmo import Tensor
from operators import AddTensor
from backpropagation import Delegate, BackwardFn, BACKWARD_VARIANCE
from ancestry import Ancestor
from gradbox import Gradbox
from common_utils import panic


@fieldwise_init
@register_passable
struct VarianceBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_VARIANCE
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool  # Track if user wanted keepdims

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad()  # Copying
        var ancestor = output.ancestry().get(0)
        ref input_tensor = ancestor.tensor()
        ref input_shape = input_tensor.shape()
        # Compute mean of input (always with keepdims=True)
        var dim = List[Int]()
        if self.axis != -100:
            dim.append(self.axis)
        var mean_val = input_tensor.mean[track_grad=False](dim, keepdims=True)

        # Compute (x - mean)
        var diff = input_tensor.__sub__[track_grad=False](mean_val)

        # Calculate divisor
        var n: Scalar[dtype]
        if self.axis != -100:
            n = Scalar[dtype](input_shape[self.axis])
        else:
            n = Scalar[dtype](input_shape.num_elements())

        var divisor = n
        if self.unbiased and n > 1:
            divisor = n - 1

        # Gradient: (2/divisor) * (x - mean)
        var local_grad = diff.__mul__[track_grad=False](2.0 / divisor)

        # If keepdims was False, we need to add dimension back to grad output
        var gradbox_ancestor: Gradbox[dtype]
        if not self.keepdims and self.axis != -100:
            # Unsqueeze gradbox to add back the reduced dimension
            gradbox_ancestor = gradbox.unsqueeze([self.axis])
        else:
            gradbox_ancestor = gradbox^

        # Broadcast to input shape (automatic broadcasting will handle it)
        gradbox_ancestor = local_grad * gradbox_ancestor

        return [(ancestor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct Variance[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        if axis != -100 and (axis < 0 or axis >= self.rank()):
            panic("Invalid axis specified for variance")
        var dim = List[Int]()
        if axis != -100:
            dim.append(axis)

        # Always compute with keepdims=True internally
        var mean_val = self.mean[track_grad=False](dim, keepdims=True)
        var diff = self.__sub__[track_grad=False](mean_val)
        var squared_diff = diff.__mul__[track_grad=False](diff)
        var sum_sq = squared_diff.sum[track_grad=False](dim, keepdims=True)

        var n: Scalar[dtype]
        if axis != -100:
            n = Scalar[dtype](self.shape()[axis])
        else:
            n = Scalar[dtype](self.numels())

        if unbiased and n > 1:
            n = n - 1

        var result = sum_sq.__truediv__[track_grad=False](n)

        # Squeeze at the end if keepdims=False
        if not keepdims and axis != -100:
            result = result.squeeze[track_grad=False]([axis])

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                backward_fn = VarianceBackward[dtype](
                    axis=axis,
                    unbiased=unbiased,
                    keepdims=keepdims,  # Store this!
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(self)

        return result^


fn main() raises:
    pass
