from tenmo import Tensor
from operators import AddTensor
from backpropagation import Delegate, BackwardFn, BACKWARD_VARIANCE
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
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad()  # Copying
        var input_tensor = output.ancestry().get(0)
        ref input_shape = input_tensor.shape()

        # Compute mean of input (always with keepdims=True)
        var dim = List[Int]()
        if self.axis != -100:
            dim.append(self.axis)

        # var mean_val = input_tensor.mean[track_grad=False](dim, keepdims=True)
        var mean_val: Tensor[dtype]
        if self.axis == -100:
            # Global mean - scalar
            # mean_val = input_tensor.mean[track_grad=False](keepdims=True)
            mean_val = input_tensor.mean[track_grad=False]()
        else:
            # Axis-specific mean - keepdims for broadcasting
            mean_val = input_tensor.mean[track_grad=False](dim, keepdims=True)

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

        # Handle gradient shape
        var gradbox_ancestor: Gradbox[dtype]

        if not self.keepdims:
            if self.axis != -100:
                # Specific axis was reduced - unsqueeze that axis
                gradbox_ancestor = gradbox.unsqueeze([self.axis])
            else:
                gradbox_ancestor = gradbox^
        else:
            gradbox_ancestor = gradbox^

        # Broadcast to input shape (automatic broadcasting will handle it)
        gradbox_ancestor = local_grad * gradbox_ancestor

        return [(input_tensor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct Variance[dtype: DType]:
    @always_inline
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

        # Compute mean - use scalar for global, keepdims for axis-specific
        var mean_val: Tensor[dtype]
        if axis == -100:
            # Global mean - scalar (0D tensor)
            # mean_val = self.mean[track_grad=False](keepdims=True)
            mean_val = self.mean[track_grad=False](keepdims=False)
        else:
            # Axis-specific mean - keepdims for broadcasting
            mean_val = self.mean[track_grad=False](dim, keepdims=True)

        var diff = self.__sub__[track_grad=False](mean_val)
        var squared_diff = diff.__mul__[track_grad=False](diff)

        # Sum over same dimensions
        var sum_sq: Tensor[dtype]
        if axis == -100:
            sum_sq = squared_diff.sum[track_grad=False](
                keepdims=True
            )  # Global sum -> scalar
        else:
            sum_sq = squared_diff.sum[track_grad=False](dim, keepdims=True)

        # Calculate divisor
        var n: Scalar[dtype]
        if axis != -100:
            n = Scalar[dtype](self.shape()[axis])
        else:
            n = Scalar[dtype](self.numels())

        if unbiased and n > 1:
            n = n - 1

        var result = sum_sq.__truediv__[track_grad=False](n)
        # Squeeze at the end if keepdims=False and axis-specific
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
                backward_fn = VarianceBackward[dtype](
                    axis=axis,
                    unbiased=unbiased,
                    keepdims=keepdims,
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(self)

        return result^

    @staticmethod
    fn forward_orig_2[
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

        # For global variance (axis=-100), compute mean as scalar
        var mean_val: Tensor[dtype]
        if axis == -100:
            # Global mean - should be scalar (0D tensor)
            mean_val = self.mean[track_grad=False]()  # No dim, no keepdims
            print("Global mean (scalar):", mean_val.item())
        else:
            # Axis-specific mean - use keepdims for broadcasting
            mean_val = self.mean[track_grad=False](dim, keepdims=True)

        var diff = self.__sub__[track_grad=False](mean_val)
        var squared_diff = diff.__mul__[track_grad=False](diff)

        # Sum over same dimensions
        var sum_sq: Tensor[dtype]
        if axis == -100:
            sum_sq = squared_diff.sum[
                track_grad=False
            ]()  # Global sum -> scalar
        else:
            sum_sq = squared_diff.sum[track_grad=False](dim, keepdims=True)

        print("Sum of squared differences:")
        sum_sq.print()

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
                    keepdims=keepdims,
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(self)

        return result^

    @always_inline
    @staticmethod
    fn forward_orig[
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
        print("What mean are we getting: ")
        mean_val.print()
        var diff = self.__sub__[track_grad=False](mean_val)
        print("diff shapes and all : ", diff)
        var squared_diff = diff.__mul__[track_grad=False](diff)
        print("squared_diff shapes ", squared_diff)
        var sum_sq = squared_diff.sum[track_grad=False](dim, keepdims=True)
        print("And the sum? ", dim.__str__())
        sum_sq.print()

        var n: Scalar[dtype]
        if axis != -100:
            n = Scalar[dtype](self.shape()[axis])
        else:
            n = Scalar[dtype](self.numels())

        if unbiased and n > 1:
            n = n - 1

        var result = sum_sq.__truediv__[track_grad=False](n)

        # Squeeze at the end if keepdims=False
        if not keepdims:
            if axis != -100:
                # Squeeze specific axis
                result = result.squeeze[track_grad=False]([axis])
            else:
                # Squeeze all dimensions for global variance (scalar result)
                result = result.squeeze[track_grad=False]()  # Squeeze all

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                backward_fn = VarianceBackward[dtype](
                    axis=axis,
                    unbiased=unbiased,
                    keepdims=keepdims,
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(self)

        return result^


fn main() raises:
    pass
