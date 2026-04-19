from tenmo import Tensor
from mnemonics import AddTensor
from backpropagation import BackwardFnArg, ArgumentType, BACKWARD_VARIANCE
from gradbox import Gradbox
from common_utils import panic
from ancestry import Ancestor

@fieldwise_init
struct VarianceBwdArg(ArgumentType):
    var axis: Int
    var unbiased: Bool
    var keepdims: Bool


@fieldwise_init
struct VarianceBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):

    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = output.ancestry().backward_fn_arg().get[VarianceBwdArg]()
        var (axis, unbiased, keepdims) = (
            bwd_arg.axis,
            bwd_arg.unbiased,
            bwd_arg.keepdims,
        )
        var gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var input_tensor = Tensor[Self.dtype](parent.buffer(), requires_grad=parent.requires_grad)
        ref input_shape = input_tensor.shape()

        # Compute mean of input (always with keepdims=True)
        var dim = List[Int]()
        if axis != -100:
            dim.append(axis)

        # var mean_val = input_tensor.mean[track_grad=False](dim, keepdims=True)
        var mean_val: Tensor[Self.dtype]
        if axis == -100:
            # Global mean - scalar
            # mean_val = input_tensor.mean[track_grad=False](keepdims=True)
            mean_val = input_tensor.mean[track_grad=False]()
        else:
            # Axis-specific mean - keepdims for broadcasting
            mean_val = input_tensor.mean[track_grad=False](dim, keepdims=True)

        var diff = input_tensor.__sub__[track_grad=False](mean_val)

        # Calculate divisor
        var n: Scalar[Self.dtype]
        if axis != -100:
            n = Scalar[Self.dtype](input_shape[axis])
        else:
            n = Scalar[Self.dtype](input_shape.num_elements())

        var divisor = n
        if unbiased and n > 1:
            divisor = n - 1

        # Gradient: (2/divisor) * (x - mean)
        var local_grad = diff.__mul__[track_grad=False](2.0 / divisor)

        # Handle gradient shape
        var gradbox_ancestor: Gradbox[Self.dtype]

        if not keepdims:
            if axis != -100:
                # Specific axis was reduced - unsqueeze that axis
                gradbox_ancestor = gradbox.unsqueeze([axis])
            else:
                gradbox_ancestor = gradbox^
        else:
            gradbox_ancestor = gradbox^

        # Broadcast to input shape (automatic broadcasting will handle it)
        gradbox_ancestor = local_grad * gradbox_ancestor

        return [(parent^, gradbox_ancestor^, AddTensor)]



@fieldwise_init
struct Variance[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        if axis != -100 and (axis < 0 or axis >= self.rank()):
            panic("Invalid axis specified for variance")

        var dim = List[Int]()
        if axis != -100:
            dim.append(axis)

        # Compute mean - use scalar for global, keepdims for axis-specific
        var mean_val: Tensor[Self.dtype]
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
        var sum_sq: Tensor[Self.dtype]
        if axis == -100:
            sum_sq = squared_diff.sum[track_grad=False](
                keepdims=True
            )  # Global sum -> scalar
        else:
            sum_sq = squared_diff.sum[track_grad=False](dim, keepdims=True)

        # Calculate divisor
        var n: Scalar[Self.dtype]
        if axis != -100:
            n = Scalar[Self.dtype](self.shape()[axis])
        else:
            n = Scalar[Self.dtype](self.numels())

        if unbiased and n > 1:
            n = n - 1

        var result = sum_sq.__truediv__[track_grad=False](n)
        # Squeeze at the end if keepdims=False and axis-specific
        if not keepdims:
            if axis != -100:
                result = result.squeeze[track_grad=False]([axis])
            else:
                result = result.squeeze[track_grad=False]()

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_VARIANCE,
                    VarianceBwdArg(
                        axis,
                        unbiased,
                        keepdims,
                    ),
                )
                result.add_ancestry(backwardFnArg^, self)

        return result^
