from tenmo import Tensor
from mnemonics import AddTensor
from backpropagation import BackwardFnArg, StdArg, BACKWARD_STD
from gradbox import Gradbox
from common_utils import panic, Epsilon
from ancestry import Ancestor


@fieldwise_init
struct StdBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[StdArg[Self.dtype]]()
        )
        var (axis, unbiased, keepdims, epsilon) = (
            bwd_arg.axis,
            bwd_arg.unbiased,
            bwd_arg.keepdims,
            bwd_arg.epsilon,
        )
        var gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var input_tensor = Tensor[Self.dtype](
            parent.buffer(), requires_grad=parent.requires_grad
        )
        ref input_shape = input_tensor.shape()

        var dim = List[Int]()
        if axis != -100:
            dim.append(axis)

        # Always use keepdims=True internally
        var mean_val = input_tensor.mean[track_grad=False](dim, keepdims=True)
        var diff = input_tensor.__sub__[track_grad=False](mean_val)

        var n: Scalar[Self.dtype]
        if axis != -100:
            n = Scalar[Self.dtype](input_shape[axis])
        else:
            n = Scalar[Self.dtype](input_shape.num_elements())

        var divisor = n
        if unbiased and n > 1:
            divisor = n - 1

        # Compute std (recompute from input for numerical stability)
        var var_val = (
            diff.__mul__[track_grad=False](diff).sum[track_grad=False](
                dim, keepdims=True
            )
        ).__truediv__[track_grad=False](divisor)
        var std_val = var_val.sqrt[track_grad=False](epsilon=epsilon)

        # Gradient: (1 / (std * divisor)) * (x - mean)
        # var local_grad = diff / ((std_val + self.epsilon) * divisor)
        var local_grad = diff.__truediv__[track_grad=False](
            (
                (std_val.__add__[track_grad=False](epsilon)).__mul__[
                    track_grad=False
                ](divisor)
            )
        )

        # Handle keepdims
        var gradbox_ancestor: Gradbox[Self.dtype]
        if not keepdims:
            if axis != -100:
                gradbox_ancestor = gradbox.unsqueeze([axis])
            else:
                var scalar_grad = gradbox.item()
                gradbox_ancestor = Gradbox[Self.dtype].full(
                    input_shape, scalar_grad, share=False
                )
        else:
            gradbox_ancestor = gradbox^

        # Broadcasting handles the rest
        gradbox_ancestor = local_grad.__mul__(gradbox_ancestor^)

        return [(parent^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
struct StdDev[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        axis: Int = -100,
        keepdims: Bool = False,
        unbiased: Bool = True,
        epsilon: Scalar[Self.dtype] = Epsilon[Self.dtype].value(),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
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

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                result.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_STD,
                    StdArg[Self.dtype](
                        axis,
                        unbiased,
                        keepdims,
                        epsilon,
                    ),
                )
                result.add_ancestry(backwardFnArg^, self)

        return result^
