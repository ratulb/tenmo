from tenmo import Tensor
from operators import AddTensor
from intlist import IntList
from validators import Validator
from backpropagation import Delegate, BackwardFn
from summation import Summer
from subtraction import Subtractor
from division import Divider
from minmax import MinMax
from layout.int_tuple import IntArray
from gradbox import Gradbox
from ancestry import Ancestor

alias SoftmaxOutput[dtype: DType] = List[Tuple[IntArray, Scalar[dtype]]]


@fieldwise_init
struct SoftmaxBackward[dtype: DType](ImplicitlyCopyable & Movable):
    var axes: IntList
    var softmax_out: SoftmaxOutput[dtype]

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.softmax_out = other.softmax_out.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes^
        self.softmax_out = other.softmax_out^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        softmax_out = Gradbox[dtype].zeros(gradbox.shape())

        # Reconstruct softmax output tensor from stored coordinate-value pairs
        for coord, value in self.softmax_out:
            softmax_out[coord] = value

        # softmax_grad = y * (g - sum(g * y, axis, keepdims=True))
        local_grad = softmax_out * (
            gradbox - (gradbox * softmax_out).sum(self.axes, keepdims=True)
        )

        ancestor = output.ancestry().get(0)
        return [(ancestor^, local_grad^, AddTensor)]


@fieldwise_init
@register_passable
struct Softmax[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        this: Tensor[dtype],
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = this.shape()
        # Normalize axes
        normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        # max_vals = this.max(normalized_axes, keepdims=True, requires_grad=False)
        max_vals = MinMax[dtype].forward[max=True, track_grad=False](
            this, normalized_axes, keepdims=True, requires_grad=False
        )
        # Numerical stability: subtract max along axes
        # stable = this - max_vals
        stable = Subtractor[dtype].forward[track_grad=False](this, max_vals)
        # Compute exponentials
        stable_exp = stable.exp()
        exp_sum = Summer[dtype].forward[track_grad=False](
            stable_exp, normalized_axes, True
        )
        out = Divider[dtype].forward[track_grad=False](stable_exp, exp_sum)

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(this.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                # Lightweight storage: only store coordinate-value pairs
                softmax_out = SoftmaxOutput[dtype](capacity=UInt(out.numels()))
                for coord in out.shape():
                    softmax_out.append((coord, out[coord]))
                backward_fn = SoftmaxBackward[dtype](
                    normalized_axes^, softmax_out^
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(this)

        return out^


# Add this alias for log softmax output storage
alias LogSoftmaxOutput[dtype: DType] = List[Tuple[IntArray, Scalar[dtype]]]

@fieldwise_init
struct LogSoftmaxBackward[dtype: DType](ImplicitlyCopyable & Movable):
    var axes: IntList
    var softmax_out: SoftmaxOutput[dtype]  # Still need regular softmax for gradient

    fn __copyinit__(out self, other: Self):
        self.axes = other.axes.copy()
        self.softmax_out = other.softmax_out.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axes = other.axes^
        self.softmax_out = other.softmax_out^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]

        # Reconstruct softmax output tensor
        softmax_out_tensor = Gradbox[dtype].zeros(gradbox.shape())
        for coord, value in self.softmax_out:
            softmax_out_tensor[coord] = value

        # Gradient for log_softmax: g - softmax(x) * sum(g, axis, keepdims=True)
        sum_grad = gradbox.sum(self.axes, keepdims=True)
        local_grad = gradbox - softmax_out_tensor * sum_grad

        ancestor = output.ancestry().get(0)
        return [(ancestor^, local_grad^, AddTensor)]


@fieldwise_init
@register_passable
struct LogSoftmax[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        this: Tensor[dtype],
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = this.shape()
        # Normalize axes
        normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        # Numerical stability: subtract max along axes
        max_vals = MinMax[dtype].forward[max=True, track_grad=False](
            this, normalized_axes, keepdims=True, requires_grad=False
        )
        stable = Subtractor[dtype].forward[track_grad=False](this, max_vals)

        # Compute exponentials and sum
        stable_exp = stable.exp()
        exp_sum = Summer[dtype].forward[track_grad=False](
            stable_exp, normalized_axes, True
        )

        # Log softmax: (x - max(x)) - log(sum(exp(x - max(x))))
        log_sum_exp = exp_sum.log(requires_grad=False)
        out = Subtractor[dtype].forward[track_grad=False](stable, log_sum_exp)

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(this.requires_grad)

            if grad_required:
                out.requires_grad_(True)

                # We still need regular softmax for the backward pass
                softmax_out = SoftmaxOutput[dtype](capacity=UInt(out.numels()))
                # Compute softmax: exp(stable) / exp_sum
                softmax_vals = Divider[dtype].forward[track_grad=False](stable_exp, exp_sum)
                for coord in softmax_vals.shape():
                    softmax_out.append((coord, softmax_vals[coord]))

                backward_fn = LogSoftmaxBackward[dtype](
                    normalized_axes^, softmax_out^
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(this)

        return out^


fn main() raises:
    print("passes")
