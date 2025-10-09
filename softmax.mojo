from tensors import Tensor
from shared import TensorLite
from operators import AddTensor
from intlist import IntList
from validators import Validator
from backpropagation import Delegate, BackwardFn
from summation import Summer
from multiplication import Multiplicator
from subtraction import Subtractor
from division import Divider
from minmax import MinMax

alias SoftmaxOutput[dtype: DType] = List[(IntList, Scalar[dtype])]


@fieldwise_init
struct SoftmaxBackward[dtype: DType](Copyable & Movable):
    var axes: IntList
    var softmax_out: SoftmaxOutput[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        incoming = output.gradients()[]
        softmax_out = Tensor[dtype].zeros(incoming.shape)
        for indices, value in self.softmax_out:
            softmax_out[indices] = rebind[Scalar[dtype]](value)

        product = Multiplicator[dtype].forward[False](incoming, softmax_out)
        sum_grad = Summer[dtype].forward[False](product, self.axes, True)

        diff = Subtractor[dtype].forward[False](incoming, sum_grad)
        grad_share = Multiplicator[dtype].forward[False](softmax_out, diff)

        ancestor = output.ancestry().get(0)
        return [(ancestor, grad_share, AddTensor)]


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
        shape = this.shape
        # Normalize axes
        normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        # max_vals = this.max(normalized_axes, keepdims=True, requires_grad=False)
        max_vals = MinMax[dtype].forward[True](
            this, normalized_axes, True, False
        )
        # Numerical stability: subtract max along axes
        # stable = this - max_vals
        stable = Subtractor[dtype].forward[False](this, max_vals)
        # Compute exponentials
        stable_exp = stable.exp()  # Revisit
        exp_sum = Summer[dtype].forward[False](
            stable_exp, normalized_axes, True
        )
        out = Divider[dtype].forward[False](stable_exp, exp_sum)

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else this.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)
                softmax_out = SoftmaxOutput[dtype](capacity=out.numels())
                for indices in out.shape:
                    softmax_out.append((indices, out[indices]))
                backward_fn = SoftmaxBackward[dtype](
                    normalized_axes, softmax_out
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(this))

        return out


fn main():
    _a = Tensor.arange(5, requires_grad=True)
    print("passes")
