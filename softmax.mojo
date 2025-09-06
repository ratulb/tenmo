from tensors import Tensor
from shared import TensorLite
from operators import AddTensor
from intlist import IntList
from validators import Validator
from backpropagation import Delegate, BackwardFn


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
        
        sum_grad = (incoming * softmax_out).sum(
            self.axes, keepdims=True, track_grad=False
        )

        grad_share = softmax_out * (incoming - sum_grad)

        ancestor = output.ancestry().get(0)[]
        return [(ancestor, grad_share, AddTensor)]


@fieldwise_init
@register_passable
struct Softmax[dtype: DType]:
    @staticmethod
    fn softmax(
        this: Tensor[dtype],
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = this.shape
        # Normalize axes
        normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        max_vals = this.max(normalized_axes, keepdims=True, requires_grad=False)
        # Numerical stability: subtract max along axes
        stable = this - max_vals
        # Compute exponentials
        stable_exp = stable.exp()
        exp_sum = stable_exp.sum(
            normalized_axes, keepdims=True, track_grad=False
        )
        # Softmax = exp(x) / sum(exp(x))
        out = stable_exp / exp_sum

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
