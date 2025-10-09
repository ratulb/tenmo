from tensors import Tensor
from intlist import IntList
from operators import AddTensor
from shared import TensorLite
from shapes import Shape
from backpropagation import Delegate, BackwardFn
from validators import Validator


@fieldwise_init
struct SumBackward[dtype: DType](Copyable & Movable):
    var axes: IntList
    var keepdims: Bool

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.grad()
        ancestor = output.ancestry().get(0)
        rank = ancestor.shape().rank()
        if rank == 0:
            return [(ancestor, gradients, AddTensor)]
        shape = ancestor.shape()

        var grad_contrib: Tensor[dtype]

        # Handle scalar gradient case (sum reduced to scalar)
        if gradients.shape == Shape():
            grad_contrib = Tensor[dtype].full(
                shape,
                gradients.item(),
                requires_grad=False,
            )
        else:
            # Handle keepdims=False case (need to reshape gradient)
            if not self.keepdims:
                # Determine axes/unsqueeze (insert dims of size 1)
                axes = gradients.shape.intlist().insert(
                    self.axes,
                    IntList.with_capacity(len(self.axes), 1),
                )
                unsqueezed_shape = Shape(axes)

                unsqueezed_grad = gradients.reshape(unsqueezed_shape)
                grad_contrib = unsqueezed_grad.broadcast_to(
                    shape, requires_grad=False
                )
            else:
                # keepdims=True: shapes match except for broadcasting
                grad_contrib = gradients.broadcast_to(
                    shape, requires_grad=False
                )

        grad_contrib.requires_grad = False

        return [
            (
                ancestor,
                grad_contrib,
                AddTensor,
            )
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@fieldwise_init
@register_passable
struct Summer[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[dtype],
        axes: IntList,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = tensor.shape.copy()
        rank = shape.rank()
        normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        out_shape = shape.compute_output_shape(normalized_axes, keepdims)
        out = Tensor[dtype].zeros(out_shape)

        if out_shape == Shape():
            if rank == 0:  # Scalar case
                out[IntList()] = tensor[IntList()]
            elif rank == len(normalized_axes) and not keepdims:  # Reducing all
                out[IntList()] = tensor.sum_all()
        else:
            reduced_shape = Shape(shape.axes_spans.select(normalized_axes))
            for out_idx in out_shape:
                var summ = Scalar[dtype](0)
                for red_idx in reduced_shape:
                    full_idx = out_idx.replace(
                        normalized_axes, red_idx
                    ) if keepdims else out_idx.insert(normalized_axes, red_idx)
                    summ += tensor[full_idx]
                out[out_idx] = summ

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else tensor.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                backward_fn = SumBackward[dtype](
                    normalized_axes, keepdims
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(tensor))

        return out


fn main():
    print("passes")
