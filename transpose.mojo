from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList
from validators import Validator


@register_passable
struct TransposeBackward[dtype: DType](Copyable):
    var axes: IntList
    
    fn __init__(out self, axes: IntList):
        self.axes = axes

    fn __copyinit__(out self, existing: Self):
        self.axes = existing.axes.copy()

        _="""fn __moveinit__(out self, deinit existing: Self):
        self.axes = existing.axes"""

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        inverted_axes = IntList.invert_permutation(self.axes)
        grad_transposed = gradients.transpose(inverted_axes)
        return [
            (
                ancestor,
                grad_transposed,
                AddTensor,
            )
        ]


@register_passable
struct Transpose[dtype: DType](Copyable):

    fn __copyinit__(out self, existing: Self):
        pass

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype], axes: IntList, requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        shape = self.shape.copy()
        normalized_axes = (
            Validator.validate_and_normalize_axes(
                shape, axes, ordered=False, fill_missing=True
            ) if len(axes)
            > 0 else IntList.range_list(shape.rank()).reversed()
        )

        # Permute shape and create default strides and permute
        var new_shape = shape.permute(normalized_axes)
        var new_strides = self.strides.permute(normalized_axes)

        base_addr = self.address() if self.owns_data else self.base.copy()
        out = Tensor[dtype](
            new_shape, base_addr, new_strides, self.offset, False
        )


        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)
                backward_fn = TransposeBackward[dtype](
                    normalized_axes
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


fn main():
    pass


