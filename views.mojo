from tenmo import Tensor
from shapes import Shape
from strides import Strides
from backpropagation import Delegate, BackwardFn
from operators import AddTensor, ZeroGrad
from validators import Validator
from ancestry import Ancestor
from gradbox import Gradbox
from layout.int_tuple import IntArray

@fieldwise_init
@register_passable
struct ViewBackward[dtype: DType](ImplicitlyCopyable):
    var shape: Shape
    var strides: Strides
    var offset: Int

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        parent = output.ancestry().get(0)
        gradbox = output.grad().copy()
        parent_tensor = parent.tensor()
        offset_delta = self.offset - parent_tensor.offset()
        parent_shape = parent.shape()
        # Flat gradbox space
        parent_gradbox = Gradbox[dtype].zeros(
            Shape(parent_shape.num_elements())
        )
        if parent_shape == Shape():
            parent_gradbox[IntArray()] = gradbox.item()
        else:
            for coord in self.shape:
                child_flat_unpinned = (self.strides * coord).sum()
                child_flat_pinned = child_flat_unpinned + offset_delta
                child_position = IntArray(size=1)
                child_position[0] = child_flat_pinned
                parent_gradbox[child_position] += gradbox[coord]

        reshaped = parent_gradbox.reshape(parent_shape)

        return [
            (parent^, reshaped^, AddTensor),
            (Ancestor(output), gradbox^, ZeroGrad),
        ]


@register_passable
struct View[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[dtype]:
        # Validate parameters and compute absolute bounds
        var abs_offset = Validator.validate_view_params(
            self, shape, strides, offset
        ) if not validated else offset

        out = Tensor[dtype].build_view(
            self.address(), shape, strides, abs_offset, requires_grad=False
        )

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)

                backward_fn = ViewBackward[dtype](
                    shape, strides, abs_offset
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main():
    pass
