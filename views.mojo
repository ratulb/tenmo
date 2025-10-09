from tensors import Tensor
from shapes import Shape
from strides import Strides
from intlist import IntList
from backpropagation import Delegate, BackwardFn
from operators import AddTensor, ZeroGrad
from shared import TensorLite
from validators import Validator


struct ViewBackward[dtype: DType](Copyable & Movable):
    var shape: Shape
    var strides: Strides
    var offset: Int

    fn __init__(out self, shape: Shape, strides: Strides, offset: Int):
        self.shape = shape
        self.strides = strides
        self.offset = offset

    fn __copyinit__(out self, existing: Self):
        self.shape = existing.shape
        self.strides = existing.strides
        self.offset = existing.offset

    fn __moveinit__(out self, deinit existing: Self):
        self.shape = existing.shape^
        self.strides = existing.strides^
        self.offset = existing.offset

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        parent = output.ancestry().get(0)
        gradients = output.grad()
        offset_delta = self.offset - parent.tensor().offset
        parent_grad = Tensor[dtype].zeros(parent.shape().num_elements())
        parent_shape = parent.shape()

        if parent_shape == Shape():
            parent_grad[0] = gradients.item()
        else:
            for coord in self.shape:
                child_flat = (coord * self.strides.to_list()).sum()
                parent_flat = child_flat + offset_delta
                parent_grad[parent_flat] += gradients[coord]

        reshaped = parent_grad.reshape(parent_shape)

        return [
            (parent, reshaped, AddTensor),
            (output, gradients, ZeroGrad),
        ]


@register_passable
struct View[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut self: Tensor[dtype],
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

        out = self.build_view(shape, strides, abs_offset, requires_grad=False)

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
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite[dtype].of(self))

        return out


fn main():
    pass
