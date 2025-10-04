from tensors import Tensor
from shared import TensorLite
from views import ViewBackward
from memory import memcpy


@register_passable
struct Contiguous[dtype: DType](Copyable):

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        dtype
    ]:
        var shape_copy = self.shape.copy()
        offset = self.offset
        numels = self.numels()
        out = Tensor[dtype](shape_copy, requires_grad=False)
        if self.is_contiguous():
            if self.owns_data:
                memcpy(out.buffer.data, self.buffer.data, numels)
            else:
                buffer = self.shared_buffer.value()[]
                memcpy(out.buffer.data, buffer.data + offset, numels)
        else:
            for idx, value in self:
                out[idx] = value

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)
                strides = self.strides
                backward_fn = ViewBackward[dtype](
                    shape_copy, strides, offset * 2
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite[dtype].of(self))

        return out


fn main():
    print("passes")
