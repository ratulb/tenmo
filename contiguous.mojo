from tenmo import Tensor
from views import ViewBackward


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
        var ndb = self.buffer.contiguous()
        var out = Tensor[dtype](ndb^, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)
                shape = self.shape()
                strides = self.strides()
                offset = self.offset()
                backward_fn = ViewBackward[dtype](
                    shape, strides, offset
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main():
    print("passes")
