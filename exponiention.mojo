from tensors import Tensor
from shared import TensorLite
from backpropagation import BackwardFn, Delegate
from operators import AddTensor
from buffers import Buffer


@fieldwise_init
struct ExponientionBackward[dtype: DType](Copyable & Movable):
    var exponent: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.grad()
        var exponent: Scalar[dtype] = rebind[Scalar[dtype]](self.exponent)
        ancestor = output.ancestry().get(0)

        # ∂(x**n)/∂x = n * x**(n-1)
        # Need to see if base_pow gets a grad_fn or not - we don't want it to have one!
        # var base_pow = self ** (scalar - 1.0)
        base_pow = ancestor.tensor() ** (exponent - 1.0)
        base_pow.requires_grad = False
        var local_grad = base_pow * exponent
        product = gradients * local_grad
        return [
            (
                ancestor,
                product,
                AddTensor,
            )
        ]


@fieldwise_init
@register_passable
struct Exponentiator[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], exponent: Scalar[dtype]) -> Tensor[dtype]:
        var out: Tensor[dtype]
        if self.is_contiguous():
            var buffer: Buffer[dtype]
            if self.owns_data:
                buffer = self.buffer
            else:
                offset = self.offset
                numels = self.numels()
                buffer = self.shared_buffer.value()[][offset : offset + numels]
            out = Tensor[dtype](
                self.shape, buffer**exponent, requires_grad=False
            )

        else:
            out = Tensor[dtype](self.shape, requires_grad=False)
            for idx, value in self:
                out[idx] = value**exponent

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = ExponientionBackward[dtype](
                    exponent
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


fn main():
    pass
