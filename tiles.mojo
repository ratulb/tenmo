from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from intlist import IntList
from operators import AddTensor
from shapes import Shape
from validators import Validator


@fieldwise_init
@register_passable
struct TileBackward[dtype: DType](Copyable):
    var repeat: IntList

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        var gradients = output.gradients()[]
        var ancestor = output.ancestry().get(0)[]
        var ancestor_shape = ancestor.shape()

        var grad_in = Tensor[dtype].zeros(ancestor_shape)
        var out_shape = gradients.shape
        var out_numels = out_shape.num_elements()

        for flat_idx in range(out_numels):
            var out_idx = out_shape.unravel_index(flat_idx)
            var ancestor_idx = IntList.with_capacity(len(out_idx))
            for d in range(len(out_idx)):
                ancestor_idx.append(out_idx[d] % ancestor_shape[d])
            grad_in[ancestor_idx] += gradients[out_idx]

        return [(ancestor, grad_in, AddTensor)]


@fieldwise_init
@register_passable
struct Tile[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        repeat: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        Validator.validate_repeat_args(
            self.shape, repeat
        )  # shape rank = len(repeat)

        var new_shape = IntList.with_capacity(self.shape.rank())
        for i in range(self.shape.rank()):
            new_shape.append(self.shape[i] * repeat[i])

        var out = Tensor[dtype](Shape(new_shape), requires_grad=False)
        var out_numels = out.numels()
        var rank = out.rank()
        var out_shape = out.shape
        var src_idx = IntList.with_capacity(rank)

        for flat_idx in range(out_numels):
            var idx = out_shape.unravel_index(flat_idx)
            src_idx.clear()
            for d in range(rank):
                src_idx.append(idx[d] % self.shape[d])
            out[idx] = self[src_idx]

        @parameter
        if track_grad:
            var grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = TileBackward[dtype](repeat).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


fn main():
    print("passes")
