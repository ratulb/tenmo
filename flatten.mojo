from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from shapes import Shape
from intlist import IntList
from common_utils import panic


@fieldwise_init
@register_passable
struct FlattenBackward[dtype: DType](Copyable):
    var start_dim: Int
    var end_dim: Int

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        tensor = ancestor.tensor()
        grad_in = Tensor[dtype].zeros(tensor.shape)

        # fast contiguous path
        if tensor.is_contiguous():
            n = gradients.shape.num_elements()
            for i in range(n):
                grad_in.buffer[i] = gradients.buffer[i]
        else:
            out_numels = gradients.shape.num_elements()
            for flat_idx in range(out_numels):
                idx = tensor.shape.unravel_index(flat_idx)
                grad_in[idx] = gradients[flat_idx]

        return [(ancestor, grad_in, AddTensor)]


@register_passable
struct Flatten[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        rank = self.rank()
        var endd = end_dim.value() if end_dim else rank - 1
        if endd < start_dim:
            panic("Flatten: end_dim must be >= start_dim")

        # compute new shape
        var new_shape = IntList()
        for i in range(start_dim):
            new_shape.append(self.shape[i])

        var flat_dim = 1
        for i in range(start_dim, endd + 1):
            flat_dim *= self.shape[i]
        new_shape.append(flat_dim)

        for i in range(endd + 1, rank):
            new_shape.append(self.shape[i])

        var out = Tensor[dtype](Shape(new_shape), requires_grad=False)

        # fast path: contiguous
        if self.is_contiguous():
            n = self.shape.num_elements()
            for i in range(n):
                if self.owns_data:
                    out.buffer[i] = self.buffer[i]
                else:
                    out.buffer[i] = self.base[].buffer[self.offset + i]
        else:
            var flat_idx = 0
            for _, value in self:
                out[flat_idx] = value
                flat_idx += 1

        # autograd hookup
        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                backward_fn = FlattenBackward[dtype](
                    start_dim, endd
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


fn main():
    print("passes")
