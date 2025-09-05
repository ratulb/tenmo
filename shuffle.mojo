from tensors import Tensor
from shared import TensorLite
from intlist import IntList
from shapes import Shape
from operators import AddTensor
from validators import Validator
from backpropagation import Delegate, BackwardFn
from random import shuffle, seed


@fieldwise_init
@register_passable
struct ShuffleBackward[dtype: DType](Copyable):
    var axis: Int
    var permutation: IntList

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        var grad = output.gradients()[]
        var ancestor = output.ancestry().get(0)[]

        var shape = grad.shape
        var axis_len = shape[self.axis]

        # Inverse permutation
        var inv_perm = IntList.filled(axis_len, 0)
        for i in range(axis_len):
            inv_perm[self.permutation[i]] = i

        # Allocate gradient w.r.t. ancestor
        var grad_in = Tensor[dtype].zeros(shape)

        # Scatter gradients back along inverse permutation
        for idx in shape:
            var src = idx.copy()
            var dst = idx.copy()
            src[self.axis] = idx[self.axis]
            dst[self.axis] = inv_perm[idx[self.axis]]
            grad_in[dst] = grad[src]

        return [(ancestor, grad_in, AddTensor)]


fn unravel_index(flat: Int, shape: Shape) -> IntList:
    rank = shape.rank()
    var idx = IntList(rank)
    var rem = flat
    for i in range(rank - 1, -1, -1):  # from last axis backward
        dim = shape[i]
        idx[i] = rem % dim
        rem //= dim
    return idx


@fieldwise_init
@register_passable
struct ShuffleForward[dtype: DType]:
    @staticmethod
    fn shuffle(
        self: Tensor[dtype],
        axis: Int = 0,
        permutation: Optional[IntList] = None,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = self.shape
        axis_len = shape[axis]

        var perm: IntList
        if permutation:
            perm = permutation.value()
            Validator.check_permutation(perm, axis_len)
        else:
            seed()
            perm = IntList.with_capacity(axis_len)
            for i in range(axis_len):
                perm.append(i)
            list = perm.tolist()
            shuffle(list)
            perm = IntList.new(list)

        # Allocate output
        var out = Tensor[dtype].zeros(shape)

        for flat in range(self.numels()):
            idx = unravel_index(flat, shape)
            src_idx = idx.copy()
            src_idx[axis] = perm[idx[axis]]
            out[idx] = self[src_idx]

        # Attach autograd info
        grad_required = (
            requires_grad.value() if requires_grad else self.requires_grad
        )
        if grad_required:
            out.requires_grad_(True)
            var backward_fn = ShuffleBackward[dtype](
                axis, perm
            ).into_backward_fn()
            out.backwardFn = Optional(backward_fn)
            out.add_ancestry(TensorLite.of(self))

        return out


fn main():
    ShuffleForward[DType.float32].shuffle(Tensor.arange(5)).print()
    print("passes")
