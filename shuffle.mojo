from tensors import Tensor
from shared import TensorLite
from intlist import IntList
from operators import AddTensor
from validators import Validator
from backpropagation import Delegate, BackwardFn
from random import shuffle, seed


@fieldwise_init
struct ShuffleBackward[dtype: DType](Copyable & Movable):
    var axis: Int
    var permutation: List[Int]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        var gradients = output.gradients()[]
        var parent = output.ancestry().get(0)[]

        var shape = gradients.shape.copy()

        # Allocate gradient w.r.t. ancestor
        var parent_grad = Tensor[dtype].zeros(
            shape
        )  # parent.shape == gradients.shape, only difference is coord postions along the permuted axis

        # Scatter gradients back using the original permutation
        # For each position in the output gradient, find where it came from in the input
        for grad_coord in shape:
            parent_coord = grad_coord.copy()
            parent_coord[self.axis] = self.permutation[grad_coord[self.axis]]
            parent_grad[parent_coord] = gradients[grad_coord]

        return [(parent, parent_grad, AddTensor)]


@register_passable
struct Shuffle[dtype: DType](Copyable):
    fn __copyinit__(out self, existing: Self):
        pass

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        perm: List[Int],  # permutation, length == axis length/span/spread
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = self.shape.copy()
        axis_length = shape[axis]

        var permutation: List[Int]
        if len(perm) > 0:
            Validator.check_permutation(perm, axis_length)
            permutation = perm
        else:
            seed()
            permutation = List[Int](capacity=axis_length)
            for i in range(axis_length):
                permutation.append(i)
            shuffle(permutation)

        # Allocate output
        var out = Tensor[dtype].zeros(shape)

        for coord in shape:
            shifted_src_coord = coord.copy()
            shifted_src_coord[axis] = permutation[coord[axis]]
            out[coord] = self[shifted_src_coord]

        # Attach autograd info
        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)

                backward_fn = ShuffleBackward[dtype](
                    axis, permutation
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


fn main() raises:
    test_shuffle()
    print("passes")


from testing import assert_true


fn test_shuffle() raises:
    print("test_shuffle")
    perm = List(2, 3, 0, 4, 1)
    a = Tensor.arange(5, requires_grad=True)
    shuffled = a.shuffle(perm=perm)
    shuffled.print()

    sliced = shuffled[1:4]
    print()
    sliced.print()
    c = sliced * 42
    c.backward()
    expected = Tensor.d1([42.0, 0.0, 0.0, 42.0, 42.0]).float()
    print()
    a.gradbox[].print()
    assert_true(a.gradbox[].all_close(expected))
    _ = a
    _ = shuffled
    _ = sliced
    _ = c
