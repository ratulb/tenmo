from tenmo import Tensor
from operators import AddTensor
from validators import Validator
from backpropagation import Delegate, BackwardFn, BACKWARD_SHUFFLE
from random import shuffle, seed
from gradbox import Gradbox


@fieldwise_init
struct ShuffleBackward[dtype: DType](ImplicitlyCopyable & Movable):
    alias TAG = BACKWARD_SHUFFLE
    var axis: Int
    var permutation: List[Int]

    fn __copyinit__(out self, other: Self):
        self.axis = other.axis
        self.permutation = other.permutation.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axis = other.axis
        self.permutation = other.permutation^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)

        var shape = gradbox.shape()

        # Allocate gradient w.r.t. ancestor
        var gradbox_parent = Gradbox[dtype].zeros(
            shape, share=False
        )  # parent.shape == gradients.shape, only difference is coord postions along the permuted axis

        # Scatter gradients back using the original permutation
        # For each position in the output gradient, find where it came from in the input
        for grad_coord in shape:
            parent_coord = grad_coord
            parent_coord[self.axis] = self.permutation[grad_coord[self.axis]]
            gradbox_parent[parent_coord] = gradbox[grad_coord]

        return [(parent^, gradbox_parent^, AddTensor)]


@fieldwise_init
@register_passable
struct Shuffle[dtype: DType](Copyable & Movable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        perm: List[Int],  # permutation, length == axis length/span/spread
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        shape = self.shape()
        axis_length = shape[axis]

        var permutation: List[Int]
        if len(perm) > 0:
            Validator.check_permutation(perm, axis_length)
            permutation = perm.copy()
        else:
            seed()
            permutation = List[Int](capacity=axis_length)
            for i in range(axis_length):
                permutation.append(i)
            shuffle(permutation)

        # Allocate output
        var out = Tensor[dtype].zeros(shape)

        for coord in shape:
            shifted_src_coord = coord
            shifted_src_coord[axis] = permutation[coord[axis]]
            out[coord] = self[shifted_src_coord]

        # Attach autograd info
        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                backward_fn = ShuffleBackward[dtype](
                    axis, permutation^
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    print("passes")
