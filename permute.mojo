from tenmo import Tensor
from backpropagation import Delegate, BackwardFn, BACKWARD_PERMUTE
from operators import AddTensor, ZeroGrad
from intarray import IntArray
from shapes import Shape
from strides import Strides
from common_utils import panic
from views import View
from ancestry import Ancestor
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct PermuteBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_PERMUTE
    var permutation: IntArray  # forward permutation used

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad()
        parent = output.ancestry().get(0)
        # Compute inverse permutation
        inverse = IntArray.filled(len(self.permutation), 0)
        for i in range(len(self.permutation)):
            inverse[self.permutation[i]] = i

        # Apply inverse permutation to gradients
        parent_gradbox = gradbox.permute(inverse)

        return [
            (parent^, parent_gradbox^, AddTensor),
            (Ancestor(output), gradbox^, ZeroGrad),
        ]


@fieldwise_init
@register_passable
struct Permute[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        if len(axes) != self.rank():
            panic("Tensor → permute: number of axes must match tensor rank")

        # Check for valid permutation
        seen = IntArray.with_capacity(len(axes))
        for axis in axes:
            if axis < 0 or axis >= self.rank():
                panic("Tensor → permute: invalid axis index")
            if axis in seen:
                panic("Tensor → permute: duplicate axis in permutation")
            seen.append(axis)

        # Create new shape and strides
        new_shape = IntArray.with_capacity(len(axes))
        new_strides = IntArray.with_capacity(len(axes))
        for axis in axes:
            new_shape.append(self.shape()[axis])
            new_strides.append(self.strides()[axis])

        # Return new view with same base but reordered axes
        out = View[dtype].forward[track_grad=False](
            self,
            shape=Shape(new_shape),
            strides=Strides(new_strides),
            offset=self.offset(),  # Permute doesn't change offset
            requires_grad=False,
            validated=True,
        )

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                permutation = axes.copy()
                backward_fn = PermuteBackward[dtype](
                    permutation
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    pass


from testing import assert_true
