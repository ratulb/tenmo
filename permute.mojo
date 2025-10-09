from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from operators import AddTensor
from intlist import IntList
from shapes import Shape
from strides import Strides
from common_utils import panic
from views import View


@fieldwise_init
struct PermuteBackward[dtype: DType](Copyable & Movable):
    var permutation: IntList  # forward permutation used

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        gradients = output.grad()
        parent = output.ancestry().get(0)
        # Compute inverse permutation
        inverse = IntList.filled(len(self.permutation), 0)
        for i in range(len(self.permutation)):
            inverse[self.permutation[i]] = i

        # Apply inverse permutation to gradients
        parent_grad = gradients.permute(inverse)

        return [(parent, parent_grad, AddTensor)]


@fieldwise_init
@register_passable
struct Permute[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut self: Tensor[dtype],
        axes: IntList,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        if len(axes) != self.shape.rank():
            panic("Tensor → permute: number of axes must match tensor rank")

        # Check for valid permutation
        seen = IntList.with_capacity(len(axes))
        for axis in axes:
            if axis < 0 or axis >= self.shape.rank():
                panic("Tensor → permute: invalid axis index")
            if axis in seen:
                panic("Tensor → permute: duplicate axis in permutation")
            seen.append(axis)

        # Create new shape and strides
        new_shape = IntList.with_capacity(len(axes))
        new_strides = IntList.with_capacity(len(axes))
        for axis in axes:
            new_shape.append(self.shape[axis])
            new_strides.append(self.strides[axis])

        # Return new view with same base but reordered axes
        out = View[dtype].forward[track_grad=False](
            self,
            shape=Shape(new_shape),
            strides=Strides(new_strides),
            offset=self.offset,  # Permute doesn't change offset
            requires_grad=False,
            validated=True,
        )

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else self.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                permutation = axes.copy()
                backward_fn = PermuteBackward[dtype](
                    permutation
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


fn main() raises:
    pass
