from tenmo import Tensor
from shapes import Shape
from strides import Strides
from backpropagation import Delegate, BackwardFn
from operators import AddTensor, ZeroGrad
from validators import Validator
from ancestry import Ancestor
from gradbox import Gradbox
from layout.int_tuple import IntArray
from common_utils import panic


@fieldwise_init
@register_passable
struct ViewBackward[dtype: DType](ImplicitlyCopyable):
    var shape: Shape
    var strides: Strides
    var offset: Int

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    # fn backward_absolute(
    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        parent = output.ancestry().get(0)
        gradbox = output.grad().copy()
        parent_shape = parent.shape()
        parent_strides = parent.strides()
        parent_offset = parent.offset()
        parent_max_index = parent.max_index()

        parent_gradbox = Gradbox[dtype].zeros(parent_shape)

        if parent_shape == Shape():
            parent_gradbox[IntArray()] = gradbox.item()
        else:
            var position = IntArray(size=parent_shape.rank())

            for child_coord in self.shape:
                # Compute absolute storage index
                var abs_index = self.offset
                for i in range(len(self.strides)):
                    abs_index += self.strides[i] * child_coord[i]

                # CRITICAL: PyTorch-style boundary check
                if abs_index < parent_offset or abs_index > parent_max_index:
                    continue  # Skip - not in parent's storage region

                # Convert to parent's logical coordinates
                var remaining = abs_index - parent_offset
                var valid = True

                for i in range(parent_shape.rank()):
                    stride = parent_strides[i]
                    dim_size = parent_shape[i]

                    if stride == 0:
                        position[i] = 0  # Broadcast dimension
                    else:
                        position[i] = remaining // stride
                        # Check if coordinate is within parent's shape bounds
                        if position[i] < 0 or position[i] >= dim_size:
                            valid = False
                            break
                        remaining = remaining % stride

                # Final check - no remainder and all coordinates valid
                if valid and remaining == 0:
                    parent_gradbox[position] += gradbox[child_coord]
        return [
            (parent^, parent_gradbox^, AddTensor),
            (Ancestor(output), gradbox^, ZeroGrad),
        ]

    fn backward_conservative(
        # fn backward(
        self,
        output: Tensor[dtype],
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        parent = output.ancestry().get(0)
        gradbox = output.grad().copy()
        parent_shape = parent.shape()
        parent_strides = parent.strides()

        parent_gradbox = Gradbox[dtype].zeros(parent_shape)

        if parent_shape == Shape():
            parent_gradbox[IntArray()] = gradbox.item()
        else:
            var position = IntArray(size=parent_shape.rank())

            for child_coord in self.shape:
                # Compute absolute storage index
                var abs_index = self.offset
                for i in range(len(self.strides)):
                    abs_index += self.strides[i] * child_coord[i]

                # Convert to parent's logical coordinates
                # Since we enforced boundaries, this MUST succeed
                var remaining = abs_index - parent.offset()

                for i in range(parent_shape.rank()):
                    parent_stride = parent_strides[i]
                    position[i] = remaining // parent_stride
                    remaining = remaining % parent_stride

                # With boundary enforcement, this should always be valid
                parent_gradbox[position] += gradbox[child_coord]

        return [
            (parent^, parent_gradbox^, AddTensor),
            (Ancestor(output), gradbox^, ZeroGrad),
        ]


@register_passable
struct View[dtype: DType](Copyable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        shape: Shape,
        strides: Strides,
        offset: Int = 0,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False,
    ) -> Tensor[dtype]:
        _="""if not self.is_contiguous():
            panic(
                "Cannot create a view from a non-contiguous tensor. Use"
                " reshape() or contiguous().view() instead"
            )"""

        var abs_offset: Int
        var abs_strides: Strides

        if not validated:
            (abs_offset, abs_strides) = Validator.validate_view_params(
                self, shape, strides, offset
            )

        else:
            abs_offset = offset
            abs_strides = strides  # already absolute

        out = Tensor[dtype].build_view(
            #self.address(),
            #UnsafePointer(to=self),
            self.unsafe_address(),
            shape,
            Optional(abs_strides),
            abs_offset,
            requires_grad=False,
        )

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = ViewBackward[dtype](
                    shape, abs_strides, abs_offset
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main():
    pass
