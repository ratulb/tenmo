from tenmo import Tensor
from shapes import Shape
from strides import Strides
from backpropagation import Delegate, BackwardFn
from operators import AddTensor, ZeroGrad
from validators import Validator
from ancestry import Ancestor
from gradbox import Gradbox
from layout.int_tuple import IntArray
from common_utils import panic, log_warning
from sys import simd_width_of


@fieldwise_init
@register_passable
struct ViewBackward[dtype: DType](ImplicitlyCopyable):
    var shape: Shape
    var strides: Strides
    var offset: Int

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        simdwidth: Int = simd_width_of[dtype]()
    ](self, output: Tensor[dtype]) -> List[
        Tuple[Ancestor[dtype], Gradbox[dtype], Int]
    ]:
        var parent = output.ancestry().get(0)
        ref gradbox = output.gradients()[]

        var parent_shape = parent.shape()
        ref parent_strides = parent.strides()
        var parent_offset = parent.offset()
        var parent_max_index = parent.max_index()
        var parent_gradbox = Gradbox[dtype].zeros(parent_shape)

        # Special case: scalar parent
        if parent_shape.rank() == 0:
            parent_gradbox[IntArray()] = gradbox.item()
        else:
            var view_rank = self.shape.rank()
            var parent_rank = parent_shape.rank()

            # Hoist metadata
            var view_data = gradbox.buffer.buffer.data
            var view_offset = gradbox.offset()
            ref view_strides = gradbox.strides()

            var parent_grad_data = parent_gradbox.buffer.buffer.data
            var parent_grad_offset = parent_gradbox.offset()
            ref parent_grad_strides = parent_gradbox.strides()

            # Check if we can use fast path (both contiguous, same shape)
            var use_fast_path = self.shape == parent_shape

            if use_fast_path:
                # Ultra-fast path: Same shape, both contiguous
                # Direct element-wise copy
                var numel = self.shape.num_elements()
                for i in range(numel):
                    parent_grad_data[parent_grad_offset + i] += view_data[
                        view_offset + i
                    ]
            else:
                # General path: Handle all view types using absolute offset mapping
                var position = IntArray(size=parent_rank)

                for child_coord in self.shape:
                    # Step 1: Compute absolute buffer index from child coordinates
                    var abs_index = self.offset
                    for i in range(view_rank):
                        abs_index += self.strides[i] * child_coord[i]

                        # CRITICAL: PyTorch-style boundary check
                        _="""if (
                            abs_index < parent_offset
                            or abs_index > parent_max_index
                        ):
                            continue  # Skip - not in parent's storage region"""

                    # Step 2: Convert absolute index to parent-relative index
                    var parent_rel_index = abs_index - parent_offset

                    # Step 3: Decompose relative index into parent coordinates
                    var remaining = parent_rel_index
                    var valid = True

                    for i in range(parent_rank):
                        var stride = parent_strides[i]
                        if stride == 0:
                            position[i] = 0  # Broadcast dimension
                        else:
                            position[i] = remaining // stride
                            if (
                                position[i] < 0
                                or position[i] >= parent_shape[i]
                            ):
                                valid = False
                                break
                            remaining = remaining % stride

                    # Step 4: Copy gradient if mapping is valid
                    if valid and remaining == 0:
                        # Compute parent gradient address
                        var parent_addr = parent_grad_offset
                        for i in range(parent_rank):
                            parent_addr += position[i] * parent_grad_strides[i]

                        # Compute view gradient address
                        var view_addr = view_offset
                        for i in range(view_rank):
                            view_addr += child_coord[i] * view_strides[i]

                        # Accumulate gradient
                        parent_grad_data[parent_addr] += view_data[view_addr]

        output.zero_grad()
        return [
            (parent^, parent_gradbox^, AddTensor),
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
        _="""if not Self.is_non_overlapping(self.shape(), self.strides()):
            log_warning("Warning: overlapping view detected")"""

        var abs_offset: Int
        var abs_strides: Strides

        if not validated:
            (abs_offset, abs_strides) = Validator.validate_view_params(
                self, shape, strides, offset
            )
        else:
            abs_offset = offset
            abs_strides = strides

        var out = Tensor[dtype].build_view(
            self.unsafe_address(),
            shape,
            Optional(abs_strides),
            abs_offset,
            requires_grad=False,
        )

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = ViewBackward[dtype](
                    shape, abs_strides, abs_offset
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^

    @staticmethod
    fn is_non_overlapping(shape: Shape, strides: Strides) -> Bool:
        """
        Fast heuristic to check if a view is non-overlapping.
        Equivalent to PyTorch's TensorImpl::is_non_overlapping_and_dense()
        (ignoring density condition).

        A view is non-overlapping if, when dimensions are sorted by
        increasing absolute stride, each stride is at least as large
        as the total span of all faster-changing dimensions.
        """
        rank = shape.rank()
        if rank == 0:
            return True

        # Pair (abs_stride, dim)
        var pairs = List[Tuple[Int, Int]](capacity=UInt(rank))
        for i in range(rank):
            pairs.append((abs(strides[i]), i))

        # Sort by stride ascending
        fn comp_fn(
            pair_a: Tuple[Int, Int], pair_b: Tuple[Int, Int]
        ) capturing -> Bool:
            return pair_a[0] < pair_b[0]

        sort[comp_fn](pairs)

        var required_stride = 1
        for abs_stride, dim in pairs:
            if shape[dim] == 0:
                continue
            if abs_stride < required_stride:
                # Overlap possible
                return False
            required_stride *= shape[dim]

        return True


fn main():
    pass
