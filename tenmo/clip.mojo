from .tensor import Tensor
from .mnemonics import AddTensor
from .backpropagation import BackwardFnArg, ClipArg, BACKWARD_CLIP
from .gradbox import Gradbox
from std.sys import simd_width_of
from .ancestry import Ancestor


@fieldwise_init
struct ClipBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """Gradient passes where min ≤ x ≤ max, blocked elsewhere."""
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[ClipArg[Self.dtype]]()
        )
        var (min_val, max_val) = bwd_arg.min_val, bwd_arg.max_val
        ref grad_output = output.gradients()[]
        var parent = output.ancestry().get(0)
        ref shape = parent.shape()
        var parent_buffer = parent.buffer()
        var parent_gradbox = Gradbox[Self.dtype].zeros(shape, share=False)

        if parent_buffer.is_contiguous():
            var src = parent_buffer.data_ptr()
            var dest = parent_gradbox.data_ptr()
            var grad_output_data = grad_output.data_ptr()
            var offset = parent_buffer.offset
            var numels = parent_buffer.numels()

            comptime simd_width = simd_width_of[Self.dtype]()

            for i in range(0, numels - simd_width + 1, simd_width):
                var x = src.load[width=simd_width](offset + i)
                var grad_out = grad_output_data.load[width=simd_width](i)

                # Mask: gradient passes only if min ≤ x ≤ max
                var in_range = x.ge(min_val) & x.le(max_val)

                var mask_float = in_range.cast[Self.dtype]()
                var grad_in = grad_out * mask_float

                dest.store[width=simd_width](i, grad_in)

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                var x = src[offset + i]
                var grad_out = grad_output_data[i]

                if x >= min_val and x <= max_val:
                    dest[i] = grad_out  # Pass through
                else:
                    dest[i] = Scalar[Self.dtype](0)  # Block
        else:
            # Non-contiguous fallback
            for coord in shape:
                var x = parent_buffer[coord]
                if x >= min_val and x <= max_val:
                    parent_gradbox[coord] = grad_output[coord]
                else:
                    parent_gradbox[coord] = Scalar[Self.dtype](0)

        return [(parent^, parent_gradbox^, AddTensor)]


@fieldwise_init
struct Clip[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        min_val: Scalar[Self.dtype],
        max_val: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Clip values: y = clamp(x, min, max)."""
        var shape = self.shape()
        var out = Tensor[Self.dtype].zeros(shape, requires_grad=False)

        if self.is_contiguous():
            var src = self.data_ptr()
            var dest = out.data_ptr()
            var offset = self.offset()
            var numels = self.numels()

            comptime simd_width = simd_width_of[Self.dtype]()

            for i in range(0, numels - simd_width + 1, simd_width):
                var x = src.load[width=simd_width](offset + i)
                var clamped = x.clamp(min_val, max_val)
                dest.store[width=simd_width](i, clamped)

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                var x = src[offset + i]
                dest[i] = x.clamp(min_val, max_val)
        else:
            # Non-contiguous fallback
            for coord in shape:
                var x = self[coord]
                out[coord] = x.clamp(min_val, max_val)

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_CLIP, ClipArg[Self.dtype](min_val, max_val)
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^
