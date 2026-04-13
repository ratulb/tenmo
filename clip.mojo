from tenmo import Tensor
from mnemonics import AddTensor
from backpropagation import BackwardFnArg, ArgumentType, BACKWARD_CLIP
from gradbox import Gradbox
from std.sys import simd_width_of


@fieldwise_init
struct ClipBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """Gradient passes where min ≤ x ≤ max, blocked elsewhere."""
        var (min_val, max_val) = output.bwd_fn_arg().arg[Tuple[Scalar[Self.dtype], Scalar[Self.dtype]]]
        ref grad_output = output.gradients()[]
        var parent = output.ancestry().get(0)
        ref shape = parent.shape()
        var parent_gradbox = Gradbox[Self.dtype].zeros(shape, share=False)

        if parent.is_contiguous():
            var src = parent.data_ptr()
            var dest = parent_gradbox.data_ptr()
            var grad_output_data = grad_output.data_ptr()
            var offset = parent.offset()
            var numels = parent.numels()

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
                var x = parent[coord]
                if x >= min_val and x <= max_val:
                    parent_gradbox[coord] = grad_output[coord]
                else:
                    parent_gradbox[coord] = Scalar[Self.dtype](0)

        return [(parent^, parent_gradbox^, AddTensor)]


@fieldwise_init
struct Clip[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                var bwd_arg = BackwardFnArg[Self.dtype](BACKWARD_CLIP, ArgumentType[Self.dtype]((
                    min_val, max_val
                )))
                out.bwdFnArg = Optional(bwd_arg^)
                out.add_ancestry(self)

        return out^
