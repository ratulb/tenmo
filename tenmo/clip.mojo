from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType

from .gradbox import Gradbox
from std.sys import simd_width_of
from .ancestry import Ancestor


@fieldwise_init
struct ClipArg[dtype: DType](ArgumentType):
    var min_val: Scalar[Self.dtype]
    var max_val: Scalar[Self.dtype]


@fieldwise_init
struct ClipBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        """Gradient passes where min ≤ x ≤ max, blocked elsewhere."""
        var bwd_arg = output.ancestry().backward_fn().get[ClipArg[Self.dtype]]()
        var (min_val, max_val) = bwd_arg.min_val, bwd_arg.max_val
        ref grad_output = output.gradients()
        var parent = output.ancestry().get(0)
        ref shape = parent.shape()
        var parent_buffer = parent.buffer()
        var parent_gradbox = Gradbox[Self.dtype].zeros(shape)

        if parent_buffer.is_contiguous():
            var src = parent_buffer.data_ptr()
            var dest = parent_gradbox.data_ptr()
            var grad_output_data = grad_output.data_ptr()
            var offset = parent_buffer.offset
            var numels = parent_buffer.numels()

            comptime simd_width = simd_width_of[Self.dtype]()

            for i in range(0, numels - simd_width + 1, simd_width):
                var x = src.unsafe_load[width=simd_width](offset + i)
                var grad_out = grad_output_data.unsafe_load[width=simd_width](i)

                # Mask: gradient passes only if min ≤ x ≤ max
                var in_range = x.ge(min_val) & x.le(max_val)

                var mask_float = in_range.cast[Self.dtype]()
                var grad_in = grad_out * mask_float

                dest.unsafe_store[width=simd_width](i, grad_in)

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                var x = src[unsafe_offset=offset + i]
                var grad_out = grad_output_data[unsafe_offset=i]

                if x >= min_val and x <= max_val:
                    dest[unsafe_offset=i] = grad_out  # Pass through
                else:
                    dest[unsafe_offset=i] = Scalar[Self.dtype](0)  # Block
        else:
            # Non-contiguous fallback: IndexIterator yields storage
            # offsets directly — no per-element IntArray alloc or
            # flatten_index. The grad output is contiguous in the common
            # case (single O(1) check up front, linear read); a strided
            # grad output keeps the scalar path.
            if grad_output.buffer().is_contiguous():
                var gout_base = grad_output.offset()
                var gout_buffer = grad_output.buffer()
                var dst = parent_gradbox.data_ptr()
                var idx = 0
                for p_off in parent_buffer.index_iterator():
                    var x = parent_buffer.storage_get[checked=False](p_off)
                    if x >= min_val and x <= max_val:
                        dst[unsafe_offset=idx] = gout_buffer.storage_get[
                            checked=False
                        ](gout_base + idx)
                    else:
                        dst[unsafe_offset=idx] = Scalar[Self.dtype](0)
                    idx += 1
            else:
                for coord in shape:
                    var x = parent_buffer[coord]
                    if x >= min_val and x <= max_val:
                        parent_gradbox[coord] = grad_output[coord]
                    else:
                        parent_gradbox[coord] = Scalar[Self.dtype](0)

        parent.update_grad(parent_gradbox^, AddTensor, None)
        parent_ids.append(parent._id)
        grad_output.zero_grad()


@fieldwise_init
struct Clip[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        min_val: Scalar[Self.dtype],
        max_val: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
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
                var x = src.unsafe_load[width=simd_width](offset + i)
                var clamped = x.clamp(min_val, max_val)
                dest.unsafe_store[width=simd_width](i, clamped)

            # Handle remainder
            for i in range(numels - numels % simd_width, numels):
                var x = src[unsafe_offset=offset + i]
                dest[unsafe_offset=i] = x.clamp(min_val, max_val)
        else:
            # Non-contiguous fallback
            for coord in shape:
                var x = self[coord]
                out[coord] = x.clamp(min_val, max_val)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn(
                    ClipArg[Self.dtype](min_val, max_val),
                    ClipBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^
