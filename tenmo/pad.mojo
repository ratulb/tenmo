# Generalized Padding Implementation for Mojo Tensor Library.

"""
Padding specification.

For N-dimensional tensor, padding is specified as a list of tuples:
[(before_0, after_0), (before_1, after_1), ..., (before_N-1, after_N-1)]

Or as a flat list (PyTorch style, applied from last to first dimension):
[before_last, after_last, before_second_last, after_second_last, ...]

Examples:
- 2D tensor (H, W): pad = [(1, 2), (3, 4)] means:
  - Dimension 0 (H): add 1 before, 2 after
  - Dimension 1 (W): add 3 before, 4 after

- 4D tensor (N, C, H, W): pad = [(0, 0), (0, 0), (1, 1), (2, 2)] means:
  - No padding on batch and channel dimensions
  - Pad H with 1 on each side
  - Pad W with 2 on each side
"""

from .tensor import Tensor
from .shared.shapes import Shape
from .gradbox import Gradbox
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType

from .shared.mnemonics import AddTensor
from .shared.panic import panic
from .shared.intarray import IntArray
from std.utils import Variant
from std.sys import simd_width_of
from std.sys.info import num_physical_cores
from std.memory import unsafe_memcpy
from .shared.indexhelper import IndexCalculator, IndexIterator
from max.algorithm import parallelize
from .ancestry import Ancestor
from std.sys import has_accelerator
from .kernels.pad_kernel import PadKernel

comptime Padding = Variant[String, Int, Tuple[Int, Int], List[Tuple[Int, Int]]]


@fieldwise_init
struct PadArg(ArgumentType):
    var pad: List[Tuple[Int, Int]]
    var mode: String

    def __init__(out self, *, copy: Self):
        self.pad = copy.pad.copy()
        self.mode = copy.mode.copy()


@fieldwise_init
struct PadBackward[dtype: DType](BackwardFnType, ImplicitlyCopyable):
    comptime datatype = Self.dtype
    """Backward pass for padding operation - handles all modes."""

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        """
                Backward pass: Accumulate gradients based on padding mode.
        """
        ref bwd_arg = output.ancestry().backward_fn().get[PadArg]()
        var pad = bwd_arg.pad.copy()
        var mode = bwd_arg.mode.copy()
        ref grad_out = output.gradients()
        var ancestor_ref = output.ancestry().get(0)

        if ancestor_ref.requires_grad:
            ref parent_shape = ancestor_ref.shape()

            # GPU BACKWARD (constant mode)
            comptime if has_accelerator():
                if mode != "constant" and grad_out.is_on_gpu():
                    panic(
                        "Pad: mode '"
                        + mode
                        + "' is not supported on GPU (constant mode only)",
                        "at PadBackward → backward",
                    )
                if mode == "constant" and grad_out.is_on_gpu():
                    try:
                        if ancestor_ref.requires_grad:
                            var gpu_device = grad_out.buffer().device()
                            var grad_parent = Gradbox[Self.dtype].zeros(
                                parent_shape, device=gpu_device
                            )
                            PadKernel[Self.dtype].launch_backward(
                                grad_out.buffer().layout(),
                                grad_out.buffer().device_state.value(),
                                grad_parent.buffer().layout(),
                                grad_parent.buffer().device_state.value(),
                                pad,
                            )
                            ancestor_ref.update_grad(
                                grad_parent^, AddTensor, None
                            )
                    except e:
                        panic("PadBackward GPU backward failed: " + String(e))
                    # Always appended (engine fanin-completion contract).
                    parent_ids.append(ancestor_ref._id)
                    grad_out.zero_grad()
                    return

            # CPU BACKWARD
            var grad_parent = Gradbox[Self.dtype].zeros(parent_shape)

            if mode == "constant":
                # The 4D SIMD path assumes pad[0]==pad[1]==(0,0) (Conv2D: no
                # batch/channel padding) — anything else takes the generic
                # path, which offsets every axis.
                if (
                    parent_shape.rank() == 4
                    and pad[0][0] == 0
                    and pad[0][1] == 0
                    and pad[1][0] == 0
                    and pad[1][1] == 0
                ):
                    Self._extract_4d_constant_simd(
                        grad_out, grad_parent, pad, parent_shape
                    )
                else:
                    Self._extract_constant(
                        grad_out, grad_parent, pad, parent_shape
                    )
            elif mode == "circular":
                Self._extract_circular(grad_out, grad_parent, pad, parent_shape)
            elif mode == "replicate":
                Self._extract_replicate(
                    grad_out, grad_parent, pad, parent_shape
                )
            elif mode == "reflect":
                Self._extract_reflect(grad_out, grad_parent, pad, parent_shape)

            ancestor_ref.update_grad(grad_parent^, AddTensor, None)

        # Always appended: parent_ids is the engine's fanin-completion
        # signal (appended set must equal ancestry set); update_grad
        # no-ops internally for untracked parents.
        parent_ids.append(ancestor_ref._id)

        grad_out.zero_grad()

    @staticmethod
    def _extract_constant(
        grad_out: Gradbox[Self.dtype],
        grad_parent: Gradbox[Self.dtype],
        pad: List[Tuple[Int, Int]],
        parent_shape: Shape,
    ):
        """Extract gradients for constant padding - simple extraction from center.
        """
        var ndim = parent_shape.rank()
        var W = parent_shape[ndim - 1]
        if W == 0:
            return

        # Fast path: unpadded last dim + both contiguous → SIMD row adds
        # instead of per-element get/set.
        if (
            pad[ndim - 1][0] == 0
            and pad[ndim - 1][1] == 0
            and grad_out.buffer().is_contiguous()
            and grad_parent.buffer().is_contiguous()
        ):
            comptime simd_w = simd_width_of[Self.dtype]()
            var outer = parent_shape.num_elements() // W
            var gout_ptr = grad_out.data_ptr()
            var gout_base = grad_out.offset()
            var gparent_ptr = grad_parent.data_ptr()
            var gparent_base = grad_parent.offset()
            var vec_end = (W // simd_w) * simd_w
            if ndim == 1:
                for w in range(0, vec_end, simd_w):
                    var acc = gparent_ptr.unsafe_load[width=simd_w](
                        gparent_base + w
                    )
                    acc += gout_ptr.unsafe_load[width=simd_w](gout_base + w)
                    gparent_ptr.unsafe_store[width=simd_w](
                        gparent_base + w, acc
                    )
                for w in range(vec_end, W):
                    var cur = gparent_ptr.unsafe_load(gparent_base + w)
                    gparent_ptr.unsafe_store(
                        gparent_base + w,
                        cur + gout_ptr.unsafe_load(gout_base + w),
                    )
                return
            # Contiguous grad_out strides (physical units).
            var rs = List[Int]()
            var run = 1
            for i in range(ndim - 1, -1, -1):
                rs.append(run)
                run *= grad_out.shape()[i]
            var outer_dims = IntArray()
            for i in range(ndim - 1):
                outer_dims.append(parent_shape[i])
            var outer_shape = Shape(outer_dims)
            for row in range(outer):
                var coords = IndexCalculator.index_to_coord(outer_shape, row)
                var src = gout_base
                for i in range(ndim - 1):
                    src += (coords[i] + pad[i][0]) * rs[ndim - 1 - i]
                var dst = gparent_base + row * W
                for w in range(0, vec_end, simd_w):
                    var acc = gparent_ptr.unsafe_load[width=simd_w](dst + w)
                    acc += gout_ptr.unsafe_load[width=simd_w](src + w)
                    gparent_ptr.unsafe_store[width=simd_w](dst + w, acc)
                for w in range(vec_end, W):
                    var cur = gparent_ptr.unsafe_load(dst + w)
                    gparent_ptr.unsafe_store(
                        dst + w, cur + gout_ptr.unsafe_load(src + w)
                    )
            return

        # Calculate offset where input data starts in padded output
        var offset_list = List[Int]()
        for i in range(ndim):
            offset_list.append(pad[i][0])

        # Iterate over parent's shape and extract gradients.
        # An odometer over the PARENT shape with the GOUT strides yields
        # each source storage address directly (plus a precomputed pad
        # displacement) — no per-element coords, no flatten_index. A plain
        # linear index would be wrong here: padding gaps break C-order
        # density between the regions. grad_parent is fresh zeros, so its
        # side stays linear (store ≡ accumulate).
        var gout_buf = grad_out.buffer()
        var gout_strides = grad_out.strides()
        var pad_const = grad_out.offset()
        for d in range(ndim):
            pad_const += offset_list[d] * gout_strides[d]
        var dst = grad_parent.data_ptr()
        var addr_it = IndexIterator(
            shape=Pointer(to=parent_shape),
            strides=Pointer(to=gout_strides),
            start_offset=pad_const,
        )
        var o = 0
        for go in addr_it:
            dst[unsafe_offset=o] = gout_buf.storage_get[checked=False](go)
            o += 1

    @staticmethod
    def _extract_4d_constant_simd(
        grad_out: Gradbox[Self.dtype],
        grad_parent: Gradbox[Self.dtype],
        pad: List[Tuple[Int, Int]],
        parent_shape: Shape,
    ):
        """
        Highly optimized 4D extraction for Conv2D.

        Input: (N, C, H_in, W_in)
        Padded: (N, C, H_pad, W_pad)

        Assumes: pad[0] = (0, 0) and pad[1] = (0, 0) (no batch/channel padding)
        Handles: Asymmetric height/width padding:
                 pad[2] = (pad_top, pad_bottom)   # Can be different!
                 pad[3] = (pad_left, pad_right)   # Can be different!

        Strategy: Extract row-by-row using SIMD from correct offsets
        """
        var N = parent_shape[0]
        var C = parent_shape[1]
        var H_in = parent_shape[2]
        var W_in = parent_shape[3]

        # Extract padding amounts (only height/width)
        var pad_top = pad[2][0]
        var pad_left = pad[3][0]

        var grad_out_shape = grad_out.shape()
        var H_pad = grad_out_shape[2]
        var W_pad = grad_out_shape[3]

        var grad_out_ptr = grad_out.data_ptr()
        var grad_in_ptr = grad_parent.data_ptr().unsafe_mut_cast[True]()

        comptime simd_w = simd_width_of[Self.dtype]()
        # Strides
        var out_stride_N = C * H_pad * W_pad
        var out_stride_C = H_pad * W_pad
        var out_stride_H = W_pad

        var in_stride_N = C * H_in * W_in
        var in_stride_C = H_in * W_in
        var in_stride_H = W_in

        # Parallelize over N×C
        var total_slices = N * C

        def extract_slice(slice_idx: Int) {imm}:
            var n = slice_idx // C
            var c = slice_idx % C

            # Source position in padded gradient (no batch/channel offset!)
            var out_nc_base = n * out_stride_N + c * out_stride_C

            # Destination position in input gradient
            var in_nc_base = n * in_stride_N + c * in_stride_C

            # Extract each row
            for h in range(H_in):
                # Source row in padded gradient (account for height and width padding)
                var out_row_start = (
                    out_nc_base + (h + pad_top) * out_stride_H + pad_left
                )

                # Destination row in input gradient
                var in_row_start = in_nc_base + h * in_stride_H

                # Copy entire row (SIMD vectorized)
                var w = 0
                var vec_end = (W_in // simd_w) * simd_w

                # Vectorized copy
                for _ in range(vec_end // simd_w):
                    var vec = grad_out_ptr.unsafe_load[width=simd_w](
                        out_row_start + w
                    )
                    grad_in_ptr.unsafe_store[width=simd_w](
                        in_row_start + w, vec
                    )
                    w += simd_w

                # Scalar tail
                for w_tail in range(vec_end, W_in):
                    grad_in_ptr[
                        unsafe_offset=in_row_start + w_tail
                    ] = grad_out_ptr[unsafe_offset=out_row_start + w_tail]

        parallelize(extract_slice, total_slices)

    @staticmethod
    def _extract_circular(
        grad_out: Gradbox[Self.dtype],
        grad_parent: Gradbox[Self.dtype],
        pad: List[Tuple[Int, Int]],
        parent_shape: Shape,
    ):
        """Extract gradients for circular padding - accumulate from all wrapped positions.
        """
        var ndim = parent_shape.rank()
        var grad_out_shape = grad_out.shape()

        # Iterate over ALL output positions and accumulate gradients
        for out_coord in grad_out_shape:
            # Map output coordinate back to input coordinate (same logic as forward)
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = parent_shape[i]

                # Same wrapping logic as forward pass
                var in_idx = (out_idx - before) % in_size
                if in_idx < 0:
                    in_idx += in_size
                in_coord.append(in_idx)

            # ACCUMULATE gradient (not replace!)
            grad_parent[in_coord] += grad_out[out_coord]

    @staticmethod
    def _extract_replicate(
        grad_out: Gradbox[Self.dtype],
        grad_parent: Gradbox[Self.dtype],
        pad: List[Tuple[Int, Int]],
        parent_shape: Shape,
    ):
        """Extract gradients for replicate padding - accumulate from all replicated positions.
        """
        var ndim = parent_shape.rank()
        var grad_out_shape = grad_out.shape()

        # Iterate over ALL output positions and accumulate gradients
        for out_coord in grad_out_shape:
            # Map output coordinate back to input coordinate (same logic as forward)
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = parent_shape[i]

                # Same clamping logic as forward pass (replicate edges)
                var in_idx = out_idx - before
                in_idx = max(0, min(in_size - 1, in_idx))
                in_coord.append(in_idx)

            # ACCUMULATE gradient
            grad_parent[in_coord] += grad_out[out_coord]

    @staticmethod
    def _extract_reflect(
        grad_out: Gradbox[Self.dtype],
        grad_parent: Gradbox[Self.dtype],
        pad: List[Tuple[Int, Int]],
        parent_shape: Shape,
    ):
        """Extract gradients for reflect padding - accumulate from all reflected positions.
        """
        var ndim = parent_shape.rank()
        var grad_out_shape = grad_out.shape()

        # Iterate over ALL output positions and accumulate gradients
        for out_coord in grad_out_shape:
            # Map output coordinate back to input coordinate (same logic as forward)
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = parent_shape[i]

                # Same reflection logic as forward pass
                var in_idx: Int
                if out_idx < before:
                    # Reflect from left border
                    in_idx = before - out_idx
                    in_idx = min(in_idx, in_size - 1)
                elif out_idx >= before + in_size:
                    # Reflect from right border
                    var offset = out_idx - (before + in_size)
                    in_idx = in_size - 2 - offset
                    in_idx = max(0, in_idx)
                else:
                    # Inside original region
                    in_idx = out_idx - before

                # Final clamp
                in_idx = max(0, min(in_size - 1, in_idx))
                in_coord.append(in_idx)

            # ACCUMULATE gradient
            grad_parent[in_coord] += grad_out[out_coord]


@fieldwise_init
struct Pad[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    """
    Generalized padding operation supporting:
    - Arbitrary dimensions.
    - Asymmetric padding (different on each side).
    - Multiple padding modes.
    - Proper gradient flow in backward pass.
    """

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        x: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        mode: String = "constant",
        value: Scalar[Self.dtype] = 0.0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """Pad tensor along specified dimensions."""
        var x_shape = x.shape()
        var ndim = x_shape.rank()

        # Validate padding specification
        if len(pad) != ndim:
            panic("Pad: padding must be specified for all dimensions")

        # Negative pads (crop semantics) are unimplemented: reject up front
        # instead of flowing into undersized-shape math and OOB downstream.
        for i in range(ndim):
            if pad[i][0] < 0 or pad[i][1] < 0:
                panic(
                    "Pad: negative padding (cropping) is not supported, got ("
                    + String(pad[i][0])
                    + ", "
                    + String(pad[i][1])
                    + ") on dimension "
                    + String(i),
                    "at Pad → forward",
                )

        # Calculate output shape
        var out_shape = List[Int]()
        for i in range(ndim):
            var before = pad[i][0]
            var after = pad[i][1]
            out_shape.append(x_shape[i] + before + after)

        # GPU PATH (constant mode only)
        comptime if has_accelerator():
            if x.is_on_gpu() and mode != "constant":
                panic(
                    "Pad: mode '"
                    + mode
                    + "' is not supported on GPU (constant mode only)",
                    "at Pad → forward",
                )
            if x.is_on_gpu() and mode == "constant":
                var result = Tensor[Self.dtype].full(
                    out_shape, value, device=x.device(), sync=sync
                )
                try:
                    PadKernel[Self.dtype].launch_forward(
                        x.buffer.layout(),
                        x.buffer.device_state.value(),
                        result.buffer.layout(),
                        result.buffer.device_state.value(),
                        pad,
                        sync=sync,
                    )
                except e:
                    panic("Pad GPU forward failed: " + String(e))

                comptime if track_grad:
                    var req_grad = requires_grad.or_else(x.requires_grad)
                    if req_grad:
                        result.requires_grad_(True)
                        var backwardFn = BackwardFn(
                            PadArg(pad.copy(), mode),
                            PadBackward[Self.dtype](),
                        )
                        backwardFn.needs_parent_data = True
                        result.add_ancestry(backwardFn^, x)

                return result^

        # CPU PATH
        var result = Tensor[Self.dtype].zeros(out_shape)

        # Apply padding based on mode
        if mode == "constant":
            # The 4D SIMD path assumes pad[0]==pad[1]==(0,0) (Conv2D: no
            # batch/channel padding) — anything else takes the generic path,
            # which offsets every axis.
            if (
                ndim == 4
                and pad[0][0] == 0
                and pad[0][1] == 0
                and pad[1][0] == 0
                and pad[1][1] == 0
            ):
                Self._pad_4d_constant_simd(x, result, pad, value)
            else:
                Self._pad_constant(x, result, pad, value)
        elif mode == "reflect":
            Self._pad_reflect(x, result, pad)
        elif mode == "replicate":
            Self._pad_replicate(x, result, pad)
        elif mode == "circular":
            Self._pad_circular(x, result, pad)
        else:
            panic("Pad: unsupported mode")

        comptime if track_grad:
            var req_grad = requires_grad.or_else(x.requires_grad)
            if req_grad:
                result.requires_grad_(True)
                # PASS MODE TO BACKWARD!
                var backwardFn = BackwardFn(
                    PadArg(pad.copy(), mode),  # Add mode here
                    PadBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                result.add_ancestry(backwardFn^, x)

        return result^

    @staticmethod
    def _pad_constant(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        value: Scalar[Self.dtype],
    ):
        """Apply constant padding (most common for CNNs)."""
        # Fill with pad value
        result.fill(value)

        # Copy input data to center region
        # We need to map input indices to output indices
        Self._copy_to_padded_region(x, result, pad)

    @staticmethod
    def _copy_to_padded_region(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ):
        """Copy input tensor to the non-padded region of output."""
        var x_shape = x.shape()
        var ndim = x_shape.rank()
        var W = x_shape[ndim - 1]
        if W == 0:
            return

        # Fast path: unpadded last dim + both contiguous → each outer
        # position maps to a contiguous W-run: bulk memcpy per row instead
        # of per-element get/set.
        if (
            pad[ndim - 1][0] == 0
            and pad[ndim - 1][1] == 0
            and x.is_contiguous()
            and result.is_contiguous()
        ):
            var outer = x.numels() // W
            var x_ptr = x.data_ptr()
            var x_base = x.offset()
            var result_ptr = result.data_ptr()
            var result_base = result.offset()
            # Contiguous result strides (physical units).
            var rs = List[Int]()
            var run = 1
            for i in range(ndim - 1, -1, -1):
                rs.append(run)
                run *= result.shape()[i]
            # rs was built back-to-front; index [ndim-1-i] == stride of dim i.
            if ndim == 1:
                unsafe_memcpy(
                    dest=result_ptr.unsafe_offset(result_base),
                    src=x_ptr.unsafe_offset(x_base),
                    count=W,
                )
                return
            var outer_dims = IntArray()
            for i in range(ndim - 1):
                outer_dims.append(x_shape[i])
            var outer_shape = Shape(outer_dims)
            for row in range(outer):
                var coords = IndexCalculator.index_to_coord(outer_shape, row)
                var dst = result_base
                for i in range(ndim - 1):
                    dst += (coords[i] + pad[i][0]) * rs[ndim - 1 - i]
                unsafe_memcpy(
                    dest=result_ptr.unsafe_offset(dst),
                    src=x_ptr.unsafe_offset(x_base + row * W),
                    count=W,
                )
            return

        # Calculate offset in output for where input data starts
        var offset_list = List[Int]()
        for i in range(ndim):
            offset_list.append(pad[i][0])  # before padding

        # Iterate over all elements of input
        for coord in x_shape:
            var result_indices = coord  # Copy it
            result_indices += offset_list
            result[result_indices] = x[coord]

    @staticmethod
    def _pad_4d_constant_simd(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        value: Scalar[Self.dtype],
    ):
        """
                Highly optimized 4D constant padding for Conv2D.

        Assumes: pad[0] = (0, 0) and pad[1] = (0, 0) (no batch/channel padding)
        Handles: Asymmetric height/width padding:
                 pad[2] = (pad_top, pad_bottom)   # Can be different!
                 pad[3] = (pad_left, pad_right)   # Can be different!

        Strategy:
        1. Fill entire output with pad value (SIMD vectorized)
        2. Copy input data row-by-row to correct offset
        3. Parallelize over N×C slices
        """
        var x_shape = x.shape()
        var result_shape = result.shape()

        var N = x_shape[0]
        var C = x_shape[1]
        var H_in = x_shape[2]
        var W_in = x_shape[3]

        var H_out = result_shape[2]
        var W_out = result_shape[3]

        # Extract padding amounts (only height/width can be asymmetric)
        var pad_top = pad[2][0]
        var pad_left = pad[3][0]
        # Note: pad[0] and pad[1] are assumed to be (0,0) for Conv2D

        var x_ptr = x.data_ptr()
        var result_ptr = result.data_ptr()

        comptime simd_w = simd_width_of[Self.dtype]()
        # Step 1: Fill with pad value (SIMD vectorized)
        var total_elements = N * C * H_out * W_out
        var fill_vec = SIMD[Self.dtype, simd_w](value)

        var i = 0
        var vec_end = (total_elements // simd_w) * simd_w

        for _ in range(vec_end // simd_w):
            result_ptr.unsafe_store[width=simd_w](i, fill_vec)
            i += simd_w

        for j in range(vec_end, total_elements):
            result_ptr[unsafe_offset=j] = value

        # Step 2: Copy input data (simplified - no batch/channel offset!)
        var x_stride_N = C * H_in * W_in
        var x_stride_C = H_in * W_in
        var x_stride_H = W_in

        var result_stride_N = C * H_out * W_out
        var result_stride_C = H_out * W_out
        var result_stride_H = W_out

        var total_slices = N * C

        def copy_slice(slice_idx: Int) {imm}:
            var n = slice_idx // C
            var c = slice_idx % C

            # Source position in input
            var x_nc_base = n * x_stride_N + c * x_stride_C

            # Destination position in output (no batch/channel offset needed!)
            var result_nc_base = n * result_stride_N + c * result_stride_C

            # Copy each row
            for h in range(H_in):
                var x_row_start = x_nc_base + h * x_stride_H

                # Destination row (account for height and width padding)
                var result_row_start = (
                    result_nc_base + (h + pad_top) * result_stride_H + pad_left
                )

                # SIMD-vectorized row copy
                var w = 0
                var vec_end = (W_in // simd_w) * simd_w

                for _ in range(vec_end // simd_w):
                    var vec = x_ptr.unsafe_load[width=simd_w](x_row_start + w)
                    result_ptr.unsafe_store[width=simd_w](
                        result_row_start + w, vec
                    )
                    w += simd_w

                # Scalar tail
                for w_tail in range(vec_end, W_in):
                    result_ptr[unsafe_offset=result_row_start + w_tail] = x_ptr[
                        unsafe_offset=x_row_start + w_tail
                    ]

        parallelize(copy_slice, total_slices)

    @staticmethod
    def _map_index(
        mode: String, out_idx: Int, before: Int, in_size: Int
    ) -> Int:
        """Scalar index map shared by the row-wise fast path.
        Mirrors the per-element formulas in the legacy _pad_replicate/
        _pad_reflect/
        _pad_circular loops exactly (including clamps)."""
        if mode == "replicate":
            var in_idx = out_idx - before
            return max(0, min(in_size - 1, in_idx))
        elif mode == "reflect":
            var in_idx: Int
            if out_idx < before:
                in_idx = before - out_idx
                in_idx = min(in_idx, in_size - 1)
            elif out_idx >= before + in_size:
                var offset = out_idx - (before + in_size)
                in_idx = in_size - 2 - offset
                in_idx = max(0, in_idx)
            else:
                in_idx = out_idx - before
            return max(0, min(in_size - 1, in_idx))
        else:  # circular
            var in_idx = (out_idx - before) % in_size
            if in_idx < 0:
                in_idx += in_size
            return in_idx

    @staticmethod
    def _pad_nd_mode(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        mode: String,
    ):
        """Row-wise fast path for replicate/reflect/circular (both contiguous).

        The last-dim map depends only on pad/sizes, so it is precomputed once
        (W_out ints). Each output row then gathers from its input row base via
        direct pointer arithmetic; maximal constant-offset runs use memcpy.
        Rows are parallelized above the house crossover gate.
        """
        var x_shape = x.shape()
        var result_shape = result.shape()
        var ndim = x_shape.rank()
        var W_in = x_shape[ndim - 1]
        var W_out = result_shape[ndim - 1]
        if W_out == 0:
            return

        # Last-dim map (identical for every row).
        var in_w = List[Int]()
        for w in range(W_out):
            in_w.append(Self._map_index(mode, w, pad[ndim - 1][0], W_in))

        # Row-major input strides (x is contiguous).
        var xst = List[Int]()
        var run = 1
        for d in range(ndim - 1, -1, -1):
            xst.append(run)
            run *= x_shape[d]
        # xst[ndim-1-d] == stride of dim d.

        var x_ptr = x.data_ptr()
        var x_base = x.offset()
        var result_ptr = result.data_ptr()
        var result_base = result.offset()
        var outer = result_shape.num_elements() // W_out
        var n_threads = num_physical_cores()

        # Outer-dim sizes for row decomposition (all but last dim).
        var outer_shape = Shape()
        if ndim > 1:
            var outer_dims = IntArray()
            for d in range(ndim - 1):
                outer_dims.append(result_shape[d])
            outer_shape = Shape(outer_dims^)

        def pad_row(r: Int) {imm}:
            var src_base = x_base
            if ndim > 1:
                var coords = IndexCalculator.index_to_coord(outer_shape, r)
                for d in range(ndim - 1):
                    var in_d = Self._map_index(
                        mode, coords[d], pad[d][0], x_shape[d]
                    )
                    src_base += in_d * xst[ndim - 1 - d]
            var dst = result_base + r * W_out
            # Constant-offset runs → memcpy; short/broken runs → scalar.
            var w = 0
            while w < W_out:
                var c = in_w[w] - w
                var w2 = w + 1
                while w2 < W_out and in_w[w2] - w2 == c:
                    w2 += 1
                if w2 - w >= 16:
                    unsafe_memcpy(
                        dest=result_ptr.unsafe_offset(dst + w),
                        src=x_ptr.unsafe_offset(src_base + in_w[w]),
                        count=w2 - w,
                    )
                else:
                    for ww in range(w, w2):
                        result_ptr[unsafe_offset=dst + ww] = x_ptr[
                            unsafe_offset=src_base + in_w[ww]
                        ]
                w = w2

        if (
            outer >= n_threads
            and result_shape.num_elements() >= n_threads * 32768
        ):
            parallelize(pad_row, outer, n_threads)
        else:
            for r in range(outer):
                pad_row(r)

    @staticmethod
    def _pad_replicate(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ):
        """Apply replicate padding - repeat edge values using coordinate iteration.
        """
        if x.is_contiguous() and result.is_contiguous() and x.numels() > 0:
            Self._pad_nd_mode(x, result, pad, "replicate")
            return
        var x_shape = x.shape()
        var result_shape = result.shape()
        var ndim = x_shape.rank()

        # Iterate over all output coordinates
        for out_coord in result_shape:
            # Map to input with edge replication
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = x_shape[i]

                # Clamp to valid input range (replicate edges)
                var in_idx = out_idx - before
                in_idx = max(0, min(in_size - 1, in_idx))
                in_coord.append(in_idx)

            result[out_coord] = x[in_coord]

    @staticmethod
    def _pad_reflect(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ):
        """Apply reflect padding - mirror at borders using coordinate iteration.
        """
        if x.is_contiguous() and result.is_contiguous() and x.numels() > 0:
            Self._pad_nd_mode(x, result, pad, "reflect")
            return
        var x_shape = x.shape()
        var result_shape = result.shape()
        var ndim = x_shape.rank()

        # Iterate over all output coordinates
        for out_coord in result_shape:
            # Map output coordinates to input coordinates with reflection
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = x_shape[i]

                var in_idx: Int
                if out_idx < before:
                    # Reflect from left border
                    in_idx = before - out_idx
                    # Clamp to avoid out of bounds
                    in_idx = min(in_idx, in_size - 1)
                elif out_idx >= before + in_size:
                    # Reflect from right border
                    var offset = out_idx - (before + in_size)
                    in_idx = in_size - 2 - offset
                    # Clamp to avoid out of bounds
                    in_idx = max(0, in_idx)
                else:
                    # Inside original region
                    in_idx = out_idx - before

                # Final clamp to ensure valid range
                in_idx = max(0, min(in_size - 1, in_idx))
                in_coord.append(in_idx)

            result[out_coord] = x[in_coord]

    @staticmethod
    def _pad_circular(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ):
        """Apply circular padding - wrap around using coordinate iteration."""
        if x.is_contiguous() and result.is_contiguous() and x.numels() > 0:
            Self._pad_nd_mode(x, result, pad, "circular")
            return
        var x_shape = x.shape()
        var result_shape = result.shape()
        var ndim = x_shape.rank()

        # Iterate over all output coordinates
        for out_coord in result_shape:
            # Map to input with wrapping
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = x_shape[i]

                # Wrap around using modulo
                var in_idx = (out_idx - before) % in_size
                if in_idx < 0:
                    in_idx += in_size
                in_coord.append(in_idx)

            result[out_coord] = x[in_coord]
