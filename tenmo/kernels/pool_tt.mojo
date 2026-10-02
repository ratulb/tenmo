"""MaxPool2d forward + backward on GPU.

Launcher contract mirrors the CPU `MaxPool2d.forward` (`tenmo/pooling.mojo`):
rank-4 input, square kernel, stride (default = kernel), symmetric padding,
output dims from the same formula. Anything else panics — the CPU core's
fail-loud layout contract, repeated here so a GPU launch can never silently
misread a view.

Handshake: the launcher wraps the input as a whole `TileTensor` over an
offset-carried
`DevicePointer` with a strided runtime `Layout` — NO `materialize_contiguous`
round-trip; strided views are consumed natively and strided-correct by
construction (the CPU core assumes dense input; this kernel is strictly more
general). Fresh output/mask buffers stay contiguous row-major (DefaultEngine,
the ConvGpu pattern). The argmax mask travels as a raw int64 pointer: it is
always a dense fresh buffer, so flat indexing buys the layout nothing
.

Backward is one thread per (n, c): zero the input slice, then a serial
scatter over the output window — race-free by construction (ties and
overlapping windows collapse/accumulate exactly like the CPU scatter) with
the same accumulation order, hence bitwise-identical grads. T4-safe: plain
global loads/stores, no shared-memory staging, no cp.async.

Migration notes:
- View-vs-copy: the input tile wraps caller storage and dies at return;
  output + mask + grad_in are fresh buffers wrapped at return via
  `with_layout_device_state` by the op layer. Nothing is mutated between
  forward and backward (mask is written once, read once).
- Rank dispatch is a single arm: NCHW is fixed at 4. Non-4D inputs panic.
  Strided inputs share one layout type (`Strided4D`: rank-4, fully dynamic
  shape+stride); fresh outputs share `Layout4D`.
- Acceptance: bitwise vs the CPU forward on dense input, gradcheck 1e-4 vs
  the CPU backward, `pool_tt` suite green on the box.
"""

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from max.gpu.host import DevicePointer
from layout import TileTensor
from layout.tile_layout import row_major, RowMajorLayout, Layout as TileLayout
from layout.tensor_engine import DevicePointerEngine
from layout.coord import Coord
from std.utils.numerics import neg_inf

from ..shared.layout import Layout
from ..shared.shapes import Shape
from ..gpu.device import DeviceState
from ..shared.panic import panic
from .kernel_helpers import elementwise_launch_config


# Rank-4 fully-dynamic row-major (fresh outputs) + fully-dynamic strided
# (views). type_of names the strided type — TypeList source spelling is not
# writable by hand.
comptime Layout4D = RowMajorLayout[Int64, Int64, Int64, Int64]
comptime Strided4D = type_of(
    TileLayout(
        Coord(Int64(0), Int64(0), Int64(0), Int64(0)),
        Coord(Int64(0), Int64(0), Int64(0), Int64(0)),
    )
)


def pool_forward_nchw[dtype: DType](
    output: TileTensor[dtype, Layout4D, MutAnyOrigin],
    input: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    mask: Pointer[Scalar[DType.int64], MutAnyOrigin],
    KH_: Int64,
    s_: Int64,
    pad_: Int64,
):
    var N = Int(input.dim[0]())
    var C = Int(input.dim[1]())
    var H_in = Int(input.dim[2]())
    var W_in = Int(input.dim[3]())
    var H_out = Int(output.dim[2]())
    var W_out = Int(output.dim[3]())
    var KH = Int(KH_)
    var s = Int(s_)
    var pad = Int(pad_)
    # Flat index == dense output position (mask buffer is always dense).
    var total = N * C * H_out * W_out
    var tid = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    var span = Int(block_dim.x) * Int(grid_dim.x)
    var idx = tid
    while idx < total:
        var tmp = idx
        var ow = tmp % W_out
        tmp //= W_out
        var oh = tmp % H_out
        tmp //= H_out
        var c = tmp % C
        var n = tmp // C
        # Strict > : first-max wins, matching the CPU generic core.
        var max_val = neg_inf[dtype]()
        var max_idx = -1
        for ky in range(KH):
            var in_y = oh * s - pad + ky
            if in_y < 0 or in_y >= H_in:
                continue
            for kx in range(KH):
                var in_x = ow * s - pad + kx
                if in_x < 0 or in_x >= W_in:
                    continue
                var val = input[n, c, in_y, in_x]
                if val > max_val:
                    max_val = val
                    max_idx = in_y * W_in + in_x
        output[n, c, oh, ow] = max_val
        mask[unsafe_offset=idx] = Int64(max_idx)
        idx += span


def pool_backward_nchw[dtype: DType](
    grad_in: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_out: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    mask: Pointer[Scalar[DType.int64], MutAnyOrigin],
    H_in_: Int64,
    W_in_: Int64,
    H_out_: Int64,
    W_out_: Int64,
):
    # One thread per (n, c): zero the input slice, then serially scatter the
    # output window — the CPU scatter's order, hence bitwise-identical grads.
    var N = Int(grad_out.dim[0]())
    var C = Int(grad_out.dim[1]())
    var H_in = Int(H_in_)
    var W_in = Int(W_in_)
    var H_out = Int(H_out_)
    var W_out = Int(W_out_)
    var tid = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    if tid < N * C:
        var n = tid // C
        var c = tid % C
        var in_base = tid * H_in * W_in
        var out_base = tid * H_out * W_out
        for i in range(H_in * W_in):
            grad_in[unsafe_offset=in_base + i] = Scalar[dtype](0)
        for oy in range(H_out):
            for ox in range(W_out):
                var m = Int(
                    mask[unsafe_offset=out_base + oy * W_out + ox]
                )
                if m >= 0:
                    grad_in[unsafe_offset=in_base + m] += grad_out[n, c, oy, ox]


struct PoolTt[dtype: DType](ImplicitlyCopyable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def forward(
        input_layout: Layout,
        input_state: DeviceState[Self.dtype],
        kernel_size: Int = 2,
        stride: Int = 2,
        padding: Int = 0,
        sync: Bool = False,
    ) raises -> Tuple[
        Layout, DeviceState[Self.dtype], DeviceState[DType.int64]
    ]:
        if input_layout.rank() != 4:
            panic("PoolTt: input must be 4D (N, C, H_in, W_in)")
        if kernel_size <= 0 or stride <= 0:
            panic("PoolTt: kernel_size and stride must be positive")
        var N = input_layout.shape[0]
        var C = input_layout.shape[1]
        var H_in = input_layout.shape[2]
        var W_in = input_layout.shape[3]
        var H_out = (H_in + 2 * padding - kernel_size) // stride + 1
        var W_out = (W_in + 2 * padding - kernel_size) // stride + 1
        if H_out <= 0 or W_out <= 0:
            panic("PoolTt: parameters lead to non-positive output size")
        # TileTensor ctors do no bounds-checking (official docs) — the
        # launcher owns size validation. Strided max index, not dense numel.
        var st = input_layout.strides
        var max_idx = (
            input_layout.offset
            + (N - 1) * st[0]
            + (C - 1) * st[1]
            + (H_in - 1) * st[2]
            + (W_in - 1) * st[3]
        )
        if len(input_state.device_buffer()) <= max_idx:
            panic("PoolTt: input buffer smaller than the strided view")
        ref gpu = input_state.get_gpu()
        var device_context = gpu[]
        var total = N * C * H_out * W_out
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            total, 1
        )
        var out_buffer = device_context.enqueue_create_buffer[Self.datatype](
            total
        )
        var mask_buffer = device_context.enqueue_create_buffer[DType.int64](
            total
        )
        # Wrap: offset-carried DevicePointer + strided runtime layout.
        # device_buffer() lends immutable; the ctor needs a mutable borrow,
        # so copy the handle (DeviceBuffer is a reference-counted handle)
        # into an owned local first.
        var in_buf = input_state.device_buffer()
        var dp = DevicePointer[Self.datatype, MutAnyOrigin](
            in_buf, input_layout.offset
        )
        var t_input = TileTensor(
            dp,
            TileLayout(
                Coord(
                    Int64(N),
                    Int64(C),
                    Int64(H_in),
                    Int64(W_in),
                ),
                Coord(Int64(st[0]), Int64(st[1]), Int64(st[2]), Int64(st[3])),
            ),
        )
        var t_output = TileTensor(
            out_buffer,
            row_major(
                Coord(Int64(N), Int64(C), Int64(H_out), Int64(W_out))
            ),
        )
        var compiled = device_context.compile_function[
            pool_forward_nchw[dtype=Self.dtype],
        ]()
        device_context.enqueue_function(
            compiled,
            t_output,
            t_input,
            mask_buffer,
            Int64(kernel_size),
            Int64(stride),
            Int64(padding),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        if sync:
            device_context.synchronize()
        var out_state = DeviceState[Self.dtype].__init__[True](
            out_buffer^, gpu
        )
        var mask_state = DeviceState[DType.int64](mask_buffer^, gpu)
        return (Layout(Shape(N, C, H_out, W_out)), out_state^, mask_state^)

    @staticmethod
    def backward(
        grad_layout: Layout,
        grad_state: DeviceState[Self.dtype],
        mask_state: DeviceState[DType.int64],
        H_in: Int,
        W_in: Int,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        if grad_layout.rank() != 4:
            panic("PoolTt backward: grad must be 4D (N, C, H_out, W_out)")
        var N = grad_layout.shape[0]
        var C = grad_layout.shape[1]
        var H_out = grad_layout.shape[2]
        var W_out = grad_layout.shape[3]
        # The scatter is mask-driven; H_in/W_in size the zeroed grad_in.
        var gst = grad_layout.strides
        var max_idx = (
            grad_layout.offset
            + (N - 1) * gst[0]
            + (C - 1) * gst[1]
            + (H_out - 1) * gst[2]
            + (W_out - 1) * gst[3]
        )
        if len(grad_state.device_buffer()) <= max_idx:
            panic("PoolTt backward: grad buffer smaller than the strided view")
        if len(mask_state.device_buffer()) < N * C * H_out * W_out:
            panic("PoolTt backward: mask buffer smaller than N*C*H_out*W_out")
        ref gpu = grad_state.get_gpu()
        var device_context = gpu[]
        var gin_buffer = device_context.enqueue_create_buffer[Self.datatype](
            N * C * H_in * W_in
        )
        var gbuf = grad_state.device_buffer()
        var dp = DevicePointer[Self.datatype, MutAnyOrigin](
            gbuf, grad_layout.offset
        )
        var t_grad = TileTensor(
            dp,
            TileLayout(
                Coord(
                    Int64(N),
                    Int64(C),
                    Int64(H_out),
                    Int64(W_out),
                ),
                Coord(
                    Int64(gst[0]), Int64(gst[1]), Int64(gst[2]), Int64(gst[3])
                ),
            ),
        )
        var compiled = device_context.compile_function[
            pool_backward_nchw[dtype=Self.dtype],
        ]()
        # One thread per (n, c): grid N*C, single-thread blocks.
        device_context.enqueue_function(
            compiled,
            gin_buffer,
            t_grad,
            mask_state.device_buffer(),
            Int64(H_in),
            Int64(W_in),
            Int64(H_out),
            Int64(W_out),
            grid_dim=N * C,
            block_dim=1,
        )
        if sync:
            device_context.synchronize()
        var gin_state = DeviceState[Self.dtype].__init__[True](
            gin_buffer^, gpu
        )
        return (Layout(Shape(N, C, H_in, W_in)), gin_state^)
