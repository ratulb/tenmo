"""Conv2D forward on GPU (rank-4 NCHW).

Launcher contract mirrors the CPU `Conv.forward` (`tenmo/conv.mojo`):
dense NCHW image + OIHW kernel, rank-4, bias `(C_out,)`, stride/dilation +
asymmetric padding, output dims from the same formula. Anything else panics —
the CPU core's fail-loud layout contract, repeated here so a GPU launch can
never silently misread a view.

Handshake (official pattern — stable v26.6 "Using TileTensor"): the launcher
builds host-side `TileTensor`s over the contiguous `DeviceBuffer`s with
runtime row-major layouts and passes them WHOLE as kernel args. Geometry rides
inside the tensors (`dim[i]()` on device); only stride/dilation/pad cross as
`Int64` config. No device-side `Coord` rebuild — the fallback (pointers +
`Int64`s cross, `Coord`/`Layout` rebuilt on device) stands by if a
`TileTensor` value ever fails to marshal through `enqueue_function` on real
hardware.

Migration notes:
- View-vs-copy: the three `TileTensor`s wrap caller storage (input, weight,
  output buffer); all die at return. `materialize_contiguous` gives compact
  row-major buffers, so strided-layout support is deferred. Output lifecycle
  is unchanged: `enqueue_create_buffer` → kernel writes →
  `DeviceState[special]` wrap.
- Backward stays on CPU (cross-device transfer nodes already exist) — this
  file is forward-only, hence no autograd surface.
- Rank dispatch is a single arm: NCHW is fixed at 4. Non-4D inputs panic.
  All three layouts share one type (`Layout4D`: rank-4, fully dynamic).
- Acceptance: 1e-6 vs the CPU forward, `cnn` suite + MNIST smoke unchanged.

Thin/thick split: this cut is thin — the host builds tensors but does no
tile-grid math beyond the output-dims formula. The thick helper (host tiling
math: `tile[]`, `vectorize`, `distribute`, `stack_allocation`, `tile_io`
copiers per the official doc) arrives with shared-memory staging.
"""

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from layout import TileTensor
from layout.tile_layout import row_major, RowMajorLayout
from layout.coord import Coord

from ..shared.layout import Layout
from ..shared.shapes import Shape
from ..gpu.transfer import materialize_contiguous
from .kernel_helpers import elementwise_launch_config
from ..gpu.device import DeviceState
from ..shared.panic import panic


# Rank-4 fully-dynamic row-major: NCHW image, OIHW kernel, NCHW output share
# one layout type, so one kernel serves all three tensors.
comptime Layout4D = RowMajorLayout[Int64, Int64, Int64, Int64]


def conv_forward_nchw[dtype: DType](
    output: TileTensor[dtype, Layout4D, MutAnyOrigin],
    image: TileTensor[dtype, Layout4D, ImmutAnyOrigin],
    weight: TileTensor[dtype, Layout4D, ImmutAnyOrigin],
    bias: Pointer[Scalar[dtype], ImmutAnyOrigin],
    stride_: Int64,
    dilation_: Int64,
    pad_top_: Int64,
    pad_left_: Int64,
):
    # Geometry rides inside the tensors — dim[i]() replaces every Int64
    # geometry arg the previous revision carried.
    var N = Int(image.dim[0]())
    var C_in = Int(image.dim[1]())
    var H_in = Int(image.dim[2]())
    var W_in = Int(image.dim[3]())
    var C_out = Int(weight.dim[0]())
    var H_out = Int(output.dim[2]())
    var W_out = Int(output.dim[3]())
    var KH = Int(weight.dim[2]())
    var KW = Int(weight.dim[3]())
    var stride = Int(stride_)
    var dilation = Int(dilation_)
    var pad_top = Int(pad_top_)
    var pad_left = Int(pad_left_)

    var out_plane = C_out * H_out * W_out
    var hw_plane = H_out * W_out
    var total = N * out_plane
    var tid = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    var span = Int(block_dim.x) * Int(grid_dim.x)
    var idx = tid
    while idx < total:
        var n = idx // out_plane
        var rem = idx - n * out_plane
        var co = rem // hw_plane
        rem -= co * hw_plane
        var oh = rem // W_out
        var ow = rem - oh * W_out
        var acc = Scalar[dtype](0)
        for ci in range(C_in):
            for kh in range(KH):
                var h = oh * stride - pad_top + kh * dilation
                if h < 0 or h >= H_in:
                    continue
                for kw in range(KW):
                    var w = ow * stride - pad_left + kw * dilation
                    if w < 0 or w >= W_in:
                        continue
                    acc += image[n, ci, h, w] * weight[co, ci, kh, kw]
        # All tensor addressing is coordinates — no flat-index writes remain.
        # The 1D bias stays a raw load (rank-1 flat buys no layout).
        output[n, co, oh, ow] = acc + bias[unsafe_offset=co]
        idx += span


struct ConvGpu[dtype: DType](ImplicitlyCopyable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch(
        image_layout: Layout,
        image_state: DeviceState[Self.dtype],
        kernel_layout: Layout,
        kernel_state: DeviceState[Self.dtype],
        bias_layout: Layout,
        bias_state: DeviceState[Self.dtype],
        stride: Int = 1,
        dilation: Int = 1,
        pad_top: Int = 0,
        pad_bottom: Int = 0,
        pad_left: Int = 0,
        pad_right: Int = 0,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        comptime assert Self.dtype != DType.bool, (
            "ConvGpu: bool convolution is meaningless (multiply-accumulate "
            "plus bias); pass a numeric dtype"
        )
        # Layout contract: rank-4 dense, C_in match, flat bias — mirrors CPU.
        if image_layout.rank() != 4:
            panic("ConvGpu: image must be 4D (N, C_in, H_in, W_in)")
        if kernel_layout.rank() != 4:
            panic("ConvGpu: kernel must be 4D (C_out, C_in, KH, KW)")
        var N = image_layout.shape[0]
        var C_in = image_layout.shape[1]
        var H_in = image_layout.shape[2]
        var W_in = image_layout.shape[3]
        var C_out = kernel_layout.shape[0]
        if kernel_layout.shape[1] != C_in:
            panic("ConvGpu: kernel input channels must match input channels")
        var KH = kernel_layout.shape[2]
        var KW = kernel_layout.shape[3]
        if bias_layout.numel() != C_out:
            panic("ConvGpu: bias must have C_out elements")
        # TileTensor ctors do no bounds-checking (official docs) — the
        # launcher owns size validation, before any device object is built.
        if len(image_state.device_buffer()) < N * C_in * H_in * W_in:
            panic("ConvGpu: image buffer smaller than N*C_in*H_in*W_in")
        if len(kernel_state.device_buffer()) < C_out * C_in * KH * KW:
            panic("ConvGpu: kernel buffer smaller than C_out*C_in*KH*KW")
        if len(bias_state.device_buffer()) < C_out:
            panic("ConvGpu: bias buffer smaller than C_out")
        var H_out = (
            H_in + pad_top + pad_bottom - (KH + (KH - 1) * (dilation - 1))
        ) // stride + 1
        var W_out = (
            W_in + pad_left + pad_right - (KW + (KW - 1) * (dilation - 1))
        ) // stride + 1
        if H_out <= 0 or W_out <= 0:
            panic("ConvGpu: parameters lead to non-positive output size")

        # Compact row-major device buffers (Friction-2-sanctioned step).
        # Every input is materialized: the kernel reads bias[co] flat, so a
        # strided/offset bias view would silently misread without this.
        var contig_image = materialize_contiguous(
            image_state, image_layout, sync=sync
        )
        var contig_kernel = materialize_contiguous(
            kernel_state, kernel_layout, sync=sync
        )
        var contig_bias = materialize_contiguous(
            bias_state, bias_layout, sync=sync
        )
        ref gpu = image_state.get_gpu()
        var device_context = gpu[]
        var total = N * C_out * H_out * W_out
        var (num_blocks, threads_per_block) = elementwise_launch_config(
            total, 1
        )
        var result_buffer = device_context.enqueue_create_buffer[
            Self.datatype
        ](total)
        # Host-built TileTensors over the device buffers, runtime layouts —
        # the official global-memory pattern: constructed on CPU, passed whole.
        var t_image = TileTensor(
            contig_image.device_buffer(),
            row_major(
                Coord(Int64(N), Int64(C_in), Int64(H_in), Int64(W_in))
            ),
        )
        var t_kernel = TileTensor(
            contig_kernel.device_buffer(),
            row_major(
                Coord(Int64(C_out), Int64(C_in), Int64(KH), Int64(KW))
            ),
        )
        var t_output = TileTensor(
            result_buffer,
            row_major(
                Coord(Int64(N), Int64(C_out), Int64(H_out), Int64(W_out))
            ),
        )
        var compiled = device_context.compile_function[
            conv_forward_nchw[dtype=Self.dtype],
        ]()
        device_context.enqueue_function(
            compiled,
            t_output,
            t_image,
            t_kernel,
            contig_bias.device_buffer(),
            Int64(stride),
            Int64(dilation),
            Int64(pad_top),
            Int64(pad_left),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )
        if sync:
            device_context.synchronize()
        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, gpu
        )
        return (Layout(Shape(N, C_out, H_out, W_out)), result_state^)
