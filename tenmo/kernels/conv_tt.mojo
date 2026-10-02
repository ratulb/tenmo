"""Direct-tiled Conv2D forward on GPU.

Launcher contract mirrors the CPU `Conv.forward` (`tenmo/conv.mojo`):
rank-4 image (N, C_in, H_in, W_in), rank-4 kernel (C_out, C_in, KH, KW),
rank-1 bias (C_out,) — bias is REQUIRED, matching the CPU core — scalar
stride and dilation shared by both spatial dims, asymmetric padding
(pad_top/bottom/left/right), output dims from the same dilated formula.
Anything else panics.

Handshake: image and kernel are wrapped as whole `TileTensor`s
over offset-carried `DevicePointer`s with strided runtime `Layout`s — NO
`materialize_contiguous` round-trip, so strided views are consumed natively
(the CPU core panics on views; this kernel is strictly more general). The
output stays contiguous row-major (DefaultEngine, the ConvGpu pattern) and
the bias travels as a raw pointer (always a dense fresh vector, so flat
indexing buys the layout nothing).

Two device functions share one grid/thread mapping (Option 2):

- `conv_tt_forward_staged`: each block cooperatively loads its input patch
  (halo included, padding fused as zeros) into a STATIC shared scratch and
  computes from shared memory. Shared allocations must be fully static
  (`stack_allocation` rejects runtime dims — probed), so the scratch is
  `[CI_T=8, 225]` flat (`row_major[8, 225]()`, the probe-proven 2-D static
  spelling), indexed `patch[ci, ph * 15 + pw]`. Covers KH/KW ≤ 7, stride ≤ 2,
  patch footprint ≤ 15 — every practical CNN layer.
- `conv_tt_forward_direct`: same mapping, halo gathered straight from
  global memory with bounds checks. No shared memory, no caps. The launcher
  picks it whenever the staged caps are exceeded, so correctness is
  universal and speed is on the common.

Grid: blocks over (n, co-tile, oh-tile, ow-tile); 128 threads per block =
CO_T(8) × OH_T(4) × OW_T(4), one output per thread. Channel reduction is
chunked over C_in in steps of CI_T=8 inside the kernel (register
accumulator survives across chunks; a barrier separates patch reuse).

T4-safe: plain global loads/stores, one static shared scratch (7.2 KB),
no cp.async. Forward-only; the backward reuses this file's
wrap and tiling vocabulary.

Backward: three grid-stride elementwise kernels, all race-free (one
thread owns one grad element — no atomics, no shared memory):
- `conv_tt_bwd_kernel_grad`: grad_kernel[oc, cig, kh, kw] is itself a
  direct conv (grad_out cross-correlated against the image), gathered
  straight from global memory. No staged caps — correct for all shapes.
- `conv_tt_bwd_image_grad`: transposed conv as a gather — one thread per
  input element solves oh/ow from (ih + pad - kh*dil) with a
  stride-divisibility + bounds check, so stride > 1 and dilation need no
  explicit upsample/dilate passes.
- `conv_tt_bwd_bias_grad`: one thread per oc sums grad_out.
Parity (not bitwise) vs the CPU backward at 1e-4: accumulation order
differs from the CPU im2col+GEMM path by design.

Migration notes:
- View-vs-copy: image/kernel tiles wrap caller storage and die at return;
  the output buffer is fresh, wrapped at return via
  `with_layout_device_state` by the op layer. Nothing is mutated between
  forward and (future) backward — the arg will carry what it needs.
- Rank dispatch is a single arm: NCHW image / OIHW kernel / C bias. Other
  ranks panic. Strided image+kernel share one layout type (`Strided4D`);
  the fresh output uses `Layout4D`.
- Acceptance: parity vs the CPU forward on dense input (atol=1e-4 — float
  accumulation order differs from the CPU im2col+GEMM path), staged caps
  exercised plus one fallback-shape test, `conv_tt` suite green on the box.
"""

from max.gpu import thread_idx, block_dim, grid_dim, block_idx, barrier
from max.gpu.host import DevicePointer
from layout import TileTensor, stack_allocation
from layout.tile_layout import row_major, RowMajorLayout, Layout as TileLayout
from layout.tensor_engine import DevicePointerEngine
from layout.coord import Coord

from ..shared.layout import Layout
from ..shared.shapes import Shape
from ..gpu.device import DeviceState
from ..shared.panic import panic
from .kernel_helpers import elementwise_launch_config


# Rank-4 fully-dynamic row-major (fresh output) + fully-dynamic strided
# (image/kernel views). type_of names the strided type.
comptime Layout4D = RowMajorLayout[Int64, Int64, Int64, Int64]
comptime Strided4D = type_of(
    TileLayout(
        Coord(Int64(0), Int64(0), Int64(0), Int64(0)),
        Coord(Int64(0), Int64(0), Int64(0), Int64(0)),
    )
)

# Staged-path geometry. CO_T × OH_T × OW_T threads per block; the shared
# scratch holds one CI_T × HP_MAX × WP_MAX patch, flattened to 2-D because
# only the 2-D static row_major spelling is probe-proven.
comptime CO_T = 8
comptime OH_T = 4
comptime OW_T = 4
comptime CI_T = 8
comptime HP_MAX = 15
comptime WP_MAX = 15
comptime K_MAX = 7
comptime STRIDE_MAX = 2
comptime THREADS_PER_BLOCK = CO_T * OH_T * OW_T


def conv_tt_forward_staged[dtype: DType](
    output: TileTensor[dtype, Layout4D, MutAnyOrigin],
    image: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    kernel: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    bias: Pointer[Scalar[dtype], MutAnyOrigin],
    N_: Int64,
    C_in_: Int64,
    H_in_: Int64,
    W_in_: Int64,
    C_out_: Int64,
    H_out_: Int64,
    W_out_: Int64,
    KH_: Int64,
    KW_: Int64,
    stride_: Int64,
    dil_: Int64,
    pad_top_: Int64,
    pad_left_: Int64,
    co_tiles_: Int64,
    oh_tiles_: Int64,
    ow_tiles_: Int64,
):
    var N = Int(N_)
    var C_in = Int(C_in_)
    var H_in = Int(H_in_)
    var W_in = Int(W_in_)
    var C_out = Int(C_out_)
    var H_out = Int(H_out_)
    var W_out = Int(W_out_)
    var KH = Int(KH_)
    var KW = Int(KW_)
    var stride = Int(stride_)
    var dil = Int(dil_)
    var pad_top = Int(pad_top_)
    var pad_left = Int(pad_left_)
    var co_tiles = Int(co_tiles_)
    var oh_tiles = Int(oh_tiles_)
    var ow_tiles = Int(ow_tiles_)
    # Block → (n, co-tile, oh-tile, ow-tile).
    var bid = Int(block_idx.x)
    var tmp = bid
    var ow_tile = tmp % ow_tiles
    tmp //= ow_tiles
    var oh_tile = tmp % oh_tiles
    tmp //= oh_tiles
    var co_tile = tmp % co_tiles
    var n = tmp // co_tiles
    var co_base = co_tile * CO_T
    var oh_base = oh_tile * OH_T
    var ow_base = ow_tile * OW_T
    # Thread → (tco, toh, tow): one output per thread.
    var tid = Int(thread_idx.x)
    var tco = tid // (OH_T * OW_T)
    var rem = tid % (OH_T * OW_T)
    var toh = rem // OW_T
    var tow = rem % OW_T
    var oc = co_base + tco
    var oh = oh_base + toh
    var ow = ow_base + tow
    # Partial edge tiles (C_out/H_out/W_out not divisible) idle their
    # OOB threads — they still load the patch and hit every barrier.
    var active = oc < C_out and oh < H_out and ow < W_out
    var H_P = (OH_T - 1) * stride + (KH - 1) * dil + 1
    var W_P = (OW_T - 1) * stride + (KW - 1) * dil + 1
    var acc = Scalar[dtype](0)
    if active:
        acc = bias[unsafe_offset=oc]
    comptime patch_l = row_major[CI_T, HP_MAX * WP_MAX]()
    var patch = stack_allocation[dtype, address_space=.SHARED](patch_l)
    # Channel reduction in CI_T chunks: cooperative patch load, barrier,
    # accumulate from shared, barrier before the next load reuses it.
    var ci_b = 0
    while ci_b < C_in:
        var chunk = C_in - ci_b
        if chunk > CI_T:
            chunk = CI_T
        var plane = H_P * W_P
        var total = chunk * plane
        var e = tid
        while e < total:
            var ci = e // plane
            var r = e % plane
            var ph = r // W_P
            var pw = r % W_P
            # Dense (step-1) coverage of the footprint: outputs in one
            # tile need off-dilation-grid cells (e.g. s=1, dil=2, odd
            # rows). The compute below reads the dilated subset.
            var ih = oh_base * stride - pad_top + ph
            var iw = ow_base * stride - pad_left + pw
            var v = Scalar[dtype](0)
            if 0 <= ih and ih < H_in and 0 <= iw and iw < W_in:
                v = image[n, ci_b + ci, ih, iw]
            patch[ci, ph * WP_MAX + pw] = v
            e += THREADS_PER_BLOCK
        barrier()
        if active:
            for ci in range(chunk):
                for kh in range(KH):
                    var ph = toh * stride + kh * dil
                    for kw in range(KW):
                        var pw = tow * stride + kw * dil
                        acc += patch[ci, ph * WP_MAX + pw] * kernel[oc, ci_b + ci, kh, kw]
        barrier()
        ci_b += CI_T
    if active:
        output[n, oc, oh, ow] = acc


def conv_tt_forward_direct[dtype: DType](
    output: TileTensor[dtype, Layout4D, MutAnyOrigin],
    image: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    kernel: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    bias: Pointer[Scalar[dtype], MutAnyOrigin],
    N_: Int64,
    C_in_: Int64,
    H_in_: Int64,
    W_in_: Int64,
    C_out_: Int64,
    H_out_: Int64,
    W_out_: Int64,
    KH_: Int64,
    KW_: Int64,
    stride_: Int64,
    dil_: Int64,
    pad_top_: Int64,
    pad_left_: Int64,
    co_tiles_: Int64,
    oh_tiles_: Int64,
    ow_tiles_: Int64,
):
    # Same grid/thread mapping as the staged kernel; the halo is gathered
    # straight from global memory. No shared memory, no caps.
    var N = Int(N_)
    var C_in = Int(C_in_)
    var H_in = Int(H_in_)
    var W_in = Int(W_in_)
    var C_out = Int(C_out_)
    var H_out = Int(H_out_)
    var W_out = Int(W_out_)
    var KH = Int(KH_)
    var KW = Int(KW_)
    var stride = Int(stride_)
    var dil = Int(dil_)
    var pad_top = Int(pad_top_)
    var pad_left = Int(pad_left_)
    var co_tiles = Int(co_tiles_)
    var oh_tiles = Int(oh_tiles_)
    var ow_tiles = Int(ow_tiles_)
    var bid = Int(block_idx.x)
    var tmp = bid
    var ow_tile = tmp % ow_tiles
    tmp //= ow_tiles
    var oh_tile = tmp % oh_tiles
    tmp //= oh_tiles
    var co_tile = tmp % co_tiles
    var n = tmp // co_tiles
    var tid = Int(thread_idx.x)
    var tco = tid // (OH_T * OW_T)
    var rem = tid % (OH_T * OW_T)
    var toh = rem // OW_T
    var tow = rem % OW_T
    var oc = co_tile * CO_T + tco
    var oh = oh_tile * OH_T + toh
    var ow = ow_tile * OW_T + tow
    if oc < C_out and oh < H_out and ow < W_out:
        var acc = bias[unsafe_offset=oc]
        for cig in range(C_in):
            for kh in range(KH):
                var ih = oh * stride - pad_top + kh * dil
                if ih < 0 or ih >= H_in:
                    continue
                for kw in range(KW):
                    var iw = ow * stride - pad_left + kw * dil
                    if iw < 0 or iw >= W_in:
                        continue
                    acc += image[n, cig, ih, iw] * kernel[oc, cig, kh, kw]
        output[n, oc, oh, ow] = acc


struct ConvTt[dtype: DType](ImplicitlyCopyable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def forward(
        image_layout: Layout,
        image_state: DeviceState[Self.dtype],
        kernel_layout: Layout,
        kernel_state: DeviceState[Self.dtype],
        bias_state: DeviceState[Self.dtype],
        stride: Int = 1,
        dilation: Int = 1,
        pad_top: Int = 0,
        pad_bottom: Int = 0,
        pad_left: Int = 0,
        pad_right: Int = 0,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        if image_layout.rank() != 4:
            panic("ConvTt: image must be 4D (N, C_in, H_in, W_in)")
        if kernel_layout.rank() != 4:
            panic("ConvTt: kernel must be 4D (C_out, C_in, KH, KW)")
        if stride <= 0 or dilation <= 0:
            panic("ConvTt: stride and dilation must be positive")
        var N = image_layout.shape[0]
        var C_in = image_layout.shape[1]
        var H_in = image_layout.shape[2]
        var W_in = image_layout.shape[3]
        var C_out = kernel_layout.shape[0]
        if kernel_layout.shape[1] != C_in:
            panic("ConvTt: kernel input channels must match input channels")
        var KH = kernel_layout.shape[2]
        var KW = kernel_layout.shape[3]
        var dil_KH = KH + (KH - 1) * (dilation - 1)
        var dil_KW = KW + (KW - 1) * (dilation - 1)
        var H_out = (H_in + pad_top + pad_bottom - dil_KH) // stride + 1
        var W_out = (W_in + pad_left + pad_right - dil_KW) // stride + 1
        if H_out <= 0 or W_out <= 0:
            panic("ConvTt: parameters lead to non-positive output size")
        if len(bias_state.device_buffer()) < C_out:
            panic("ConvTt: bias buffer smaller than C_out")
        # TileTensor ctors do no bounds-checking — the launcher owns size
        # validation. Strided max index, not dense numel.
        var ist = image_layout.strides
        var image_max = (
            image_layout.offset
            + (N - 1) * ist[0]
            + (C_in - 1) * ist[1]
            + (H_in - 1) * ist[2]
            + (W_in - 1) * ist[3]
        )
        if len(image_state.device_buffer()) <= image_max:
            panic("ConvTt: image buffer smaller than the strided view")
        var kst = kernel_layout.strides
        var kernel_max = (
            kernel_layout.offset
            + (C_out - 1) * kst[0]
            + (C_in - 1) * kst[1]
            + (KH - 1) * kst[2]
            + (KW - 1) * kst[3]
        )
        if len(kernel_state.device_buffer()) <= kernel_max:
            panic("ConvTt: kernel buffer smaller than the strided view")
        # Staged caps: static scratch covers KH/KW ≤ K_MAX, stride ≤
        # STRIDE_MAX, and the halo footprint of one OH_T × OW_T tile.
        var H_P = (OH_T - 1) * stride + (KH - 1) * dilation + 1
        var W_P = (OW_T - 1) * stride + (KW - 1) * dilation + 1
        var staged = (
            KH <= K_MAX
            and KW <= K_MAX
            and stride <= STRIDE_MAX
            and H_P <= HP_MAX
            and W_P <= WP_MAX
        )
        ref gpu = image_state.get_gpu()
        var device_context = gpu[]
        var co_tiles = (C_out + CO_T - 1) // CO_T
        var oh_tiles = (H_out + OH_T - 1) // OH_T
        var ow_tiles = (W_out + OW_T - 1) // OW_T
        var num_blocks = N * co_tiles * oh_tiles * ow_tiles
        var out_buffer = device_context.enqueue_create_buffer[Self.datatype](
            N * C_out * H_out * W_out
        )
        var in_buf = image_state.device_buffer()
        var image_dp = DevicePointer[Self.datatype, MutAnyOrigin](
            in_buf, image_layout.offset
        )
        var t_image = TileTensor(
            image_dp,
            TileLayout(
                Coord(
                    Int64(N),
                    Int64(C_in),
                    Int64(H_in),
                    Int64(W_in),
                ),
                Coord(Int64(ist[0]), Int64(ist[1]), Int64(ist[2]), Int64(ist[3])),
            ),
        )
        var kern_buf = kernel_state.device_buffer()
        var kernel_dp = DevicePointer[Self.datatype, MutAnyOrigin](
            kern_buf, kernel_layout.offset
        )
        var t_kernel = TileTensor(
            kernel_dp,
            TileLayout(
                Coord(
                    Int64(C_out),
                    Int64(C_in),
                    Int64(KH),
                    Int64(KW),
                ),
                Coord(Int64(kst[0]), Int64(kst[1]), Int64(kst[2]), Int64(kst[3])),
            ),
        )
        var t_output = TileTensor(
            out_buffer,
            row_major(
                Coord(Int64(N), Int64(C_out), Int64(H_out), Int64(W_out))
            ),
        )
        if staged:
            var compiled = device_context.compile_function[
                conv_tt_forward_staged[dtype=Self.dtype],
            ]()
            device_context.enqueue_function(
                compiled,
                t_output,
                t_image,
                t_kernel,
                bias_state.device_buffer(),
                Int64(N),
                Int64(C_in),
                Int64(H_in),
                Int64(W_in),
                Int64(C_out),
                Int64(H_out),
                Int64(W_out),
                Int64(KH),
                Int64(KW),
                Int64(stride),
                Int64(dilation),
                Int64(pad_top),
                Int64(pad_left),
                Int64(co_tiles),
                Int64(oh_tiles),
                Int64(ow_tiles),
                grid_dim=num_blocks,
                block_dim=THREADS_PER_BLOCK,
            )
        else:
            var compiled = device_context.compile_function[
                conv_tt_forward_direct[dtype=Self.dtype],
            ]()
            device_context.enqueue_function(
                compiled,
                t_output,
                t_image,
                t_kernel,
                bias_state.device_buffer(),
                Int64(N),
                Int64(C_in),
                Int64(H_in),
                Int64(W_in),
                Int64(C_out),
                Int64(H_out),
                Int64(W_out),
                Int64(KH),
                Int64(KW),
                Int64(stride),
                Int64(dilation),
                Int64(pad_top),
                Int64(pad_left),
                Int64(co_tiles),
                Int64(oh_tiles),
                Int64(ow_tiles),
                grid_dim=num_blocks,
                block_dim=THREADS_PER_BLOCK,
            )
        if sync:
            device_context.synchronize()
        var out_state = DeviceState[Self.dtype].__init__[True](
            out_buffer^, gpu
        )
        return (Layout(Shape(N, C_out, H_out, W_out)), out_state^)

    @staticmethod
    def backward(
        grad_layout: Layout,
        grad_state: DeviceState[Self.dtype],
        image_layout: Layout,
        image_state: DeviceState[Self.dtype],
        kernel_layout: Layout,
        kernel_state: DeviceState[Self.dtype],
        N: Int,
        C_in: Int,
        H_in: Int,
        W_in: Int,
        C_out: Int,
        KH: Int,
        KW: Int,
        H_out: Int,
        W_out: Int,
        stride: Int,
        dilation: Int,
        pad_top: Int,
        pad_left: Int,
        sync: Bool = False,
    ) raises -> Tuple[
        Layout,
        DeviceState[Self.dtype],
        Layout,
        DeviceState[Self.dtype],
        Layout,
        DeviceState[Self.dtype],
    ]:
        # Fail loud on shape drift: the gradbox must match the saved
        # forward geometry exactly (guides the engine, never silent).
        if grad_layout.rank() != 4:
            panic("ConvTt backward: grad must be 4D (N, C_out, H_out, W_out)")
        if (
            grad_layout.shape[0] != N
            or grad_layout.shape[1] != C_out
            or grad_layout.shape[2] != H_out
            or grad_layout.shape[3] != W_out
        ):
            panic("ConvTt backward: grad shape does not match saved forward")
        if image_layout.rank() != 4 or kernel_layout.rank() != 4:
            panic("ConvTt backward: image and kernel must be rank-4")
        var gst = grad_layout.strides
        var grad_max = (
            grad_layout.offset
            + (N - 1) * gst[0]
            + (C_out - 1) * gst[1]
            + (H_out - 1) * gst[2]
            + (W_out - 1) * gst[3]
        )
        if len(grad_state.device_buffer()) <= grad_max:
            panic("ConvTt backward: grad buffer smaller than strided view")
        var ist = image_layout.strides
        var image_max = (
            image_layout.offset
            + (N - 1) * ist[0]
            + (C_in - 1) * ist[1]
            + (H_in - 1) * ist[2]
            + (W_in - 1) * ist[3]
        )
        if len(image_state.device_buffer()) <= image_max:
            panic("ConvTt backward: image buffer smaller than strided view")
        var kst = kernel_layout.strides
        var kernel_max = (
            kernel_layout.offset
            + (C_out - 1) * kst[0]
            + (C_in - 1) * kst[1]
            + (KH - 1) * kst[2]
            + (KW - 1) * kst[3]
        )
        if len(kernel_state.device_buffer()) <= kernel_max:
            panic("ConvTt backward: kernel buffer smaller than strided view")
        ref gpu = grad_state.get_gpu()
        var device_context = gpu[]
        var gi_buffer = device_context.enqueue_create_buffer[Self.datatype](
            N * C_in * H_in * W_in
        )
        var gk_buffer = device_context.enqueue_create_buffer[Self.datatype](
            C_out * C_in * KH * KW
        )
        var gb_buffer = device_context.enqueue_create_buffer[Self.datatype](
            C_out
        )
        var gbuf = grad_state.device_buffer()
        var grad_dp = DevicePointer[Self.datatype, MutAnyOrigin](
            gbuf, grad_layout.offset
        )
        var t_grad = TileTensor(
            grad_dp,
            TileLayout(
                Coord(
                    Int64(N),
                    Int64(C_out),
                    Int64(H_out),
                    Int64(W_out),
                ),
                Coord(Int64(gst[0]), Int64(gst[1]), Int64(gst[2]), Int64(gst[3])),
            ),
        )
        var in_buf = image_state.device_buffer()
        var image_dp = DevicePointer[Self.datatype, MutAnyOrigin](
            in_buf, image_layout.offset
        )
        var t_image = TileTensor(
            image_dp,
            TileLayout(
                Coord(
                    Int64(N),
                    Int64(C_in),
                    Int64(H_in),
                    Int64(W_in),
                ),
                Coord(Int64(ist[0]), Int64(ist[1]), Int64(ist[2]), Int64(ist[3])),
            ),
        )
        var kern_buf = kernel_state.device_buffer()
        var kernel_dp = DevicePointer[Self.datatype, MutAnyOrigin](
            kern_buf, kernel_layout.offset
        )
        var t_kernel = TileTensor(
            kernel_dp,
            TileLayout(
                Coord(
                    Int64(C_out),
                    Int64(C_in),
                    Int64(KH),
                    Int64(KW),
                ),
                Coord(Int64(kst[0]), Int64(kst[1]), Int64(kst[2]), Int64(kst[3])),
            ),
        )
        var compiled_gk = device_context.compile_function[
            conv_tt_bwd_kernel_grad[dtype=Self.dtype],
        ]()
        var (gk_blocks, gk_threads) = elementwise_launch_config(
            C_out * C_in * KH * KW, 1
        )
        device_context.enqueue_function(
            compiled_gk,
            gk_buffer,
            t_grad,
            t_image,
            Int64(N),
            Int64(C_in),
            Int64(H_in),
            Int64(W_in),
            Int64(C_out),
            Int64(KH),
            Int64(KW),
            Int64(H_out),
            Int64(W_out),
            Int64(stride),
            Int64(dilation),
            Int64(pad_top),
            Int64(pad_left),
            grid_dim=gk_blocks,
            block_dim=gk_threads,
        )
        var compiled_gi = device_context.compile_function[
            conv_tt_bwd_image_grad[dtype=Self.dtype],
        ]()
        var (gi_blocks, gi_threads) = elementwise_launch_config(
            N * C_in * H_in * W_in, 1
        )
        device_context.enqueue_function(
            compiled_gi,
            gi_buffer,
            t_grad,
            t_kernel,
            Int64(N),
            Int64(C_in),
            Int64(H_in),
            Int64(W_in),
            Int64(C_out),
            Int64(KH),
            Int64(KW),
            Int64(H_out),
            Int64(W_out),
            Int64(stride),
            Int64(dilation),
            Int64(pad_top),
            Int64(pad_left),
            grid_dim=gi_blocks,
            block_dim=gi_threads,
        )
        var compiled_gb = device_context.compile_function[
            conv_tt_bwd_bias_grad[dtype=Self.dtype],
        ]()
        var (gb_blocks, gb_threads) = elementwise_launch_config(C_out, 1)
        device_context.enqueue_function(
            compiled_gb,
            gb_buffer,
            t_grad,
            Int64(N),
            Int64(C_out),
            Int64(H_out),
            Int64(W_out),
            grid_dim=gb_blocks,
            block_dim=gb_threads,
        )
        if sync:
            device_context.synchronize()
        var gi_state = DeviceState[Self.dtype].__init__[True](
            gi_buffer^, gpu
        )
        var gk_state = DeviceState[Self.dtype].__init__[True](
            gk_buffer^, gpu
        )
        var gb_state = DeviceState[Self.dtype].__init__[True](
            gb_buffer^, gpu
        )
        return (
            Layout(Shape(N, C_in, H_in, W_in)),
            gi_state^,
            Layout(Shape(C_out, C_in, KH, KW)),
            gk_state^,
            Layout(Shape(C_out)),
            gb_state^,
        )


def conv_tt_bwd_kernel_grad[dtype: DType](
    grad_kernel: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_out: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    image: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    N_: Int64,
    C_in_: Int64,
    H_in_: Int64,
    W_in_: Int64,
    C_out_: Int64,
    KH_: Int64,
    KW_: Int64,
    H_out_: Int64,
    W_out_: Int64,
    stride_: Int64,
    dil_: Int64,
    pad_top_: Int64,
    pad_left_: Int64,
):
    # One thread per (oc, cig, kh, kw): flat idx IS the dense OIHW offset.
    var N = Int(N_)
    var C_in = Int(C_in_)
    var H_in = Int(H_in_)
    var W_in = Int(W_in_)
    var C_out = Int(C_out_)
    var KH = Int(KH_)
    var KW = Int(KW_)
    var H_out = Int(H_out_)
    var W_out = Int(W_out_)
    var stride = Int(stride_)
    var dil = Int(dil_)
    var pad_top = Int(pad_top_)
    var pad_left = Int(pad_left_)
    var total = C_out * C_in * KH * KW
    var tid = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    var span = Int(block_dim.x) * Int(grid_dim.x)
    var idx = tid
    while idx < total:
        var tmp = idx
        var kw = tmp % KW
        tmp //= KW
        var kh = tmp % KH
        tmp //= KH
        var cig = tmp % C_in
        var oc = tmp // C_in
        var acc = Scalar[dtype](0)
        for n in range(N):
            for oh in range(H_out):
                var ih = oh * stride - pad_top + kh * dil
                if ih < 0 or ih >= H_in:
                    continue
                for ow in range(W_out):
                    var iw = ow * stride - pad_left + kw * dil
                    if iw < 0 or iw >= W_in:
                        continue
                    acc += grad_out[n, oc, oh, ow] * image[n, cig, ih, iw]
        grad_kernel[unsafe_offset=idx] = acc
        idx += span


def conv_tt_bwd_image_grad[dtype: DType](
    grad_image: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_out: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    kernel: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    N_: Int64,
    C_in_: Int64,
    H_in_: Int64,
    W_in_: Int64,
    C_out_: Int64,
    KH_: Int64,
    KW_: Int64,
    H_out_: Int64,
    W_out_: Int64,
    stride_: Int64,
    dil_: Int64,
    pad_top_: Int64,
    pad_left_: Int64,
):
    # One thread per input element, gathering over the (oc, kh, kw)
    # contributions whose windows cover it. The divisibility check runs
    # before the bounds check: negative numerators with zero remainder
    # still fail bounds (oh < 0), so trunc-remainder sign is harmless.
    var N = Int(N_)
    var C_in = Int(C_in_)
    var H_in = Int(H_in_)
    var W_in = Int(W_in_)
    var C_out = Int(C_out_)
    var KH = Int(KH_)
    var KW = Int(KW_)
    var H_out = Int(H_out_)
    var W_out = Int(W_out_)
    var stride = Int(stride_)
    var dil = Int(dil_)
    var pad_top = Int(pad_top_)
    var pad_left = Int(pad_left_)
    var total = N * C_in * H_in * W_in
    var tid = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    var span = Int(block_dim.x) * Int(grid_dim.x)
    var idx = tid
    while idx < total:
        var tmp = idx
        var iw = tmp % W_in
        tmp //= W_in
        var ih = tmp % H_in
        tmp //= H_in
        var cig = tmp % C_in
        var n = tmp // C_in
        var acc = Scalar[dtype](0)
        for oc in range(C_out):
            for kh in range(KH):
                var num_h = ih + pad_top - kh * dil
                if num_h % stride != 0:
                    continue
                var oh = num_h // stride
                if oh < 0 or oh >= H_out:
                    continue
                for kw in range(KW):
                    var num_w = iw + pad_left - kw * dil
                    if num_w % stride != 0:
                        continue
                    var ow = num_w // stride
                    if ow < 0 or ow >= W_out:
                        continue
                    acc += grad_out[n, oc, oh, ow] * kernel[oc, cig, kh, kw]
        grad_image[unsafe_offset=idx] = acc
        idx += span


def conv_tt_bwd_bias_grad[dtype: DType](
    grad_bias: Pointer[Scalar[dtype], MutAnyOrigin],
    grad_out: TileTensor[
        dtype, Strided4D, MutAnyOrigin, Engine=DevicePointerEngine[
            element_width=1
        ]
    ],
    N_: Int64,
    C_out_: Int64,
    H_out_: Int64,
    W_out_: Int64,
):
    var N = Int(N_)
    var C_out = Int(C_out_)
    var H_out = Int(H_out_)
    var W_out = Int(W_out_)
    var tid = Int(thread_idx.x) + Int(block_dim.x) * Int(block_idx.x)
    var span = Int(block_dim.x) * Int(grid_dim.x)
    var oc = tid
    while oc < C_out:
        var acc = Scalar[dtype](0)
        for n in range(N):
            for oh in range(H_out):
                for ow in range(W_out):
                    acc += grad_out[n, oc, oh, ow]
        grad_bias[unsafe_offset=oc] = acc
        oc += span
