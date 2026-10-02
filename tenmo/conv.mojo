# From-scratch Conv2D: fused pad-by-index, blocked im2col, GEMM core.
#
# Design:
#   forward:  im2col (blocked, pad fused as zero-fill) → GEMM → repack+bias
#   backward: dK = P^T @ cols        (same GEMM, strided-A path)
#             dX = col2im(dY @ W)     (N-parallel scatter, disjoint fast path)
#             db = column sums over packed dY
#
# Layout contract: inputs must be contiguous with zero offset (fail loud —
# the old core silently assumed this). Gradbox payloads out are always fresh
# contiguous offset-0 (boundary asserts in Backward.invoke/update_grad hold).
#
# GPU: forward dispatches to ConvGpu (tenmo/kernels/conv_gpu.mojo) when the
# image is GPU-resident. Backward has no GPU kernels — ConvBackward transfers
# grads + parent data to CPU, runs this same core, and ships grads back.

from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType
from .ancestry import Ancestor
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .shared.shapes import Shape
from .shared.strides import Strides
from .shared.buffers import Buffer
from .shared.layout import Layout
from .shared.panic import panic
from .matmul_cpu import MmCpu2d
from .kernels.conv_gpu import ConvGpu

from std.sys import simd_width_of, has_accelerator
from std.sys.intrinsics import strided_load
from std.sys.info import num_physical_cores
from std.memory import unsafe_memcpy
from max.algorithm import parallelize


@fieldwise_init
struct ConvBwdArg(ArgumentType):
    var N: Int
    var C_in: Int
    var H_in: Int
    var W_in: Int
    var C_out: Int
    var KH: Int
    var KW: Int
    var H_out: Int
    var W_out: Int
    var stride: Int
    var dilation: Int
    var pad_top: Int
    var pad_left: Int


@fieldwise_init
struct ConvBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var bwd_arg = output.ancestry().backward_fn().get[ConvBwdArg]()
        var N = bwd_arg.N
        var C_in = bwd_arg.C_in
        var H_in = bwd_arg.H_in
        var W_in = bwd_arg.W_in
        var C_out = bwd_arg.C_out
        var KH = bwd_arg.KH
        var KW = bwd_arg.KW
        var H_out = bwd_arg.H_out
        var W_out = bwd_arg.W_out
        var stride = bwd_arg.stride
        var dilation = bwd_arg.dilation
        var pad_top = bwd_arg.pad_top
        var pad_left = bwd_arg.pad_left

        var image = output.ancestry().get(0)
        var kernel = output.ancestry().get(1)
        var bias = output.ancestry().get(2)

        # GPU PATH: CPU-fallback backward
        # No GPU conv-backward kernels exist: shadow grads + parent data to
        # CPU, run the shared core below, ship grads back to the parents'
        # device. Forward guarantees all three parents share the image's
        # device, so one handle covers every write-back.
        comptime if has_accelerator():
            if output.gradients().is_on_gpu():
                if (
                    not image.is_on_gpu()
                    or not kernel.is_on_gpu()
                    or not bias.is_on_gpu()
                ):
                    panic(
                        "ConvBackward: mixed devices — image, kernel and"
                        " bias must share one GPU"
                    )
                if not image.buffer().device_state:
                    panic("ConvBackward: GPU parent without device state")
                try:
                    var gpu = image.buffer().device_state.value().get_gpu()
                    var dy_cpu = output.gradients().buffer().to_cpu(sync=True)
                    var img_cpu = image.buffer().to_cpu(sync=True)
                    var kern_cpu = kernel.buffer().to_cpu(sync=True)
                    var grads = Self._cpu_gradients(
                        dy_cpu^,
                        img_cpu^,
                        kern_cpu^,
                        N,
                        C_in,
                        H_in,
                        W_in,
                        C_out,
                        KH,
                        KW,
                        H_out,
                        W_out,
                        stride,
                        dilation,
                        pad_top,
                        pad_left,
                        kernel.requires_grad,
                        image.requires_grad,
                        bias.requires_grad,
                    )
                    if kernel.requires_grad:
                        var gk = grads[0]
                        var gk_gpu = Gradbox[Self.dtype](
                            gk.buffer().to_gpu(gpu)
                        )
                        kernel.update_grad(gk_gpu^, AddTensor, None)
                    parent_ids.append(kernel._id)
                    if image.requires_grad:
                        var gi = grads[1]
                        var gi_gpu = Gradbox[Self.dtype](
                            gi.buffer().to_gpu(gpu)
                        )
                        image.update_grad(gi_gpu^, AddTensor, None)
                    parent_ids.append(image._id)
                    if bias.requires_grad:
                        var gb = grads[2]
                        var gb_gpu = Gradbox[Self.dtype](
                            gb.buffer().to_gpu(gpu)
                        )
                        bias.update_grad(gb_gpu^, AddTensor, None)
                    parent_ids.append(bias._id)
                    output.gradients().zero_grad()
                except e:
                    panic(
                        "ConvBackward GPU fallback failed: " + String(e),
                        "at ConvBackward → backward",
                    )
                return

        # CPU PATH
        ref dy_ref = output.gradients()
        var grads = Self._cpu_gradients(
            dy_ref.buffer().copy(),
            image.buffer().copy(),
            kernel.buffer().copy(),
            N,
            C_in,
            H_in,
            W_in,
            C_out,
            KH,
            KW,
            H_out,
            W_out,
            stride,
            dilation,
            pad_top,
            pad_left,
            kernel.requires_grad,
            image.requires_grad,
            bias.requires_grad,
        )

        # dK (same parent_ids contract as before: id always appended).
        var gk = grads[0]
        if kernel.requires_grad:
            kernel.update_grad(gk, AddTensor, None)
        parent_ids.append(kernel._id)

        # dX (same contract).
        var gi = grads[1]
        if image.requires_grad:
            image.update_grad(gi, AddTensor, None)
        parent_ids.append(image._id)

        # db (same contract).
        var gb = grads[2]
        if bias.requires_grad:
            bias.update_grad(gb, AddTensor, None)
        parent_ids.append(bias._id)

        dy_ref.zero_grad()

    @staticmethod
    def _cpu_gradients(
        dy_ndb_in: NDBuffer[Self.dtype],
        img_ndb: NDBuffer[Self.dtype],
        kern_ndb: NDBuffer[Self.dtype],
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
        need_kernel: Bool,
        need_image: Bool,
        need_bias: Bool,
    ) -> Tuple[Gradbox[Self.dtype], Gradbox[Self.dtype], Gradbox[Self.dtype]]:
        """Shared CPU conv-backward core: (grad_kernel, grad_image, grad_bias).

        Inputs are CPU NDBuffers (caller shadows GPU parents via to_cpu
        first). Skipped entries return a 1-element placeholder — the caller
        only applies grads whose need_* flag is set.
        """
        var M = N * H_out * W_out
        var K = C_in * KH * KW
        var HW_out = H_out * W_out

        # dY work buffer: gradbox is guaranteed contiguous+offset-0 by the
        # invoke boundary assert in debug builds; materialize defensively so
        # release builds never silently misread a strided gradbox.
        # O(1) alias; materialize only when the boundary invariant fails.
        var dy_ndb = dy_ndb_in.copy()
        var dy_ok = dy_ndb.is_contiguous() and dy_ndb.offset == 0
        if not dy_ok:
            dy_ndb = dy_ndb_in.contiguous(owned=True)
        var dy_ptr = dy_ndb.data_ptr()

        # Pack dY (N,C_out,H_out,W_out) C-order → P (M, C_out) contiguous.
        # Reads are strided (co-slices), writes contiguous; parallel over m.
        var P_buf = Buffer[Self.dtype](M * C_out)
        var P_ptr = P_buf.unsafe_ptr()

        def pack_dy_row(m: Int) {imm}:
            var n = m // HW_out
            var rem = m - n * HW_out
            var oy = rem // W_out
            var ox = rem - oy * W_out
            var src_base = (n * C_out) * HW_out + oy * W_out + ox
            var dst_base = m * C_out
            for co in range(C_out):
                P_ptr[unsafe_offset=dst_base + co] = dy_ptr[
                    unsafe_offset=src_base + co * HW_out
                ]

        if M >= num_physical_cores():
            parallelize(pack_dy_row, M, num_physical_cores())
        else:
            for m in range(M):
                pack_dy_row(m)

        # dK = P^T @ cols
        var grad_kernel = Gradbox[Self.dtype].zeros(Shape(1))
        if need_kernel:
            # Recompute im2col cols (M, K) from the stored image.
            var cols_buf = Conv[Self.dtype].im2col(
                img_ndb.copy(),
                N,
                C_in,
                H_in,
                W_in,
                H_out,
                W_out,
                KH,
                KW,
                stride,
                dilation,
                pad_top,
                pad_left,
            )
            # A = P^T (C_out, M) as a strided view (path 1b: strided A,
            # contiguous B); B = cols (M, K) contiguous.
            var A_layout = Layout(
                Shape(C_out, M), Strides(1, C_out)
            )
            var B_layout = Layout(Shape(M, K))
            var C_buf = MmCpu2d[Self.dtype].tiled_matmul(
                A_layout, P_buf.copy(), B_layout, cols_buf^
            )[1]
            # C (C_out, K) C-order reinterprets directly as
            # (C_out, C_in, KH, KW) — identical memory order, zero copy.
            var dk_ndb = NDBuffer[Self.dtype](
                C_buf^, Shape(C_out, C_in, KH, KW)
            )
            grad_kernel = Gradbox[Self.dtype](dk_ndb^)

        # dX = col2im(P @ W)
        var grad_image = Gradbox[Self.dtype].zeros(Shape(1))
        if need_image:
            # B = kernel (C_out, K) — already row-major in memory.
            var A_layout = Layout(Shape(M, C_out))
            var B_layout = Layout(Shape(C_out, K))
            var kern_shadow = kern_ndb.copy()
            var dcols_buf = MmCpu2d[Self.dtype].tiled_matmul(
                A_layout, P_buf.copy(), B_layout, kern_shadow.buffer.copy()
            )[1]
            var dx_buf = Conv[Self.dtype].col2im(
                dcols_buf^,
                M,
                K,
                N,
                C_in,
                H_in,
                W_in,
                H_out,
                W_out,
                KH,
                KW,
                stride,
                dilation,
                pad_top,
                pad_left,
            )
            var dx_ndb = NDBuffer[Self.dtype](
                dx_buf^, Shape(N, C_in, H_in, W_in)
            )
            grad_image = Gradbox[Self.dtype](dx_ndb^)

        # db = column sums over P
        # Two defects fixed here. BOTH were
        # invisible to the suite because every test seeds dY uniformly,
        # which makes a row-window sum and a column sum numerically equal.
        #
        # (1) The column sum must be STRIDED. P is (M, C_out) row-major, so
        #     one column's elements are C_out apart. The old code loaded a
        #     CONTIGUOUS window of simdwidth elements at `co + m * C_out` —
        #     a horizontal slice spanning columns co..co+simdwidth-1 — then
        #     summed the lanes. When M < simdwidth that window never loads
        #     and the scalar tail is exact (why small convs passed); for
        #     M >= simdwidth the result was a sum over a sparse subset of
        #     rows. Now a true strided load.
        # (2) `parallelize` over C_out corrupts db when the packed P
        #     (M*C_out) has < 16 elements on this build — observed M/2 or
        #     foreign memory. Every observed failure had P < 16; every
        #     P >= 16 config was exact (sweep of 100+ shapes x 3 reps). The
        #     serial loop is exact at any size, so gate the parallel branch
        #     on total packed elements.
        var grad_bias = Gradbox[Self.dtype].zeros(Shape(1))
        if need_bias:
            var db_buf = Buffer[Self.dtype](C_out)
            var db_ptr = db_buf.unsafe_ptr()
            comptime sw = simd_width_of[Self.dtype]()

            def bias_col(co: Int) {imm}:
                # Strided column sum: lane j accumulates P[m + j, co].
                var acc = SIMD[Self.dtype, sw](0)
                var m = 0
                while m + sw <= M:
                    acc += strided_load[sw](
                        P_ptr.unsafe_offset(co + m * C_out), C_out
                    )
                    m += sw
                var total = Scalar[Self.dtype](0)
                for v in range(sw):
                    total += acc[v]
                while m < M:
                    total += P_ptr[unsafe_offset=co + m * C_out]
                    m += 1
                db_ptr[unsafe_offset=co] = total

            if C_out >= num_physical_cores() and M * C_out >= 16:
                parallelize(bias_col, C_out, num_physical_cores())
            else:
                for co in range(C_out):
                    bias_col(co)
            var db_ndb = NDBuffer[Self.dtype](db_buf^, Shape(C_out))
            grad_bias = Gradbox[Self.dtype](db_ndb^)

        return (grad_kernel^, grad_image^, grad_bias^)


@fieldwise_init
struct Conv[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    """From-scratch Conv2D core: im2col + GEMM + fused bias epilogue."""

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        kernel: Tensor[Self.dtype],
        bias: Tensor[Self.dtype],
        stride: Int = 1,
        dilation: Int = 1,
        pad_top: Int = 0,
        pad_bottom: Int = 0,
        pad_left: Int = 0,
        pad_right: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        # Layout contract: dense NCHW, contiguous, zero offset. Fail loud —
        # the old core silently assumed this and misread views.
        # (The GPU branch below runs first: ConvGpu materializes views
        # itself, so GPU inputs skip the contiguity gate.)
        ref image_shape = image.shape()
        ref kernel_shape = kernel.shape()
        if image_shape.rank() != 4:
            panic("Conv: image must be 4D (N, C_in, H_in, W_in)")
        if kernel_shape.rank() != 4:
            panic("Conv: kernel must be 4D (C_out, C_in, KH, KW)")
        var N = image_shape[0]
        var C_in = image_shape[1]
        var H_in = image_shape[2]
        var W_in = image_shape[3]
        var C_out = kernel_shape[0]
        if kernel_shape[1] != C_in:
            panic("Conv: kernel input channels must match input channels")
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]
        if not bias.shape() == Shape(C_out):
            panic("Conv: bias must have shape (C_out,)")

        # GPU forward (direct convolution kernel; backward falls back to CPU
        # via transfers in ConvBackward).
        comptime if has_accelerator():
            if image.is_on_gpu():
                return Self.forward_gpu[track_grad=track_grad](
                    image,
                    kernel,
                    bias,
                    stride,
                    dilation,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    requires_grad=requires_grad,
                    sync=sync,
                )

        if not image.is_contiguous() or image.offset() != 0:
            panic("Conv: image must be contiguous with zero offset")
        if not kernel.is_contiguous() or kernel.offset() != 0:
            panic("Conv: kernel must be contiguous with zero offset")

        var H_out = (H_in + pad_top + pad_bottom - (KH + (KH - 1) * (dilation - 1))) // stride + 1
        var W_out = (W_in + pad_left + pad_right - (KW + (KW - 1) * (dilation - 1))) // stride + 1
        if H_out <= 0 or W_out <= 0:
            panic("Conv: parameters lead to non-positive output size")

        var M = N * H_out * W_out
        var K = C_in * KH * KW

        # 1. Blocked im2col with fused pad (OOB reads → 0).
        var cols_buf = Self.im2col(
            image.buffer,
            N,
            C_in,
            H_in,
            W_in,
            H_out,
            W_out,
            KH,
            KW,
            stride,
            dilation,
            pad_top,
            pad_left,
        )

        # 2. Repack kernel (C_out, K) C-order → (K, C_out) contiguous.
        var kern_buf = Self.repack_kernel(
            kernel.buffer, C_out, K
        )

        # 3. GEMM: C (M, C_out) = cols (M, K) @ kern (K, C_out).
        # Both sides contiguous → path 1a (SIMD+FMA+unroll+prefetch).
        var A_layout = Layout(Shape(M, K))
        var B_layout = Layout(Shape(K, C_out))
        var result = MmCpu2d[Self.dtype].tiled_matmul(
            A_layout, cols_buf^, B_layout, kern_buf^
        )
        var C_buf = result[1]

        # 4. Repack C (M, C_out) → (N, C_out, H_out, W_out) + fused bias.
        var out_buf = Self.repack_output_bias(
            C_buf^,
            bias.buffer,
            N,
            C_out,
            H_out,
            W_out,
        )
        var out_ndb = NDBuffer[Self.dtype](
            out_buf^, Shape(N, C_out, H_out, W_out)
        )
        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(
                image.requires_grad or kernel.requires_grad or bias.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn(
                    ConvBwdArg(
                        N,
                        C_in,
                        H_in,
                        W_in,
                        C_out,
                        KH,
                        KW,
                        H_out,
                        W_out,
                        stride,
                        dilation,
                        pad_top,
                        pad_left,
                    ),
                    ConvBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, image, kernel, bias)

        return out^

    @staticmethod
    def forward_gpu[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        kernel: Tensor[Self.dtype],
        bias: Tensor[Self.dtype],
        stride: Int = 1,
        dilation: Int = 1,
        pad_top: Int = 0,
        pad_bottom: Int = 0,
        pad_left: Int = 0,
        pad_right: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        """GPU conv forward via ConvGpu. Same contract as forward.

        Mixed devices fail loud — no silent per-op transfers (the
        internally-created zeros bias already lives on image.device();
        move the model once with to_gpu).
        """
        if not kernel.is_on_gpu() or not bias.is_on_gpu():
            panic(
                "Conv: mixed devices — move image, kernel and bias to the"
                " same GPU"
            )
        # Re-derive dims (forward validated rank/channels/bias already).
        ref image_shape = image.shape()
        ref kernel_shape = kernel.shape()
        var N = image_shape[0]
        var C_in = image_shape[1]
        var H_in = image_shape[2]
        var W_in = image_shape[3]
        var C_out = kernel_shape[0]
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]
        try:
            var result = ConvGpu[Self.dtype].launch(
                image.buffer.layout(),
                image.buffer.device_state.value(),
                kernel.buffer.layout(),
                kernel.buffer.device_state.value(),
                bias.buffer.layout(),
                bias.buffer.device_state.value(),
                stride,
                dilation,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                sync=sync,
            )
            var out_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                result[0], result[1]
            )
            var H_out = out_ndb.shape[2]
            var W_out = out_ndb.shape[3]
            var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

            comptime if track_grad:
                var grad_required = requires_grad.or_else(
                    image.requires_grad
                    or kernel.requires_grad
                    or bias.requires_grad
                )
                if grad_required:
                    out.requires_grad_(True)
                    var backwardFn = BackwardFn(
                        ConvBwdArg(
                            N,
                            C_in,
                            H_in,
                            W_in,
                            C_out,
                            KH,
                            KW,
                            H_out,
                            W_out,
                            stride,
                            dilation,
                            pad_top,
                            pad_left,
                        ),
                        ConvBackward[Self.dtype](),
                    )
                    backwardFn.needs_parent_data = True
                    out.add_ancestry(backwardFn^, image, kernel, bias)

            return out^
        except e:
            panic("Conv GPU forward failed: " + String(e))
            # Unreachable — satisfies definite assignment.
            return Tensor[Self.dtype].zeros(Shape(1))

    @staticmethod
    def im2col(
        image_buf: NDBuffer[Self.dtype],
        N: Int,
        C_in: Int,
        H_in: Int,
        W_in: Int,
        H_out: Int,
        W_out: Int,
        KH: Int,
        KW: Int,
        stride: Int,
        dilation: Int,
        pad_top: Int,
        pad_left: Int,
    ) -> Buffer[Self.dtype]:
        """Gather (M, K) patch rows with fused zero-pad. Parallel over rows."""
        var M = N * H_out * W_out
        var K = C_in * KH * KW
        var cols = Buffer[Self.dtype](M * K)
        var cols_ptr = cols.unsafe_ptr()
        # Rebind: params are immutable, data_ptr needs ref access.
        var img = image_buf
        var img_ptr = img.data_ptr()
        var img_off = img.offset
        var HW_in = H_in * W_in
        var C_in_HW = C_in * HW_in

        def fill_row(m: Int) {imm}:
            var n = m // (H_out * W_out)
            var rem = m - n * (H_out * W_out)
            var oy = rem // W_out
            var ox = rem - oy * W_out
            var dst = m * K
            var q = 0
            for ci in range(C_in):
                var src_ci = img_off + n * C_in_HW + ci * HW_in
                for ky in range(KH):
                    var sy = oy * stride + ky * dilation - pad_top
                    if sy < 0 or sy >= H_in:
                        # Fully padded row segment → zeros.
                        for _ in range(KW):
                            cols_ptr[unsafe_offset=dst + q] = Scalar[
                                Self.dtype
                            ](0)
                            q += 1
                        continue
                    var sx_base = ox * stride - pad_left
                    if dilation == 1 and sx_base >= 0 and sx_base + KW <= W_in:
                        # In-bounds dense run. Small runs inline as scalar
                        # stores (a memcpy call costs more than a few
                        # stores); large runs use bulk copy.
                        var src = src_ci + sy * W_in + sx_base
                        if KW <= 16:
                            for kx in range(KW):
                                cols_ptr[unsafe_offset=dst + q + kx] = img_ptr[
                                    unsafe_offset=src + kx
                                ]
                            q += KW
                        else:
                            unsafe_memcpy(
                                dest=cols_ptr.unsafe_offset(dst + q),
                                src=img_ptr.unsafe_offset(src),
                                count=KW,
                            )
                            q += KW
                    else:
                        for kx in range(KW):
                            var sx = sx_base + kx * dilation
                            if sx < 0 or sx >= W_in:
                                cols_ptr[unsafe_offset=dst + q] = Scalar[
                                    Self.dtype
                                ](0)
                            else:
                                cols_ptr[unsafe_offset=dst + q] = img_ptr[
                                    unsafe_offset=src_ci + sy * W_in + sx
                                ]
                            q += 1

        if M >= num_physical_cores():
            parallelize(fill_row, M, num_physical_cores())
        else:
            for m in range(M):
                fill_row(m)
        return cols^

    @staticmethod
    def repack_kernel(
        kernel_buf: NDBuffer[Self.dtype], C_out: Int, K: Int
    ) -> Buffer[Self.dtype]:
        """Transpose-pack (C_out, K) C-order → (K, C_out) contiguous."""
        var out = Buffer[Self.dtype](K * C_out)
        var out_ptr = out.unsafe_ptr()
        # Rebind: params are immutable, data_ptr needs ref access.
        var kern = kernel_buf
        var kern_ptr = kern.data_ptr()
        var kern_off = kern.offset

        def pack_co(co: Int) {imm}:
            var src = kern_off + co * K
            for k in range(K):
                out_ptr[unsafe_offset=k * C_out + co] = kern_ptr[
                    unsafe_offset=src + k
                ]

        if C_out >= num_physical_cores():
            parallelize(pack_co, C_out, num_physical_cores())
        else:
            for co in range(C_out):
                pack_co(co)
        return out^

    @staticmethod
    def repack_output_bias(
        C_buf: Buffer[Self.dtype],
        bias_buf: NDBuffer[Self.dtype],
        N: Int,
        C_out: Int,
        H_out: Int,
        W_out: Int,
    ) -> Buffer[Self.dtype]:
        """Repack GEMM C (M, C_out) → NCHW with fused bias add.

        Co-outer traversal: for a fixed channel, m-order writes are
        SEQUENTIAL (one HW_out run per batch), while the old m-outer order
        scattered with H_out*W_out stride — same element count but ~16x
        the cache-line traffic. Reads from C stride by C_out (L1-resident
        working set); bias hoists to a scalar per channel.
        """
        var HW_out = H_out * W_out
        var out = Buffer[Self.dtype](N * C_out * HW_out)
        var out_ptr = out.unsafe_ptr()
        # Rebind: params are immutable, data_ptr needs ref access.
        var Cb = C_buf
        var C_ptr = Cb.unsafe_ptr()
        var bb = bias_buf
        var bias_ptr = bb.data_ptr()
        var bias_off = bb.offset

        def repack_co(co: Int) {imm}:
            var b = bias_ptr[unsafe_offset=bias_off + co]
            var dst_ch = co * HW_out
            for n in range(N):
                var m_base = n * HW_out
                var dst_base = (n * C_out) * HW_out + dst_ch
                var src_base = m_base * C_out + co
                for r in range(HW_out):
                    out_ptr[unsafe_offset=dst_base + r] = C_ptr[
                        unsafe_offset=src_base + r * C_out
                    ] + b

        parallelize(repack_co, C_out, num_physical_cores())
        return out^

    @staticmethod
    def col2im(
        dcols_buf: Buffer[Self.dtype],
        M: Int,
        K: Int,
        N: Int,
        C_in: Int,
        H_in: Int,
        W_in: Int,
        H_out: Int,
        W_out: Int,
        KH: Int,
        KW: Int,
        stride: Int,
        dilation: Int,
        pad_top: Int,
        pad_left: Int,
    ) -> Buffer[Self.dtype]:
        """Scatter dCols (M, K) rows into dX (N, C_in, H_in, W_in), zero-init.

        Parallel over N (batch slices are disjoint — no races). When
        stride covers the dilated kernel, windows are disjoint and rows
        parallelize freely (fast path).
        """
        var HW_in = H_in * W_in
        var dx = Buffer[Self.dtype].zeros(N * C_in * HW_in)
        var dx_ptr = dx.unsafe_ptr()
        # Rebind: params are immutable, unsafe_ptr needs ref access.
        var dcb = dcols_buf
        var dc_ptr = dcb.unsafe_ptr()
        var dil_KH = KH + (KH - 1) * (dilation - 1)
        var dil_KW = KW + (KW - 1) * (dilation - 1)
        var disjoint = stride >= dil_KH and stride >= dil_KW
        comptime sw = simd_width_of[Self.dtype]()

        def scatter_batch(n: Int) {imm}:
            var dx_n = n * C_in * HW_in
            for oy in range(H_out):
                for ox in range(W_out):
                    var m = (n * H_out + oy) * W_out + ox
                    var src = m * K
                    for ci in range(C_in):
                        var dx_ci = dx_n + ci * HW_in
                        var q_base = src + (ci * KH) * KW
                        for ky in range(KH):
                            var sy = oy * stride + ky * dilation - pad_top
                            if sy < 0 or sy >= H_in:
                                continue
                            var sx_base = ox * stride - pad_left
                            var q = q_base + ky * KW
                            if (
                                dilation == 1
                                and sx_base >= 0
                                and sx_base + KW <= W_in
                            ):
                                # Dense run: SIMD accumulate.
                                var dst = dx_ci + sy * W_in + sx_base
                                var c = 0
                                while c + sw <= KW:
                                    var acc = dx_ptr.unsafe_load[width=sw](
                                        dst + c
                                    )
                                    acc += dc_ptr.unsafe_load[width=sw](q + c)
                                    dx_ptr.unsafe_store[width=sw](dst + c, acc)
                                    c += sw
                                while c < KW:
                                    dx_ptr[unsafe_offset=dst + c] += dc_ptr[
                                        unsafe_offset=q + c
                                    ]
                                    c += 1
                            else:
                                for kx in range(KW):
                                    var sx = sx_base + kx * dilation
                                    if sx < 0 or sx >= W_in:
                                        continue
                                    dx_ptr[
                                        unsafe_offset=dx_ci + sy * W_in + sx
                                    ] += dc_ptr[unsafe_offset=q + kx]

        if disjoint:
            # Windows never overlap: every row writes a disjoint region.
            def scatter_row(m: Int) {imm}:
                var n = m // (H_out * W_out)
                var rem = m - n * (H_out * W_out)
                var oy = rem // W_out
                var ox = rem - oy * W_out
                var dx_n = n * C_in * HW_in
                var src = m * K
                for ci in range(C_in):
                    var dx_ci = dx_n + ci * HW_in
                    var q_base = src + (ci * KH) * KW
                    for ky in range(KH):
                        var sy = oy * stride + ky * dilation - pad_top
                        var sx_base = ox * stride - pad_left
                        var q = q_base + ky * KW
                        # Disjoint + in-bounds by construction... except pad
                        # edges still clip: keep the bounds check.
                        for kx in range(KW):
                            var sx = sx_base + kx * dilation
                            if (
                                sy < 0
                                or sy >= H_in
                                or sx < 0
                                or sx >= W_in
                            ):
                                continue
                            dx_ptr[unsafe_offset=dx_ci + sy * W_in + sx] += (
                                dc_ptr[unsafe_offset=q + kx]
                            )

            parallelize(scatter_row, M, num_physical_cores())
        elif N >= num_physical_cores():
            parallelize(scatter_batch, N, num_physical_cores())
        else:
            for n in range(N):
                scatter_batch(n)
        return dx^
