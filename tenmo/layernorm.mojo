# LayerNorm — tenmo/layernorm.mojo
#
# Normalizes each token's feature vector (last dim, size D) to zero mean and
# ~unit variance, then applies a learnable per-feature affine (gamma, beta).
#
# NOTATION (math <-> code) — used in every comment below
#   mu     = mean(x) over the last dim                  Welford, pass 1
#   v      = mean((x - mu)^2), biased (unbiased=False)  Welford, pass 1
#   sigma  = sqrt(v + eps)  =  1 / rstd                 eps is INSIDE the sqrt
#   y      = x_hat = (x - mu) / sigma                   saved for backward
#   out    = gamma * x_hat + beta
#   g      = d_x_hat = upstream * gamma                 grad arriving at x_hat
#
# With eps > 0 the variance of x_hat is v / (v + eps): just under 1, not
# exactly 1.
#
# FORWARD — execution path (two passes, no composed Tensor ops)
#
#   Pass 1  Welford: mean and biased variance in a single pass over x.
#           Stride-safe; runs on CPU and GPU.
#   Pass 2  Fused normalize: rstd, x_hat and out in one sweep.
#           GPU: LayerNormKernel.  CPU: normalize_cpu (SIMD over D, parallel
#           over rows when the input is large enough).
#
# Reference semantics — what the two passes compute (composed-op form, kept
# for readability; this is NOT the code path):
#   mean  = x.mean(axis=-1, keepdims=True)
#   var   = x.variance(axis=-1, keepdims=True, unbiased=False)
#   rstd  = (var + eps).pow(-0.5)                 # 1 / sigma
#   x_hat = (x - mean) * rstd
#   out   = gamma * x_hat + beta
#
# Saved into LayerNormBwdArg (by-products of pass 2 — no extra work):
#   x_hat  needed for d_gamma and for the dx three-term formula
#   rstd   needed for the dx three-term formula
#   gamma  needed for d_x_hat = upstream * gamma
#
# BACKWARD — where the three-term formula comes from (composes Tensor ops)
#
#   Chain rule through mu, v, sigma (d = D):
#     d mu    / d x_j = 1/d
#     d v     / d x_j = (2/d)(x_j - mu)       mu-dependence cancels because
#                                             sum_k (x_k - mu) = 0
#     d sigma / d x_j = (x_j - mu) / (d * sigma)
#
#   Product rule on y_i = (x_i - mu) / sigma, then substitute
#   x_i - mu = sigma * y_i:
#     d y_i / d x_j = (1/sigma) * (delta_ij - 1/D - y_i*y_j/D)
#
#   Contract with g  (dx_j = sum_i g_i * d y_i / d x_j):
#
#     dx = (1/sigma) * ( g - mean(g) - y * mean(g * y) )      means over D
#        = rstd * (d_x_hat - mean(d_x_hat) - x_hat * mean(d_x_hat * x_hat))
#
#   Exact WITH eps, provided sigma = sqrt(v + eps) and y is the actual saved
#   x_hat. (Closed form checked against central finite differences, float64,
#   eps = 1e-5: max abs error ~6e-10.)
#
#   Properties of dx:
#     - mean(dx) == 0 exactly (because sum(y) == 0).
#     - dx . y == 0 only in the limit eps -> 0. For eps > 0 it equals
#       (D / sigma) * mean(g * y) * eps / (v + eps): tiny but nonzero.
#     - Reading it: the gradient at x_hat loses its mean and its component
#       along x_hat, then is rescaled by 1/sigma.
#
#   Parameter gradients (sum over every axis except the last):
#     d_beta  = sum(upstream)
#     d_gamma = sum(upstream * x_hat)
#     d_x_hat = upstream * gamma
#
# PRECONDITIONS (assumed by the kernels, enforced by forward)
#   - x: made contiguous by forward() if needed (one copy for strided views).
#   - gamma, beta: forward() materializes a contiguous, offset-0 copy when
#     handed a sliced/offset view (zero-cost alias for ordinary parameters).
#     normalize_cpu / the GPU kernel read them flat from index 0, and the
#     materialized copy is what backward saves, keeping both passes
#     consistent. Only the shape is validated beyond that.

from .tensor import Tensor
from .shared.shapes import Shape
from .gradbox import Gradbox
from .ancestry import Ancestor
from .backpropagation import BackwardFn, ArgumentType, BackwardFnType
from .ndbuffer import NDBuffer
from .shared.buffers import Buffer
from .shared.mnemonics import AddTensor
from .gpu.device import GPU
from .shared.panic import panic
from .kernels.layernorm_kernel import LayerNormKernel
from .shared.intarray import IntArray
from .layer_trait import LayerTrait
from .named_parameter import NamedParameter
from .welford import Welford
from .shared.constants import LAYERNORM_DEFAULT_EPS
from std.sys import has_accelerator, simd_width_of
from max.algorithm import parallelize
from std.sys.info import num_physical_cores
from std.math import rsqrt


@fieldwise_init
struct LayerNormalizer[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    def normalize(
        x: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        var_: NDBuffer[Self.dtype],
        gamma: NDBuffer[Self.dtype],
        beta: NDBuffer[Self.dtype],
        eps: Scalar[Self.dtype],
        sync: Bool = True,
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ]:
        """Fused normalize: rstd + x_hat + out in a single pass.

        Pass 2 of LayerNorm forward — Welford (Pass 1) already ran.
        Computes, per row:
            rstd  = 1 / sqrt(var + eps)
            x_hat = (x - mean) * rstd
            out   = gamma * x_hat + beta
        Returns (out_ndb, x_hat_ndb, rstd_ndb).
        out and x_hat are shape (*, D). rstd is shape (*, 1).
        x_hat and rstd are returned because backward needs them; they are
        already produced by this pass, so saving them costs nothing extra.

        Args:
         x:     Input (*, D). Must be contiguous.
         mean:  Per-row mean (*, 1) from Welford.
         var_:  Per-row biased variance (*, 1) from Welford.
         gamma: Scale (D,). Contiguous, offset 0.
         beta:  Shift (D,). Contiguous, offset 0.
         eps:   Numerical stability constant (added to var inside the sqrt).
         sync:  Whether to synchronize the GPU operation.
        """
        comptime if has_accelerator():
            if x.is_on_gpu():
                try:
                    var (out_pair, x_hat_pair, rstd_pair) = LayerNormKernel[
                        Self.dtype
                    ].launch(
                        x.layout(),
                        x.device_state.value(),
                        mean.layout(),
                        mean.device_state.value(),
                        var_.layout(),
                        var_.device_state.value(),
                        gamma.layout(),
                        gamma.device_state.value(),
                        beta.layout(),
                        beta.device_state.value(),
                        eps,
                        sync=sync,
                    )
                    var out_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        out_pair[0], out_pair[1]
                    )
                    var x_hat_ndb = NDBuffer[
                        Self.dtype
                    ].with_layout_device_state(x_hat_pair[0], x_hat_pair[1])
                    var rstd_ndb = NDBuffer[
                        Self.dtype
                    ].with_layout_device_state(rstd_pair[0], rstd_pair[1])
                    return out_ndb, x_hat_ndb, rstd_ndb
                except e:
                    print(e)
                    panic("LayerNormalizer.normalize → GPU operation failed")
                    return (
                        NDBuffer[Self.dtype].Empty(),
                        NDBuffer[Self.dtype].Empty(),
                        NDBuffer[Self.dtype].Empty(),
                    )  # unreachable
        return LayerNormalizer[Self.dtype].normalize_cpu(
            x, mean, var_, gamma, beta, eps
        )

    @staticmethod
    def normalize_cpu(
        x: NDBuffer[Self.dtype],
        mean: NDBuffer[Self.dtype],
        var_: NDBuffer[Self.dtype],
        gamma: NDBuffer[Self.dtype],
        beta: NDBuffer[Self.dtype],
        eps: Scalar[Self.dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ]:
        """CPU fused normalize — SIMD over D, parallel over rows (gated).

        Layout assumptions: x is contiguous (its offset is honoured);
        mean/var_ are the fresh contiguous (*, 1) outputs of Welford, indexed
        by flat row; gamma/beta are contiguous and indexed from 0.
        """
        var out_shape = x.shape
        var D = out_shape[-1]
        var outer_size = x.numels() // D  # number of tokens (rows)
        var rstd_shape = out_shape[0:-1] + [1]

        var out_buf = Buffer[Self.dtype](x.numels())
        var x_hat_buf = Buffer[Self.dtype](x.numels())
        var rstd_buf = Buffer[Self.dtype](outer_size)

        comptime SIMD_WIDTH = simd_width_of[Self.dtype]()
        var simd_end = D - (D % SIMD_WIDTH)  # SIMD body; scalar tail after

        def ln_row(row: Int) {imm}:
            var row_mean = mean.buffer[row]
            var row_var = var_.buffer[row]
            # rstd = 1 / sigma = rsqrt(var + eps). The conditional is a
            # defensive guard: with var >= 0 and eps > 0, safe_var is always
            # positive, so it only matters if rounding ever hands us a tiny
            # negative variance.
            var safe_var = row_var + eps
            var rstd = rsqrt(
                safe_var if safe_var
                > Scalar[Self.dtype](0) else Scalar[Self.dtype](eps)
            )
            rstd_buf[row] = rstd

            var row_base = row * D
            var x_base = x.offset + row_base
            ref x_buffer = x.buffer
            ref gamma_buffer = gamma.buffer
            ref beta_buffer = beta.buffer
            # SIMD body: x_hat = (x - mu) * rstd ; out = gamma * x_hat + beta
            for i in range(0, simd_end, SIMD_WIDTH):
                var x_vec = x_buffer.load[simdwidth=SIMD_WIDTH](x_base + i)
                var gamma_vec = gamma_buffer.load[simdwidth=SIMD_WIDTH](i)
                var beta_vec = beta_buffer.load[simdwidth=SIMD_WIDTH](i)
                var x_hat_vec = (x_vec - row_mean) * rstd
                var out_vec = gamma_vec * x_hat_vec + beta_vec
                x_hat_buf.store[simdwidth=SIMD_WIDTH](row_base + i, x_hat_vec)
                out_buf.store[simdwidth=SIMD_WIDTH](row_base + i, out_vec)
            # Scalar tail: the last D % SIMD_WIDTH features of the row.
            for i in range(simd_end, D):
                var x_i = x_buffer[x_base + i]
                var x_hat_i = (x_i - row_mean) * rstd
                var out_i = gamma_buffer[i] * x_hat_i + beta_buffer[i]
                x_hat_buf[row_base + i] = x_hat_i
                out_buf[row_base + i] = out_i

        # Rows are independent, so parallelism needs no synchronization.
        # Gated: only when there are enough rows and enough total work to
        # pay for the thread hand-off.
        var n_threads = num_physical_cores()
        if outer_size >= n_threads and outer_size * D >= n_threads * 1024:
            parallelize(ln_row, outer_size, n_threads)
        else:
            for row in range(outer_size):
                ln_row(row)

        var out_ndb = NDBuffer[Self.dtype](out_buf^, out_shape)
        var x_hat_ndb = NDBuffer[Self.dtype](x_hat_buf^, out_shape)
        var rstd_ndb = NDBuffer[Self.dtype](rstd_buf^, rstd_shape)
        return (out_ndb^, x_hat_ndb^, rstd_ndb^)


# Backward argument — saved from forward, zero recomputation in backward


@fieldwise_init
struct LayerNormBwdArg[dtype: DType](ArgumentType):
    var x_hat: NDBuffer[Self.dtype]  # y = (x - mu)/sigma          (*, D)
    var rstd: NDBuffer[Self.dtype]  # 1/sigma = rsqrt(var + eps)   (*, 1)
    # gamma is read again in backward (d_x_hat = upstream * gamma), so it must
    # not be modified in place between forward and backward.
    var gamma: NDBuffer[Self.dtype]  # learnable scale             (D,)
    var normalized_shape: Int  # D — last dim size


# Fused CPU dx — backward analogue of normalize_cpu (pass 2)


@fieldwise_init
struct LayerNormDxCpu[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    def fused_dx(
        upstream: NDBuffer[Self.dtype],
        x_hat: NDBuffer[Self.dtype],
        rstd: NDBuffer[Self.dtype],
        gamma: NDBuffer[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """Fused dx: rstd * (g - mean(g) - x_hat * mean(g * x_hat)) per row.

        Same three-term formula as the composed backward path, but each row
        is computed in two sweeps with scalar accumulators (m1, m2) instead
        of ~6 full-size (*, D) temporaries (d_x_hat, means, term copies,
        bracket). SIMD over D, parallel over rows (gated, same rule as
        normalize_cpu).

        Layout assumptions mirror normalize_cpu: upstream/x_hat honour
        their offset (both are contiguous, offset-0 by construction);
        rstd is indexed by flat row; gamma is contiguous, indexed from 0
        (guaranteed by forward's layout guard).
        """
        var out_shape = upstream.shape
        var D = out_shape[-1]
        var outer_size = upstream.numels() // D  # number of tokens (rows)
        var out_buf = Buffer[Self.dtype](upstream.numels())

        comptime SIMD_WIDTH = simd_width_of[Self.dtype]()
        var simd_end = D - (D % SIMD_WIDTH)  # SIMD body; scalar tail after

        def dx_row(row: Int) {imm}:
            var r = rstd.buffer[row]
            var up_base = upstream.offset + row * D
            var y_base = x_hat.offset + row * D
            ref up_buffer = upstream.buffer
            ref y_buffer = x_hat.buffer
            ref gamma_buffer = gamma.buffer
            # Pass A: scalar row means m1 = mean(g), m2 = mean(g * y),
            # with g = upstream * gamma.
            var m1 = Scalar[Self.dtype](0)
            var m2 = Scalar[Self.dtype](0)
            for i in range(0, simd_end, SIMD_WIDTH):
                var u_vec = up_buffer.load[simdwidth=SIMD_WIDTH](up_base + i)
                var g_vec = gamma_buffer.load[simdwidth=SIMD_WIDTH](i)
                var y_vec = y_buffer.load[simdwidth=SIMD_WIDTH](y_base + i)
                var gg = u_vec * g_vec
                m1 += gg.reduce_add()
                m2 += (gg * y_vec).reduce_add()
            for i in range(simd_end, D):
                var gg = up_buffer[up_base + i] * gamma_buffer[i]
                m1 += gg
                m2 += gg * y_buffer[y_base + i]
            m1 /= Scalar[Self.dtype](D)
            m2 /= Scalar[Self.dtype](D)
            # Pass B: dx = rstd * (g - m1 - y * m2).
            var dx_base = row * D
            for i in range(0, simd_end, SIMD_WIDTH):
                var u_vec = up_buffer.load[simdwidth=SIMD_WIDTH](up_base + i)
                var g_vec = gamma_buffer.load[simdwidth=SIMD_WIDTH](i)
                var y_vec = y_buffer.load[simdwidth=SIMD_WIDTH](y_base + i)
                var gg = u_vec * g_vec
                var dx_vec = (gg - m1 - y_vec * m2) * r
                out_buf.store[simdwidth=SIMD_WIDTH](dx_base + i, dx_vec)
            for i in range(simd_end, D):
                var gg = up_buffer[up_base + i] * gamma_buffer[i]
                out_buf[dx_base + i] = (
                    gg - m1 - y_buffer[y_base + i] * m2
                ) * r

        # Rows are independent, so parallelism needs no synchronization.
        # Same gate as normalize_cpu.
        var n_threads = num_physical_cores()
        if outer_size >= n_threads and outer_size * D >= n_threads * 1024:
            parallelize(dx_row, outer_size, n_threads)
        else:
            for row in range(outer_size):
                dx_row(row)

        return NDBuffer[Self.dtype](out_buf^, out_shape)


# Backward
@fieldwise_init
struct LayerNormBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        """Backward for out = gamma * x_hat + beta.

        Produces three gradients: d_x, d_gamma, d_beta. See the file header
        for the derivation of the d_x three-term formula.
        """
        ref arg = (
            output.ancestry().backward_fn().get[LayerNormBwdArg[Self.dtype]]()
        )
        ref gradbox = output.gradients()  # upstream dL/d(out)  (*, D)

        var input_ancestor = output.ancestry().get(0)  # x
        var gamma_ancestor = output.ancestry().get(1)  # gamma
        var beta_ancestor = output.ancestry().get(2)  # beta

        var upstream = Tensor[Self.dtype](gradbox.buffer())  # (*, D)
        var x_hat = Tensor[Self.dtype](arg.x_hat)  # y = (x-mu)/sigma  (*, D)
        var rstd = Tensor[Self.dtype](arg.rstd)  # 1/sigma            (*, 1)
        var gamma = Tensor[Self.dtype](arg.gamma)  # (D,)

        # dL/dβ = sum(upstream) over all non-D dims
        # out = gamma * x_hat + beta, so beta receives upstream unchanged,
        # summed over every token.
        # Reduce over all axes 0..rank-2 sequentially, leaving shape (D,)
        # e.g. upstream (B, T, D) -> sum axis=0 -> (T, D) -> sum axis=0 -> (D,)
        var d_beta_t = upstream.copy()
        for _ax in range(upstream.rank() - 1):
            d_beta_t = d_beta_t.sum[track_grad=False](axes=[0], keepdims=False)
        var d_beta_ndb = d_beta_t.buffer

        # dL/dγ = sum(upstream * x_hat) over all non-D dims
        # gamma scales x_hat feature-wise, so its gradient is the
        # upstream/x_hat product summed over every token.
        var ux = upstream.__mul__[track_grad=False](x_hat)  # (*, D)
        var d_gamma_t = ux.copy()
        for _ax in range(ux.rank() - 1):
            d_gamma_t = d_gamma_t.sum[track_grad=False](
                axes=[0], keepdims=False
            )
        var d_gamma_ndb = d_gamma_t.buffer

        # dL/dx — three-term formula
        #   dx = rstd * ( g - mean(g) - x_hat * mean(g * x_hat) )
        # with g = d_x_hat = upstream * gamma, means taken over D.
        # Exact with eps (rstd and x_hat are the saved, eps-inclusive values).
        # Effect: g loses its mean and its component along x_hat, then is
        # rescaled by 1/sigma.
        #
        # CPU runs the fused per-row kernel (two sweeps, zero (*, D)
        # temporaries). GPU stays on the composed Tensor ops below (fused
        # GPU kernel deferred — the CPU result gates it).
        var use_fused_dx = True
        comptime if has_accelerator():
            if output.is_on_gpu():
                use_fused_dx = False

        def compute_dx() {imm} -> NDBuffer[Self.dtype]:
            if use_fused_dx:
                return LayerNormDxCpu[Self.dtype].fused_dx(
                    gradbox.buffer(), arg.x_hat, arg.rstd, arg.gamma
                )
            # g = d_x_hat = upstream * gamma   broadcast gamma (D,) over (*, D)
            var d_x_hat = upstream.__mul__[track_grad=False](gamma)  # (*, D)

            # term 2 input: mean(g, axis=-1, keepdims=True)
            var mean_d_x_hat = d_x_hat.mean[track_grad=False](
                axes=[-1], keepdims=True
            )  # (*, 1)

            # term 3 input: mean(g * x_hat, axis=-1, keepdims=True)
            var mean_d_x_hat_x_hat = d_x_hat.__mul__[track_grad=False](
                x_hat
            ).mean[track_grad=False](
                axes=[-1], keepdims=True
            )  # (*, 1)

            # bracket = g - mean(g) - x_hat * mean(g * x_hat)
            var term1 = d_x_hat.copy()  # g
            var term2 = mean_d_x_hat.copy()  # mean(g)      (*, 1) broadcasts
            var term3 = x_hat.__mul__[track_grad=False](
                mean_d_x_hat_x_hat  # mean(g * x_hat)  (*, 1) broadcasts
            )  # x_hat * mean(g * x_hat)
            var bracket = term1.__sub__[track_grad=False](term2).__sub__[
                track_grad=False
            ](term3)

            # dx = rstd * bracket = bracket / sigma
            var d_x_c = rstd.__mul__[track_grad=False](bracket)  # (*, D)
            return d_x_c.buffer

        # dx = rstd * bracket = bracket / sigma
        var d_x = Tensor[Self.dtype](compute_dx())  # (*, D)

        # Wrap into Gradbox and update parents
        var d_input = Gradbox[Self.dtype](d_x.buffer)
        var d_gamma = Gradbox[Self.dtype](d_gamma_ndb)
        var d_beta = Gradbox[Self.dtype](d_beta_ndb)

        input_ancestor.update_grad(d_input^, AddTensor, None)
        parent_ids.append(input_ancestor._id)
        gamma_ancestor.update_grad(d_gamma^, AddTensor, None)
        parent_ids.append(gamma_ancestor._id)
        beta_ancestor.update_grad(d_beta^, AddTensor, None)
        parent_ids.append(beta_ancestor._id)

        gradbox.zero_grad()


@fieldwise_init
struct LayerNormForward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        gamma: Tensor[Self.dtype],
        beta: Tensor[Self.dtype],
        # Default matches LayerNorm module init and torch.nn.LayerNorm —
        # NOT machine epsilon: a direct call with defaults must agree with
        # the module call style.
        eps: Scalar[Self.dtype] = Scalar[Self.dtype](LAYERNORM_DEFAULT_EPS),
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var D = self.shape()[-1]

        if gamma.shape() != Shape(D) or beta.shape() != Shape(D):
            panic("LayerNorm: gamma and beta shape mismatch")

        # Gamma/beta layout guard
        # Pass 2 reads gamma/beta flat from index 0 (no offset/stride
        # handling). Ordinary parameters are already contiguous with
        # offset 0, so this is a zero-cost alias on the fast path — only
        # a sliced/offset view pays one D-element copy.
        var gamma_buf = gamma.buffer
        if not gamma_buf.is_contiguous() or gamma_buf.offset != 0:
            # owned=True: the default fast path would alias a
            # contiguous+offset view back to itself (offset preserved).
            # GPU note: contiguous_device_state's fast path assumes
            # offset-0 storage; an offset GPU view stays latent until
            # the fused GPU kernel lands (CPU gates it).
            gamma_buf = gamma_buf.contiguous(owned=True)
        var beta_buf = beta.buffer
        if not beta_buf.is_contiguous() or beta_buf.offset != 0:
            beta_buf = beta_buf.contiguous(owned=True)

        # Pass 1: Welford — mean + var in single pass
        # Pass 2 (normalize_cpu) indexes x flat and requires contiguous
        # storage ("must be contiguous" per its docstring); Welford is
        # stride-safe but shares this buffer for consistency. The guard
        # keeps the fast path zero-cost on CPU and GPU (the GPU
        # contiguous() leg always materialises) — only strided views pay
        # one copy. `var` is an alias (no copy) when already contiguous.
        var x_buf = self.buffer
        if not x_buf.is_contiguous():
            x_buf = x_buf.contiguous()
        var (mean_ndb, var_ndb) = Welford[Self.dtype].forward(
            x_buf,
            IntArray(self.rank() - 1),  # reduce over the last axis only
            unbiased=False,  # biased variance (divide by D), as LayerNorm defines it
            keepdims=True,  # (*, 1), so pass 2 can index by flat row
            sync=False,
        )

        # Pass 2: fused normalize — rstd + x_hat + out in single pass
        # rstd = rsqrt(var + eps) computed inside the kernel; saved for backward.
        # x_hat is written in pass 2 anyway, so saving it for backward is free.
        var (out_ndb, x_hat_ndb, rstd_ndb) = LayerNormalizer[
            Self.dtype
        ].normalize(
            x_buf,
            mean_ndb,
            var_ndb,
            gamma_buf,
            beta_buf,
            eps,
            sync=False,
        )

        comptime if has_accelerator():
            if out_ndb.is_on_gpu() and sync:
                out_ndb.sync()
        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        # Autograd wiring
        comptime if track_grad:
            var grad_required = requires_grad.or_else(
                self.requires_grad or gamma.requires_grad or beta.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                var bwd_arg = LayerNormBwdArg[Self.dtype](
                    x_hat=x_hat_ndb^,  # (*, D) — free by-product of pass 2
                    rstd=rstd_ndb^,  # (*, 1) — free by-product of pass 2
                    gamma=gamma_buf^,  # (D,) — post-guard copy
                    normalized_shape=D,
                )
                var backwardFn = BackwardFn(
                    bwd_arg^,
                    LayerNormBackward[Self.dtype](),
                )
                out.add_ancestry(backwardFn^, self, gamma, beta)

        return out^

@fieldwise_init
struct LayerNorm[dtype: DType](LayerTrait):
    """Layer normalization over the last dimension.

    Normalizes each token's features to zero mean and ~unit variance, then
    applies learnable scale (gamma) and shift (beta).

    Used in transformer blocks as Pre-LayerNorm:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    What each part does
    -------------------
    Normalization  (x - mu) / sigma,  sigma = sqrt(var + eps):
        The part that matters. Forward: every token's features get mean 0 and
        variance ~1. Backward: the gradient loses its mean and its component
        along the normalized output, then is rescaled by 1/sigma (derivation
        in the file header). The output is also invariant to the scale of x,
        so the weights feeding this layer are effectively scale-free, and each
        Pre-LN sublayer sees a fixed-scale input while the residual stream
        keeps growing. Mean-subtraction looks dispensable in practice
        (RMSNorm drops it and is reported to match); the rescaling is the
        load-bearing half.

    gamma (scale):
        Not needed for expressivity in Pre-LN: the next op is linear (Q/K/V
        or the MLP's first layer), so gamma and beta fold into its weight and
        bias. What it plausibly adds is a per-feature reparameterization with
        its own gradients and optimizer state (and, in many setups, exclusion
        from weight decay). One case where it is NOT absorbable: a final
        LayerNorm feeding an output head tied to the token embedding.

    beta (shift):
        The weakest case — also absorbable into the next layer's bias, and
        many modern LLMs drop it. Kept for parity with torch.nn.LayerNorm.

    Not there to "undo internal covariate shift": that explanation was
    tested directly and did not hold up.

    Args:
        normalized_shape: Size of the last dimension D.
        eps:              Numerical stability constant, added to the variance
                          inside the sqrt. Default 1e-5.
    """

    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype

    var gamma: Tensor[Self.dtype]  # (normalized_shape,) — ones init
    var beta: Tensor[Self.dtype]  # (normalized_shape,) — zeros init
    var normalized_shape: Int
    var eps: Scalar[Self.dtype]
    var training: Bool

    def __init__(
        out self,
        normalized_shape: Int,
        eps: Scalar[Self.dtype] = Scalar[Self.dtype](LAYERNORM_DEFAULT_EPS),
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.training = True
        # gamma=1, beta=0: at init the affine is the identity, so the layer
        # starts as pure normalization.
        self.gamma = Tensor[Self.dtype].ones(
            Shape(normalized_shape), requires_grad=True
        )
        self.beta = Tensor[Self.dtype].zeros(
            Shape(normalized_shape), requires_grad=True
        )

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        # Training records autograd ancestry; eval skips it (no grad tracking).
        if self.training:
            return LayerNormForward[Self.dtype].forward[track_grad=True](
                x, self.gamma, self.beta, self.eps, sync=sync
            )
        else:
            return LayerNormForward[Self.dtype].forward[track_grad=False](
                x, self.gamma, self.beta, self.eps, sync=sync
            )

    def parameters(
        ref self,
    ) -> List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[Pointer[Tensor[Self.dtype], MutAnyOrigin]]()
        params.append(
            Pointer(to=self.gamma)
            .unsafe_mut_cast[True]()
            .as_unsafe_any_origin()
        )
        params.append(
            Pointer(to=self.beta).unsafe_mut_cast[True]().as_unsafe_any_origin()
        )
        return params^

    def named_parameters(
        ref self, prefix: String
    ) -> List[NamedParameter[Self.dtype]]:
        var result = List[NamedParameter[Self.dtype]]()
        var g = Pointer(to=self.gamma).unsafe_mut_cast[True]()
        result.append(
            NamedParameter(
                prefix + "gamma",
                g.as_unsafe_any_origin().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
            )
        )
        var b = Pointer(to=self.beta).unsafe_mut_cast[True]()
        result.append(
            NamedParameter(
                prefix + "beta",
                b.as_unsafe_any_origin().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
            )
        )
        return result^

    def num_parameters(self) -> Int:
        return self.gamma.numels() + self.beta.numels()

    def train(mut self):
        """Set to training mode."""
        self.training = True

    def eval(mut self):
        """Set to evaluation mode — no gradient tracking."""
        self.training = False

    def to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """Move gamma and beta to GPU as permanent GPU leaves."""
        var out = self
        out.gamma = out.gamma.to_gpu(gpu=gpu, stop_grad=True)
        out.beta = out.beta.to_gpu(gpu=gpu, stop_grad=True)
        return out^

    def to_cpu(self) raises -> Self:
        """Move gamma and beta back to CPU after training."""
        var out = self
        out.gamma = out.gamma.to_cpu(stop_grad=True)
        out.beta = out.beta.to_cpu(stop_grad=True)
        return out^

