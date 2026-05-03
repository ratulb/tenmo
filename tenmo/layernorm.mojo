# =============================================================================
# LayerNorm forward — tenmo/layernorm.mojo
#
# Composed from existing Tenmo ops — no new kernel needed for first version.
# Welford already powers variance() under the hood.
#
# Forward:
#   mean  = x.mean(axis=-1, keepdims=True)
#   var   = x.variance(axis=-1, keepdims=True, unbiased=False)
#   rstd  = (var + eps).pow(-0.5)          # 1/sqrt(var+eps)
#   x_hat = (x - mean) * rstd
#   out   = gamma * x_hat + beta
#
# Saved into LayerNormBwdArg (all free from forward):
#   x_hat  — needed for d_gamma and dx three-term formula
#   rstd   — needed for dx three-term formula
#   gamma  — needed for dx three-term formula
#
# Backward (three-term formula over last dim D):
#   d_beta  = sum(upstream,       dims=all_except_last)
#   d_gamma = sum(upstream*x_hat, dims=all_except_last)
#   d_x_hat = upstream * gamma                             # (*, D)
#   d_x     = rstd * (d_x_hat
#                     - mean(d_x_hat,       axis=-1, keepdims=True)
#                     - x_hat * mean(d_x_hat * x_hat, axis=-1, keepdims=True))
#
# =============================================================================

from .tensor import Tensor
from .shapes import Shape
from .gradbox import Gradbox
from .ancestry import Ancestor
from .backpropagation import BackwardFnArg, BACKWARD_LAYER_NORM
from .ndbuffer import NDBuffer
from .mnemonics import LAYER_NORM, AddTensor
from .device import GPU
from .common_utils import panic
from std.utils import Variant
from std.sys import has_accelerator


# =============================================================================
# Backward argument — saved from forward, zero recomputation in backward
# =============================================================================


@fieldwise_init
struct LayerNormBwdArg[dtype: DType](ArgumentType):
    var x_hat: NDBuffer[Self.dtype]  # normalized input  (*, D) keepdims shape
    var rstd: NDBuffer[Self.dtype]  # 1/sqrt(var+eps)   (*, 1) keepdims shape
    var gamma: NDBuffer[Self.dtype]  # learnable scale   (D,)
    var normalized_shape: Int  # D — last dim size


# =============================================================================
# Backward
# =============================================================================
@fieldwise_init
struct LayerNormBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref arg = (
            output.ancestry()
            .backward_fn_arg()
            .get[LayerNormBwdArg[Self.dtype]]()
        )
        ref gradbox = output.gradients()[]  # upstream dL/dy  (*, D)

        var input_ancestor = output.ancestry().get(0)  # x
        var gamma_ancestor = output.ancestry().get(1)  # gamma
        var beta_ancestor = output.ancestry().get(2)  # beta

        var upstream = Tensor[Self.dtype](gradbox.buffer)  # (*, D)
        var x_hat = Tensor[Self.dtype](arg.x_hat)  # (*, D)
        var rstd = Tensor[Self.dtype](arg.rstd)  # (*, 1)
        var gamma = Tensor[Self.dtype](arg.gamma)  # (D,)

        # ── dL/dβ = sum(upstream) over all non-D dims ──────────────────────
        # Reduce over all axes 0..rank-2 sequentially, leaving shape (D,)
        # e.g. upstream (B, T, D) -> sum axis=0 -> (T, D) -> sum axis=0 -> (D,)
        var d_beta_t = upstream
        for _ax in range(upstream.rank() - 1):
            d_beta_t = d_beta_t.sum[track_grad=False](axes=[0], keepdims=False)
        var d_beta_ndb = d_beta_t.buffer

        # ── dL/dγ = sum(upstream * x_hat) over all non-D dims ──────────────
        var ux = upstream.__mul__[track_grad=False](x_hat)  # (*, D)
        var d_gamma_t = ux
        for _ax in range(ux.rank() - 1):
            d_gamma_t = d_gamma_t.sum[track_grad=False](
                axes=[0], keepdims=False
            )
        var d_gamma_ndb = d_gamma_t.buffer

        # ── dL/dx — three-term formula ──────────────────────────────────────
        # d_x_hat = upstream * gamma   broadcast gamma (D,) over (*, D)
        var d_x_hat = upstream.__mul__[track_grad=False](gamma)  # (*, D)

        # mean(d_x_hat, axis=-1, keepdims=True)
        var mean_d_x_hat = d_x_hat.mean[track_grad=False](
            axes=[-1], keepdims=True
        )  # (*, 1)

        # mean(d_x_hat * x_hat, axis=-1, keepdims=True)
        var mean_d_x_hat_x_hat = d_x_hat.__mul__[track_grad=False](x_hat).mean[
            track_grad=False
        ](
            axes=[-1], keepdims=True
        )  # (*, 1)

        # dx = rstd * (d_x_hat - mean_d_x_hat - x_hat * mean_d_x_hat_x_hat)
        var term1 = d_x_hat
        var term2 = mean_d_x_hat  # (*, 1) broadcasts
        var term3 = x_hat.__mul__[track_grad=False](
            mean_d_x_hat_x_hat  # (*, 1) broadcasts
        )
        var bracket = term1.__sub__[track_grad=False](term2).__sub__[
            track_grad=False
        ](term3)

        var d_x = rstd.__mul__[track_grad=False](bracket)  # (*, D)

        # ── Wrap into Gradbox and return ────────────────────────────────────
        var d_input = Gradbox[Self.dtype](d_x.buffer, share=False)
        var d_gamma = Gradbox[Self.dtype](d_gamma_ndb, share=False)
        var d_beta = Gradbox[Self.dtype](d_beta_ndb, share=False)

        return [
            (input_ancestor^, d_input^, AddTensor),
            (gamma_ancestor^, d_gamma^, AddTensor),
            (beta_ancestor^, d_beta^, AddTensor),
        ]


@fieldwise_init
struct LayerNormForward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        gamma: Tensor[Self.dtype],
        beta: Tensor[Self.dtype],
        eps: Scalar[Self.dtype] = Scalar[Self.dtype](1e-5),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var D = self.shape()[-1]

        if gamma.shape() != Shape(D) or beta.shape() != Shape(D):
            panic("LayerNorm: gamma and beta shape mismatch")

        # ── Pass 1: Welford — mean + var in single pass ──────────────────
        var (mean_ndb, var_ndb) = self.buffer.welford(
            axis=self.rank() - 1, unbiased=False, keepdims=True
        )

        # ── Pass 2: fused normalize — rstd + x_hat + out in single pass ──
        # rstd = rsqrt(var + eps) computed inside kernel — saved for backward
        # x_hat saved for backward — written in pass 2 anyway, zero extra cost
        var (out_ndb, x_hat_ndb, rstd_ndb) = self.buffer.layernorm_normalize(
            mean_ndb, var_ndb, gamma.buffer, beta.buffer, eps
        )

        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        # ── Autograd wiring ──────────────────────────────────────────────
        comptime if track_grad:
            var grad_required = requires_grad.or_else(
                self.requires_grad or gamma.requires_grad or beta.requires_grad
            )
            if grad_required:
                out.requires_grad_(True)
                var bwd_arg = LayerNormBwdArg[Self.dtype](
                    x_hat=x_hat_ndb^,  # (*, D) — free from pass 2
                    rstd=rstd_ndb^,  # (*, 1) — free from pass 2
                    gamma=gamma.buffer,  # (D,)
                    normalized_shape=D,
                )
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_LAYER_NORM, bwd_arg^
                )
                out.add_ancestry(backwardFnArg^, self, gamma, beta)

        return out^


@fieldwise_init
struct LayerNorm[dtype: DType](ImplicitlyCopyable & Movable):
    """Layer normalization over the last dimension.

    Normalizes inputs to zero mean and unit variance per token/position,
    then applies learnable scale (gamma) and shift (beta).

    Used in transformer blocks as Pre-LayerNorm:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))

    Args:
        normalized_shape: Size of the last dimension D.
        eps:              Numerical stability constant. Default 1e-5.
    """

    var gamma: Tensor[Self.dtype]  # (normalized_shape,) — ones init
    var beta: Tensor[Self.dtype]  # (normalized_shape,) — zeros init
    var normalized_shape: Int
    var eps: Scalar[Self.dtype]
    var training: Bool

    fn __init__(
        out self,
        normalized_shape: Int,
        eps: Scalar[Self.dtype] = Scalar[Self.dtype](1e-5),
    ):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.training = True
        self.gamma = Tensor[Self.dtype].ones(
            Shape(normalized_shape), requires_grad=True
        )
        self.beta = Tensor[Self.dtype].zeros(
            Shape(normalized_shape), requires_grad=True
        )

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if self.training:
            return LayerNormForward[Self.dtype].forward[track_grad=True](
                x, self.gamma, self.beta, self.eps
            )
        else:
            return LayerNormForward[Self.dtype].forward[track_grad=False](
                x, self.gamma, self.beta, self.eps
            )

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        var params = List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()
        params.append(
            UnsafePointer(to=self.gamma).unsafe_mut_cast[True]().as_any_origin()
        )
        params.append(
            UnsafePointer(to=self.beta).unsafe_mut_cast[True]().as_any_origin()
        )
        return params^

    fn num_parameters(self) -> Int:
        return self.gamma.numels() + self.beta.numels()

    fn train(mut self):
        """Set to training mode."""
        self.training = True

    fn eval(mut self):
        """Set to evaluation mode — no gradient tracking."""
        self.training = False

    fn to_gpu(deinit self, gpu: Optional[GPU] = None) raises -> Self:
        """Move gamma and beta to GPU as permanent GPU leaves."""
        var out = self^
        out.gamma = out.gamma.to_gpu(gpu=gpu, stop_grad=True)
        out.beta = out.beta.to_gpu(gpu=gpu, stop_grad=True)
        return out^

    fn to_cpu(deinit self) raises -> Self:
        """Move gamma and beta back to CPU after training."""
        var out = self^
        out.gamma = out.gamma.to_cpu(stop_grad=True)
        out.beta = out.beta.to_cpu(stop_grad=True)
        return out^

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), LAYER_NORM)
