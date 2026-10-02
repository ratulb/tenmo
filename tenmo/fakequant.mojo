"""Fake quantization with a straight-through estimator (STE).

    out = s * clamp(round(x / s), qmin, qmax)

One op, not a composition. The reason is the gradient, and it is worth
stating plainly because it is the whole design constraint.

Writing `k = clamp(round(x/s), qmin, qmax)`, we have `out = s * k`. Because
`k` is piecewise constant in both `x` and `s`, the TRUE derivative is

    d out/d s = k      (a.e., saturated cells included)
    d out/d x = 0      (a.e.)

The quantizer is a staircase, so the true input gradient is zero almost
everywhere and training would never move. This op **deliberately lies**:
it overrides `d/dx` to 1 while leaving `d/ds` at its true value. That is
what "straight-through" means — the estimator, not the derivative.

Which forces two parents with two different rules:

| Parent   | Gradient       | Arithmetic               | Needs saved data? |
|----------|----------------|--------------------------|-------------------|
| `x`      | `upstream`     | none — pure hand-off     | no                |
| `scale`  | `upstream * k` | one multiply, then reduce| yes (`k`)         |

The `x` edge is a pure hand-off: no arithmetic, no dtype conversion. That is
precisely why this op needs no int8 gradbox, so the open question of whether
`Gradbox[int8]` can hold a meaningful gradient is not a blocker here.

It must be a new op rather than a relaxation of `to_dtype`. Widening
`to_dtype`'s `trackable` rule would make it claim grad-tracking across
integer casts for *every* caller, including `Embedding`'s index path where
int targets correctly return leaves. One honest op beats one dishonest
shared helper.

**Why `round` and not `floor(x + 0.5)`:** Mojo's `round` is half-to-even
(banker's rounding), while `floor(x + 0.5)` rounds half *up*; they disagree
on every tie. Ties are exactly reachable in quantization because a dyadic
scale puts `x/s` on a `.5` precisely. See `tenmo/roundfloor.mojo`.

CPU only. The gradient math is device-independent, but the forward needs the
round/clip GPU kernels, which do not exist yet, so a device tensor is
rejected loudly rather than silently producing wrong numbers.

Finite differences note: `round` is discontinuous, so an FD probe on this
function is only valid *strictly inside* one quantization cell. Straddling a
step returns garbage -- an early draft of this design came back with the
wrong sign for `d/ds` for exactly that reason.
"""

from .tensor import Tensor
from .shared.shapes import Shape
from .shared.mnemonics import AddTensor
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType
from .gradbox import Gradbox
from .ancestry import Ancestor
from .shared.panic import panic
from .ndbuffer import NDBuffer
from .roundfloor import Round


# Saved forward data


@fieldwise_init
struct FakeQuantBwdArg[dtype: DType](ArgumentType):
    """Saved forward data. Only `k` — the quantized levels, pre-`s`-multiply.

    `k` is needed solely for `d/ds = k`. The `x` edge needs nothing, because
    straight-through means the upstream gradient is handed over untouched.
    """

    var k: NDBuffer[Self.dtype]
    # Whether `scale` is a scalar (per-tensor) or x-shaped (per-element).
    #
    # This is a payload field rather than a `scale_ancestor.shape()` lookup on
    # purpose: `to_ancestor` never populates `Ancestor.ndb`, and with
    # `needs_parent_data = False` it is never populated later either, so
    # touching `.shape()` on a parent dereferences an empty Optional and
    # segfaults. Carrying the flag also costs nothing.
    var scale_is_scalar: Bool


# Scale-gradient reduction
#
# `scale` broadcasts against `x` in the forward, so the product
# `upstream * k` has x's shape and must be reduced back to scale's shape
# before it can be accumulated. Only two forms are supported:
#
#   scale.shape() == ()          -> sum every axis   (per-tensor)
#   scale.shape() == x.shape()   -> identity          (per-element)
#
# Per-channel scales shaped (1,C) against an (B,C) activation are the natural
# next case and are deliberately NOT supported yet: they need a
# right-aligned reshape plus a keepdims sum plus a squeeze, and guessing at
# that reduction silently produces a wrong gradient. It panics instead.


# Backward handler


@fieldwise_init
struct FakeQuantBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        # NOTE: no `raises` here. All 72 other handlers are non-raising, and the
        # BackwardFnType trait declares `def backward(...)` without it, with a
        # `pass` body. A raising override does NOT dispatch to the override --
        # the erased fn pointer calls the trait default, which does nothing.
        # That failure is SILENT: ancestry registers, parents append, backward()
        # runs, and every gradient stays zero. Keep this non-raising; anything
        # that can fail is validated in forward instead.
        ref bwd_arg = output.ancestry().backward_fn().get[FakeQuantBwdArg[Self.dtype]]()
        ref gradbox = output.gradients()  # upstream dL/d(out)

        var x_ancestor = output.ancestry().get(0)  # x
        var scale_ancestor = output.ancestry().get(1)  # scale

        # d/dx = 1 (STE) — a deliberate lie, see the file header
        # Pure hand-off. No arithmetic, no dtype conversion: this is why the
        # op needs no int8 gradbox. The output's own gradbox is already
        # contiguous with zero offset, which is what update_grad's
        # handler-boundary invariant requires.
        x_ancestor.update_grad(gradbox, AddTensor, None)
        parent_ids.append(x_ancestor._id)

        # d/ds = k — the TRUE derivative, unlike the x edge
        # One elementwise multiply against the saved levels, then a reduction
        # back to scale's shape (identity or full sum).
        var upstream = Tensor[Self.dtype](gradbox.buffer())
        var k = Tensor[Self.dtype](bwd_arg.k)
        var d_scale = upstream.__mul__[track_grad=False](k)

        # Reduce `upstream * k` (x's shape) back to scale's shape. The owner
        # must stay alive across the Gradbox construction: binding `.buffer` off
        # a temporary dies at end of statement and leaves a dangling ref.
        var d_scale_owner: Tensor[Self.dtype]
        if bwd_arg.scale_is_scalar:
            # Per-tensor: one scale for the whole tensor, so every axis reduces.
            d_scale_owner = d_scale.sum[track_grad=False]()
        else:
            # Per-element: forward already validated the shapes match.
            d_scale_owner = d_scale
        var d_scale_gradbox = Gradbox[Self.dtype](d_scale_owner.buffer)

        scale_ancestor.update_grad(d_scale_gradbox^, AddTensor, None)
        parent_ids.append(scale_ancestor._id)

        gradbox.zero_grad()


# Forward


@fieldwise_init
struct FakeQuant[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        scale: Tensor[Self.dtype],
        qmin: Scalar[Self.dtype],
        qmax: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """out = scale * clamp(round(self / scale), qmin, qmax).

        `round` and the clip are deliberately NOT grad-carrying (see
        `tenmo/roundfloor.mojo`): the true derivative is 0 a.e. and the STE
        overrides it at the node level here. Wiring ancestry per sub-op would
        hand back zeros for every input gradient.
        """
        if self.is_on_gpu() or scale.is_on_gpu():
            panic(
                "Tensor.fake_quant: CPU only for now — the round/clip GPU",
                " kernels do not exist yet.",
            )

        # Validate the scale shape here, not in backward: a failing backward
        # must stay non-raising (see the note in FakeQuantBackward).
        var scale_is_scalar = scale.shape().rank() == 0
        if not scale_is_scalar and scale.shape() != self.shape():
            panic(
                "Tensor.fake_quant: unsupported scale shape (rank ",
                String(scale.shape().rank()),
                ") against input rank ",
                String(self.shape().rank()),
                ". Supported: a scalar scale (per-tensor) or a scale with the",
                " same shape as the input (per-element). Per-channel (1,C)",
                " scales are not implemented yet.",
            )

        var z = self.__truediv__[track_grad=False](scale)
        var k = Round[Self.dtype].forward(z).clip[track_grad=False](qmin, qmax)
        var out = scale.__mul__[track_grad=False](k)

        comptime if track_grad:
            # The scale counts too. LSQ-style training learns ONLY the scale and
            # keeps activations frozen, so gating on self.requires_grad alone
            # would silently register no ancestry and train nothing.
            var tracked = self.requires_grad or scale.requires_grad
            var grad_required = requires_grad.or_else(tracked)
            if grad_required:
                out.requires_grad_(True)
                # `k` is a fresh non-grad tensor (round and the
                # track_grad=False clip both return leaves), so aliasing its
                # buffer is safe. `copy()` is an O(1) refcount bump.
                var bwd_arg = FakeQuantBwdArg[Self.dtype](
                    k.buffer.copy(), scale_is_scalar
                )
                var backwardFn = BackwardFn(
                    bwd_arg^, FakeQuantBackward[Self.dtype]()
                )
                out.add_ancestry(backwardFn^, self, scale)

        return out^
