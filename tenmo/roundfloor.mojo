"""Round / floor — non-differentiable elementwise unary ops (CPU).

Both quantize a float to an integer *level* while staying in floating
point. `round` goes to the nearest level, ties to even; `floor` truncates
toward -inf.

Neither carries a gradient. The true derivative is 0 almost everywhere and
undefined at the integers, so a grad-tracking version would silently zero
every gradient passing through it -- the reason a quantizer must be a
single node that fabricates its own gradient rather than a chain of
differentiated ops (see `tenmo/fakequant.mojo`). Both therefore return a
bare leaf and register no ancestry, following `argmax`/`argmin`
(`tensor.mojo`) rather than the grad-carrying unary ops.

CPU only: GPU kernels do not exist yet, so a device tensor is rejected
loudly instead of silently producing host-side garbage from a device
pointer.

Semantics verified on this toolchain (Mojo 1.1.0):

- `round` is **half-to-even** (banker's rounding), not C's
  half-away-from-zero: `round(2.5)=2`, `round(3.5)=4`, `round(-2.5)=-2`,
  `round(-3.5)=-4`. So `floor(x + 0.5)` is NOT a substitute -- it rounds
  half up and disagrees on every tie.
- `round(-0.5) = -0.0`, so signed zero can reach the output.
- `round`/`floor` are **free functions** on both `Scalar` and `SIMD`
  (elementwise). There is no method form -- `v.round()` is a compile error.

CPU only. The SIMD loop shape mirrors `clip.mojo`: contiguous fast path plus
an `index_iterator` fallback for strided views.
"""

from .tensor import Tensor
from .shared.panic import panic
from std.sys import simd_width_of
from std.math import round, floor, Roundable, Floorable


@fieldwise_init
struct Round[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    """Round to nearest integer level, ties to even. Non-differentiable."""

    @staticmethod
    def forward(
        self: Tensor[Self.dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Rounds x. Always returns a leaf -- no ancestry is registered."""
        return _unary_round_floor[is_round=True](self, "Tensor.round")


@fieldwise_init
struct Floor[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    """Truncate toward -inf. Non-differentiable."""

    @staticmethod
    def forward(
        self: Tensor[Self.dtype],
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Floors x. Always returns a leaf -- no ancestry is registered."""
        return _unary_round_floor[is_round=False](self, "Tensor.floor")


@always_inline
def _apply[
    T: Roundable & Floorable,
    is_round: Bool,
](x: T) -> T:
    """Shared implementation.
    `is_round` is a comptime flag, NOT a function parameter. Passing `round` /
    `floor` as a `def(...)` value fails: they are an overload set, and selecting
    one by the parameter's signature does not resolve on this pin
    ("missing required argument: 'move'" at the call site). Instead `_apply`
    calls them directly, where the argument's concrete type drives normal
    overload resolution. Calling round/floor directly works on both Scalar and
    SIMD; see the note above.
    """
    comptime if is_round:
        return round(x)
    else:
        return floor(x)


@always_inline
def _unary_round_floor[
    dtype: DType,
    is_round: Bool,
](
    self: Tensor[dtype],
    label: String,
) -> Tensor[dtype] where dtype.is_floating_point():
    if self.is_on_gpu():
        panic(
            label,
            " has no GPU kernel yet (CPU only). Move the tensor to CPU first,",
            ". No GPU kernel exists yet for this op.",
        )

    var shape = self.shape()
    var out = Tensor[dtype].zeros(shape, requires_grad=False)

    if self.is_contiguous():
        var src = self.data_ptr()
        var dest = out.data_ptr()
        var offset = self.offset()
        var numels = self.numels()

        comptime simd_width = simd_width_of[dtype]()

        for i in range(0, numels - simd_width + 1, simd_width):
            var x = src.unsafe_load[width=simd_width](offset + i)
            dest.unsafe_store[width=simd_width](i, _apply[is_round=is_round](x))

        # Remainder: numels % simd_width lanes, scalar path.
        for i in range(numels - numels % simd_width, numels):
            var x = src[unsafe_offset=offset + i]
            dest[unsafe_offset=i] = _apply[is_round=is_round](x)
    else:
        # Strided view: IndexIterator yields storage offsets directly, so
        # there is no per-element IntArray alloc and no flatten_index. Same
        # approach as clip.mojo's non-contiguous path.
        var src_buffer = self.buffer
        var dest = out.data_ptr()
        var idx = 0
        for p_off in src_buffer.index_iterator():
            dest[unsafe_offset=idx] = _apply[is_round=is_round](
                src_buffer.storage_get[checked=False](p_off)
            )
            idx += 1

    # No ancestry, deliberately. `track_grad` is intentionally absent from
    # the public signature: a parameter that silently does nothing is worse
    # than a compile error when a caller expects a graph edge.
    return out^
