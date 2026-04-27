# =============================================================================
# product.mojo
# =============================================================================
#
# Product reduction with full grad flow — CPU and GPU.
# Follows Mean/MeanBackward patterns exactly.
#
# FORWARD
# ───────
# Product.forward:
#   1. Validate and normalize axes.
#   2. Call tensor.buffer.product(normalized_axes, keepdims)
#      → NDBuffer dispatches to GPU (Reduction.launch_product) or CPU.
#   3. Wrap result in Tensor.
#   4. If grad required: store BackwardFnArg(BACKWARD_PRODUCT, ProductArg)
#      and register ancestry.
#
# BACKWARD
# ────────
# ProductBackward.backward:
#   grad_input[i] = grad_out * excl_product[i]
#
#   excl_product[i] = product of all elements in i's reduction slice
#                     except element i itself.
#
#   Obtained from ProductArg:
#     store_excl_product=True  → ProductArg.excl_product is Some(ndb) — use directly
#     store_excl_product=False → ProductArg.excl_product is None — recompute
#
#   Zero handling (via zero_counts, always stored):
#     zero_count == 0  → grad_input[i] = grad_out * excl_product[i]
#     zero_count == 1  → same formula — excl_product handles correctly:
#                        excl[zero_pos]  = product of non-zero others (non-zero grad)
#                        excl[non_zero]  = 0 (contains the zero → grad = 0)
#     zero_count >= 2  → grad_input[i] = 0 (excl_product also gives 0 here)
#   No special-casing needed — zero semantics fall out of excl_product naturally.
#
# NDBuffer changes
# ────────────────
#   reduce[mean: Bool]     →  reduce[op_code: Int]   (SUM / MEAN only)
#   NEW: product(axes, keepdims)  dispatches to GPU or CPU product path
#   NEW: product_cpu(axes, keepdims) — CPU reference implementation
#
# =============================================================================

#from .mnemonics import AddTensor, PRODUCT, BACKWARD_PRODUCT
from .mnemonics import AddTensor
from .reduction_kernel import ProductArg
from .backpropagation import BackwardFnArg, BACKWARD_PRODUCT
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .tensor import Tensor
from .ancestry import Ancestor
from .shapes import Shape
from .common_utils import panic
from .validators import Validator


struct ProductBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var bwd_arg = output.ancestry().backward_fn_arg().get[ProductArg[Self.dtype]]()
        ref gradbox = output.gradients()[]
        ref gradbox_shape = gradbox.shape()
        var ancestor = output.ancestry().get(0)
        ref ancestor_shape = ancestor.shape()

        # ── Step 1: expand grad to input shape (same pattern as MeanBackward) ─
        var expanded = gradbox.copy()

        if gradbox_shape == Shape():
            # Full reduction to scalar — broadcast scalar grad to input shape
            var scalar_grad = gradbox.item()
            expanded = Gradbox[Self.dtype].full(
                ancestor_shape,
                scalar_grad,
                share=False,
                device=gradbox.device(),
            )
        elif not bwd_arg.keepdims:
            # Re-insert the reduced axes as size-1 dims before broadcasting
            expanded = expanded.reshape(
                Shape(
                    gradbox_shape.intarray().insert(
                        bwd_arg.axes,
                        IntArray.filled(len(bwd_arg.axes), 1),
                    )
                )
            )

        # expanded is now broadcastable to ancestor_shape
        var grad_broadcast = expanded.broadcast_to(ancestor_shape)

        # ── Step 2: get or recompute excl_product ─────────────────────────────
        var excl_ndb: NDBuffer[Self.dtype]

        if bwd_arg.excl_product:
            # Stored during forward — use directly (GPU or CPU NDBuffer)
            excl_ndb = bwd_arg.excl_product.value()
        else:
            # Recompute from stored input
            # NDBuffer.compute_excl_product dispatches GPU or CPU
            excl_ndb = bwd_arg.input.compute_excl_product(
                bwd_arg.axes, bwd_arg.keepdims
            )

        # ── Step 3: grad_input = grad_broadcast * excl_product ───────────────
        # NDBuffer multiply — device-aware, stays on GPU if on GPU.
        # Zero handling falls out naturally from excl_product values:
        #   zero_count >= 2 → excl_product = 0 everywhere in slice
        #   zero_count == 1 → excl_product = 0 for non-zero elements,
        #                     product-of-others for the zero element
        #   zero_count == 0 → standard product-of-others
        # No explicit zero_counts check needed here.
        var grad_input = grad_broadcast.buffer * excl_ndb

        var gradbox_ancestor = Gradbox[Self.dtype](grad_input^, share=False)
        return [
            (
                ancestor^,
                gradbox_ancestor^,
                AddTensor,
            )
        ]


# =============================================================================
# SECTION 3 — Product forward
# =============================================================================
#
# Mirrors Mean.forward exactly.
# store_excl_product comptime flag flows from forward into ProductArg —
# backward reads the same flag's effect via excl_product being Some or None.
# =============================================================================

@fieldwise_init
struct Product[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True,
        store_excl_product: Bool = True,
    ](
        tensor: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Compute product reduction along axes.

        All dtypes supported. Accumulates in float64 log-space for
        overflow safety — no silent wraparound for any dtype.

        Precision note: int64/uint64 values beyond 2^53 are approximate
        in the float64 accumulator. All other types are exact.

        store_excl_product=True  (default):
            excl_product computed during forward and stored in ProductArg.
            Backward is fast — no second kernel launch.
            Memory cost: one input-shaped buffer of dtype.

        store_excl_product=False:
            excl_product recomputed during backward.
            Less memory, slower backward.

        Args:
            tensor:             Input tensor.
            axes:               Reduction axes (unnormalised).
            keepdims:           Keep reduced dimensions.
            requires_grad:      Override grad tracking.
            store_excl_product: Comptime flag — store or recompute excl_product.

        Returns:
            Output tensor with product applied.
        """
        var normalized_axes = Validator.validate_and_normalize_axes(
            tensor.shape(), axes
        )

        # NDBuffer.product dispatches to GPU or CPU (see Section 4)
        var result = tensor.buffer.product[store_excl_product](
            normalized_axes, keepdims
        )
        var ndb     = result[0]   # NDBuffer — product output
        var bwd_arg = result[1]   # ProductArg — for backward

        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_PRODUCT, bwd_arg^
                )
                out.add_ancestry(backwardFnArg^, tensor)

        return out^

