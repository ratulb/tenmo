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
#   2. Call Product.product(normalized_axes, keepdims)
#      → dispatches to GPU (Reduction.launch_product) or CPU.
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
#     via Product.compute_excl_product.
#
#   Zero handling (via zero_counts, always stored):
#     zero_count == 0  → grad_input[i] = grad_out * excl_product[i]
#     zero_count == 1  → same formula — excl_product handles correctly:
#                        excl[zero_pos]  = product of non-zero others (non-zero grad)
#                        excl[non_zero]  = 0 (contains the zero → grad = 0)
#     zero_count >= 2  → grad_input[i] = 0 (excl_product also gives 0 here)
#   No special-casing needed — zero semantics fall out of excl_product naturally.
#
# =============================================================================

from .intarray import IntArray
from .mnemonics import AddTensor
from tenmo.kernels.reduction_kernel import ProductArg, Reduction
from .backpropagation import BackwardFnArg, BACKWARD_PRODUCT
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .tensor import Tensor
from .ancestry import Ancestor
from .shapes import Shape
from .common_utils import panic
from .validators import Validator
from .shared.scalar_ops import ScalarOps
from std.sys.info import has_accelerator
from std.math import log, exp


@fieldwise_init
struct Product[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    # ── Helper: CPU exclusive product (leaf) ───────────────────────────────

    @staticmethod
    def excl_product_cpu(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> NDBuffer[Self.dtype]:
        var excl = NDBuffer[Self.dtype].zeros(ndb.shape)
        var f64_zero = Scalar[DType.float64](0)
        var out_shape = ndb.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var reduction_axes_shape = ndb.shape.reduced_shape(normalized_axes)

        if out_shape == Shape():
            var total_log = f64_zero
            var total_neg = 0
            var total_zero = 0
            for idx in ndb.index_iterator():
                var val = ndb.buffer[idx].cast[DType.float64]()
                if val == f64_zero:
                    total_zero += 1
                else:
                    if val < f64_zero:
                        total_neg += 1
                    total_log += log(abs(val))

            for idx in ndb.index_iterator():
                var val = ndb.buffer[idx].cast[DType.float64]()
                excl.set(
                    idx,
                    ScalarOps[Self.dtype].excl_one_cpu(
                        val, total_log, total_neg, total_zero, f64_zero
                    ),
                )
        else:
            for out_coord in out_shape:
                var total_log = f64_zero
                var total_neg = 0
                var total_zero = 0
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    var val = ndb[self_coord].cast[DType.float64]()
                    if val == f64_zero:
                        total_zero += 1
                    else:
                        if val < f64_zero:
                            total_neg += 1
                        total_log += log(abs(val))

                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    var val = ndb[self_coord].cast[DType.float64]()
                    excl[self_coord] = ScalarOps[Self.dtype].excl_one_cpu(
                        val, total_log, total_neg, total_zero, f64_zero
                    )

        return excl^

    # ── Helper: CPU product reduction (float64 log-space) ──────────────────

    @staticmethod
    def product_cpu[
        store_excl_product: Bool = True,
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> Tuple[NDBuffer[Self.dtype], ProductArg[Self.dtype]]:
        var out_shape = ndb.shape.compute_output_shape(
            normalized_axes, keepdims, validated=True
        )
        var out = NDBuffer[Self.dtype].zeros(out_shape)
        var zero_counts = NDBuffer[DType.int32].zeros(out_shape)

        var f64_zero = Scalar[DType.float64](0)

        if out_shape == Shape():
            var log_abs_sum = f64_zero
            var neg_count = 0
            var zero_count = 0
            for idx in ndb.index_iterator():
                var val = ndb.buffer[idx].cast[DType.float64]()
                if val == f64_zero:
                    zero_count += 1
                else:
                    if val < f64_zero:
                        neg_count += 1
                    log_abs_sum += log(abs(val))
            zero_counts[IntArray()] = Scalar[DType.int32](zero_count)
            if zero_count > 0:
                out[IntArray()] = Scalar[Self.dtype](0)
            else:
                var sign = Scalar[DType.float64](
                    -1 if neg_count % 2 == 1 else 1
                )
                out[IntArray()] = ScalarOps[Self.dtype].cast_result[Self.dtype](
                    sign * exp(log_abs_sum)
                )
        else:
            var reduction_axes_shape = ndb.shape.reduced_shape(normalized_axes)
            for out_coord in out_shape:
                var log_abs_sum = f64_zero
                var neg_count = 0
                var zero_count = 0
                for red_coord in reduction_axes_shape:
                    var self_coord = out_coord.replace(
                        normalized_axes, red_coord
                    ) if keepdims else out_coord.insert(
                        normalized_axes, red_coord
                    )
                    var val = ndb[self_coord].cast[DType.float64]()
                    if val == f64_zero:
                        zero_count += 1
                    else:
                        if val < f64_zero:
                            neg_count += 1
                        log_abs_sum += log(abs(val))
                zero_counts[out_coord] = Scalar[DType.int32](zero_count)
                if zero_count > 0:
                    out[out_coord] = Scalar[Self.dtype](0)
                else:
                    var sign = Scalar[DType.float64](
                        -1 if neg_count % 2 == 1 else 1
                    )
                    out[out_coord] = ScalarOps[Self.dtype].cast_result[
                        Self.dtype
                    ](sign * exp(log_abs_sum))

        var excl_optional: Optional[NDBuffer[Self.dtype]] = None
        comptime if store_excl_product:
            excl_optional = Optional(
                Product.excl_product_cpu(ndb, normalized_axes, keepdims)
            )

        var reduced_volume = ndb.shape.reduced_shape(normalized_axes).product()

        var arg = ProductArg[Self.dtype](
            input=ndb,
            excl_product=excl_optional^,
            zero_counts=zero_counts^,
            axes=normalized_axes,
            keepdims=keepdims,
            reduced_volume=reduced_volume,
        )

        return (out^, arg^)

    # ── Helper: exclusive product recompute (for backward) ─────────────────

    @staticmethod
    def compute_excl_product(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> NDBuffer[Self.dtype]:
        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var (_, productArg) = Reduction[Self.dtype].launch_product[
                        store_excl_product=True
                    ](ndb, normalized_axes, keepdims)
                    return productArg.excl_product.value()
                except e:
                    panic(
                        "Product.compute_excl_product — GPU failed: ",
                        String(e),
                    )
                    return NDBuffer[Self.dtype].Empty()
        return Product.excl_product_cpu(ndb, normalized_axes, keepdims)

    # ── Main product dispatch (GPU / CPU) ──────────────────────────────────

    @always_inline
    @staticmethod
    def product[
        store_excl_product: Bool = True,
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool = False,
    ) -> Tuple[NDBuffer[Self.dtype], ProductArg[Self.dtype]]:
        var out: NDBuffer[Self.dtype]
        var arg: ProductArg[Self.dtype]

        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var result = Reduction[Self.dtype].launch_product[
                        store_excl_product
                    ](ndb, normalized_axes, keepdims)
                    out = result[0]
                    arg = result[1]
                except e:
                    print(e)
                    panic("Product.product — GPU operation failed: ", String(e))
                    out = NDBuffer[Self.dtype].Empty()
                    arg = ProductArg[Self.dtype].Empty()
            else:
                (out, arg) = Product.product_cpu[store_excl_product](
                    ndb, normalized_axes, keepdims
                )
        else:
            (out, arg) = Product.product_cpu[store_excl_product](
                ndb, normalized_axes, keepdims
            )

        return (out^, arg^)

    # ── Scalar product of all elements (CPU only) ──────────────────────────

    @always_inline
    @staticmethod
    def product_all(
        ndb: NDBuffer[Self.dtype],
    ) -> Scalar[Self.dtype]:
        if ndb.is_contiguous():
            var start = ndb.offset
            var end = start + ndb.numels()
            return ndb.buffer.product(start, end)
        else:
            var product: Scalar[Self.dtype] = Scalar[Self.dtype](1)
            for index in ndb.index_iterator():
                product *= ndb.buffer[index]
            return product

    # ── Forward entry point ────────────────────────────────────────────────

    @always_inline
    @staticmethod
    def forward[
        track_grad: Bool = True,
        store_excl_product: Bool = True,
    ](
        tensor: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
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
            sync:               Whether to synchronize the GPU operation.

        Returns:
            Output tensor with product applied.
        """
        var normalized_axes = Validator.validate_and_normalize_axes(
            tensor.shape(), axes
        )

        var result = Product.product[store_excl_product](
            tensor.buffer, normalized_axes, keepdims
        )
        var ndb = result[0]
        var bwd_arg = result[1]

        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_PRODUCT, bwd_arg^
                )
                backwardFnArg.needs_parent_data = True
                out.add_ancestry(backwardFnArg^, tensor)

        return out^


struct ProductBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[ProductArg[Self.dtype]]()
        )
        ref gradbox = output.gradients()
        var gradbox_shape = gradbox.shape()
        var ancestor = output.ancestry().get(0)
        ref ancestor_shape = ancestor.shape()

        # ── Step 1: expand grad to input shape ─────────────────────────────
        var expanded = gradbox.copy()

        if gradbox_shape == Shape():
            var scalar_grad = gradbox.item()
            expanded = Gradbox[Self.dtype].full(
                ancestor_shape,
                scalar_grad,
                
                device=gradbox.device(),
            )
        elif not bwd_arg.keepdims:
            expanded = expanded.reshape(
                Shape(
                    gradbox_shape.intarray().insert(
                        bwd_arg.axes,
                        IntArray.filled(len(bwd_arg.axes), 1),
                    )
                )
            )

        var grad_broadcast = expanded.broadcast_to(ancestor_shape)

        # ── Step 2: get or recompute excl_product ──────────────────────────
        var excl_ndb: NDBuffer[Self.dtype]

        if bwd_arg.excl_product:
            excl_ndb = bwd_arg.excl_product.value()
        else:
            excl_ndb = Product.compute_excl_product(
                bwd_arg.input, bwd_arg.axes, bwd_arg.keepdims
            )

        # ── Step 3: grad_input = grad_broadcast * excl_product ─────────────
        var grad_input = grad_broadcast.buffer().arithmetic_ops[Multiply](excl_ndb)

        var gradbox_ancestor = Gradbox[Self.dtype](grad_input^)
        ancestor.update_grad(gradbox_ancestor^, AddTensor, None)
        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()
