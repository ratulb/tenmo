# product.mojo
#
# Product reduction with full grad flow — CPU and GPU.
# Follows Mean/MeanBackward patterns exactly.
#
# FORWARD
# Product.forward:
#   1. Validate and normalize axes.
#   2. Call Product.product(normalized_axes, keepdims)
#      → dispatches to GPU (Reduction.launch_product) or CPU.
#   3. Wrap result in Tensor.
#   4. If grad required: store BackwardFn(ProductArg) and register ancestry.
#
# BACKWARD
# ProductBackward():
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

from .shared.intarray import IntArray
from .shared.mnemonics import AddTensor, Multiply
from .kernels.reduction_kernel import ReductionKernel
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType

from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .tensor import Tensor
from .ancestry import Ancestor
from .shared.shapes import Shape
from .shared.panic import panic
from .shared.reduction_walk import ReductionWalk


@fieldwise_init
struct ProductArg[dtype: DType](ArgumentType):
    """Backward argument for product reduction.

    Always stored:
        input        — original forward input (needed for recompute path)
        zero_counts  — per-output-element zero count (int32, on same device)
        axes         — reduction axes
        keepdims     — keepdims flag from forward
        reduced_volume — elements per slice (for excl recompute)

    Conditionally stored (store_excl_product comptime flag AND grad need):
        excl_product — input-shaped buffer of per-element exclusive products
                       None if store_excl_product=False (recomputed in backward)
                       or when no backward will run (non-grad forward —
                       forward gates the store on the resolved requires_grad)

    Defined here (consumer side) because it is stored in BackwardFn via
    ArgumentType as a long-lived payload. NDBuffer.buffer pairs are only
    valid for the enclosing call, so the kernel returns raw pairs and this
    module assembles the NDBuffers via with_layout_buffer.
    """

    var input: NDBuffer[Self.dtype]
    var excl_product: Optional[NDBuffer[Self.dtype]]
    var zero_counts: NDBuffer[DType.int32]
    var axes: IntArray
    var keepdims: Bool
    var reduced_volume: Int

    @staticmethod
    def Empty() -> ProductArg[Self.dtype]:
        return ProductArg[Self.dtype](
            NDBuffer[Self.dtype].Empty(),
            None,
            NDBuffer[DType.int32].Empty(),
            IntArray(),
            False,
            0,
        )


from .validators import Validator
from .shared.scalar_ops import ScalarOps
from std.sys.info import has_accelerator
from std.math import log, exp
from max.algorithm import parallelize
from std.sys.info import num_physical_cores


@fieldwise_init
struct Product[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    # Helper: CPU exclusive product (leaf)

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

            var li = 0
            for idx in ndb.index_iterator():
                var val = ndb.buffer[idx].cast[DType.float64]()
                excl.set(
                    li,
                    ScalarOps[Self.dtype].excl_one_cpu(
                        val, total_log, total_neg, total_zero, f64_zero
                    ),
                )
                li += 1
        else:
            var walk = ReductionWalk.build(
                ndb.shape, normalized_axes, ndb.strides,
                keepdims,
            )
            var ndb_offset = ndb.offset
            var ndb_buf = ndb.buffer
            var num_out = out_shape.num_elements()
            var n_threads = num_physical_cores()

            def excl_scan(oi: Int) {imm}:
                var base = walk.base_offset_from_flat(out_shape, oi, ndb_offset)
                var iter = walk.make_odometer(base, 0)
                var total_log = f64_zero
                var total_neg = 0
                var total_zero = 0
                for _ in range(walk.volume):
                    var val = ndb_buf[iter.off].cast[DType.float64]()
                    if val == f64_zero:
                        total_zero += 1
                    else:
                        if val < f64_zero:
                            total_neg += 1
                        total_log += log(abs(val))
                    _ = iter.advance(walk)

                # Second pass: write each element's excl value at its logical
                # (row-major) position in the input-shaped contiguous `excl`.
                var base_flat = walk.base_logical_from_flat(out_shape, oi)
                var witer = walk.make_odometer(base, base_flat)
                for _ in range(walk.volume):
                    var val = ndb_buf[witer.off].cast[DType.float64]()
                    excl.buffer[witer.logical_pos] = ScalarOps[
                        Self.dtype
                    ].excl_one_cpu(
                        val, total_log, total_neg, total_zero, f64_zero
                    )
                    _ = witer.advance(walk)

            # log/abs is expensive — parallel when work exceeds ~threads*1024.
            if (
                num_out >= n_threads
                and num_out * walk.volume >= n_threads * 1024
            ):
                parallelize(excl_scan, num_out, n_threads)
            else:
                for oi in range(num_out):
                    excl_scan(oi)

        return excl^

    # Helper: CPU product reduction (float64 log-space)

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
        # zero_counts is stored but only read on grad paths that keep it;
        # non-grad forwards (store_excl_product=False) skip the alloc+fill.
        # (comptime gate: no runtime branch.)
        var zero_counts: NDBuffer[DType.int32]
        comptime if store_excl_product:
            zero_counts = NDBuffer[DType.int32].zeros(out_shape)
        else:
            zero_counts = NDBuffer[DType.int32].Empty()

        var f64_zero = Scalar[DType.float64](0)
        var reduction_axes_shape = ndb.shape.reduced_shape(
            normalized_axes
        )
        var reduced_volume = reduction_axes_shape.product()
        var num_output_elements = out_shape.num_elements()
        var n_threads = num_physical_cores()

        var walk = ReductionWalk.build(
            ndb.shape, normalized_axes, ndb.strides,
            keepdims,
        )
        var ndb_offset = ndb.offset
        var ndb_buf = ndb.buffer

        def compute_prod(out_flat_idx: Int) {imm}:
            var base = walk.base_offset_from_flat(
                out_shape, out_flat_idx, ndb_offset
            )
            var iter = walk.make_odometer(base, 0)
            var log_abs_sum = f64_zero
            var neg_count = 0
            var zero_count = 0
            for _ in range(walk.volume):
                var val = ndb_buf[iter.off].cast[DType.float64]()
                if val == f64_zero:
                    zero_count += 1
                else:
                    if val < f64_zero:
                        neg_count += 1
                    log_abs_sum += log(abs(val))
                _ = iter.advance(walk)
            comptime if store_excl_product:
                zero_counts.buffer[out_flat_idx] = Scalar[DType.int32](
                    zero_count
                )
            if zero_count > 0:
                out.buffer[out_flat_idx] = Scalar[Self.dtype](0)
            else:
                var sign = Scalar[DType.float64](
                    -1 if neg_count % 2 == 1 else 1
                )
                out.buffer[out_flat_idx] = ScalarOps[Self.dtype].cast_result[
                    Self.dtype
                ](sign * exp(log_abs_sum))

        if out_shape == Shape():
            Product._global_product_cpu[store_excl_product](
                ndb, out, zero_counts
            )
        else:
            # Each output element scans reduced_volume elements through log/abs
            # (expensive): measured crossover ≈ threads * 1024 total work. Also
            # require at least one output per thread.
            if (
                num_output_elements >= n_threads
                and num_output_elements * walk.volume >= n_threads * 1024
            ):
                parallelize(compute_prod, num_output_elements, n_threads)
            else:
                for out_flat_idx in range(num_output_elements):
                    compute_prod(out_flat_idx)

        var excl_optional: Optional[NDBuffer[Self.dtype]] = None
        comptime if store_excl_product:
            excl_optional = Optional(
                Product.excl_product_cpu(ndb, normalized_axes, keepdims)
            )

        var arg = ProductArg[Self.dtype](
            input=ndb,
            excl_product=excl_optional^,
            zero_counts=zero_counts^,
            axes=normalized_axes,
            keepdims=keepdims,
            reduced_volume=reduced_volume,
        )

        return (out^, arg^)

    @staticmethod
    def _global_product_cpu[
        store_zero_counts: Bool = True,
    ](
        ndb: NDBuffer[Self.dtype],
        out_ndb: NDBuffer[Self.dtype],
        zero_counts: NDBuffer[DType.int32],
    ):
        """Reduce all elements to a scalar product (log-space).

        Segmented when contiguous and large (log-sum / neg count / zero count
        are all additive), else serial over the logical index space.
        """
        var f64_zero = Scalar[DType.float64](0)
        var n_threads = num_physical_cores()
        var total_elements = ndb.numels()
        var n_segments = 1 if total_elements < n_threads * 1024 else n_threads
        var seg_log = NDBuffer[DType.float64].zeros(Shape(n_segments))
        var seg_neg = NDBuffer[DType.int32].zeros(Shape(n_segments))
        var seg_zero = NDBuffer[DType.int32].zeros(Shape(n_segments))

        def seg_prod(seg: Int) {imm}:
            var r0 = seg * total_elements // n_segments
            var r1 = (seg + 1) * total_elements // n_segments
            var l_log = f64_zero
            var l_neg = 0
            var l_zero = 0
            for i in range(r0, r1):
                var val = ndb.buffer[ndb.offset + i].cast[
                    DType.float64
                ]()
                if val == f64_zero:
                    l_zero += 1
                else:
                    if val < f64_zero:
                        l_neg += 1
                    l_log += log(abs(val))
            seg_log.buffer[seg] = l_log
            seg_neg.buffer[seg] = Scalar[DType.int32](l_neg)
            seg_zero.buffer[seg] = Scalar[DType.int32](l_zero)

        if ndb.is_contiguous() and total_elements >= n_threads * 1024:
            parallelize(seg_prod, n_segments, n_threads)
            var log_abs_sum = f64_zero
            var neg_count = 0
            var zero_count = 0
            for seg in range(n_segments):
                log_abs_sum += seg_log.buffer[seg]
                neg_count += Int(seg_neg.buffer[seg])
                zero_count += Int(seg_zero.buffer[seg])
            comptime if store_zero_counts:
                zero_counts[IntArray()] = Scalar[DType.int32](zero_count)
            if zero_count > 0:
                out_ndb[IntArray()] = Scalar[Self.dtype](0)
            else:
                var sign = Scalar[DType.float64](
                    -1 if neg_count % 2 == 1 else 1
                )
                out_ndb[IntArray()] = ScalarOps[Self.dtype].cast_result[
                    Self.dtype
                ](sign * exp(log_abs_sum))
        else:
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
            comptime if store_zero_counts:
                zero_counts[IntArray()] = Scalar[DType.int32](zero_count)
            if zero_count > 0:
                out_ndb[IntArray()] = Scalar[Self.dtype](0)
            else:
                var sign = Scalar[DType.float64](
                    -1 if neg_count % 2 == 1 else 1
                )
                out_ndb[IntArray()] = ScalarOps[Self.dtype].cast_result[
                    Self.dtype
                ](sign * exp(log_abs_sum))

    # Helper: exclusive product recompute (for backward)

    @staticmethod
    def compute_excl_product(
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool,
    ) -> NDBuffer[Self.dtype]:
        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var launch_result = ReductionKernel[Self.dtype].launch_product[
                        store_excl_product=True
                    ](
                        ndb.layout(),
                        ndb.device_state.value(),
                        normalized_axes,
                        keepdims,
                    )
                    var excl_optional = launch_result[2]
                    var (excl_layout, excl_storage) = excl_optional.value()
                    return NDBuffer[Self.dtype].with_layout_device_state(
                        excl_layout, excl_storage
                    )
                except e:
                    panic(
                        "Product.compute_excl_product — GPU failed: ",
                        String(e),
                    )
                    return NDBuffer[Self.dtype].Empty()
        return Product.excl_product_cpu(ndb, normalized_axes, keepdims)

    # Main product dispatch (GPU / CPU)

    @always_inline
    @staticmethod
    def product[
        store_excl_product: Bool = True,
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool = False,
        sync: Bool = True,
    ) -> Tuple[NDBuffer[Self.dtype], ProductArg[Self.dtype]]:
        var out: NDBuffer[Self.dtype]
        var arg: ProductArg[Self.dtype]

        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    # Hoisted once: reused for the launch and the shape read
                    # below (layout() rebuilds + deep-copies per call).
                    var ndb_layout = ndb.layout()
                    var (out_pair, zero_pair, excl_optional) = ReductionKernel[
                        Self.dtype
                    ].launch_product[store_excl_product](
                        ndb_layout,
                        ndb.device_state.value(),
                        normalized_axes,
                        keepdims,
                        sync=sync,
                    )
                    var out_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        out_pair[0], out_pair[1]
                    )
                    var zero_ndb = NDBuffer[
                        DType.int32
                    ].with_layout_device_state(zero_pair[0], zero_pair[1])
                    out = out_ndb
                    var reduced_volume = (
                        ndb_layout.shape.reduced_shape(normalized_axes)
                        .product()
                    )
                    var excl_ndb: Optional[NDBuffer[Self.dtype]] = None
                    if excl_optional:
                        var (excl_layout, excl_storage) = excl_optional.value()
                        excl_ndb = Optional(
                            NDBuffer[Self.dtype].with_layout_device_state(
                                excl_layout, excl_storage
                            )
                        )
                    arg = ProductArg[Self.dtype](
                        input=ndb,
                        excl_product=excl_ndb,
                        zero_counts=zero_ndb,
                        axes=normalized_axes,
                        keepdims=keepdims,
                        reduced_volume=reduced_volume,
                    )
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

    # Scalar product of all elements (CPU only)

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

    # Forward entry point

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
            Only for grad-required forwards (forward gates the store on the
            resolved requires_grad) — non-grad forwards skip it entirely.
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

        # Only store the exclusive product when backward will actually run.
        # `grad_required` is runtime (tensor.requires_grad), so the product
        # dispatch must branch at runtime — excl computation is gated on it
        # instead of being paid for every forward (3x data traffic -> 1x).
        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                var result = Product.product[store_excl_product](
                    tensor.buffer, normalized_axes, keepdims, sync=sync
                )
                var ndb = result[0]
                var bwd_arg = result[1]
                var out = Tensor[Self.dtype](ndb^, requires_grad=False)
                out.requires_grad_(True)
                var backwardFn = BackwardFn(
                    bwd_arg^,
                    ProductBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, tensor)
                return out^

            var result = Product.product[False](
                tensor.buffer, normalized_axes, keepdims, sync=sync
            )
            var ndb = result[0]
            return Tensor[Self.dtype](ndb^, requires_grad=False)

        var result = Product.product[False](
            tensor.buffer, normalized_axes, keepdims, sync=sync
        )
        var ndb = result[0]
        return Tensor[Self.dtype](ndb^, requires_grad=False)


@fieldwise_init
struct ProductBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var bwd_arg = (
            output.ancestry().backward_fn().get[ProductArg[Self.dtype]]()
        )
        ref gradbox = output.gradients()
        var gradbox_shape = gradbox.shape()
        var ancestor = output.ancestry().get(0)
        ref ancestor_shape = ancestor.shape()

        # Step 1: expand grad to input shape
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

        # Step 2: get or recompute excl_product
        var excl_ndb: NDBuffer[Self.dtype]

        if bwd_arg.excl_product:
            excl_ndb = bwd_arg.excl_product.value()
        else:
            excl_ndb = Product.compute_excl_product(
                bwd_arg.input, bwd_arg.axes, bwd_arg.keepdims
            )

        # Step 3: grad_input = grad_broadcast * excl_product
        var grad_input = grad_broadcast.buffer().arithmetic_ops[Multiply](
            excl_ndb
        )

        var gradbox_ancestor = Gradbox[Self.dtype](grad_input^)
        ancestor.update_grad(gradbox_ancestor^, AddTensor, None)
        parent_ids.append(ancestor._id)
        gradbox.zero_grad()
