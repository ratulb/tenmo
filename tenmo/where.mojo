from .tensor import Tensor
from .shared.mnemonics import AddTensor, Multiply, ReverseSubtract
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType

from .gradbox import Gradbox
from .ancestry import Ancestor
from .ndbuffer import NDBuffer
from .shared.shapes import Shape
from std.sys import has_accelerator
from .kernels.where_kernel import WhereKernel
from .shared.broadcasthelper import ShapeBroadcaster
from .shared.panic import panic


@fieldwise_init
struct WhereArg[dtype: DType](ArgumentType):
    var condition: NDBuffer[Self.dtype]
    var a_requires_grad: Bool
    var b_requires_grad: Bool
    # Parent shapes at forward time: backward reduces each side's
    # output-shaped grad back to these (torch-style un-broadcast).
    # Stored (not read from ancestors) so needs_parent_data stays False.
    var a_shape: Shape
    var b_shape: Shape


@fieldwise_init
struct WhereBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref bwd_arg = (
            output.ancestry().backward_fn().get[WhereArg[Self.dtype]]()
        )
        ref condition = bwd_arg.condition
        var a_requires_grad = bwd_arg.a_requires_grad
        var b_requires_grad = bwd_arg.b_requires_grad

        ref gradbox = output.gradients()
        var grad_ndb = gradbox.buffer()

        var num_parents = len(output.ancestry())
        if num_parents == 0:
            gradbox.zero_grad()
            return

        var ancestor_index = 0

        if a_requires_grad:
            var parent = output.ancestry().get(ancestor_index)
            if parent.requires_grad:
                var grad_a_ndb = grad_ndb.arithmetic_ops[Multiply](condition)
                var grad_a = Gradbox[Self.dtype](grad_a_ndb^)
                # Un-broadcast: grad is at output shape; the parent may be
                # smaller (forward broadcasts a/b/cond). Mirror
                # BroadcastToBackward: sum over broadcast axes, reshape rest.
                if grad_a.shape() != bwd_arg.a_shape:
                    var axes = ShapeBroadcaster.broadcast_mask(
                        bwd_arg.a_shape, grad_a.shape()
                    ).indices_of(1)
                    grad_a = grad_a.sum(axes=axes, keepdims=True)
                    if grad_a.shape() != bwd_arg.a_shape:
                        grad_a = grad_a.reshape(bwd_arg.a_shape)
                parent.update_grad(grad_a^, AddTensor, None)
            parent_ids.append(parent._id)
            ancestor_index += 1

        if b_requires_grad:
            var parent = output.ancestry().get(ancestor_index)
            if parent.requires_grad:
                # 1 - condition without a full-ones alloc: scalar reverse
                # subtract computes it elementwise in one pass.
                var one_minus_cond = condition.scalar_ops[ReverseSubtract](
                    Scalar[Self.dtype](1.0), sync=False
                )
                var grad_b_ndb = grad_ndb.arithmetic_ops[Multiply](
                    one_minus_cond^
                )
                var grad_b = Gradbox[Self.dtype](grad_b_ndb^)
                # Un-broadcast (see a-branch above).
                if grad_b.shape() != bwd_arg.b_shape:
                    var axes_b = ShapeBroadcaster.broadcast_mask(
                        bwd_arg.b_shape, grad_b.shape()
                    ).indices_of(1)
                    grad_b = grad_b.sum(axes=axes_b, keepdims=True)
                    if grad_b.shape() != bwd_arg.b_shape:
                        grad_b = grad_b.reshape(bwd_arg.b_shape)
                parent.update_grad(grad_b^, AddTensor, None)
            parent_ids.append(parent._id)

        gradbox.zero_grad()


def bool_to_float_ndb[
    dtype: DType,
](cond_ndb: NDBuffer[DType.bool],) raises -> NDBuffer[dtype]:
    var shape = cond_ndb.shape
    var numels = cond_ndb.numels()
    var out = NDBuffer[dtype].zeros(shape)
    comptime if has_accelerator():
        if cond_ndb.is_on_gpu():
            var gpu = cond_ndb.device_state.value().get_gpu()
            var cpu_ndb = cond_ndb.to_cpu(sync=True)
            var src = cpu_ndb.data_ptr().unsafe_mut_cast[True]()
            var dst = out.data_ptr().unsafe_mut_cast[True]()
            for i in range(numels):
                dst[unsafe_offset=i] = Scalar[dtype](1.0) if src[
                    unsafe_offset=i
                ] else Scalar[dtype](0.0)
            return out.to_gpu(gpu)
    var src = cond_ndb.data_ptr().unsafe_mut_cast[True]()
    var dst = out.data_ptr().unsafe_mut_cast[True]()
    for i in range(numels):
        dst[unsafe_offset=i] = Scalar[dtype](1.0) if src[
            unsafe_offset=i
        ] else Scalar[dtype](0.0)
    return out^


def expand_to_shape[
    dtype: DType,
](ndb: NDBuffer[dtype], target: Shape,) -> NDBuffer[dtype]:
    if ndb.shape == target:
        return ndb.contiguous() if ndb.is_on_gpu() else ndb.copy()
    var expanded = ndb.broadcast_to(target)
    return expanded.contiguous()


def cpu_where[
    dtype: DType,
](
    cond_ndb: NDBuffer[DType.bool],
    a_ndb: NDBuffer[dtype],
    b_ndb: NDBuffer[dtype],
) -> NDBuffer[dtype]:
    var shape = cond_ndb.shape
    var numels = cond_ndb.numels()
    var out = NDBuffer[dtype].zeros(shape)
    var c = cond_ndb.data_ptr().unsafe_mut_cast[True]()
    var a = a_ndb.data_ptr().unsafe_mut_cast[True]()
    var b = b_ndb.data_ptr().unsafe_mut_cast[True]()
    var o = out.data_ptr().unsafe_mut_cast[True]()
    for i in range(numels):
        o[unsafe_offset=i] = a[unsafe_offset=i] if c[unsafe_offset=i] else b[
            unsafe_offset=i
        ]
    return out^


@fieldwise_init
struct Where[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def _compute_forward[
        track_grad: Bool = True,
    ](
        condition: Tensor[DType.bool],
        a_tensor: Tensor[Self.dtype],
        b_tensor: Tensor[Self.dtype],
        a_requires_grad_flag: Bool,
        b_requires_grad_flag: Bool,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var cond_shape = condition.buffer.shape
        var a_shape = a_tensor.buffer.shape
        var b_shape = b_tensor.buffer.shape
        var ab_shape = ShapeBroadcaster.broadcast_shape(a_shape, b_shape)
        var output_shape = ShapeBroadcaster.broadcast_shape(
            ab_shape, cond_shape
        )

        var cond_ndb = expand_to_shape(condition.buffer, output_shape)
        var a_ndb = expand_to_shape(a_tensor.buffer, output_shape)
        var b_ndb = expand_to_shape(b_tensor.buffer, output_shape)

        var out_ndb: NDBuffer[Self.dtype]
        # Mixed-device guard: a CPU condition with GPU operands would
        # silently read device memory as host in cpu_where below. (The
        # reverse mix is fine — the GPU branch transfers operands.)
        if (
            not cond_ndb.is_on_gpu()
            and (a_ndb.is_on_gpu() or b_ndb.is_on_gpu())
        ):
            panic(
                "Where — mixed devices: CPU condition with GPU operands."
                " Move the condition to GPU (or all inputs to CPU)."
            )
        comptime if has_accelerator():
            if cond_ndb.is_on_gpu():
                # GPU device errors (transfer/compile/launch) surface as raises
                # from the host API; follow the SumMeanReduction protocol and
                # convert them into a panic, keeping where non-raising. The
                # `.Empty()` assignment is unreachable — it only satisfies the
                # compiler's definite-assignment check after the panic.
                try:
                    var gpu = cond_ndb.device_state.value().gpu
                    if not a_ndb.is_on_gpu():
                        a_ndb = a_ndb.to_gpu(gpu)
                    if not b_ndb.is_on_gpu():
                        b_ndb = b_ndb.to_gpu(gpu)
                    var result = WhereKernel[Self.dtype].launch_forward(
                        a_ndb.layout(),
                        a_ndb.device_state.value(),
                        b_ndb.layout(),
                        b_ndb.device_state.value(),
                        cond_ndb.layout(),
                        cond_ndb.device_state.value(),
                        sync=sync,
                    )
                    out_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        result[0], result[1]
                    )
                except e:
                    print(e)
                    panic("Where — GPU operation failed in `where`.")
                    out_ndb = NDBuffer[Self.dtype].Empty()
            else:
                out_ndb = cpu_where(cond_ndb, a_ndb, b_ndb)
        else:
            out_ndb = cpu_where(cond_ndb, a_ndb, b_ndb)

        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = a_requires_grad_flag or b_requires_grad_flag
            if grad_required:
                out.requires_grad_(True)
                var cond_float: NDBuffer[Self.dtype]
                try:
                    cond_float = bool_to_float_ndb[Self.dtype](cond_ndb)
                except e:
                    print(e)
                    panic(
                        "Where — GPU gradient-flag preparation failed in"
                        " `where`."
                    )
                    cond_float = NDBuffer[Self.dtype].Empty()
                var where_arg = WhereArg[Self.dtype](
                    cond_float^,
                    a_requires_grad_flag,
                    b_requires_grad_flag,
                    a_tensor.buffer.shape,
                    b_tensor.buffer.shape,
                )
                var backwardFn = BackwardFn(
                    where_arg^,
                    WhereBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = False

                if a_requires_grad_flag and b_requires_grad_flag:
                    out.add_ancestry(backwardFn^, a_tensor, b_tensor)
                elif a_requires_grad_flag:
                    out.add_ancestry(backwardFn^, a_tensor)
                elif b_requires_grad_flag:
                    out.add_ancestry(backwardFn^, b_tensor)

        return out^

    @staticmethod
    def forward[
        track_grad: Bool = True,
    ](
        condition: Tensor[DType.bool],
        a: Tensor[Self.dtype],
        b: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var a_rg = requires_grad.or_else(a.requires_grad)
        var b_rg = requires_grad.or_else(b.requires_grad)
        return Where[Self.dtype]._compute_forward[track_grad](
            condition, a, b, a_rg, b_rg, sync=sync
        )

    @staticmethod
    def forward[
        track_grad: Bool = True,
    ](
        condition: Tensor[DType.bool],
        a: Scalar[Self.dtype],
        b: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var b_shape = b.buffer.shape
        var a_tensor = Tensor[Self.dtype].full(b_shape, a, requires_grad=False)
        var b_rg = requires_grad.or_else(b.requires_grad)
        return Where[Self.dtype]._compute_forward[track_grad](
            condition, a_tensor, b, False, b_rg, sync=sync
        )

    @staticmethod
    def forward[
        track_grad: Bool = True,
    ](
        condition: Tensor[DType.bool],
        a: Tensor[Self.dtype],
        b: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var a_shape = a.buffer.shape
        var b_tensor = Tensor[Self.dtype].full(a_shape, b, requires_grad=False)
        var a_rg = requires_grad.or_else(a.requires_grad)
        return Where[Self.dtype]._compute_forward[track_grad](
            condition, a, b_tensor, a_rg, False, sync=sync
        )

    @staticmethod
    def forward[
        track_grad: Bool = True,
    ](
        condition: Tensor[DType.bool],
        a: Scalar[Self.dtype],
        b: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        _ = requires_grad
        var cond_shape = condition.buffer.shape
        var a_tensor = Tensor[Self.dtype].full(
            cond_shape, a, requires_grad=False
        )
        var b_tensor = Tensor[Self.dtype].full(
            cond_shape, b, requires_grad=False
        )
        return Where[Self.dtype]._compute_forward[track_grad](
            condition, a_tensor, b_tensor, False, False, sync=sync
        )
