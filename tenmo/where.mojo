from .tensor import Tensor
from .mnemonics import AddTensor, Multiply, Subtract
from .backpropagation import BackwardFnArg, BACKWARD_WHERE, WhereArg
from .gradbox import Gradbox
from .ancestry import Ancestor
from .ndbuffer import NDBuffer
from .common_utils import panic
from std.sys import has_accelerator
from .kernels.where_kernel import WhereGpuKernel
from .broadcasthelper import ShapeBroadcaster


@fieldwise_init
struct WhereBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref bwd_arg = output.ancestry().backward_fn_arg().get[WhereArg[Self.dtype]]()
        ref condition = bwd_arg.condition
        var a_requires_grad = bwd_arg.a_requires_grad
        var b_requires_grad = bwd_arg.b_requires_grad

        ref gradbox = output.gradients()
        var grad_ndb = gradbox.buffer()

        var num_parents = len(output.ancestry())
        if num_parents == 0:
            if not retain_graph:
                gradbox.zero_grad()
            return

        var ancestor_index = 0

        if a_requires_grad:
            var parent = output.ancestry().get(ancestor_index)
            if parent.requires_grad:
                var grad_a_ndb = grad_ndb.arithmetic_ops[Multiply](condition)
                var grad_a = Gradbox[Self.dtype](grad_a_ndb^)
                parent.update_grad(grad_a^, AddTensor, None)
            parent_ids.append(parent._id)
            ancestor_index += 1

        if b_requires_grad:
            var parent = output.ancestry().get(ancestor_index)
            if parent.requires_grad:
                var cond_shape = condition.shape
                var ones = NDBuffer[Self.dtype].full(
                    cond_shape, Scalar[Self.dtype](1.0)
                )
                var one_minus_cond = ones.arithmetic_ops[Subtract](condition)
                var grad_b_ndb = grad_ndb.arithmetic_ops[Multiply](one_minus_cond^)
                var grad_b = Gradbox[Self.dtype](grad_b_ndb^)
                parent.update_grad(grad_b^, AddTensor, None)
            parent_ids.append(parent._id)

        if not retain_graph:
            gradbox.zero_grad()


def bool_to_float_ndb[
    dtype: DType,
](
    cond_ndb: NDBuffer[DType.bool],
) -> NDBuffer[dtype]:
    var shape = cond_ndb.shape
    var numels = cond_ndb.numels()
    var out = NDBuffer[dtype].zeros(shape)
    var src = cond_ndb.data_ptr().unsafe_mut_cast[True]()
    var dst = out.data_ptr().unsafe_mut_cast[True]()
    for i in range(numels):
        dst[i] = Scalar[dtype](1.0) if src[i] else Scalar[dtype](0.0)
    return out^


def expand_to_shape[
    dtype: DType,
](
    ndb: NDBuffer[dtype],
    target: Shape,
) raises -> NDBuffer[dtype]:
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
        o[i] = a[i] if c[i] else b[i]
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
    ) raises -> Tensor[Self.dtype]:
        var cond_shape = condition.buffer.shape
        var a_shape = a_tensor.buffer.shape
        var b_shape = b_tensor.buffer.shape
        var ab_shape = ShapeBroadcaster.broadcast_shape(a_shape, b_shape)
        var output_shape = ShapeBroadcaster.broadcast_shape(ab_shape, cond_shape)

        var cond_ndb = expand_to_shape(condition.buffer, output_shape)
        var a_ndb = expand_to_shape(a_tensor.buffer, output_shape)
        var b_ndb = expand_to_shape(b_tensor.buffer, output_shape)

        var out_ndb: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if cond_ndb.is_on_gpu():
                out_ndb = WhereGpuKernel[Self.dtype].launch_forward(
                    a_ndb, b_ndb, cond_ndb, False, False,
                    Scalar[Self.dtype](0), Scalar[Self.dtype](0),
                    sync=sync,
                )
            else:
                out_ndb = cpu_where(cond_ndb, a_ndb, b_ndb)
        else:
            out_ndb = cpu_where(cond_ndb, a_ndb, b_ndb)

        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = a_requires_grad_flag or b_requires_grad_flag
            if grad_required:
                out.requires_grad_(True)
                var cond_float = bool_to_float_ndb[Self.dtype](cond_ndb)
                var where_arg = WhereArg[Self.dtype](
                    cond_float^, a_requires_grad_flag, b_requires_grad_flag
                )
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_WHERE, where_arg^
                )
                backwardFnArg.needs_parent_data = False

                if a_requires_grad_flag and b_requires_grad_flag:
                    out.add_ancestry(backwardFnArg^, a_tensor, b_tensor)
                elif a_requires_grad_flag:
                    out.add_ancestry(backwardFnArg^, a_tensor)
                elif b_requires_grad_flag:
                    out.add_ancestry(backwardFnArg^, b_tensor)

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
    ) raises -> Tensor[Self.dtype]:
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
    ) raises -> Tensor[Self.dtype]:
        var b_shape = b.buffer.shape
        var a_tensor = Tensor[Self.dtype].full(
            b_shape, a, requires_grad=False
        )
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
    ) raises -> Tensor[Self.dtype]:
        var a_shape = a.buffer.shape
        var b_tensor = Tensor[Self.dtype].full(
            a_shape, b, requires_grad=False
        )
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
    ) raises -> Tensor[Self.dtype]:
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
