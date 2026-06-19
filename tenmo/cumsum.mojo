from .tensor import Tensor
from .mnemonics import AddTensor
from .backpropagation import BackwardFnArg, BACKWARD_CUMSUM, Integer
from .gradbox import Gradbox
from .ancestry import Ancestor
from .ndbuffer import NDBuffer
from .common_utils import panic
from std.sys import has_accelerator
from .kernels.cumsum_kernel import CumsumGpuKernel


@fieldwise_init
struct CumsumBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ) raises:
        var axis = output.ancestry().backward_fn_arg().get[Integer]().value
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)

        var grad_ndb: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if gradbox.is_on_gpu():
                var shape = gradbox.shape()
                var rank = shape.rank()
                var outer: Int = 1
                for i in range(axis):
                    outer *= shape[i]
                var axis_size = shape[axis]
                var inner: Int = 1
                for i in range(axis + 1, rank):
                    inner *= shape[i]
                grad_ndb = CumsumGpuKernel[Self.dtype].launch_backward(
                    gradbox.buffer(), axis, outer, axis_size, inner
                )
            else:
                grad_ndb = cumsum_backward_cpu[Self.dtype](
                    gradbox.buffer(), axis
                )
        else:
            grad_ndb = cumsum_backward_cpu[Self.dtype](
                gradbox.buffer(), axis
            )
        var gradbox_ancestor = Gradbox[Self.dtype](grad_ndb^)

        if parent.requires_grad:
            parent.update_grad(gradbox_ancestor^, AddTensor, None)
        parent_ids.append(parent._id)

        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct Cumsum[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True,
    ](
        self: Tensor[Self.dtype],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var shape = self.shape()
        var rank = shape.rank()
        var norm_axis = axis
        if norm_axis < 0:
            norm_axis = rank + norm_axis
        if norm_axis < 0 or norm_axis >= rank:
            panic(
                "cumsum: invalid axis "
                + String(axis)
                + " for tensor of rank "
                + String(rank)
            )

        var ndb: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if self.buffer.is_on_gpu():
                var outer: Int = 1
                for i in range(norm_axis):
                    outer *= shape[i]
                var axis_size = shape[norm_axis]
                var inner: Int = 1
                for i in range(norm_axis + 1, rank):
                    inner *= shape[i]
                try:
                    ndb = CumsumGpuKernel[Self.dtype].launch(
                        self.buffer, norm_axis, outer, axis_size, inner, sync
                    )
                except e:
                    panic(
                        "cumsum GPU forward failed: " + String(e)
                    )
                    ndb = NDBuffer[Self.dtype].Empty()
            else:
                ndb = cumsum_cpu[Self.dtype](self.buffer, norm_axis)
        else:
            ndb = cumsum_cpu[Self.dtype](self.buffer, norm_axis)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].integer_arg(
                    BACKWARD_CUMSUM, norm_axis
                )
                backwardFnArg.needs_parent_data = False
                out.add_ancestry(backwardFnArg^, self)

        return out^


def cumsum_cpu[
    dtype: DType,
](
    inp: NDBuffer[dtype],
    axis: Int,
) -> NDBuffer[dtype]:
    var shape = inp.shape
    var rank = shape.rank()
    var axis_size = shape[axis]
    var out = NDBuffer[dtype].zeros(shape)

    if axis_size <= 1:
        if axis_size == 1:
            out.copy_from_alike(inp)
        return out^

    var inner: Int = 1
    for i in range(axis + 1, rank):
        inner *= shape[i]

    var outer: Int = 1
    for i in range(axis):
        outer *= shape[i]

    if inp.is_contiguous():
        var in_ptr = inp.data_ptr()
        var in_offset = inp.offset
        var out_ptr = out.data_ptr()
        var out_offset = out.offset

        for o in range(outer):
            for k in range(axis_size):
                for i in range(inner):
                    var idx = (o * axis_size + k) * inner + i
                    if k == 0:
                        out_ptr[out_offset + idx] = in_ptr[in_offset + idx]
                    else:
                        out_ptr[out_offset + idx] = (
                            out_ptr[out_offset + idx - inner]
                            + in_ptr[in_offset + idx]
                        )
    else:
        var flat_idx = 0
        for buf_idx in inp.index_iterator():
            var coord_k = (flat_idx // inner) % axis_size
            if coord_k == 0:
                out.data_ptr()[out.offset + flat_idx] = inp.buffer[buf_idx]
            else:
                out.data_ptr()[out.offset + flat_idx] = (
                    out.data_ptr()[out.offset + flat_idx - inner]
                    + inp.buffer[buf_idx]
                )
            flat_idx += 1

    return out^


def cumsum_backward_cpu[
    dtype: DType,
](
    grad: NDBuffer[dtype],
    axis: Int,
) -> NDBuffer[dtype]:
    var shape = grad.shape
    var rank = shape.rank()
    var axis_size = shape[axis]
    var out = NDBuffer[dtype].zeros(shape)

    if axis_size <= 1:
        if axis_size == 1:
            out.copy_from_alike(grad)
        return out^

    var inner: Int = 1
    for i in range(axis + 1, rank):
        inner *= shape[i]

    var outer: Int = 1
    for i in range(axis):
        outer *= shape[i]

    if grad.is_contiguous():
        var in_ptr = grad.data_ptr()
        var in_offset = grad.offset
        var out_ptr = out.data_ptr()
        var out_offset = out.offset

        for o in range(outer):
            for k in range(axis_size - 1, -1, -1):
                for i in range(inner):
                    var idx = (o * axis_size + k) * inner + i
                    if k == axis_size - 1:
                        out_ptr[out_offset + idx] = in_ptr[in_offset + idx]
                    else:
                        out_ptr[out_offset + idx] = (
                            in_ptr[in_offset + idx]
                            + out_ptr[out_offset + idx + inner]
                        )
    else:
        var flat_idx = 0
        for buf_idx in grad.index_iterator():
            var coord_k = (flat_idx // inner) % axis_size
            if coord_k == axis_size - 1:
                out.data_ptr()[out.offset + flat_idx] = grad.buffer[buf_idx]
            else:
                out.data_ptr()[out.offset + flat_idx] = (
                    grad.buffer[buf_idx]
                    + out.data_ptr()[out.offset + flat_idx + inner]
                )
            flat_idx += 1

    return out^
