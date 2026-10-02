from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .shared.shapes import Shape
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType

from .validators import Validator
from .shared.intarray import IntArray
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .ancestry import Ancestor
from .kernels.minmax_kernel import MinMaxKernel
from .shared.panic import panic
from .minmax_reducer import MinMaxReducer
from std.sys.info import has_accelerator


@fieldwise_init
struct MinMaxArg[dtype: DType](ArgumentType):
    var axes: IntArray
    var keepdims: Bool
    var mask: NDBuffer[Self.dtype]


@fieldwise_init
struct MinMaxBackward[dtype: DType](BackwardFnType, ImplicitlyCopyable):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var bwd_arg = (
            output.ancestry().backward_fn().get[MinMaxArg[Self.dtype]]()
        )
        var (axes, keepdims, mask) = (
            bwd_arg.axes,
            bwd_arg.keepdims,
            bwd_arg.mask,
        )
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        var shape = ancestor.shape()
        var mask_grad = Gradbox[Self.dtype](mask)

        if shape.rank() == 0:
            # Rank-0 out == x (single element): dx = upstream * mask. The
            # mask is all-ones here, but upstream scaling still applies —
            # without it dx is always 1 (e.g. (2*x.max()).sum() must give
            # dx=2). Mirrors the main path's grad_expanded * mask_grad.
            var grad_contrib = gradbox * mask_grad
            ancestor.update_grad(grad_contrib^, AddTensor, None)
            parent_ids.append(ancestor._id)
            # Mirror the main path: clear the consumed grad unless the
            # caller retains intermediates, else stale grad persists on
            # the output across repeat backward() calls.
            gradbox.zero_grad()
            return

        var grad_expanded: Gradbox[Self.dtype]
        if gradbox.shape() == Shape():
            grad_expanded = Gradbox[Self.dtype].full(
                shape, gradbox.item(), device=mask.device()
            )
        elif not keepdims:
            grad_expanded = gradbox.unsqueeze(axes).broadcast_to(
                shape,
            )
        else:
            grad_expanded = gradbox.broadcast_to(shape)

        var grad_contrib = grad_expanded * mask_grad
        ancestor.update_grad(grad_contrib^, AddTensor, None)
        parent_ids.append(ancestor._id)
        gradbox.zero_grad()


@fieldwise_init
struct MinMax[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        max: Bool, track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var shape = self.shape()
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        var tracking_grad = track_grad and requires_grad.or_else(
            self.requires_grad
        )
        var (result_ndb, mask_ndb) = Self.minmax[is_max=max](
            self.buffer, normalized_axes, keepdims, tracking_grad, sync=sync
        )
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn(
                    MinMaxArg[Self.dtype](normalized_axes, keepdims, mask_ndb^),
                    MinMaxBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^

    @always_inline
    @staticmethod
    def minmax[
        is_max: Bool
    ](
        ndb: NDBuffer[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        paired: Bool = False,
        sync: Bool = True,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        ref shape = ndb.shape
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var (result_pair, mask_pair) = MinMaxKernel[
                        Self.dtype
                    ].launch[is_max=is_max](
                        ndb.layout(),
                        ndb.device_state.value(),
                        normalized_axes,
                        keepdims,
                        sync=sync,
                    )
                    var result_ndb = NDBuffer[
                        Self.dtype
                    ].with_layout_device_state(result_pair[0], result_pair[1])
                    var mask_ndb = NDBuffer[
                        Self.dtype
                    ].with_layout_device_state(mask_pair[0], mask_pair[1])
                    return result_ndb, mask_ndb
                except e:
                    print(e)
                    panic("MinmaxNdBuffer minmax: gpu path failed")
                    return (
                        NDBuffer[Self.dtype].Empty(),
                        NDBuffer[Self.dtype].Empty(),
                    )
            else:
                return Self.minmax_cpu[is_max](
                    ndb, normalized_axes, keepdims, paired
                )
        else:
            return Self.minmax_cpu[is_max](
                ndb, normalized_axes, keepdims, paired
            )

    @staticmethod
    def minmax_cpu[
        is_max: Bool
    ](
        ndb: NDBuffer[Self.dtype],
        normalized_axes: IntArray,
        keepdims: Bool = False,
        paired: Bool = False,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        var result_ndb = MinMaxReducer[Self.dtype].reduce_minmax[is_max](
            ndb, normalized_axes, keepdims
        )
        if paired:
            var mask_ndb = MinMaxReducer[Self.dtype].build_minmax_mask[is_max](
                ndb, result_ndb, normalized_axes, keepdims
            )
            return result_ndb, mask_ndb
        else:
            return result_ndb, NDBuffer[Self.dtype].Empty()
