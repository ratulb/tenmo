from .tensor import Tensor
from .mnemonics import AddTensor
from .shapes import Shape
from .backpropagation import BackwardFnArg, MinMaxArg, BACKWARD_MINMAX
from .validators import Validator
from .intarray import IntArray
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .ancestry import Ancestor
from tenmo.kernels.minmax_kernel import ReductionMinMax
from .common_utils import panic
from std.sys.info import has_accelerator


@fieldwise_init
struct MinMaxBackward[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var bwd_arg = (
            output.ancestry().backward_fn_arg().get[MinMaxArg[Self.dtype]]()
        )
        var (axes, keepdims, mask) = (
            bwd_arg.axes,
            bwd_arg.keepdims,
            bwd_arg.mask,
        )
        var gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        var shape = ancestor.shape()
        var mask_grad = Gradbox[Self.dtype](mask)

        if shape.rank() == 0:
            ancestor.update_grad(mask_grad^, AddTensor, None)
            parent_ids.append(ancestor._id)
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
        if not retain_graph:
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
        sync: Bool = False,
    ) -> Tensor[Self.dtype]:
        var shape = self.shape()
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)
        var tracking_grad = track_grad and requires_grad.or_else(
            self.requires_grad
        )
        var (result_ndb, mask_ndb) = Self.minmax[is_max=max](
            self.buffer, normalized_axes, keepdims, tracking_grad
        )
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_MINMAX,
                    MinMaxArg[Self.dtype](normalized_axes, keepdims, mask_ndb^),
                )
                backwardFnArg.needs_parent_data = True
                out.add_ancestry(backwardFnArg^, self)

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
        sync: Bool = False,
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        ref shape = ndb.shape
        var normalized_axes = Validator.validate_and_normalize_axes(shape, axes)

        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var (result_ndb, mask_ndb) = ReductionMinMax[
                        Self.dtype
                    ].launch[is_max=is_max](ndb, normalized_axes, keepdims, sync=sync)
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
