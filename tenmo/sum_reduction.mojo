from .tensor import Tensor
from .mnemonics import AddTensor, SUM
from .intarray import IntArray
from .shapes import Shape
from .backpropagation import BackwardFnArg, ReductionArg, BACKWARD_SUM
from .validators import Validator
from .sum_mean_reduction import SumMeanReduction
from .gradbox import Gradbox
from .ancestry import Ancestor


@fieldwise_init
struct SumBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref bwd_arg = output.ancestry().backward_fn_arg().get[ReductionArg]()
        var (axes, keepdims) = bwd_arg.axes, bwd_arg.keepdims
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        ref shape = ancestor.shape()

        var grad_contrib: Gradbox[Self.dtype]

        if gradbox.shape() == Shape():
            grad_contrib = Gradbox[Self.dtype].full(
                shape, gradbox.item(), device=gradbox.device()
            )
        else:
            if not keepdims:
                axes = (
                    gradbox.shape()
                    .intarray()
                    .insert(
                        axes,
                        IntArray.filled(len(axes), 1),
                    )
                )
                unsqueezed_shape = Shape(axes)
                unsqueezed_grad = gradbox.reshape(unsqueezed_shape)
                grad_contrib = unsqueezed_grad.broadcast_to(shape)
            else:
                grad_contrib = gradbox.broadcast_to(shape)

        if ancestor.requires_grad:
            ancestor.update_grad(grad_contrib^, AddTensor, None)

        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct Summer[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var shape = tensor.shape()
        var reduction_axes = Validator.normalize_reduction_axes(shape, axes)
        var nd_buffer = SumMeanReduction[Self.dtype].reduce[op_code=SUM](tensor.buffer, reduction_axes, keepdims)
        var out = Tensor[Self.dtype](nd_buffer^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_SUM, ReductionArg(reduction_axes, keepdims)
                )
                out.add_ancestry(backwardFnArg^, tensor)

        return out^
