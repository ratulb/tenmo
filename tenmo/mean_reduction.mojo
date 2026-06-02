from .tensor import Tensor
from .intarray import IntArray
from .mnemonics import AddTensor, MEAN
from .shapes import Shape
from .backpropagation import BackwardFnArg, ReductionArg, BACKWARD_MEAN
from .validators import Validator
from .sum_mean_reduction import SumMeanReduction
from .gradbox import Gradbox
from .common_utils import panic
from .ancestry import Ancestor


struct MeanBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var bwd_arg = output.ancestry().backward_fn_arg().get[ReductionArg]()
        ref gradbox = output.gradients()
        ref gradbox_shape = gradbox.shape()
        var ancestor = output.ancestry().get(0)
        ref ancestor_shape = ancestor.shape()

        var grad_contrib: Gradbox[Self.dtype]
        if gradbox_shape == Shape():
            scalar_grad = gradbox.item() / Scalar[Self.dtype](
                ancestor_shape.num_elements()
            )
            grad_contrib = Gradbox[Self.dtype].full(
                ancestor_shape,
                scalar_grad,
                
                device=gradbox.device(),
            )
        else:
            var expanded = gradbox.copy()
            if not bwd_arg.keepdims:
                expanded = expanded.reshape(
                    Shape(
                        gradbox_shape.intarray().insert(
                            bwd_arg.axes,
                            IntArray.filled(len(bwd_arg.axes), 1),
                        )
                    )
                )
            var broadcasted = expanded.broadcast_to(ancestor_shape)
            var count = ancestor_shape.reduced_shape(bwd_arg.axes).product()
            count = count if count > 0 else 1
            grad_contrib = broadcasted / Scalar[Self.dtype](count)

        if ancestor.requires_grad:
            ancestor.update_grad(grad_contrib, AddTensor, None)
        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct Mean[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @always_inline
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        normalized_axes = Validator.validate_and_normalize_axes(
            tensor.shape(), axes
        )
        var ndb = SumMeanReduction[Self.dtype].reduce[op_code=MEAN](tensor.buffer, normalized_axes, keepdims)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_MEAN, ReductionArg(normalized_axes, keepdims)
                )
                out.add_ancestry(backwardFnArg^, tensor)

        return out^

    @always_inline
    @staticmethod
    def forward(
        gradbox: Gradbox[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
    ) -> Gradbox[Self.dtype]:
        var gradbox_shape = gradbox.shape()
        normalized_axes = Validator.validate_and_normalize_axes(
            gradbox_shape, axes
        )
        var ndb = SumMeanReduction[Self.dtype].reduce[op_code=MEAN](gradbox.buffer, normalized_axes, keepdims)
        var out = Gradbox[Self.dtype](ndb^)

        return out^
