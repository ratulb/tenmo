from .tensor import Tensor
from .shared.intarray import IntArray
from .shared.mnemonics import AddTensor, MEAN
from .shared.shapes import Shape
from .backpropagation import BackwardFn, BackwardFnType

from .validators import Validator
from .sum_mean_reduction import ReductionArg, SumMeanReduction
from .gradbox import Gradbox
from .ancestry import Ancestor


@fieldwise_init
struct MeanBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var bwd_arg = output.ancestry().backward_fn().get[ReductionArg]()
        ref gradbox = output.gradients()
        var gradbox_shape = gradbox.shape()
        var ancestor = output.ancestry().get(0)
        ref ancestor_shape = ancestor.shape()

        var grad_contrib: Gradbox[Self.dtype]
        if gradbox_shape == Shape():
            var scalar_grad = gradbox.item() / Scalar[Self.dtype](
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
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var normalized_axes = Validator.validate_and_normalize_axes(
            tensor.shape(), axes
        )
        var ndb = SumMeanReduction[Self.dtype].reduce[op_code=MEAN](
            tensor.buffer, normalized_axes, keepdims, sync=sync
        )
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(tensor.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn(
                    ReductionArg(normalized_axes, keepdims),
                    MeanBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, tensor)

        return out^

    @always_inline
    @staticmethod
    def forward(
        gradbox: Gradbox[Self.dtype],
        axes: IntArray,
        keepdims: Bool = False,
        sync: Bool = False,
    ) -> Gradbox[Self.dtype]:
        var gradbox_shape = gradbox.shape()
        var normalized_axes = Validator.validate_and_normalize_axes(
            gradbox_shape, axes
        )
        var ndb = SumMeanReduction[Self.dtype].reduce[op_code=MEAN](
            gradbox.buffer(), normalized_axes, keepdims, sync=sync
        )
        var out = Gradbox[Self.dtype](ndb^)

        return out^
