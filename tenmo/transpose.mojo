from .tensor import Tensor
from .backpropagation import BackwardFnArg, IntArrayArg, BACKWARD_TRANSPOSE
from .mnemonics import AddTensor
from .validators import Validator
from .gradbox import Gradbox
from .intarray import IntArray
from .ancestry import Ancestor


@fieldwise_init
struct TransposeBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var axes = output.ancestry().backward_fn_arg().get[IntArrayArg]().array
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        var inverted_axes = IntArray.invert_permutation(axes^)
        var gradbox_transposed_contiguous = gradbox.transpose(inverted_axes^)
        if ancestor.requires_grad:
            ancestor.update_grad(
                gradbox_transposed_contiguous^, AddTensor, None
            )
        parent_ids.append(ancestor._id)
        gradbox.zero_grad()


@fieldwise_init
struct Transpose[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        mut self: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var transposed_ndb = self.buffer.transpose(axes, shared=True)
        var out = Tensor[Self.dtype](transposed_ndb^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                ref shape = self.shape()
                var normalized_axes = (
                    Validator.validate_and_normalize_axes(
                        shape, axes, ordered=False, fill_missing=True
                    ) if len(axes)
                    > 0 else IntArray.range(0, shape.rank()).reversed()
                )
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].from_intarray(
                    BACKWARD_TRANSPOSE, normalized_axes
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^
