from .tensor import Tensor
from .backpropagation import BackwardFn, IntArrayArg, BackwardFnType

from .shared.mnemonics import AddTensor
from .validators import Validator
from .shared.intarray import IntArray
from .ancestry import Ancestor


@fieldwise_init
struct TransposeBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var axes = output.ancestry().backward_fn().get[IntArrayArg]().array
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        var inverted_axes = IntArray.invert_permutation(axes^)
        var gradbox_transposed_contiguous = gradbox.transpose(inverted_axes^)
        if ancestor.requires_grad:
            ancestor.update_grad(
                gradbox_transposed_contiguous^, AddTensor, None
            )
        parent_ids.append(ancestor._id)
        # View conduit: always cleared.
        gradbox.zero_grad()


@fieldwise_init
struct Transpose[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var transposed_ndb = self.buffer.transpose(axes)
        var out = Tensor[Self.dtype](transposed_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                ref shape = self.shape()
                var normalized_axes = (
                    Validator.validate_and_normalize_axes(
                        shape, axes, ordered=False, fill_missing=True
                    ) if len(axes)
                    > 0 else IntArray.range(start=shape.rank()-1, end=-1, step=-1)
                )
                out.requires_grad_(True)
                var backwardFn = BackwardFn.from_intarray[Self.dtype](
                    normalized_axes,
                    TransposeBackward[Self.dtype](),
                )
                out.add_ancestry(backwardFn^, self)

        return out^
