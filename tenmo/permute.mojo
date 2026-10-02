from .tensor import Tensor
from .backpropagation import BackwardFn, IntArrayArg, BackwardFnType

from .shared.mnemonics import AddTensor
from .shared.intarray import IntArray
from .validators import Validator
from .ancestry import Ancestor


@fieldwise_init
struct PermuteBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        """
        Backward pass for permute.
        Apply inverse permutation to upstream gradients.
        GPU safe: Gradbox.permute materialises via contiguous(owned=True)
                  → contiguous_device_state() on GPU.
        """
        var permutation = (
            output.ancestry().backward_fn().get[IntArrayArg]().array
        )
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)

        # Invert the forward permutation
        var inverted = IntArray.invert_permutation(permutation)

        # Apply inverse permutation — GPU safe via NDBuffer.permute
        var parent_gradbox = gradbox.permute(inverted^)
        if parent.requires_grad:
            parent.update_grad(parent_gradbox^, AddTensor, None)
        parent_ids.append(parent._id)
        # View conduit: grad already forwarded to base above; always cleared.
        gradbox.zero_grad()


@fieldwise_init
struct Permute[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        # Normalize up front, mirroring Transpose: negative axes must be
        # canonical before use — NDBuffer.permute builds geometry from raw
        # indices and the stored arg feeds invert_permutation in backward.
        # (ordered=False preserves the permutation order.)
        var normalized_axes = Validator.validate_and_normalize_axes(
            self.shape(), axes, ordered=False
        )
        # NDBuffer.permute reads (a copy of) the normalized axes for
        # geometry; the original is stored below for backward inversion.
        var result_ndb = self.buffer.permute(
            IntArray(copy=normalized_axes)
        )
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.from_intarray[Self.dtype](
                    normalized_axes, PermuteBackward[Self.dtype]()
                )
                out.add_ancestry(backwardFn^, self)

        return out^
