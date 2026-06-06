from .tensor import Tensor
from .backpropagation import BackwardFnArg, IntArrayArg, BACKWARD_PERMUTE
from .mnemonics import AddTensor
from .intarray import IntArray
from .shapes import Shape
from .strides import Strides
from .common_utils import panic
from .views import View
from .gradbox import Gradbox
from .ancestry import Ancestor


@fieldwise_init
struct PermuteBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        """
        Backward pass for permute.
        Apply inverse permutation to upstream gradients.
        GPU safe: Gradbox.permute → NDBuffer.permute(shared=False)
                  → contiguous() → contiguous_device_state() on GPU.
        """
        var permutation = (
            output.ancestry().backward_fn_arg().get[IntArrayArg]().array
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
        gradbox.zero_grad()


@fieldwise_init
struct Permute[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        mut self: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
        sync: Bool = False,
    ) -> Tensor[Self.dtype]:
        # NDBuffer.permute(shared=True) handles validation + view creation
        # GPU safe — just reorders shape/strides metadata, no data movement
        var result_ndb = self.buffer.permute(axes, shared=True)
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].from_intarray(
                    BACKWARD_PERMUTE, axes
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^
