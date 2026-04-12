from tenmo import Tensor
from backpropagation import IntArrayArg, BACKWARD_PERMUTE
from mnemonics import AddTensor, ZeroGrad
from intarray import IntArray
from shapes import Shape
from strides import Strides
from common_utils import panic
from views import View
from gradbox import Gradbox


@fieldwise_init
struct PermuteBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """
        Backward pass for permute.
        Apply inverse permutation to upstream gradients.
        GPU safe: Gradbox.permute → NDBuffer.permute(shared=False)
                  → contiguous() → contiguous_device_state() on GPU.
        """
        var permutation = output.fn_arg().arg[IntArrayArg].axes
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)

        # Invert the forward permutation
        var inverted = IntArray.invert_permutation(permutation)

        # Apply inverse permutation — GPU safe via NDBuffer.permute
        var parent_gradbox = gradbox.permute(inverted^)

        return [
            (parent^, parent_gradbox^, AddTensor),
            (output, gradbox, ZeroGrad),
        ]


@fieldwise_init
struct Permute[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut self: Tensor[Self.dtype],
        axes: IntArray,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        # NDBuffer.permute(shared=True) handles validation + view creation
        # GPU safe — just reorders shape/strides metadata, no data movement
        var result_ndb = self.buffer.permute(axes, shared=True)
        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var bwd_fn_arg = IntArrayArg(
                    axes.copy()
                ).into_arg[Self.dtype](BACKWARD_PERMUTE)
                out.fnArg = Optional(bwd_fn_arg^)
                out.add_ancestry(self)

        return out^
