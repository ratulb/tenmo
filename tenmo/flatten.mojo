from tensor import Tensor
from backpropagation import BackwardFnArg, BACKWARD_FLATTEN
from mnemonics import AddTensor
from common_utils import panic
from gradbox import Gradbox
from ancestry import Ancestor


@fieldwise_init
struct FlattenBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        ancestor_shape = ancestor.shape()
        # Just reshape gradient back to original
        var reshaped_grad = gradbox.reshape(ancestor_shape)
        return [
            (ancestor, reshaped_grad^, AddTensor),
        ]


struct FlattenForward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        start_dim: Int = 0,
        end_dim: Optional[Int] = None,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        flattened_buffer = self.buffer.flatten(start_dim, end_dim)
        out = Tensor[Self.dtype](flattened_buffer^, requires_grad=False)

        comptime if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_FLATTEN
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^
