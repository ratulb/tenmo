from tenmo import Tensor
from backpropagation import Delegate, BackwardFn, BACKWARD_FLATTEN
from mnemonics import AddTensor
from common_utils import panic
from gradbox import Gradbox


@fieldwise_init
struct FlattenBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
    comptime TAG = BACKWARD_FLATTEN

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        ancestor_shape = ancestor.shape()
        # Just reshape gradient back to original
        var reshaped_grad = gradbox.reshape(ancestor_shape)
        return [
            (ancestor^, reshaped_grad^, AddTensor),
        ]


struct FlattenForward[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                backward_fn = FlattenBackward[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^
