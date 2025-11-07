from tenmo import Tensor
from operators import AddTensor
from backpropagation import Delegate, BackwardFn
from ancestry import Ancestor
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct ReLUBackward[dtype: DType](ImplicitlyCopyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        ancestor = output.ancestry().get(0)
        input_tensor = ancestor.tensor()
        shape = ancestor.shape()
        gradbox_ancestor = Gradbox[dtype].zeros(
            shape, share=False
        )
        var zero = Scalar[dtype](0)
        for coord in shape:
            if input_tensor[coord] > zero:
                gradbox_ancestor[coord] = gradbox[coord]

        return [(ancestor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
@register_passable
struct ReLU[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[
        dtype
    ]:
        shape = self.shape()
        out = Tensor[dtype].zeros(shape, requires_grad=False)
        zero = Scalar[dtype](0)
        for coord in shape:
            out[coord] = self[coord] if self[coord] > zero else zero

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(self.requires_grad)

            if grad_required:
                out.requires_grad_(True)
                backward_fn = ReLUBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


fn main() raises:
    print("passes")
