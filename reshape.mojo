from tensors import Tensor
from shared import TensorLite
from operators import AddTensor, ZeroGrad
from backpropagation import BackwardFn, Delegate
from shapes import Shape
from validators import Validator


@fieldwise_init
@register_passable
struct ReshapeBackward[dtype: DType](Copyable):
    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        reshaped = gradients.reshape(ancestor.shape())

        return [
            (ancestor, reshaped, AddTensor),
            (output, gradients, ZeroGrad),
        ]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))


@fieldwise_init
@register_passable
struct Reshape[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensor: Tensor[dtype],
        new_shape: Shape,
        requires_grad: Optional[Bool] = None,
        validated: Bool = False
    ) -> Tensor[dtype]:
        shape = new_shape if validated else Validator.validate_and_construct_new_shape(
            tensor.shape, new_shape.intlist()
        )

        buffer = tensor.buffer if tensor.owns_data else tensor.base[].buffer
        out = Tensor[dtype](shape, buffer)

        @parameter
        if track_grad:
            grad_required = (
                requires_grad.value() if requires_grad else tensor.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)
                backward_fn = ReshapeBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite[dtype].of(tensor))

        return out


fn main() raises:

    test_reshape_preserves_grad_accumulation()
    print("\npasses\n")


from testing import assert_true

fn test_reshape_preserves_grad_accumulation() raises:
    print("test_reshape_preserves_grad_accumulation")
    # Chained reshape should still accumulate gradients
    a = Tensor.of(1.0, 2.0, 3.0, requires_grad=True)
    print("shapes: ", Shape.of(3), Shape.of(1, 3))
    b = a.reshape(Shape.of(3))
    c = b.reshape(Shape.of(1, 3))

    #d = c.sum()
    c.backward()
    a.gradbox[].print()
    assert_true((a.gradbox[] == Tensor.of(1.0, 1, 1)).all_true()) 
