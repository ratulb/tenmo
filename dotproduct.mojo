from tensors import Tensor
from shared import TensorLite
from common_utils import panic
from backpropagation import Delegate, BackwardFn
from operators import (
    AddTensor,
)


@fieldwise_init
@register_passable
struct DotBackward[dtype: DType](Copyable & Movable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.grad().item()  # Scalar
        var grad_outputs: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]

        if ancestor_1.requires_grad():
            tensor = ancestor_2.tensor()

            buffer = tensor.buffer.unbox() if tensor.is_dense() else tensor.contiguous[
                False
            ](
                requires_grad=False
            ).buffer.unbox()
            buffer = gradients * buffer
            outgoing = Tensor[dtype](
                ancestor_1.shape(), buffer.box(), requires_grad=False
            )
            grad_outputs.append(
                (
                    ancestor_1,
                    outgoing^,
                    AddTensor,
                )
            )

        if ancestor_2.requires_grad():
            tensor = ancestor_1.tensor()
            buffer = tensor.buffer.unbox() if tensor.is_dense() else tensor.contiguous[
                False
            ](
                requires_grad=False
            ).buffer.unbox()
            buffer = gradients * buffer
            outgoing = Tensor[dtype](
                ancestor_2.shape(), buffer.box(), requires_grad=False
            )

            grad_outputs.append(
                (
                    ancestor_2,
                    outgoing^,
                    AddTensor,
                )
            )

        return grad_outputs


@register_passable
struct Dot[dtype: DType](Copyable):
    fn __copyinit__(out self, existing: Self):
        pass

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[dtype],
        other: Tensor[dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[dtype]:
        rank1 = self.rank()
        rank2 = other.rank()
        if not rank1 == rank2 and not rank1 <= 1:
            panic("Tensor → dot: not supported for rank > 1")
        numels1 = self.numels()
        numels2 = other.numels()
        if not numels1 == numels2:
            panic(
                "Tensor → dot: size does not match",
                numels1.__str__(),
                numels2.__str__(),
            )
        var out: Tensor[dtype]
        if self.is_dense() and other.is_dense():
            scalar = self.buffer.unbox().dot(other.buffer.unbox())
            out = Tensor[dtype].scalar(scalar, requires_grad=False)
        else:
            scalar = Scalar[dtype](0)
            for idx in self.shape:
                scalar += self[idx] * other[idx]
            out = Tensor[dtype].scalar(scalar, requires_grad=False)

        @parameter
        if track_grad:
            grad_required = requires_grad.value() if requires_grad else (
                self.requires_grad or other.requires_grad
            )

            if grad_required:
                out.requires_grad_(True)
                backward_fn = DotBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(
                    TensorLite[dtype].of(self), TensorLite[dtype].of(other)
                )

        return out


fn main():
    print("passes")
