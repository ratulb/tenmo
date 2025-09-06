from tensors import Tensor
from intlist import IntList
from operators import AddTensor, SubtractTensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from buffers import Buffer


@fieldwise_init
@register_passable
struct SubBackward[dtype: DType](Copyable):
    var signs: IntList

    fn __init__(out self):
        self.signs = IntList.Empty

    fn negate(mut self, neg: Bool):
        if neg:
            self.signs.append(1)
        else:
            self.signs.append(0)

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        count = len(output.ancestry())
        grad_outputs = List[Tuple[TensorLite[dtype], Tensor[dtype], Int]](
            capacity=count
        )
        for i in range(count):
            ancestor = output.ancestry().get(i)[]
            grad_outputs.append(
                (
                    ancestor,
                    gradients,
                    AddTensor if self.signs[i] == 0 else SubtractTensor,
                )
            )
        return grad_outputs


@fieldwise_init
@register_passable
struct SubLeftRightBackwardScalar[dtype: DType](Copyable):
    var negate: Bool

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)[]
        return [
            (ancestor, gradients, SubtractTensor if self.negate else AddTensor)
        ]


@fieldwise_init
@register_passable
struct SubtractScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out: Tensor[dtype]
        if self.owns_data:
            out = Tensor[dtype](
                self.shape, self.buffer - scalar, self.requires_grad
            )
        else:
            buffer = Buffer[dtype](self.numels())
            out = Tensor[dtype](self.shape, buffer, self.requires_grad)
            for indices in self.shape:
                out[indices] = self[indices] - scalar

        @parameter
        if track_grad:
            if self.requires_grad:
                backward_fn = SubLeftRightBackwardScalar[dtype](
                    False
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


@fieldwise_init
@register_passable
struct SubtractFromScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out: Tensor[dtype]
        if self.owns_data:
            out = Tensor[dtype](
                self.shape, scalar - self.buffer, self.requires_grad
            )
        else:
            buffer = Buffer[dtype](self.numels())
            out = Tensor[dtype](self.shape, buffer, self.requires_grad)
            for indices in self.shape:
                out[indices] = scalar - self[indices]

        @parameter
        if track_grad:
            if self.requires_grad:
                backward_fn = SubLeftRightBackwardScalar[dtype](
                    True
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


fn main():
    print("passes")
