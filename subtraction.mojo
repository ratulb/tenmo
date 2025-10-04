from tensors import Tensor
from intlist import IntList
from operators import AddTensor, SubtractTensor, Subtract, scalar_ops
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from buffers import Buffer
from broadcastbackward import BroadcastBackward
from common_utils import panic


struct SubBackward[dtype: DType](Copyable & Movable):
    var signs: IntList

    fn __init__(out self):
        self.signs = IntList.Empty

    fn __copyinit__(out self, existing: Self):
        self.signs = existing.signs.copy()

    fn __moveinit__(out self, deinit existing: Self):
        self.signs = existing.signs

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
        gradients = output.grad()
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
        gradients = output.grad()
        ancestor = output.ancestry().get(0)[]
        return [
            (ancestor, gradients, SubtractTensor if self.negate else AddTensor)
        ]


@register_passable
struct SubtractScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out: Tensor[dtype]
        shape = self.shape.copy()

        if self.owns_data:
            out = Tensor[dtype](shape, self.buffer - scalar, False)
        else:
            out = Tensor[dtype](shape, requires_grad=False)
            for idx, value in self:
                out[idx] = value - scalar

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = SubLeftRightBackwardScalar[dtype](
                    False
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


@register_passable
struct SubtractFromScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out: Tensor[dtype]
        shape = self.shape.copy()
        if self.owns_data:
            out = Tensor[dtype](shape, scalar - self.buffer, False)
        else:
            out = Tensor[dtype](shape, requires_grad=False)
            for idx, value in self:
                out[idx] = scalar - value

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = SubLeftRightBackwardScalar[dtype](
                    True
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


@register_passable
struct Subtractor[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], other: Tensor[dtype]) -> Tensor[dtype]:
        if not self.broadcastable(other):
            panic(
                "Tensor →__sub__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
                "→ at Subtractor → forward",
            )

        var out: Tensor[dtype]
        this_shape = self.shape.copy()
        if self.shape != other.shape:
            out = self.broadcast_op(other, scalar_ops[dtype, Subtract])
        else:
            if self.owns_data and other.owns_data:
                buffer = self.buffer - other.buffer
                out = Tensor[dtype](this_shape, buffer, False)
            else:
                out = Tensor[dtype].zeros(this_shape, False)
                for coord in this_shape:
                    out[coord] = self[coord] - other[coord]

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad

            if requires_grad:
                out.requires_grad_(True)

                if self.shape == other.shape:
                    sub_backward = SubBackward[dtype]()
                    if self.requires_grad:
                        out.add_ancestry(TensorLite.of(self))
                        sub_backward.negate(False)
                    if other.requires_grad:
                        out.add_ancestry(TensorLite.of(other))
                        sub_backward.negate(True)
                    backward_fn = sub_backward.into_backward_fn()
                    out.backwardFn = Optional(backward_fn)

                else:
                    backward_fn = BroadcastBackward[
                        dtype, AddTensor, SubtractTensor, False
                    ]().into_backward_fn()

                    out.backwardFn = Optional(backward_fn)
                    out.add_ancestry(TensorLite.of(self), TensorLite.of(other))

        return out


fn main():
    print("passes")
