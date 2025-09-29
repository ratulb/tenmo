from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from intlist import IntList
from operators import AddTensor, Add, scalar_ops
from buffers import Buffer
from broadcastbackward import BroadcastBackward
from common_utils import panic


@fieldwise_init
@register_passable
struct AddBackwardScalar[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.grad()
        ancestor = output.ancestry().get(0)[]
        if ancestor.shape() != gradients.shape:
            gradients = gradients.reshape(ancestor.shape())
        # Gradient of addition is 1 → just pass through incoming grad
        return [(ancestor, gradients, AddTensor)]


@fieldwise_init
@register_passable
struct AddBackward[dtype: DType](Copyable):
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
            grad_outputs.append((ancestor, gradients, AddTensor))
        return grad_outputs


@fieldwise_init
@register_passable
struct AddScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out: Tensor[dtype]
        if self.is_dense():
            buffer = self.buffer.unbox() + scalar
            out = Tensor[dtype](
                self.shape, buffer.box(), requires_grad=self.requires_grad
            )
        else:
            buffer = Buffer[dtype](self.numels())
            out = Tensor[dtype](
                self.shape, buffer.box(), requires_grad=self.requires_grad
            )
            for coord in self.shape:
                out[coord] = self[coord] + scalar

        @parameter
        if track_grad:
            if self.requires_grad:
                backward_fn = AddBackwardScalar[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite[dtype].of(self))

        return out


# Element wise addition of two tensors
@fieldwise_init
@register_passable
struct Adder[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], other: Tensor[dtype]) -> Tensor[dtype]:
        if self.address() == other.address():
            return self.__mul__(2)
        if not self.broadcastable(other):
            panic(
                "Tensor__add__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
                "at Addition → forward",
            )

        var out: Tensor[dtype]
        if self.shape != other.shape:
            out = self.broadcast_op(other, scalar_ops[dtype, Add])
        else:
            if self.is_dense() and other.is_dense():
                buffer = self.buffer.unbox() + other.buffer.unbox()
                out = Tensor[dtype](
                    self.shape, buffer.box(), requires_grad=False
                )
            else:
                out = Tensor[dtype].zeros(self.shape, requires_grad=False)
                for coord in self.shape:
                    out[coord] = self[coord] + other[coord]

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)

                if self.shape == other.shape:
                    backward_fn = AddBackward[dtype]().into_backward_fn()
                    out.backwardFn = Optional(backward_fn)
                    if self.requires_grad:
                        out.add_ancestry(TensorLite[dtype].of(self))
                    if other.requires_grad:
                        out.add_ancestry(TensorLite[dtype].of(other))
                else:
                    backward_fn = BroadcastBackward[
                        dtype, AddTensor, AddTensor, False
                    ]().into_backward_fn()

                    out.backwardFn = Optional(backward_fn)
                    out.add_ancestry(TensorLite.of(self), TensorLite.of(other))

        return out


fn main():
    print("passes")
