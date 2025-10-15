from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from intlist import IntList
from operators import AddTensor, Multiply, scalar_ops
from buffers import Buffer
from broadcastbackward import BroadcastBackward
from common_utils import panic, id


@fieldwise_init
@register_passable
struct MulBackwardScalar[dtype: DType](Copyable & Movable):
    var factor: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        gradients = output.gradients()[]
        ancestor = output.ancestry().get(0)
        scaled_gradients = gradients * self.factor
        return [
            (
                ancestor,
                scaled_gradients,
                AddTensor,
            )
        ]


@fieldwise_init
@register_passable
struct MultiplyBackward[dtype: DType](Copyable & Movable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: TensorLite[dtype]
    ) -> List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]:
        gradients = output.gradients()[]
        var grad_outputs: List[
            Tuple[TensorLite[dtype], Tensor[dtype], Int]
        ] = []

        count = len(output.ancestry())
        ancestor_1 = output.ancestry().get(0)

        if count == 1:  # B = A * A, A is the only ancestor of B
            tensor_1 = ancestor_1.tensor()
            product = Multiplicator[dtype].forward[False](gradients, tensor_1)
            product = MultiplyScalar[dtype].forward[False](
                product, Scalar[dtype](2)
            )
            return [(ancestor_1, product, AddTensor)]

        ancestor_2 = output.ancestry().get(1)

        if ancestor_1.requires_grad():
            product = gradients * ancestor_2.tensor()

            product.requires_grad = False
            grad_outputs.append(
                (
                    ancestor_1,
                    product,
                    AddTensor,
                )
            )

        if ancestor_2.requires_grad():
            product = gradients * ancestor_1.tensor()
            product.requires_grad = False
            grad_outputs.append(
                (
                    ancestor_2,
                    product,
                    AddTensor,
                )
            )

        return grad_outputs


@fieldwise_init
@register_passable
struct MultiplyScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], factor: Scalar[dtype]) -> Tensor[dtype]:
        var buffer: Buffer[dtype]
        if self.is_contiguous():
            if self.owns_data:
                buffer = self.buffer * factor
            else:
                offset = self.offset
                numels = self.numels()
                buffer = (
                    self.shared_buffer.value()[][offset : offset + numels]
                    * factor
                )
        else:
            idx = 0
            buffer = Buffer[dtype](self.numels())
            for coord in self.shape:
                buffer[idx] = self[coord] * factor
                idx += 1

        out = Tensor[dtype](self.shape, buffer^, requires_grad=False)

        @parameter
        if track_grad:
            out.requires_grad_(True)
            if self.requires_grad:
                backward_fn = MulBackwardScalar[dtype](
                    factor
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(self)

        return out


# Element wise multiplication of two tensors
@fieldwise_init
@register_passable
struct Multiplicator[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], other: Tensor[dtype]) -> Tensor[dtype]:
        if not self.broadcastable(other):
            panic(
                "Tensor → __mul__(self * other) → dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
                "at Multiplicator → forward",
            )
        var out: Tensor[dtype]
        if self.shape != other.shape:
            out = self.broadcast_op(other, scalar_ops[dtype, Multiply])
        else:
            var buffer: Buffer[dtype]
            if self.owns_data and other.owns_data:
                buffer = self.buffer * other.buffer
            else:
                buffer = Buffer[dtype](self.numels())
                idx = 0
                for coord in self.shape:
                    buffer[idx] = self[coord] * other[coord]
                    idx += 1

            out = Tensor[dtype](self.shape, buffer, requires_grad=False)

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)

                if self.shape == other.shape:
                    backward_fn = MultiplyBackward[dtype]().into_backward_fn()
                    out.backwardFn = Optional(backward_fn)
                    if id(self) == id(other):  # B = A * A, self == other == A
                        out.add_ancestry(self)
                    else:
                        out.add_ancestry(self, other)
                else:
                    backward_fn = BroadcastBackward[
                        dtype, AddTensor, AddTensor, True
                    ]().into_backward_fn()
                    out.backwardFn = Optional(backward_fn)
                    out.add_ancestry(self, other)

        return out


fn main():
    print("passes")
