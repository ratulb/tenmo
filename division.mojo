from tensors import Tensor
from shared import TensorLite
from backpropagation import Delegate, BackwardFn
from intlist import IntList
from operators import AddTensor, SubtractTensor, Divide, scalar_ops
from buffers import Buffer
from broadcastbackward import BroadcastBackward
from common_utils import panic


@fieldwise_init
@register_passable
struct TrueDivBackwardScalar[dtype: DType](Copyable):
    var factor: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.grad()
        var divisor: Scalar[dtype] = rebind[Scalar[dtype]](self.factor)
        ancestor = output.ancestry().get(0)[]
        # ∂(x / s)/∂x = 1/s → incoming_grad / scalar
        var divided = gradients / divisor
        return [
            (
                ancestor,
                divided,
                AddTensor,
            )
        ]


@fieldwise_init
@register_passable
struct RightTrueDivBackwardScalar[dtype: DType](Copyable):
    var scalar: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        var scalar: Scalar[dtype] = rebind[Scalar[dtype]](self.scalar)
        ancestor = output.ancestry().get(0)[]
        squared = ancestor.tensor().__pow__(2)
        squared_reciprocal = 1.0 / squared
        grad = (gradients * scalar) * squared_reciprocal

        return [
            (
                ancestor,
                grad,
                SubtractTensor,
            )
        ]


@fieldwise_init
@register_passable
struct DivideBackward[dtype: DType](Copyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward[
        dtype: DType
    ](self, output: TensorLite[dtype]) -> List[
        Tuple[TensorLite[dtype], Tensor[dtype], Int]
    ]:
        gradients = output.gradients()[]
        ancestor_1 = output.ancestry().get(0)[]
        ancestor_2 = output.ancestry().get(1)[]

        outgoing_grads = List[Tuple[TensorLite[dtype], Tensor[dtype], Int]]()

        if ancestor_1.requires_grad():
            tensor_2 = ancestor_2.tensor()
            tensor_2.requires_grad_(False)
            tensor_2_reciprocal = 1 / tensor_2
            tensor_1_shape = ancestor_1.shape()
            tensor_1_grad = gradients * tensor_2_reciprocal
            if tensor_1_grad.shape != tensor_1_shape:
                tensor_1_grad = Tensor[dtype].sum_over_broadcasted_axes(
                    tensor_1_grad, tensor_1_shape
                )
            outgoing_grads.append((ancestor_1, tensor_1_grad, AddTensor))

        if ancestor_2.requires_grad():
            tensor_1 = ancestor_1.tensor()
            tensor_2 = ancestor_2.tensor()
            tensor_1.requires_grad_(False)
            tensor_2.requires_grad_(False)
            tensor_2_squared = tensor_2 * tensor_2
            tensor_2_squared_reciprocal = 1 / tensor_2_squared
            tensor_2_grad = tensor_1 * tensor_2_squared_reciprocal
            tensor_2_grad = gradients * tensor_2_grad
            if tensor_2_grad.shape != tensor_2.shape:
                tensor_2_grad = Tensor[dtype].sum_over_broadcasted_axes(
                    tensor_2_grad, tensor_2.shape
                )
            outgoing_grads.append((ancestor_2, tensor_2_grad, SubtractTensor))

        return outgoing_grads


@fieldwise_init
@register_passable
struct DivideScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __rtruediv__ is for numeric data types only",
        ]()

        var buffer: Buffer[dtype]
        if self.is_dense():
            buffer = self.buffer.unbox()
        else:
            idx = 0
            buffer = Buffer[dtype](self.numels())
            for coord in self.shape:
                buffer[idx] = self[coord]
                idx += 1
        buffer = scalar / buffer
        out = Tensor[dtype](self.shape, buffer.box(), requires_grad=False)

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = RightTrueDivBackwardScalar[dtype](
                    scalar
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


@fieldwise_init
@register_passable
struct DivideByScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
        constrained[
            dtype.is_numeric(),
            "Tensor → __truediv__ is for numeric data types only",
        ]()

        if scalar == Scalar[dtype](0):
            panic("Tensor → __truediv__ : canot divide by " + scalar.__str__())

        var buffer: Buffer[dtype]
        if self.is_dense():
            buffer = self.buffer.unbox()
        else:
            idx = 0
            buffer = Buffer[dtype](self.numels())
            for coord in self.shape:
                buffer[idx] = self[coord]
                idx += 1
        buffer = buffer / scalar
        out = Tensor[dtype](self.shape, buffer.box(), requires_grad=False)

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)

                backward_fn = TrueDivBackwardScalar[dtype](
                    scalar
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self))

        return out


# Element wise division of two tensors
@fieldwise_init
@register_passable
struct Divider[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], other: Tensor[dtype]) -> Tensor[dtype]:
        if not self.broadcastable(other):
            panic(
                "Tensor →__truediv__(self * other): dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
                "at Divider → forward",
            )

        var out: Tensor[dtype]
        if self.shape != other.shape:
            out = self.broadcast_op(other, scalar_ops[dtype, Divide])
        else:
            var buffer: Buffer[dtype]
            if self.is_dense() and other.is_dense():
                buffer = self.buffer.unbox() / other.buffer.unbox()
            else:
                buffer = Buffer[dtype](self.numels())
                idx = 0
                for coord in self.shape:
                    buffer[idx] = self[coord] / other[coord]
                    idx += 1

            out = Tensor[dtype](self.shape, buffer.box(), requires_grad=False)

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)
                backward_fn = DivideBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn)
                out.add_ancestry(TensorLite.of(self), TensorLite.of(other))

        return out


fn main():
    print("passes")
