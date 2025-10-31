from tenmo import Tensor
from backpropagation import Delegate, BackwardFn
from intlist import IntList
from operators import AddTensor, SubtractTensor, Divide, ReverseDivide
from common_utils import panic
from ancestry import Ancestor
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct TrueDivBackwardScalar[dtype: DType](ImplicitlyCopyable):
    var factor: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        ancestor = output.ancestry().get(0)
        # ∂(x / s)/∂x = 1/s → incoming_grad / scalar
        var divided = gradbox / self.factor
        return [
            (
                ancestor^,
                divided^,
                AddTensor,
            )
        ]


@fieldwise_init
@register_passable
struct RightTrueDivBackwardScalar[dtype: DType](ImplicitlyCopyable):
    var scalar: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        ancestor = output.ancestry().get(0)
        tensor = ancestor.tensor()
        squared = tensor.__pow__(2)
        squared_reciprocal = 1.0 / squared
        gradbox = (gradbox * self.scalar) * squared_reciprocal

        return [
            (
                ancestor^,
                gradbox^,
                SubtractTensor,
            )
        ]


@fieldwise_init
@register_passable
struct DivideBackward[dtype: DType](ImplicitlyCopyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        ancestor_top = output.ancestry().get(0)
        ancestor_bottom = output.ancestry().get(1)

        grad_shares = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]](
            capacity=2
        )

        if ancestor_top.requires_grad():
            tensor_bottom = ancestor_bottom.tensor()
            tensor_bottom.requires_grad_(False)
            tensor_bottom_reciprocal = 1 / tensor_bottom
            tensor_top_shape = ancestor_top.shape()
            tensor_top_gradbox = gradbox * tensor_bottom_reciprocal
            if tensor_top_gradbox.shape() != tensor_top_shape:
                tensor_top_gradbox = (
                    tensor_top_gradbox.sum_over_broadcasted_axes(
                        tensor_top_shape
                    )
                )
            grad_shares.append(
                (ancestor_top.copy(), tensor_top_gradbox^, AddTensor)
            )

        if ancestor_bottom.requires_grad():
            tensor_top = ancestor_top.tensor()
            tensor_bottom = ancestor_bottom.tensor()
            tensor_top.requires_grad_(False)
            tensor_bottom.requires_grad_(False)
            tensor_bottom_squared = tensor_bottom * tensor_bottom
            tensor_bottom_squared_reciprocal = 1 / tensor_bottom_squared
            tensor_bottom_grad = tensor_top * tensor_bottom_squared_reciprocal
            tensor_bottom_gradbox = gradbox * tensor_bottom_grad
            if tensor_bottom_gradbox.shape() != tensor_bottom.shape():
                tensor_bottom_gradbox = (
                    tensor_bottom_gradbox.sum_over_broadcasted_axes(
                        tensor_bottom.shape()
                    )
                )
            grad_shares.append(
                (ancestor_bottom^, tensor_bottom_gradbox^, SubtractTensor)
            )

        return grad_shares^


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

        nd_buffer = self.buffer.scalar_ops[ReverseDivide](scalar)
        var out = Tensor[dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = RightTrueDivBackwardScalar[dtype](
                    scalar
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


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

        nd_buffer = self.buffer.scalar_ops[Divide](scalar)
        var out = Tensor[dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)

                backward_fn = TrueDivBackwardScalar[dtype](
                    scalar
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


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
                + self.shape().__str__()
                + " <=> "
                + other.shape().__str__(),
                "at Divider → forward",
            )
        nd_buffer = self.buffer.arithmetic_ops[Divide](other.buffer)
        var out = Tensor[dtype](nd_buffer^, requires_grad=False)

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)
                backward_fn = DivideBackward[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self, other)

        return out^


fn main():
    print("passes")
