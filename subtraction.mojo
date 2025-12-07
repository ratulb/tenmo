from tenmo import Tensor
from intarray import IntArray
from operators import AddTensor, SubtractTensor, Subtract, ReverseSubtract
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_SUB,
    BACKWARD_SUBTRACT_BROADCAST,
    BACKWARD_SUB_SCALAR,
)
from common_utils import panic
from gradbox import Gradbox
from broadcastbackward import BroadcastBackward


@register_passable
struct SubBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_SUB
    var signs: IntArray

    fn __init__(out self):
        self.signs = IntArray()

    fn __copyinit__(out self, existing: Self):
        self.signs = existing.signs.copy()

    fn negate(mut self, neg: Bool):
        if neg:
            self.signs.append(1)
        else:
            self.signs.append(0)

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        count = len(output.ancestry())
        grad_shares = List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]](
            capacity=count
        )
        for i in range(count):
            var ancestor = output.ancestry().get(i)
            grad_shares.append(
                (
                    ancestor^,
                    gradbox,
                    AddTensor if self.signs[i] == 0 else SubtractTensor,
                )
            )
        return grad_shares^


@fieldwise_init
@register_passable
struct SubLeftRightBackwardScalar[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_SUB_SCALAR
    var negate: Bool

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        ref ancestor = output.ancestry().get(0)
        return [
            (
                ancestor,
                gradbox,
                SubtractTensor if self.negate else AddTensor,
            )
        ]


alias SubtractBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=False,
    lhs_op=AddTensor,
    rhs_op=SubtractTensor,
    TAG=BACKWARD_SUBTRACT_BROADCAST,
]


@register_passable
struct SubtractScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[Subtract](scalar), requires_grad=False
        )

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = SubLeftRightBackwardScalar[Self.dtype](
                    False
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


@register_passable
struct SubtractFromScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[Self.dtype]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[ReverseSubtract](scalar), requires_grad=False
        )

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = SubLeftRightBackwardScalar[Self.dtype](
                    True
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


@register_passable
struct Subtractor[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        if not self.broadcastable(other):
            panic(
                "Tensor →__sub__(self, other): dimension mismatch: "
                + self.shape().__str__()
                + " <=> "
                + other.shape().__str__(),
                "→ at Subtractor → forward",
            )

        var out: Tensor[Self.dtype] = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer),
            requires_grad=False,
        )

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad

            if requires_grad:
                out.requires_grad_(True)

                if self.shape() == other.shape():
                    sub_backward = SubBackward[Self.dtype]()
                    if self.requires_grad:
                        out.add_ancestry(self)
                        sub_backward.negate(False)
                    if other.requires_grad:
                        out.add_ancestry(other)
                        sub_backward.negate(True)
                    backward_fn = sub_backward.into_backward_fn()
                    out.backwardFn = Optional(backward_fn^)

                else:
                    backward_fn = SubtractBroadcastBackward[
                        Self.dtype
                    ]().into_backward_fn()

                    out.backwardFn = Optional(backward_fn^)
                    out.add_ancestry(self, other)

        return out^


fn main():
    print("passes")
