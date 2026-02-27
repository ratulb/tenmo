from tenmo import Tensor
from intarray import IntArray
from mnemonics import AddTensor, SubtractTensor, Subtract, ReverseSubtract
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
from sys import has_accelerator
from binary_forward import BinaryOperation
from scalar_forward import ScalarOperation


@register_passable
struct SubBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_SUB
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
    comptime TAG = BACKWARD_SUB_SCALAR
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


comptime SubtractBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=False,
    lhs_op=AddTensor,
    rhs_op=SubtractTensor,
    TAG=BACKWARD_SUBTRACT_BROADCAST,
]


@fieldwise_init
@register_passable
struct SubtractScalar[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out = ScalarOperation[Self.dtype].forward[Subtract](self, scalar)

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


@fieldwise_init
@register_passable
struct SubtractFromScalar[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out = ScalarOperation[Self.dtype].forward[ReverseSubtract](
            self, scalar
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


@fieldwise_init
@register_passable
struct Subtractor[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        if not self.broadcastable(other):
            panic(
                "Tensor subtraction dimension mismatch: cannot broadcast shape "
                + self.shape().__str__()
                + " with "
                + other.shape().__str__(),
                "at Subtractor → forward",
            )

        var out = BinaryOperation[Self.dtype].forward[Subtract](self, other)

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


from common_utils import now
from testing import assert_true


fn main() raises:
    comptime dtype = DType.float32
    a1 = Tensor[dtype].rand(5000, 1000)
    b1 = Tensor[dtype].rand(5000, 1000)
    a = a1.transpose(0, 1)
    b = b1.transpose(0, 1)
    start = now()
    r1 = a - b
    print("CPU subtract took: ", (now() - start) * 1000, "ms")
    start = now()
    ag = a.to_gpu()
    bg = b.to_gpu()
    print("Transfer to gpu took: ", (now() - start) * 1000, "ms")
    start = now()
    r2 = ag - bg
    print("Overall GPU took: ", (now() - start) * 1000, "ms")
    assert_true(r1.all_close(r2))
