from tenmo import Tensor
from intarray import IntArray
from mnemonics import AddTensor, SubtractTensor, Subtract, ReverseSubtract
from backpropagation import (
    BackwardFnArg,
    Boolean,
    IntArrayArg,
    BACKWARD_SUB,
    BACKWARD_SUB_SCALAR,
    BACKWARD_SUBTRACT_BROADCAST,
)
from common_utils import panic
from gradbox import Gradbox
from broadcastbackward import BroadcastBackward


@fieldwise_init
struct SubBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var signs = output.backward_fn_arg().get[IntArrayArg]().array
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
                    AddTensor if signs[i] == 0 else SubtractTensor,
                )
            )
        return grad_shares^


@fieldwise_init
struct SubLeftRightBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var negate = output.backward_fn_arg().get[Boolean]().is_true
        ref gradbox = output.gradients()[]
        ref ancestor = output.ancestry().get(0)
        return [
            (
                ancestor,
                gradbox,
                SubtractTensor if negate else AddTensor,
            )
        ]


comptime SubtractBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=False,
    lhs_op=AddTensor,
    rhs_op=SubtractTensor,
]


@fieldwise_init
struct SubtractScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[Subtract](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                out.backwardFnArg = Optional(
                    BackwardFnArg[Self.dtype].boolean_arg(
                        BACKWARD_SUB_SCALAR, False
                    )
                )
                out.add_ancestry(self)

        return out^


@fieldwise_init
struct SubtractFromScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[ReverseSubtract](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                out.backwardFnArg = Optional(
                    BackwardFnArg[Self.dtype].boolean_arg(
                        BACKWARD_SUB_SCALAR, True
                    )
                )
                out.add_ancestry(self)

        return out^


@fieldwise_init
struct Subtractor[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        if not self.broadcastable(other):
            panic(
                "Tensor subtraction dimension mismatch: cannot broadcast shape "
                + String(self.shape())
                + " with "
                + String(other.shape()),
                "at Subtractor → forward",
            )

        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer),
            requires_grad=False,
        )

        comptime if track_grad:
            requires_grad = self.requires_grad or other.requires_grad

            if requires_grad:
                out.requires_grad_(True)

                if self.shape() == other.shape():
                    var signs = IntArray()
                    if self.requires_grad:
                        out.add_ancestry(self)
                        signs.append(0)
                    if other.requires_grad:
                        out.add_ancestry(other)
                        signs.append(1)
                    out.backwardFnArg = Optional(
                        BackwardFnArg[Self.dtype].from_intarray(
                            BACKWARD_SUB, signs
                        )
                    )
                else:
                    out.add_ancestry(self, other)
                    out.backwardFnArg = Optional(
                        BackwardFnArg[Self.dtype].null_arg(
                            BACKWARD_SUBTRACT_BROADCAST
                        )
                    )
        return out^

fn main() raises:
    print("passes")
