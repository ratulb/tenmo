from tenmo import Tensor
from intarray import IntArray
from mnemonics import AddTensor, SubtractTensor, Subtract, ReverseSubtract
from backpropagation import (
    SubtractArg,
    BooleanArg,
    FnArg,
    BACKWARD_SUB_SCALAR,
)
from common_utils import panic
from gradbox import Gradbox
from broadcastbackward import BroadcastBackward

@fieldwise_init
struct SubBackward[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var signs = output.fn_arg().arg[SubtractArg].signs
        if len(signs) == 0:
            return SubtractBroadcastBackward[Self.dtype].backward(output)
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
struct SubLeftRightBackwardScalar[dtype: DType](RegisterPassable, ImplicitlyCopyable):

    @staticmethod
    fn backward(
        output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var negate = output.fn_arg().arg[BooleanArg].is_true
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
struct SubtractScalar[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                out.fnArg = Optional(FnArg[Self.dtype].boolean(False, BACKWARD_SUB_SCALAR))
                out.add_ancestry(self)

        return out^


@fieldwise_init
struct SubtractFromScalar[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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
                out.fnArg = Optional(FnArg[Self.dtype].boolean(True, BACKWARD_SUB_SCALAR))
                out.add_ancestry(self)

        return out^


@fieldwise_init
struct Subtractor[dtype: DType](RegisterPassable, ImplicitlyCopyable):
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

                var subract_arg = SubtractArg()
                if self.shape() == other.shape():
                    if self.requires_grad:
                        out.add_ancestry(self)
                        subract_arg.negate(False)
                    if other.requires_grad:
                        out.add_ancestry(other)
                        subract_arg.negate(True)
                else:
                    out.add_ancestry(self, other)

                out.fnArg = Optional(subract_arg.into_arg[Self.dtype]())
        return out^


from common_utils import now
from std.testing import assert_true


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
    assert_true(r1.all_close(r2.to_cpu()))
