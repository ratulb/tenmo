from tenmo import Tensor
from backpropagation import (
    BackwardFnArg,
    BACKWARD_ADD,
    BACKWARD_ADD_SCALAR,
    BACKWARD_ADD_BROADCAST,
)
from mnemonics import AddTensor, Add
from common_utils import panic
from gradbox import Gradbox
from broadcastbackward import BroadcastBackward


@fieldwise_init
struct AddBackwardScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        if ancestor.shape() != gradbox.shape():
            gradbox = gradbox.reshape(ancestor.shape())
        # Gradient of addition is 1 → just pass through incoming grad
        return [(ancestor^, gradbox, AddTensor)]


comptime AddBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=False,
    lhs_op=AddTensor,
    rhs_op=AddTensor,
]


@fieldwise_init
struct AddBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var gradbox = output.gradients()[]
        count = len(output.ancestry())

        var grad_shares = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ](capacity=count)

        if count == 1:
            var ancestor = output.ancestry().get(0)
            grad_shares.append((ancestor^, gradbox^, AddTensor))
        else:
            var ancestor_lhs = output.ancestry().get(0)
            var ancestor_rhs = output.ancestry().get(1)
            lhs_requires_grad = ancestor_lhs.requires_grad
            rhs_requires_grad = ancestor_rhs.requires_grad

            if lhs_requires_grad and rhs_requires_grad:
                grad_shares.append((ancestor_lhs^, gradbox, AddTensor))
                grad_shares.append((ancestor_rhs^, gradbox^, AddTensor))

            elif lhs_requires_grad and not rhs_requires_grad:
                grad_shares.append((ancestor_lhs^, gradbox^, AddTensor))

            elif not lhs_requires_grad and rhs_requires_grad:
                grad_shares.append((ancestor_rhs^, gradbox^, AddTensor))

            else:
                pass

        return grad_shares^


@fieldwise_init
struct AddScalar[dtype: DType](Copyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[Add](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                out.backwardFnArg = Optional(
                    BackwardFnArg[Self.dtype].null_arg(BACKWARD_ADD_SCALAR)
                )
                out.add_ancestry(self)

        return out^


# Element wise addition of two tensors - would broadcast if required
@fieldwise_init
struct Adder[dtype: DType](Copyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        if not self.broadcastable(other):
            panic(
                "Tensor addition dimension mismatch: cannot broadcast shape "
                + String(self.shape())
                + " with "
                + String(other.shape()),
                "at Adder → forward",
            )

        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Add](other.buffer),
            requires_grad=False,
        )

        comptime if track_grad:
            if self.requires_grad or other.requires_grad:
                out.requires_grad_(True)

                if self.shape() == other.shape():
                    out.backwardFnArg = Optional(
                        BackwardFnArg[Self.dtype].null_arg(BACKWARD_ADD)
                    )
                    if self.requires_grad:
                        out.add_ancestry(self)
                    if other.requires_grad:
                        out.add_ancestry(other)
                else:
                    out.backwardFnArg = Optional(
                        BackwardFnArg[Self.dtype].null_arg(
                            BACKWARD_ADD_BROADCAST
                        )
                    )
                    out.add_ancestry(self, other)

        return out^


from common_utils import now
from std.testing import assert_true
from shapes import Shape


fn main() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].full(Shape.of(3, 3), 2, requires_grad=True)
    var B = Tensor[dtype].full(Shape.of(3, 1), 2, requires_grad=True)
    var C = A + B
    C.backward()
    A.grad().print()
