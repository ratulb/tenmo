from .tensor import Tensor
from .backpropagation import (
    BackwardFnArg,
    ScalarArg,
    BACKWARD_MULTIPLY,
    BACKWARD_MULTIPLY_SCALAR,
    BACKWARD_MULTIPLY_BROADCAST,
)
from .mnemonics import AddTensor, Multiply
from .common_utils import panic, id
from .gradbox import Gradbox
from .broadcastbackward import BroadcastBackward
from .ancestry import Ancestor
from std.os.atomic import Atomic, Consistency, fence


@fieldwise_init
struct MultiplyBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var factor = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        scaled_gradbox = gradbox * factor
        return [
            (
                ancestor^,
                scaled_gradbox^,
                AddTensor,
            )
        ]


@fieldwise_init
struct MultiplyBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        count = len(output.ancestry())
        var ancestor_lhs = output.ancestry().get(0)

        if count == 1:  # B = A * A, A is the only ancestor of B
            var gradbox_prod = (
                Gradbox[Self.dtype](ancestor_lhs.buffer(), share=False)
                * gradbox
            )
            gradbox_prod = gradbox_prod * Scalar[Self.dtype](2)
            return [(ancestor_lhs^, gradbox_prod^, AddTensor)]

        var grad_shares = List[
            Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]
        ](capacity=2)

        var ancestor_rhs = output.ancestry().get(1)

        if ancestor_lhs.requires_grad:
            var gradbox_prod = gradbox * Gradbox[Self.dtype](
                ancestor_rhs.buffer(), share=False
            )

            grad_shares.append(
                (
                    ancestor_lhs,
                    gradbox_prod^,
                    AddTensor,
                )
            )

        if ancestor_rhs.requires_grad:
            var gradbox_prod = gradbox * Gradbox[Self.dtype](
                ancestor_lhs.buffer(), share=False
            )
            grad_shares.append(
                (
                    ancestor_rhs^,
                    gradbox_prod^,
                    AddTensor,
                )
            )

        return grad_shares^


comptime MultiplyBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=True,
    lhs_op=AddTensor,
    rhs_op=AddTensor,
]


@fieldwise_init
struct MultiplyScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], factor: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out: Tensor[Self.dtype] = Tensor[Self.dtype](
            self.buffer.scalar_ops[Multiply](factor), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_MULTIPLY_SCALAR, factor
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


# Element wise multiplication of two tensors
@fieldwise_init
struct Multiplicator[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        if not self.broadcastable(other):
            panic(
                "Tensor multiplication dimension mismatch: cannot broadcast"
                " shape "
                + self.shape().__str__()
                + " with "
                + other.shape().__str__(),
                "at Multiplicator → forward",
            )

        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer),
            requires_grad=False,
        )

        comptime if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)

                if self.shape() == other.shape():
                    var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                        BACKWARD_MULTIPLY
                    )
                    if id(self) == id(other):  # B = A * A, self == other == A
                        out.add_ancestry(backwardFnArg^, self)
                    else:
                        out.add_ancestry(backwardFnArg^, self, other)
                else:
                    var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                        BACKWARD_MULTIPLY_BROADCAST
                    )
                    out.add_ancestry(backwardFnArg^, self, other)

        return out^

    @staticmethod
    fn forward(
        self: Tensor[Self.dtype], other: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        var nd_buffer = self.buffer.arithmetic_ops[Multiply](other.buffer)
        return Gradbox[Self.dtype](nd_buffer^, share=False)


from .shapes import Shape


fn main() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].full(Shape.of(3, 3), 2, requires_grad=True)
    var B = Tensor[dtype].full(Shape.of(3, 1), 2, requires_grad=True)
    var C = A * B
    # var C = A * 10
    C.backward(42)
    A.grad().print()  # grad() call detaches
    B.grad().print()
    print("passes")
