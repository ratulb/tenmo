from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_MULTIPLY,
    BACKWARD_MULTIPLY_SCALAR,
    BACKWARD_MULTIPLY_BROADCAST,
)
from mnemonics import AddTensor, Multiply
from common_utils import panic, id
from gradbox import Gradbox
from broadcastbackward import BroadcastBackward
from sys import has_accelerator
from scalar_forward import ScalarOperation
from binary_forward import BinaryOperation


@fieldwise_init
@register_passable
struct MultiplyBackwardScalar[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_MULTIPLY_SCALAR
    var factor: Scalar[Self.dtype]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        scaled_gradbox = gradbox * self.factor

        return [
            (
                ancestor^,
                scaled_gradbox^,
                AddTensor,
            )
        ]


@fieldwise_init
@register_passable
struct MultiplyBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_MULTIPLY

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        # ref gradbox = output.gradients()[]
        var gradbox = output.grad()
        count = len(output.ancestry())
        var ancestor_lhs = output.ancestry().get(0)

        if count == 1:  # B = A * A, A is the only ancestor of B
            gradbox_prod = ancestor_lhs * gradbox
            gradbox_prod = gradbox_prod * Scalar[Self.dtype](2)
            return [(ancestor_lhs^, gradbox_prod^, AddTensor)]

        var grad_shares = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ](capacity=2)

        var ancestor_rhs = output.ancestry().get(1)

        if ancestor_lhs.requires_grad:
            var gradbox_prod = gradbox * ancestor_rhs

            grad_shares.append(
                (
                    ancestor_lhs,
                    gradbox_prod^,
                    AddTensor,
                )
            )

        if ancestor_rhs.requires_grad:
            var gradbox_prod = gradbox * ancestor_lhs
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
    TAG=BACKWARD_MULTIPLY_BROADCAST,
]


@fieldwise_init
@register_passable
struct MultiplyScalar[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], factor: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out = ScalarOperation[Self.dtype].forward[Multiply](self, factor)

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = MultiplyBackwardScalar[Self.dtype](
                    factor
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


# Element wise multiplication of two tensors
@fieldwise_init
@register_passable
struct Multiplicator[dtype: DType]:
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

        var out = BinaryOperation[Self.dtype].forward[Multiply](self, other)

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)

                if self.shape() == other.shape():
                    backward_fn = MultiplyBackward[
                        Self.dtype
                    ]().into_backward_fn()
                    out.backwardFn = Optional(backward_fn^)
                    if id(self) == id(other):  # B = A * A, self == other == A
                        out.add_ancestry(self)
                    else:
                        out.add_ancestry(self, other)
                else:
                    backward_fn = MultiplyBroadcastBackward[
                        Self.dtype
                    ]().into_backward_fn()
                    out.backwardFn = Optional(backward_fn^)
                    out.add_ancestry(self, other)

        return out^

    @staticmethod
    fn forward(
        self: Tensor[Self.dtype], other: Gradbox[Self.dtype]
    ) -> Gradbox[Self.dtype]:
        var nd_buffer = self.buffer.arithmetic_ops[Multiply](other.buffer)
        return Gradbox[Self.dtype](nd_buffer^, share=False)


from common_utils import now
from testing import assert_true


fn main() raises:
    comptime dtype = DType.float32
    a = Tensor[dtype].arange(5000000)
    b = Tensor[dtype].arange(5000000)
    start = now()
    r1 = a * b
    print("CPU took: ", (now() - start) * 1000, "ms")
    start = now()
    ag = a.to_gpu()
    bg = b.to_gpu()
    print("to_gpu took: ", (now() - start) * 1000, "ms")
    start = now()
    r2 = ag * bg
    print("Overall GPU took: ", (now() - start) * 1000, "ms")
    assert_true(r1.all_close(r2))
