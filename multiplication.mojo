from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_MULTIPLY,
    BACKWARD_MULTIPLY_SCALAR,
    BACKWARD_MULTIPLY_BROADCAST,
)
from operators import AddTensor, Multiply
from common_utils import panic, id
from gradbox import Gradbox
from broadcastbackward import BroadcastBackward


@fieldwise_init
@register_passable
struct MultiplyBackwardScalar[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_MULTIPLY_SCALAR
    var factor: Scalar[dtype]

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
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
    alias TAG = BACKWARD_MULTIPLY

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        # ref gradbox = output.gradients()[]
        var gradbox = output.grad()
        count = len(output.ancestry())
        var ancestor_lhs = output.ancestry().get(0)

        if count == 1:  # B = A * A, A is the only ancestor of B
            gradbox_prod = ancestor_lhs * gradbox
            gradbox_prod = gradbox_prod * Scalar[dtype](2)
            return [(ancestor_lhs^, gradbox_prod^, AddTensor)]

        var grad_shares = List[Tuple[Tensor[dtype], Gradbox[dtype], Int]](
            capacity=2
        )

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


alias MultiplyBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=True,
    lhs_op=AddTensor,
    rhs_op=AddTensor,
    TAG=BACKWARD_MULTIPLY_BROADCAST,
]


@fieldwise_init
@register_passable
struct MultiplyScalar[dtype: DType]:
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], factor: Scalar[dtype]) -> Tensor[dtype]:
        var out: Tensor[dtype] = Tensor[dtype](
            self.buffer.scalar_ops[Multiply](factor), requires_grad=False
        )

        @parameter
        if track_grad:
            out.requires_grad_(True)
            if self.requires_grad:
                backward_fn = MultiplyBackwardScalar[dtype](
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
    ](self: Tensor[dtype], other: Tensor[dtype]) -> Tensor[dtype]:
        if not self.broadcastable(other):
            panic(
                "Tensor → __mul__(self * other) → dimension mismatch: "
                + self.shape().__str__()
                + " <=> "
                + other.shape().__str__(),
                "at Multiplicator → forward",
            )

        var out: Tensor[dtype] = Tensor[dtype](
            self.buffer.arithmetic_ops[Multiply](other.buffer),
            requires_grad=False,
        )

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)

                if self.shape() == other.shape():
                    backward_fn = MultiplyBackward[dtype]().into_backward_fn()
                    out.backwardFn = Optional(backward_fn^)
                    if id(self) == id(other):  # B = A * A, self == other == A
                        out.add_ancestry(self)
                    else:
                        out.add_ancestry(self, other)
                else:
                    backward_fn = MultiplyBroadcastBackward[
                        dtype
                    ]().into_backward_fn()
                    out.backwardFn = Optional(backward_fn^)
                    out.add_ancestry(self, other)

        return out^

    @staticmethod
    fn forward(self: Tensor[dtype], other: Gradbox[dtype]) -> Gradbox[dtype]:
        var nd_buffer = self.buffer.arithmetic_ops[Multiply](other.buffer)
        return Gradbox[dtype](nd_buffer^, share=False)
