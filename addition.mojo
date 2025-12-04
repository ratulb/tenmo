from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_ADD,
    BACKWARD_ADD_SCALAR,
    BACKWARD_ADD_BROADCAST,
)
from operators import AddTensor, Add
from common_utils import panic, id
from gradbox import Gradbox
from broadcastbackward import BroadcastBackward


@fieldwise_init
@register_passable
struct AddBackwardScalar[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_ADD_SCALAR

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        if ancestor.shape() != gradbox.shape():
            gradbox = gradbox.reshape(ancestor.shape())
        # Gradient of addition is 1 → just pass through incoming grad
        return [(ancestor^, gradbox, AddTensor)]


alias AddBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=False,
    lhs_op=AddTensor,
    rhs_op=AddTensor,
    TAG=BACKWARD_ADD_BROADCAST,
]


@fieldwise_init
@register_passable
struct AddBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_ADD

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        var gradbox = output.grad()
        count = len(output.ancestry())

        var grad_shares = List[Tuple[Tensor[dtype], Gradbox[dtype], Int]](
            capacity=UInt(count)
        )

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
@register_passable
struct AddScalar[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], scalar: Scalar[dtype]) -> Tensor[dtype]:
        var out: Tensor[dtype] = Tensor[dtype](
            self.buffer.scalar_ops[Add](scalar), requires_grad=False
        )

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = AddBackwardScalar[dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


# Element wise addition of two tensors - would broadcast if required
@fieldwise_init
@register_passable
struct Adder[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[dtype], other: Tensor[dtype]) -> Tensor[dtype]:
        if id(self) == id(other):
            return self.__mul__(Scalar[dtype](2))
        if not self.broadcastable(other):
            panic(
                "Tensor__add__(self, other): dimension mismatch: "
                + self.shape().__str__()
                + " <=> "
                + other.shape().__str__(),
                "at Addition → forward",
            )

        var out: Tensor[dtype] = Tensor[dtype](
            self.buffer.arithmetic_ops[Add](other.buffer),
            requires_grad=False,
        )

        @parameter
        if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)
                # No broadcasting happened
                if self.shape() == other.shape():
                    backward_fn = AddBackward[dtype]().into_backward_fn()
                    out.backwardFn = Optional(backward_fn^)
                    if self.requires_grad:
                        out.add_ancestry(self)
                    if other.requires_grad:
                        out.add_ancestry(other)
                else:
                    # Broadcasting happened
                    backward_fn = AddBroadcastBackward[
                        dtype
                    ]().into_backward_fn()

                    out.backwardFn = Optional(backward_fn^)
                    out.add_ancestry(self, other)

        return out^


fn main():
    print("passes")
