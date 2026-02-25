from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_ADD,
    BACKWARD_ADD_SCALAR,
    BACKWARD_ADD_BROADCAST,
)
from mnemonics import AddTensor, Add
from common_utils import panic, id
from gradbox import Gradbox
from broadcastbackward import BroadcastBackward
from sys import has_accelerator
from arithmetic_ops_kernel import ArithmeticOpsKernel


@fieldwise_init
@register_passable
struct AddBackwardScalar[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_ADD_SCALAR

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
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
    TAG=BACKWARD_ADD_BROADCAST,
]


@fieldwise_init
@register_passable
struct AddBackward[dtype: DType](ImplicitlyCopyable):
    comptime TAG = BACKWARD_ADD

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, read output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var gradbox = output.grad()
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
@register_passable
struct AddScalar[dtype: DType](Copyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out: Tensor[Self.dtype] = Tensor[Self.dtype](
            self.buffer.scalar_ops[Add](scalar), requires_grad=False
        )

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = AddBackwardScalar[Self.dtype]().into_backward_fn()
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
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        if not self.broadcastable(other):
            panic(
                "Tensor addition dimension mismatch: cannot broadcast shape "
                + self.shape().__str__()
                + " with "
                + other.shape().__str__(),
                "at Adder.forward",
            )

        var out: Tensor[Self.dtype]

        @parameter
        if has_accelerator():
            if self.is_on_gpu() and other.is_on_gpu():
                try:
                    out = ArithmeticOpsKernel[Self.dtype].launch[Add](
                        self, other
                    )
                except e:
                    print(e)
                    print("Adder - GPU operation failed. Failling back on CPU")
                    out = Tensor[Self.dtype](
                        self.buffer.arithmetic_ops[Add](other.buffer),
                        requires_grad=False,
                    )
            else:
                out = Tensor[Self.dtype](
                    self.buffer.arithmetic_ops[Add](other.buffer),
                    requires_grad=False,
                )
        else:
            out = Tensor[Self.dtype](
                self.buffer.arithmetic_ops[Add](other.buffer),
                requires_grad=False,
            )

        @parameter
        if track_grad:
            if self.requires_grad or other.requires_grad:
                out.requires_grad_(True)

                if self.shape() == other.shape():
                    var bwd = AddBackward[Self.dtype]().into_backward_fn()
                    out.backwardFn = Optional(bwd^)
                    if self.requires_grad:
                        out.add_ancestry(self)
                    if other.requires_grad:
                        out.add_ancestry(other)
                else:
                    var bwd = AddBroadcastBackward[
                        Self.dtype
                    ]().into_backward_fn()
                    out.backwardFn = Optional(bwd^)
                    out.add_ancestry(self, other)

        return out^
