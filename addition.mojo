from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_ADD,
    BACKWARD_ADD_SCALAR,
    BACKWARD_ADD_BROADCAST,
)
from mnemonics import AddTensor, Add
from common_utils import panic
from gradbox import Gradbox
from broadcastbackward import BroadcastBackward
from sys import has_accelerator
from scalar_forward import ScalarOperation
from binary_forward import BinaryOperation


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
        var out = ScalarOperation[Self.dtype].forward[Add](self, scalar)

        @parameter
        if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                backward_fn = AddBackwardScalar[Self.dtype]().into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                if self.is_on_cpu():
                    out.add_ancestry(self)
                else:
                    print("GPU tensor origin is being added")
                    out.add_ancestry(self.ancestry().origin())

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
                "at Adder → forward",
            )

        var out = BinaryOperation[Self.dtype].forward[Add](self, other)

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


from common_utils import now
from testing import assert_true
from shapes import Shape

fn main() raises:
    comptime dtype = DType.float32
    _="""a = Tensor[dtype].arange(5000000)
    b = Tensor[dtype].arange(5000000)
    start = now()
    r1 = a - b
    print("CPU took: ", (now() - start) * 1000, "ms")
    start = now()
    ag = a.to_gpu()
    bg = b.to_gpu()
    print("to_gpu took: ", (now() - start) * 1000, "ms")
    start = now()
    r2 = ag - bg
    print("Overall GPU took: ", (now() - start) * 1000, "ms")
    assert_true(r1.all_close(r2))
    print()"""
    print("The meaty part")

    A = Tensor[dtype].full(Shape.of(3, 3), 2, requires_grad=True)
    print("A's id: ", A.id())
    a = A.to_gpu(requires_grad=True)
    a.ancestry().print()
    expected = Tensor[dtype].full(Shape.of(3, 3), 2) + 42
    b = a + 42
    assert_true(b.all_close(expected), "Scalar add assertion failed")
    b.ancestry().print()
    b.backward()
    a.grad().print()
