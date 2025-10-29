from tenmo import Tensor
from backpropagation import Delegate, BackwardFn
from operators import AddTensor, Add, scalar_ops

# from broadcastbackward import BroadcastBackward
from common_utils import panic
from gradbox import Gradbox
from ancestry import Ancestor


@fieldwise_init
@register_passable
struct AddBackwardScalar[dtype: DType](ImplicitlyCopyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        ancestor = output.ancestry().get(0)
        if ancestor.shape() != gradbox.shape():
            gradbox = gradbox.reshape(ancestor.shape())
        # Gradient of addition is 1 → just pass through incoming grad
        return [(ancestor^, gradbox^, AddTensor)]


@fieldwise_init
@register_passable
struct AddBroadcastBackward[dtype: DType](ImplicitlyCopyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        var grad_shares = List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]](
            capacity=2
        )

        ancestor_lhs = output.ancestry().get(0)
        ancestor_rhs = output.ancestry().get(1)
        lhs_requires_grad = ancestor_lhs.requires_grad()
        rhs_requires_grad = ancestor_rhs.requires_grad()

        tensor_lhs = ancestor_lhs.tensor()
        tensor_rhs = ancestor_rhs.tensor()

        if ancestor_lhs.requires_grad():
            lhs_share = tensor_lhs.upstream_grad_share[augment=False](
                tensor_rhs, gradbox.as_tensor(requires_grad=False)
            )
            grad_shares.append(
                (
                    ancestor_lhs^,
                    lhs_share^.as_gradbox(share=False),
                    AddTensor,
                )
            )

        if ancestor_rhs.requires_grad():
            rhs_share = tensor_rhs.upstream_grad_share[augment=False](
                tensor_lhs, gradbox.as_tensor(requires_grad=False)
            )
            grad_shares.append(
                (
                    ancestor_rhs^,
                    rhs_share^.as_gradbox(share=False),
                    AddTensor,
                )
            )

        return grad_shares


@fieldwise_init
@register_passable
struct AddBackward[dtype: DType](ImplicitlyCopyable):
    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self))

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Ancestor[dtype], Gradbox[dtype], Int]]:
        gradbox = output.grad().copy()
        count = len(output.ancestry())

        var grad_shares = List[Tuple[TensorLite[dtype], Tensor[dtype], Int]](
            cacpacity=UInt(count)
        )

        if count == 1:
            ancestor = output.ancestry().get(0)
            grad.shares.append((ancestor^, gradbox^, AddTensor))
        else:
            ancestor_lhs = output.ancestry().get(0)
            ancestor_rhs = output.ancestry().get(1)
            lhs_requires_grad = ancestor_lhs.requires_grad()
            rhs_requires_grad = ancestor_rhs.requires_grad()

            if lhs_requires_grad and rhs_requires_grad:
                grad_shares.append((ancestor_lhs^, gradbox.copy(), AddTensor))
                grad_shares.append((ancestor_rhs^, gradbox^, AddTensor))

            elif lhs_requires_grad and not rhs.requires_grad:
                grad_shares.append((ancestor_lhs^, gradbox^, AddTensor))

            elif not lhs_requires_grad and rhs.requires_grad:
                grad_shares.append((ancestor_rhs^, gradbox^, AddTensor))

            else:
                pass

        return grad_shares


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
        if self.address() == other.address():
            # return self.__mul__(2)
            pass
        _ = """if not self.broadcastable(other):
            panic(
                "Tensor__add__(self, other): dimension mismatch: "
                + self.shape.__str__()
                + " <=> "
                + other.shape.__str__(),
                "at Addition → forward",
            )"""

        var out: Tensor[dtype] = Tensor[dtype](
            self.buffer.arithmetic_ops[Add](other.buffer),
            requires_grad=False,
        )
        # if self.shape != other.shape:
        # out = self.broadcast_op(other, scalar_ops[dtype, Add])
        # else:
        _ = """if self.owns_data and other.owns_data:
            buffer = self.buffer + other.buffer
            out = Tensor[dtype](self.shape, buffer^, requires_grad=False)
        else:
            out = Tensor[dtype].zeros(self.shape, requires_grad=False)
            for coord in self.shape:
                out[coord] = self[coord] + other[coord]"""

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


fn main():
    pass
