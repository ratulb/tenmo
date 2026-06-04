from .tensor import Tensor
from .backpropagation import (
    BackwardFnArg,
    BACKWARD_ADD,
    BACKWARD_ADD_SCALAR,
    BACKWARD_ADD_BROADCAST,
)
from .mnemonics import AddTensor, Add
from .common_utils import panic
from .gradbox import Gradbox
from .broadcast import BroadcastBackward
from .ancestry import Ancestor


@fieldwise_init
struct AddBackwardScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        if not output.parents:
            panic("Addition add scalar backward: parent_refs is None!")
        ref parent = output.ancestry().get(0)
        if parent.requires_grad:
            ref gradbox = output.gradients()
            if parent.shape() != gradbox.shape():
                var reshaped = gradbox.reshape(parent.shape())
                parent.update_grad(reshaped, AddTensor, None)
            else:
                parent.update_grad(gradbox, AddTensor, None)
        parent_ids.append(parent._id)
        if not retain_graph:
            output.gradients().zero_grad()


comptime AddBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=False,
    lhs_op=AddTensor,
    rhs_op=AddTensor,
]


@fieldwise_init
struct AddScalar[dtype: DType](Copyable, RegisterPassable):
    @staticmethod
    def forward[
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
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_ADD_SCALAR
                )
                backwardFnArg.needs_parent_data = True
                out.add_ancestry(backwardFnArg^, self)
        return out^


# Element wise addition of two tensors - would broadcast if required
@fieldwise_init
struct Adder[dtype: DType](Copyable, RegisterPassable):
    @staticmethod
    def forward[
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
                    var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                        BACKWARD_ADD
                    )
                    if self.requires_grad and other.requires_grad:
                        out.add_ancestry(backwardFnArg^, self, other)
                    elif self.requires_grad:
                        out.add_ancestry(backwardFnArg^, self)
                    else:
                        out.add_ancestry(backwardFnArg^, other)
                else:
                    var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                        BACKWARD_ADD_BROADCAST
                    )
                    backwardFnArg.needs_parent_data = True
                    out.add_ancestry(backwardFnArg^, self, other)

        return out^


@fieldwise_init
struct AddBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var gradbox = output.gradients()
        count = len(output.ancestry())

        if count == 1:
            var ancestor = output.ancestry().get(0)
            ancestor.update_grad(gradbox^, AddTensor, None)
            parent_ids.append(ancestor._id)
        else:
            var ancestor_lhs = output.ancestry().get(0)
            var ancestor_rhs = output.ancestry().get(1)
            lhs_requires_grad = ancestor_lhs.requires_grad
            rhs_requires_grad = ancestor_rhs.requires_grad

            if lhs_requires_grad and rhs_requires_grad:
                ancestor_lhs.update_grad(gradbox, AddTensor, None)
                parent_ids.append(ancestor_lhs._id)
                ancestor_rhs.update_grad(gradbox, AddTensor, None)
                parent_ids.append(ancestor_rhs._id)

            elif lhs_requires_grad and not rhs_requires_grad:
                ancestor_lhs.update_grad(gradbox, AddTensor, None)
                parent_ids.append(ancestor_lhs._id)

            elif not lhs_requires_grad and rhs_requires_grad:
                ancestor_rhs.update_grad(gradbox, AddTensor, None)
                parent_ids.append(ancestor_rhs._id)

            else:
                pass
        if not retain_graph:
            gradbox.zero_grad()
