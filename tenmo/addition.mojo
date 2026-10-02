from .tensor import Tensor
from .backpropagation import (
    BackwardFnType,
    BackwardFn,
)
from .shared.mnemonics import AddTensor, Add
from .shared.panic import panic
from .broadcast import BroadcastBackward
from .ancestry import Ancestor


@fieldwise_init
struct AddBackwardScalar[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        if not output.parents:
            panic("Addition add scalar backward: parent_refs is None!")
        var parent = output.ancestry().get(0)
        if parent.requires_grad:
            ref gradbox = output.gradients()
            # No reshape: scalar-add preserves shape exactly, and every
            # shape-changing downstream backward restores out-shape before
            # accumulating (same contract as AddBackward's count==1 path).
            parent.update_grad(gradbox, AddTensor, None)
        parent_ids.append(parent._id)
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
    ](
        self: Tensor[Self.dtype], scalar: Scalar[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[Add](scalar, sync=sync), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.null_arg[Self.dtype](
                    AddBackwardScalar[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)
        return out^


# Element wise addition of two tensors - would broadcast if required
@fieldwise_init
struct Adder[dtype: DType](Copyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype], other: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if not self.broadcastable(other):
            panic(
                "Tensor addition dimension mismatch: cannot broadcast shape "
                + String(self.shape())
                + " with "
                + String(other.shape()),
                "at Adder → forward",
            )

        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Add](other.buffer, sync=sync),
            requires_grad=False,
        )

        comptime if track_grad:
            if self.requires_grad or other.requires_grad:
                out.requires_grad_(True)
                if self.shape() == other.shape():
                    var backwardFn = BackwardFn.null_arg[Self.dtype](
                        AddBackward[Self.dtype]()
                    )
                    if self.requires_grad and other.requires_grad:
                        out.add_ancestry(backwardFn^, self, other)
                    elif self.requires_grad:
                        out.add_ancestry(backwardFn^, self)
                    else:
                        out.add_ancestry(backwardFn^, other)
                else:
                    var backwardFn = BackwardFn.null_arg[Self.dtype](
                        AddBroadcastBackward[Self.dtype](),
                    )
                    backwardFn.needs_parent_data = True
                    # Both parents: BroadcastBackward unconditionally gets
                    # ancestry(0) and ancestry(1); per-parent compute is
                    # already guarded on requires_grad inside.
                    out.add_ancestry(backwardFn^, self, other)

        return out^


@fieldwise_init
struct AddBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var gradbox = output.gradients()
        var count = len(output.ancestry())

        if count == 1:
            var ancestor = output.ancestry().get(0)
            ancestor.update_grad(gradbox^, AddTensor, None)
            parent_ids.append(ancestor._id)
        else:
            var ancestor_lhs = output.ancestry().get(0)
            var ancestor_rhs = output.ancestry().get(1)
            var lhs_requires_grad = ancestor_lhs.requires_grad
            var rhs_requires_grad = ancestor_rhs.requires_grad

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
        gradbox.zero_grad()
