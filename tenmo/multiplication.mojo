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
from .broadcast import BroadcastBackward
from .ancestry import Ancestor
from std.atomic import Atomic, Ordering, fence


@fieldwise_init
struct MultiplyBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var factor = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        scaled_gradbox = gradbox * factor
        ancestor.update_grad(scaled_gradbox^, AddTensor, None)
        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct MultiplyBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref gradbox = output.gradients()
        count = len(output.ancestry())
        var ancestor_lhs = output.ancestry().get(0)

        if count == 1:  # B = A * A, A is the only ancestor of B
            var gradbox_prod = (
                Gradbox[Self.dtype](ancestor_lhs.buffer())
                * gradbox
            )
            gradbox_prod = gradbox_prod * Scalar[Self.dtype](2)
            ancestor_lhs.update_grad(gradbox_prod^, AddTensor, None)
            parent_ids.append(ancestor_lhs._id)
            if not retain_graph:
                gradbox.zero_grad()
            return

        var ancestor_rhs = output.ancestry().get(1)

        if ancestor_lhs.requires_grad:
            var gradbox_prod = gradbox * Gradbox[Self.dtype](
                ancestor_rhs.buffer(), 
            )
            ancestor_lhs.update_grad(gradbox_prod^, AddTensor, None)
            parent_ids.append(ancestor_lhs._id)

        if ancestor_rhs.requires_grad:
            var gradbox_prod = gradbox * Gradbox[Self.dtype](
                ancestor_lhs.buffer(), 
            )
            ancestor_rhs.update_grad(gradbox_prod^, AddTensor, None)
            parent_ids.append(ancestor_rhs._id)
        if not retain_graph:
            gradbox.zero_grad()


comptime MultiplyBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=True,
    lhs_op=AddTensor,
    rhs_op=AddTensor,
]


@fieldwise_init
struct MultiplyScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], factor: Scalar[Self.dtype], sync: Bool = True) -> Tensor[
        Self.dtype
    ]:
        var out: Tensor[Self.dtype] = Tensor[Self.dtype](
            self.buffer.scalar_ops[Multiply](factor, sync=sync), requires_grad=False
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
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype], sync: Bool = True) -> Tensor[
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
            self.buffer.arithmetic_ops[Multiply](other.buffer, sync=sync),
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
                    backwardFnArg.needs_parent_data = True
                    if id(self) == id(other):  # B = A * A, self == other == A
                        out.add_ancestry(backwardFnArg^, self)
                    else:
                        out.add_ancestry(backwardFnArg^, self, other)
                else:
                    var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                        BACKWARD_MULTIPLY_BROADCAST
                    )
                    backwardFnArg.needs_parent_data = True
                    out.add_ancestry(backwardFnArg^, self, other)

        return out^

    @staticmethod
    def forward(
        self: Tensor[Self.dtype], other: Gradbox[Self.dtype], sync: Bool = True
    ) -> Gradbox[Self.dtype]:
        var nd_buffer = self.buffer.arithmetic_ops[Multiply](other.buffer(), sync=sync)
        return Gradbox[Self.dtype](nd_buffer^)
