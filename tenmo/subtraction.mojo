from .tensor import Tensor
from .intarray import IntArray
from .mnemonics import AddTensor, SubtractTensor, Subtract, ReverseSubtract
from .backpropagation import (
    BackwardFnArg,
    Boolean,
    IntArrayArg,
    BACKWARD_SUB,
    BACKWARD_SUB_SCALAR,
    BACKWARD_SUBTRACT_BROADCAST,
)
from .common_utils import panic
from .gradbox import Gradbox
from .broadcast import BroadcastBackward
from .ancestry import Ancestor


@fieldwise_init
struct SubBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var signs = output.ancestry().backward_fn_arg().get[IntArrayArg]().array
        ref gradbox = output.gradients()
        count = len(output.ancestry())
        for i in range(count):
            var ancestor = output.ancestry().get(i)
            var op_code = AddTensor if signs[i] == 0 else SubtractTensor
            ancestor.update_grad(gradbox, op_code, None)
            parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct SubLeftRightBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var negate = output.ancestry().backward_fn_arg().get[Boolean]().is_true
        ref gradbox = output.gradients()
        ref ancestor = output.ancestry().get(0)
        var op_code = SubtractTensor if negate else AddTensor
        ancestor.update_grad(gradbox, op_code, None)
        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


comptime SubtractBroadcastBackward[dtype: DType] = BroadcastBackward[
    dtype,
    augment=False,
    lhs_op=AddTensor,
    rhs_op=SubtractTensor,
]


@fieldwise_init
struct SubtractScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[Subtract](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].boolean_arg(
                    BACKWARD_SUB_SCALAR, False
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


@fieldwise_init
struct SubtractFromScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[ReverseSubtract](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].boolean_arg(
                    BACKWARD_SUB_SCALAR, True
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


@fieldwise_init
struct Subtractor[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        if not self.broadcastable(other):
            panic(
                "Tensor subtraction dimension mismatch: cannot broadcast shape "
                + String(self.shape())
                + " with "
                + String(other.shape()),
                "at Subtractor → forward",
            )

        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer),
            requires_grad=False,
        )

        comptime if track_grad:
            requires_grad = self.requires_grad or other.requires_grad

            if requires_grad:
                out.requires_grad_(True)

                if self.shape() == other.shape():
                    var signs = IntArray()
                    var backwardFnArg: BackwardFnArg[Self.dtype]

                    if self.requires_grad and other.requires_grad:
                        signs.append(0, 1)
                        backwardFnArg = BackwardFnArg[Self.dtype].from_intarray(
                            BACKWARD_SUB, signs
                        )
                        out.add_ancestry(backwardFnArg^, self, other)
                    elif self.requires_grad:
                        signs.append(0)
                        backwardFnArg = BackwardFnArg[Self.dtype].from_intarray(
                            BACKWARD_SUB, signs
                        )
                        out.add_ancestry(backwardFnArg^, self)
                    else:
                        signs.append(1)
                        backwardFnArg = BackwardFnArg[Self.dtype].from_intarray(
                            BACKWARD_SUB, signs
                        )

                        out.add_ancestry(backwardFnArg^, other)

                else:
                    var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                        BACKWARD_SUBTRACT_BROADCAST
                    )
                    out.add_ancestry(backwardFnArg^, self, other)

        return out^

    @staticmethod
    def forward(
        self: Tensor[Self.dtype], other: Gradbox[Self.dtype]
    ) -> Tensor[Self.dtype]:
        if self.shape() != other.shape():
            panic(
                "Tensor subtraction(gradbox) dimension mismatch: shapes don't"
                " match "
                + String(self.shape())
                + " with "
                + String(other.shape()),
                "at Subtractor → forward",
            )

        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer),
            requires_grad=False,
        )

        return out^
