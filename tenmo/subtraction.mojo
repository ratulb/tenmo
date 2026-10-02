from .tensor import Tensor
from .shared.intarray import IntArray
from .shared.mnemonics import (
    AddTensor,
    SubtractTensor,
    Subtract,
    ReverseSubtract,
)
from .backpropagation import (
    BackwardFnType,
    BackwardFn,
    Boolean,
    IntArrayArg,
)
from .shared.panic import panic
from .gradbox import Gradbox
from .broadcast import BroadcastBackward
from .ancestry import Ancestor


@fieldwise_init
struct SubBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var signs = output.ancestry().backward_fn().get[IntArrayArg]().array
        ref gradbox = output.gradients()
        var count = len(output.ancestry())
        for i in range(count):
            var ancestor = output.ancestry().get(i)
            var op_code = AddTensor if signs[i] == 0 else SubtractTensor
            ancestor.update_grad(gradbox, op_code, None)
            parent_ids.append(ancestor._id)
        gradbox.zero_grad()


@fieldwise_init
struct SubLeftRightBackwardScalar[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var negate = output.ancestry().backward_fn().get[Boolean]().is_true
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        var op_code = SubtractTensor if negate else AddTensor
        ancestor.update_grad(gradbox, op_code, None)
        parent_ids.append(ancestor._id)
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
    ](
        self: Tensor[Self.dtype], scalar: Scalar[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[Subtract](scalar, sync=sync),
            requires_grad=False,
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.boolean_arg[Self.dtype](
                    False,
                    SubLeftRightBackwardScalar[Self.dtype](),
                )
                out.add_ancestry(backwardFn^, self)

        return out^


@fieldwise_init
struct SubtractFromScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype], scalar: Scalar[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[ReverseSubtract](scalar, sync=sync),
            requires_grad=False,
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.boolean_arg[Self.dtype](
                    True,
                    SubLeftRightBackwardScalar[Self.dtype](),
                )
                out.add_ancestry(backwardFn^, self)

        return out^


@fieldwise_init
struct Subtractor[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype], other: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if not self.broadcastable(other):
            panic(
                "Tensor subtraction dimension mismatch: cannot broadcast shape "
                + String(self.shape())
                + " with "
                + String(other.shape()),
                "at Subtractor → forward",
            )

        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer, sync=sync),
            requires_grad=False,
        )

        comptime if track_grad:
            var requires_grad = self.requires_grad or other.requires_grad

            if requires_grad:
                out.requires_grad_(True)

                if self.shape() == other.shape():
                    var signs = IntArray()
                    var backwardFn: BackwardFn

                    if self.requires_grad and other.requires_grad:
                        signs.append(0, 1)
                        backwardFn = BackwardFn.from_intarray[Self.dtype](
                            signs, SubBackward[Self.dtype]()
                        )
                        out.add_ancestry(backwardFn^, self, other)
                    elif self.requires_grad:
                        signs.append(0)
                        backwardFn = BackwardFn.from_intarray[Self.dtype](
                            signs, SubBackward[Self.dtype]()
                        )
                        out.add_ancestry(backwardFn^, self)
                    else:
                        signs.append(1)
                        backwardFn = BackwardFn.from_intarray[Self.dtype](
                            signs, SubBackward[Self.dtype]()
                        )

                        out.add_ancestry(backwardFn^, other)

                else:
                    var backwardFn = BackwardFn.null_arg[Self.dtype](
                        SubtractBroadcastBackward[Self.dtype](),
                    )
                    backwardFn.needs_parent_data = True
                    out.add_ancestry(backwardFn^, self, other)

        return out^

    @staticmethod
    def forward(
        self: Tensor[Self.dtype], other: Gradbox[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        # No shape check here: arithmetic_ops validates broadcastability
        # (same as Multiplicator's Tensor-Gradbox path).
        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Subtract](other.buffer(), sync=sync),
            requires_grad=False,
        )

        return out^
