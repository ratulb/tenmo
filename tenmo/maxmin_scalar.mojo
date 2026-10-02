from .tensor import Tensor
from .shared.mnemonics import AddTensor, MAX, MIN
from .backpropagation import (
    BackwardFnType,
    BackwardFn,
    ScalarArg,
)
from .gradbox import Gradbox
from std.sys import has_accelerator
from .ndbuffer import NDBuffer
from .shared.mnemonics import GreaterThan, LessThan, Equal, Add, Multiply
from .ancestry import Ancestor


@fieldwise_init
struct MaxBackwardScalar[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var scalar = (
            output.ancestry().backward_fn().get[ScalarArg[Self.dtype]]().value
        )
        ref gradbox = output.gradients()
        var parent_ref = output.ancestry().get(0)
        var parent = Tensor[Self.dtype](
            parent_ref.buffer(), requires_grad=parent_ref.requires_grad
        )

        # Work at NDBuffer level — avoids pulling in GPU kernel launchers
        # Tie convention mirrors torch.maximum with an untracked scalar
        # operand (verified torch 2.11.0+cpu): grad 1 where x > s, 0.5
        # where x == s (the other half flows to the scalar), 0 below.
        var gt_bool: NDBuffer[DType.bool]
        var eq_bool: NDBuffer[DType.bool]

        comptime if has_accelerator():
            if parent.is_on_gpu():
                gt_bool = parent.buffer.compare_scalar[GreaterThan](scalar)
                eq_bool = parent.buffer.compare_scalar[Equal](scalar)
            else:
                gt_bool = parent.buffer.compare_scalar_cpu[GreaterThan](
                    scalar
                )
                eq_bool = parent.buffer.compare_scalar_cpu[Equal](scalar)
        else:
            gt_bool = parent.buffer.compare_scalar_cpu[GreaterThan](scalar)
            eq_bool = parent.buffer.compare_scalar_cpu[Equal](scalar)

        var mask_gt = gt_bool.to_dtype[Self.dtype]()
        var mask_eq = eq_bool.to_dtype[Self.dtype]()
        var mask = mask_gt.arithmetic_ops[Add](
            mask_eq.scalar_ops[Multiply](Scalar[Self.dtype](0.5))
        )
        # wrap mask as Gradbox and multiply
        var grad_input = Gradbox[Self.dtype](
            mask.arithmetic_ops[Multiply](gradbox.buffer()),
        )

        parent_ref.update_grad(grad_input^, AddTensor, None)
        parent_ids.append(parent_ref._id)

        gradbox.zero_grad()


@fieldwise_init
struct MinBackwardScalar[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var scalar = (
            output.ancestry().backward_fn().get[ScalarArg[Self.dtype]]().value
        )
        ref gradbox = output.gradients()
        var parent_ref = output.ancestry().get(0)
        var parent = Tensor[Self.dtype](
            parent_ref.buffer(), requires_grad=parent_ref.requires_grad
        )

        # Tie convention mirrors torch.minimum with an untracked scalar
        # operand (verified torch 2.11.0+cpu): grad 1 where x < s, 0.5
        # where x == s, 0 above.
        var lt_bool: NDBuffer[DType.bool]
        var eq_bool: NDBuffer[DType.bool]

        comptime if has_accelerator():
            if parent.is_on_gpu():
                lt_bool = parent.buffer.compare_scalar[LessThan](scalar)
                eq_bool = parent.buffer.compare_scalar[Equal](scalar)
            else:
                lt_bool = parent.buffer.compare_scalar_cpu[LessThan](scalar)
                eq_bool = parent.buffer.compare_scalar_cpu[Equal](scalar)
        else:
            lt_bool = parent.buffer.compare_scalar_cpu[LessThan](scalar)
            eq_bool = parent.buffer.compare_scalar_cpu[Equal](scalar)

        var mask_lt = lt_bool.to_dtype[Self.dtype]()
        var mask_eq = eq_bool.to_dtype[Self.dtype]()
        var mask = mask_lt.arithmetic_ops[Add](
            mask_eq.scalar_ops[Multiply](Scalar[Self.dtype](0.5))
        )
        var grad_input = Gradbox[Self.dtype](
            mask.arithmetic_ops[Multiply](gradbox.buffer()),
        )
        parent_ref.update_grad(grad_input^, AddTensor, None)
        parent_ids.append(parent_ref._id)

        gradbox.zero_grad()


@fieldwise_init
struct MaxScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        scalar: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[MAX](scalar, sync=sync), requires_grad=False
        )

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.scalar_arg[Self.dtype](
                    scalar, MaxBackwardScalar[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^


@fieldwise_init
struct MinScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        scalar: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[MIN](scalar, sync=sync), requires_grad=False
        )

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn = BackwardFn.scalar_arg[Self.dtype](
                    scalar, MinBackwardScalar[Self.dtype]()
                )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^
