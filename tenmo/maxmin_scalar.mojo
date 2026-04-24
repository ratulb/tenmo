from .tensor import Tensor
from .mnemonics import AddTensor, MAX, MIN
from .backpropagation import (
    BackwardFnArg,
    ScalarArg,
    BACKWARD_MAX_SCALAR,
    BACKWARD_MIN_SCALAR,
)
from .gradbox import Gradbox
from std.sys import has_accelerator
from .ndbuffer import NDBuffer
from .mnemonics import GreaterThan, LessThan
from .ancestry import Ancestor


@fieldwise_init
struct MaxBackwardScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var scalar = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()[]
        var parent_ref = output.ancestry().get(0)
        var parent = Tensor[Self.dtype](
            parent_ref.buffer(), requires_grad=parent_ref.requires_grad
        )

        # Work at NDBuffer level — avoids pulling in GPU kernel launchers
        var mask_bool: NDBuffer[DType.bool]

        comptime if has_accelerator():
            if parent.is_on_gpu():
                mask_bool = parent.buffer.compare_scalar[GreaterThan](scalar)
            else:
                mask_bool = parent.buffer.compare_scalar_cpu[GreaterThan](
                    scalar
                )
        else:
            mask_bool = parent.buffer.compare_scalar_cpu[GreaterThan](scalar)

        var mask_float = mask_bool.to_dtype[Self.dtype]()
        # wrap mask_float as Gradbox and multiply
        var grad_input = Gradbox[Self.dtype](
            mask_float * gradbox.buffer, share=False
        )

        return [(parent_ref^, grad_input^, AddTensor)]



@fieldwise_init
struct MinBackwardScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        var scalar = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()[]
        var parent_ref = output.ancestry().get(0)
        var parent = Tensor[Self.dtype](
            parent_ref.buffer(), requires_grad=parent_ref.requires_grad
        )

        var mask_bool: NDBuffer[DType.bool]

        comptime if has_accelerator():
            if parent.is_on_gpu():
                mask_bool = parent.buffer.compare_scalar[LessThan](scalar)
            else:
                mask_bool = parent.buffer.compare_scalar_cpu[LessThan](scalar)
        else:
            mask_bool = parent.buffer.compare_scalar_cpu[LessThan](scalar)

        var mask_float = mask_bool.to_dtype[Self.dtype]()
        var grad_input = Gradbox[Self.dtype](
            mask_float * gradbox.buffer, share=False
        )
        return [(parent_ref^, grad_input^, AddTensor)]


@fieldwise_init
struct MaxScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        scalar: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[MAX](scalar), requires_grad=False
        )

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_MAX_SCALAR, scalar
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


@fieldwise_init
struct MinScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        scalar: Scalar[Self.dtype],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[MIN](scalar), requires_grad=False
        )

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_MIN_SCALAR, scalar
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^

