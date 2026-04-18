from tenmo import Tensor
from mnemonics import AddTensor, MAX, MIN
from backpropagation import (
    BackwardFnArg,
    ScalarArg,
    BACKWARD_MAX_SCALAR,
    BACKWARD_MIN_SCALAR,
)
from gradbox import Gradbox
from std.sys import has_accelerator
from ndbuffer import NDBuffer
from mnemonics import GreaterThan, LessThan
from ancestors_newest import AncestorRef
# ── MaxBackwardScalar ─────────────────────────────────────────────────────────


@fieldwise_init
struct MaxBackwardScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):

    @staticmethod
    fn backward(
        output: AncestorRef[Self.dtype],
    ) -> List[Tuple[AncestorRef[Self.dtype], Gradbox[Self.dtype], Int]]:
        var scalar = output.backward_fn_arg().get[ScalarArg[Self.dtype]]().value
        ref gradbox = output.gradients()[]
        var parent_ref = output.ancestry().get(0)
        var parent = Tensor[Self.dtype](parent_ref.buffer(), requires_grad=parent_ref.requires_grad)

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

# ── MinBackwardScalar ─────────────────────────────────────────────────────────


@fieldwise_init
struct MinBackwardScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):

    @staticmethod
    fn backward(
        output: AncestorRef[Self.dtype],
    ) -> List[Tuple[AncestorRef[Self.dtype], Gradbox[Self.dtype], Int]]:
        var scalar = output.backward_fn_arg().get[ScalarArg[Self.dtype]]().value
        ref gradbox = output.gradients()[]
        var parent_ref = output.ancestry().get(0)
        var parent = Tensor[Self.dtype](parent_ref.buffer(), requires_grad=parent_ref.requires_grad)

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


# ── MaxScalar forward ─────────────────────────────────────────────────────────


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
                var backwardFnArg =
                    BackwardFnArg[Self.dtype].scalar_arg(
                        BACKWARD_MAX_SCALAR, scalar

                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


# ── MinScalar forward ─────────────────────────────────────────────────────────


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
                var backwardFnArg =
                    BackwardFnArg[Self.dtype].scalar_arg(
                        BACKWARD_MIN_SCALAR, scalar

                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


from std.testing import assert_true


fn main() raises:
    comptime dtype = DType.float32

    # Test Max
    var a = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
    var b = a.max(4.0)
    b.print()
    assert_true(b == Tensor[dtype].d1([4.0, 5.0, 4.0, 7.0, 4.0]))

    b.backward()
    a.grad().print()
    assert_true(a.grad() == Tensor[dtype].d1([0.0, 1.0, 0.0, 1.0, 0.0]))

    # Test Min
    var c = Tensor[dtype].d1([1.0, 5.0, 3.0, 7.0, 2.0], requires_grad=True)
    var d = c.min(4.0)
    d.print()
    assert_true(d == Tensor[dtype].d1([1.0, 4.0, 3.0, 4.0, 2.0]))

    d.backward()
    c.grad().print()
    assert_true(c.grad() == Tensor[dtype].d1([1.0, 0.0, 1.0, 0.0, 1.0]))

    comptime if has_accelerator():
        print("test_maxmin_gpu_parity_min")
        var a_cpu = Tensor[dtype].d2(
            [[1.0, 3.0, 7.0], [8.0, 2.0, 5.0]], requires_grad=True
        )
        var a_gpu = a_cpu.to_gpu()

        var b_cpu = a_cpu.min(4.0)
        b_cpu.print()
