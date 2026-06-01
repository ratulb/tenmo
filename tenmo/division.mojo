from .tensor import Tensor
from .backpropagation import (
    BackwardFnArg,
    ScalarArg,
    BACKWARD_DIVIDE,
    BACKWARD_DIV_SCALAR,
    BACKWARD_RIGHT_DIV_SCALAR,
)
from tenmo.mnemonics import AddTensor, SubtractTensor, Divide, ReverseDivide
from tenmo.common_utils import panic
from tenmo.gradbox import Gradbox
from tenmo.ancestry import Ancestor
from tenmo.ndbuffer import NDBuffer
from tenmo.buffers import Buffer
from tenmo.kernels.division_kernel import DivisionKernel
from tenmo.sum_mean_reduction import SumMeanReduction
from tenmo.intarray import IntArray
from tenmo.indexhelper import IndexCalculator
from std.sys import simd_width_of, has_accelerator


# =============================================================================
# Scalar element functions — pure Scalar[dtype] transformations
# =============================================================================


@fieldwise_init
struct DivElement[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def rdiv_scalar_backward_element(
        grad_out: Scalar[Self.dtype],
        x: Scalar[Self.dtype],
        scalar: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        return scalar * grad_out / (x * x)

    @staticmethod
    def divide_backward_x_element(
        grad_out: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        return grad_out / y

    @staticmethod
    def divide_backward_y_element(
        grad_out: Scalar[Self.dtype],
        x: Scalar[Self.dtype],
        y: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        return grad_out * x / (y * y)


# =============================================================================
# Buffer SIMD kernels  — contiguous CPU only
# =============================================================================


@fieldwise_init
struct DivBuffer[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def rdiv_scalar_backward(
        grad_output: Buffer[Self.dtype],
        divisor: Buffer[Self.dtype],
        scalar: Scalar[Self.dtype],
    ) -> Buffer[Self.dtype]:
        """Fused backward for s / x. Computes s * grad_output / x² in one pass.
        NOTE: Positive sign because the caller uses SubtractTensor (negation).
        """
        var extent = grad_output.size
        var grad_out = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var s_vec = SIMD[Self.dtype, simd_width](scalar)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var g = grad_output.load[simdwidth=simd_width](idx)
            var x = divisor.load[simdwidth=simd_width](idx)
            var x_sq = x * x
            grad_out.store[simdwidth=simd_width](idx, s_vec * g / x_sq)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var s_s = scalar
            for i in range(remainder):
                var idx = start_idx + i
                var g = grad_output[idx]
                var x = divisor[idx]
                grad_out[idx] = s_s * g / (x * x)

        return grad_out^

    @staticmethod
    def divide_backward(
        grad_output: Buffer[Self.dtype],
        x: Buffer[Self.dtype],
        y: Buffer[Self.dtype],
    ) -> Tuple[Buffer[Self.dtype], Buffer[Self.dtype]]:
        """Fused backward for x / y. Computes both gradients in one pass.
        grad_x = grad_output / y
        grad_y = -grad_output * x / y²
        """
        var extent = grad_output.size
        var grad_x_buf = Buffer[Self.dtype](extent)
        var grad_y_buf = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var g = grad_output.load[simdwidth=simd_width](idx)
            var xv = x.load[simdwidth=simd_width](idx)
            var yv = y.load[simdwidth=simd_width](idx)
            var y_sq = yv * yv
            grad_x_buf.store[simdwidth=simd_width](idx, g / yv)
            grad_y_buf.store[simdwidth=simd_width](idx, g * xv / y_sq)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            for i in range(remainder):
                var idx = start_idx + i
                var g = grad_output[idx]
                var xv = x[idx]
                var yv = y[idx]
                grad_x_buf[idx] = g / yv
                grad_y_buf[idx] = g * xv / (yv * yv)

        return (grad_x_buf^, grad_y_buf^)


# =============================================================================
# NDBuffer-level device dispatch + CPU fallback
# =============================================================================


@fieldwise_init
struct DivNdBuffer[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def rdiv_scalar_backward(
        grad_output: NDBuffer[Self.dtype],
        divisor: NDBuffer[Self.dtype],
        scalar: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        """Fused backward for s / x. Returns gradient for divisor.
        Device-aware: GPU → DivisionKernel.launch_rdiv_scalar_backward.
        CPU → _rdiv_scalar_backward_cpu.
        """
        result: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if grad_output.is_on_gpu():
                try:
                    result = DivisionKernel[
                        Self.dtype
                    ].launch_rdiv_scalar_backward(grad_output, divisor, scalar)
                except e:
                    print(e)
                    panic(
                        "DivNdBuffer rdiv_scalar_backward → GPU operation"
                        " failed"
                    )
                    result = NDBuffer[Self.dtype].Empty()
            else:
                result = Self._rdiv_scalar_backward_cpu(
                    grad_output, divisor, scalar
                )
        else:
            result = Self._rdiv_scalar_backward_cpu(
                grad_output, divisor, scalar
            )
        return result^

    @staticmethod
    @always_inline
    def _rdiv_scalar_backward_cpu(
        grad_output: NDBuffer[Self.dtype],
        divisor: NDBuffer[Self.dtype],
        scalar: Scalar[Self.dtype],
    ) -> NDBuffer[Self.dtype]:
        if grad_output.is_contiguous() and divisor.is_contiguous():
            var buf = DivBuffer[Self.dtype].rdiv_scalar_backward(
                grad_output.buffer, divisor.buffer, scalar
            )
            return NDBuffer[Self.dtype](buf^, grad_output.shape)
        else:
            var extent = grad_output.numels()
            var grad_buf = Buffer[Self.dtype](extent)
            var idx = 0
            for coord in grad_output.index_iterator():
                grad_buf[idx] = DivElement[
                    Self.dtype
                ].rdiv_scalar_backward_element(
                    grad_output.buffer[coord],
                    divisor.buffer[coord],
                    scalar,
                )
                idx += 1
            return NDBuffer[Self.dtype](grad_buf^, grad_output.shape)

    @staticmethod
    def divide_backward(
        grad_output: NDBuffer[Self.dtype],
        x: NDBuffer[Self.dtype],
        y: NDBuffer[Self.dtype],
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        """Fused backward for x / y. Returns (grad_x, grad_y).
        Device-aware: GPU → DivisionKernel.launch_divide_backward.
        CPU → _divide_backward_cpu.
        """
        result_x: NDBuffer[Self.dtype]
        result_y: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if grad_output.is_on_gpu():
                try:
                    (result_x, result_y) = DivisionKernel[
                        Self.dtype
                    ].launch_divide_backward(grad_output, x, y)
                except e:
                    print(e)
                    panic("DivNdBuffer divide_backward → GPU operation failed")
                    result_x = NDBuffer[Self.dtype].Empty()
                    result_y = NDBuffer[Self.dtype].Empty()
            else:
                (result_x, result_y) = Self._divide_backward_cpu(
                    grad_output, x, y
                )
        else:
            (result_x, result_y) = Self._divide_backward_cpu(grad_output, x, y)
        return (result_x^, result_y^)

    @staticmethod
    @always_inline
    def _divide_backward_cpu(
        grad_output: NDBuffer[Self.dtype],
        x: NDBuffer[Self.dtype],
        y: NDBuffer[Self.dtype],
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        # Fast path: same-shape contiguous → buffer-level SIMD kernel
        if (
            grad_output.is_contiguous()
            and x.is_contiguous()
            and y.is_contiguous()
            and grad_output.shape == x.shape
            and x.shape == y.shape
        ):
            var (buf_x, buf_y) = DivBuffer[Self.dtype].divide_backward(
                grad_output.buffer, x.buffer, y.buffer
            )
            return (
                NDBuffer[Self.dtype](buf_x^, grad_output.shape),
                NDBuffer[Self.dtype](buf_y^, grad_output.shape),
            )
        # Fallback: broadcast operands to match output via stride-0 views,
        # then iterate flat with coordinate conversion. broadcast_to handles
        # stride=0 automatically — no manual rank-checking needed.
        var out_shape = grad_output.shape
        var extent = grad_output.numels()
        var gx_buf = Buffer[Self.dtype](extent)
        var gy_buf = Buffer[Self.dtype](extent)
        var x_bc = x if x.shape == out_shape else x.broadcast_to(out_shape)
        var y_bc = y if y.shape == out_shape else y.broadcast_to(out_shape)
        var idx = 0
        for flat_idx in grad_output.index_iterator():
            var g = grad_output.buffer[flat_idx]
            var coord = IndexCalculator.index_to_coord(out_shape, flat_idx)
            gx_buf[idx] = DivElement[Self.dtype].divide_backward_x_element(
                g, y_bc[coord]
            )
            gy_buf[idx] = DivElement[Self.dtype].divide_backward_y_element(
                g, x_bc[coord], y_bc[coord]
            )
            idx += 1
        return (
            NDBuffer[Self.dtype](gx_buf^, out_shape),
            NDBuffer[Self.dtype](gy_buf^, out_shape),
        )


@fieldwise_init
struct TrueDivBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var scalar = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        ref gradbox = output.gradients()[]
        ancestor = output.ancestry().get(0)
        # ∂(x / s)/∂x = 1/s → incoming_grad / scalar
        var divided = gradbox / scalar
        ancestor.update_grad(divided^, AddTensor, None)
        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct RightTrueDivBackwardScalar[dtype: DType](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var scalar = (
            output.ancestry()
            .backward_fn_arg()
            .get[ScalarArg[Self.dtype]]()
            .value
        )
        var gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        # Fused: -s * grad_output / x² in one pass
        var grad_ndb = DivNdBuffer[Self.dtype].rdiv_scalar_backward(
            gradbox.buffer, ancestor.buffer(), scalar
        )
        var grad_parent = Gradbox[Self.dtype](grad_ndb^, share=False)
        ancestor.update_grad(grad_parent^, SubtractTensor, None)
        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct DivideBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref gradbox = output.gradients()[]
        var ancestor_top = output.ancestry().get(0)
        var ancestor_bottom = output.ancestry().get(1)
        var buffer_top = ancestor_top.buffer()
        var buffer_bottom = ancestor_bottom.buffer()

        # Fused: both gradients in one pass
        var (grad_num, grad_den) = DivNdBuffer[Self.dtype].divide_backward(
            gradbox.buffer, buffer_top, buffer_bottom
        )

        if ancestor_top.requires_grad:
            if grad_num.shape != buffer_top.shape:
                grad_num = SumMeanReduction[
                    Self.dtype
                ].sum_over_broadcasted_axes(grad_num, buffer_top.shape)
            ancestor_top.update_grad(
                Gradbox[Self.dtype](grad_num^, share=False),
                AddTensor,
                None,
            )
            parent_ids.append(ancestor_top._id)

        if ancestor_bottom.requires_grad:
            if grad_den.shape != buffer_bottom.shape:
                grad_den = SumMeanReduction[
                    Self.dtype
                ].sum_over_broadcasted_axes(grad_den, buffer_bottom.shape)
            ancestor_bottom.update_grad(
                Gradbox[Self.dtype](grad_den^, share=False),
                SubtractTensor,
                None,
            )
            parent_ids.append(ancestor_bottom._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct DivideScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → __rtruediv__ is for numeric data types only"

        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[ReverseDivide](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_RIGHT_DIV_SCALAR, scalar
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


@fieldwise_init
struct DivideByScalar[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], scalar: Scalar[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        comptime assert (
            Self.dtype.is_numeric()
        ), "Tensor → __truediv__ is for numeric data types only"

        if scalar == Scalar[Self.dtype](0):
            panic("Tensor → __truediv__ : canot divide by " + String(scalar))

        var out = Tensor[Self.dtype](
            self.buffer.scalar_ops[Divide](scalar), requires_grad=False
        )

        comptime if track_grad:
            if self.requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].scalar_arg(
                    BACKWARD_DIV_SCALAR, scalar
                )
                out.add_ancestry(backwardFnArg^, self)

        return out^


# Element wise division of two tensors
@fieldwise_init
struct Divider[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](self: Tensor[Self.dtype], other: Tensor[Self.dtype]) -> Tensor[
        Self.dtype
    ]:
        if not self.broadcastable(other):
            panic(
                "Tensor division dimension mismatch: cannot broadcast shape "
                + String(self.shape())
                + " with "
                + String(other.shape()),
                "at Divider → forward",
            )

        var out = Tensor[Self.dtype](
            self.buffer.arithmetic_ops[Divide](other.buffer),
            requires_grad=False,
        )

        comptime if track_grad:
            requires_grad = self.requires_grad or other.requires_grad
            if requires_grad:
                out.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].null_arg(
                    BACKWARD_DIVIDE
                )
                out.add_ancestry(backwardFnArg^, self, other)

        return out^
