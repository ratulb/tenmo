from .tensor import Tensor
from .shared.shapes import Shape
from .gradbox import Gradbox
from .backpropagation import BackwardFn, ArgumentType, BackwardFnType

from .layer_trait import LayerTrait
from .shared.mnemonics import AddTensor
from .shared.buffers import Buffer
from .ndbuffer import NDBuffer
from std.utils.numerics import neg_inf
from .shared.panic import panic
from max.algorithm import parallelize
from .ancestry import Ancestor


@fieldwise_init
struct MaxPool2dBwdArg(ArgumentType):
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var input_shape: Shape
    var argmax_mask: NDBuffer[DType.int64]


@fieldwise_init
struct MaxPool2dBackward[dtype: DType](BackwardFnType, ImplicitlyCopyable):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var bwd_arg = output.ancestry().backward_fn().get[MaxPool2dBwdArg]()
        var (kernel_size, stride, padding, input_shape, argmax_mask) = (
            bwd_arg.kernel_size,
            bwd_arg.stride,
            bwd_arg.padding,
            bwd_arg.input_shape,
            bwd_arg.argmax_mask,
        )
        ref grad_output = output.gradients()

        var input_tensor_ref = output.ancestry().get(0)

        # Grad is computed only if needed, but the id is ALWAYS appended:
        # parent_ids is the engine's fanin-completion signal (appended set
        # must equal ancestry set).
        if input_tensor_ref.requires_grad:
            var N = input_shape[0]
            var C = input_shape[1]
            var H_in = input_shape[2]
            var W_in = input_shape[3]

            var output_shape = grad_output.shape()
            var H_out = output_shape[2]
            var W_out = output_shape[3]

            var grad_input = Gradbox[Self.dtype].zeros(input_shape)

            # Direct buffer access
            var grad_out_ptr = grad_output.data_ptr()
            var grad_in_ptr = (
                grad_input.data_ptr()
                .unsafe_mut_cast[True]()
                .unsafe_origin_cast[MutAnyOrigin]()
            )
            var argmax_ptr = argmax_mask.data_ptr()

            # Precompute strides
            var grad_out_stride_N = C * H_out * W_out
            var grad_out_stride_C = H_out * W_out
            var grad_out_stride_H = W_out

            var grad_in_stride_N = C * H_in * W_in
            var grad_in_stride_C = H_in * W_in

            def scatter_gradients_optimized(idx: Int) {imm}:
                var n = idx // C
                var c = idx % C

                var grad_out_nc_base = (
                    n * grad_out_stride_N + c * grad_out_stride_C
                )
                var grad_in_nc_base = (
                    n * grad_in_stride_N + c * grad_in_stride_C
                )
                var argmax_nc_base = n * C * H_out * W_out + c * H_out * W_out

                # Process all output positions
                for out_y in range(H_out):
                    var grad_out_y_base = (
                        grad_out_nc_base + out_y * grad_out_stride_H
                    )
                    var argmax_y_base = argmax_nc_base + out_y * W_out

                    for out_x in range(W_out):
                        # Direct memory access
                        var max_idx = Int(
                            argmax_ptr[unsafe_offset=argmax_y_base + out_x]
                        )

                        if max_idx >= 0:
                            var grad_val = grad_out_ptr[
                                unsafe_offset=grad_out_y_base + out_x
                            ]

                            # Accumulate gradient at max position
                            grad_in_ptr[
                                unsafe_offset=grad_in_nc_base + max_idx
                            ] += grad_val

            parallelize(scatter_gradients_optimized, N * C)
            input_tensor_ref.update_grad(grad_input^, AddTensor, None)
        parent_ids.append(input_tensor_ref._id)

        grad_output.zero_grad()


@fieldwise_init
struct MaxPool2d[dtype: DType](LayerTrait & RegisterPassable):
    """
        Batched, multi-channel 2D Max Pooling.

    1. Direct buffer access (eliminates indexing overhead)
    2. Unrolled loops for common kernel sizes (2×2, 3×3)
    3. Precomputed strides
    4. Cache-friendly memory access
    """

    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype
    var training: Bool
    var kernel_size: Int
    var stride: Int
    var padding: Int

    def __init__(
        out self,
        kernel_size: Int = 2,
        stride: Optional[Int] = None,
        padding: Int = 0,
    ):
        self.training = True
        self.kernel_size = kernel_size
        self.stride = stride.or_else(kernel_size)
        self.padding = padding

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        if self.training:
            return Self.forward[track_grad=True](
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                sync=sync,
            )
        else:
            return Self.forward[track_grad=False](
                x,
                self.kernel_size,
                self.stride,
                self.padding,
                sync=sync,
            )

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        input_tensor: Tensor[Self.dtype],
        kernel_size: Int = 2,
        stride: Optional[Int] = None,
        padding: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        ref input_shape = input_tensor.shape()
        if input_shape.rank() != 4:
            panic("MaxPool2d expects 4D input: (N, C, H_in, W_in)")

        var N = input_shape[0]
        var C = input_shape[1]
        var H_in = input_shape[2]
        var W_in = input_shape[3]

        var KH = kernel_size
        var KW = kernel_size
        var s = stride.or_else(kernel_size)
        var pad = padding

        var H_out = (H_in + 2 * pad - KH) // s + 1
        var W_out = (W_in + 2 * pad - KW) // s + 1

        if H_out <= 0 or W_out <= 0:
            panic("Invalid MaxPool2d parameters")

        # Uninitialized allocs: every element of both is overwritten by the
        # pool kernels below (one output + one argmax per position), so the
        # zeros fills were pure overhead. The mask is still only *kept* (in
        # BwdArg) when grad is tracked.
        var output = Tensor[Self.dtype](
            NDBuffer[Self.dtype](
                Buffer[Self.dtype](N * C * H_out * W_out),
                Shape(N, C, H_out, W_out),
            ),
            requires_grad=False,
        )
        var argmax_mask = NDBuffer[DType.int64](
            Buffer[DType.int64](N * C * H_out * W_out),
            Shape(N, C, H_out, W_out),
        )

        # Direct buffer access
        var input_ptr = (
            input_tensor.data_ptr()
            .unsafe_mut_cast[True]()
            .unsafe_origin_cast[MutAnyOrigin]()
        )
        var output_ptr = output.data_ptr()
        var argmax_ptr = argmax_mask.data_ptr()

        # Precompute strides
        var in_stride_N = C * H_in * W_in
        var in_stride_C = H_in * W_in
        var in_stride_H = W_in

        var out_stride_N = C * H_out * W_out
        var out_stride_C = H_out * W_out
        var out_stride_H = W_out

        # Specialize for common kernel sizes
        if KH == 2 and KW == 2:
            # Optimized 2×2 pooling (most common)
            Self._pool_2x2(
                input_ptr,
                output_ptr,
                argmax_ptr,
                N,
                C,
                H_in,
                W_in,
                H_out,
                W_out,
                s,
                pad,
                in_stride_N,
                in_stride_C,
                in_stride_H,
                out_stride_N,
                out_stride_C,
                out_stride_H,
            )
        elif KH == 3 and KW == 3:
            # Optimized 3×3 pooling
            Self._pool_3x3(
                input_ptr,
                output_ptr,
                argmax_ptr,
                N,
                C,
                H_in,
                W_in,
                H_out,
                W_out,
                s,
                pad,
                in_stride_N,
                in_stride_C,
                in_stride_H,
                out_stride_N,
                out_stride_C,
                out_stride_H,
            )
        else:
            # Generic pooling for arbitrary kernel sizes
            Self._pool_generic(
                input_ptr,
                output_ptr,
                argmax_ptr,
                N,
                C,
                H_in,
                W_in,
                H_out,
                W_out,
                KH,
                KW,
                s,
                pad,
                in_stride_N,
                in_stride_C,
                in_stride_H,
                out_stride_N,
                out_stride_C,
                out_stride_H,
            )

        # Setup gradient tracking
        comptime if track_grad:
            var grad_required = requires_grad.or_else(
                input_tensor.requires_grad
            )
            if grad_required:
                output.requires_grad_(True)
                var backwardFn = BackwardFn(
                    MaxPool2dBwdArg(
                        kernel_size,
                        s,  # Stride
                        pad,
                        input_shape,
                        argmax_mask,
                    ),
                    MaxPool2dBackward[Self.dtype](),
                )
                output.add_ancestry(backwardFn^, input_tensor)

        return output^

    # OPTIMIZED 2×2 POOLING (fully unrolled)
    @staticmethod
    def _pool_2x2(
        input_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        output_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        argmax_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
        N: Int,
        C: Int,
        H_in: Int,
        W_in: Int,
        H_out: Int,
        W_out: Int,
        stride: Int,
        pad: Int,
        in_stride_N: Int,
        in_stride_C: Int,
        in_stride_H: Int,
        out_stride_N: Int,
        out_stride_C: Int,
        out_stride_H: Int,
    ):
        """Fully unrolled 2×2 max pooling."""

        def pool_nc(idx: Int) {imm}:
            var n = idx // C
            var c = idx % C

            var in_nc_base = n * in_stride_N + c * in_stride_C
            var out_nc_base = n * out_stride_N + c * out_stride_C
            var argmax_nc_base = n * C * H_out * W_out + c * H_out * W_out

            for out_y in range(H_out):
                var in_y_start = out_y * stride - pad
                var out_y_base = out_nc_base + out_y * out_stride_H
                var argmax_y_base = argmax_nc_base + out_y * W_out

                for out_x in range(W_out):
                    var in_x_start = out_x * stride - pad

                    # Unrolled 2×2 window
                    var max_val = neg_inf[Self.dtype]()
                    var max_idx = -1

                    # Position (0, 0)
                    var in_y0 = in_y_start
                    var in_x0 = in_x_start
                    if (
                        in_y0 >= 0
                        and in_y0 < H_in
                        and in_x0 >= 0
                        and in_x0 < W_in
                    ):
                        var idx0 = in_nc_base + in_y0 * in_stride_H + in_x0
                        var val0 = input_ptr[unsafe_offset=idx0]
                        if val0 > max_val:
                            max_val = val0
                            max_idx = in_y0 * W_in + in_x0

                    # Position (0, 1)
                    var in_x1 = in_x_start + 1
                    if (
                        in_y0 >= 0
                        and in_y0 < H_in
                        and in_x1 >= 0
                        and in_x1 < W_in
                    ):
                        var idx1 = in_nc_base + in_y0 * in_stride_H + in_x1
                        var val1 = input_ptr[unsafe_offset=idx1]
                        if val1 > max_val:
                            max_val = val1
                            max_idx = in_y0 * W_in + in_x1

                    # Position (1, 0)
                    var in_y1 = in_y_start + 1
                    if (
                        in_y1 >= 0
                        and in_y1 < H_in
                        and in_x0 >= 0
                        and in_x0 < W_in
                    ):
                        var idx2 = in_nc_base + in_y1 * in_stride_H + in_x0
                        var val2 = input_ptr[unsafe_offset=idx2]
                        if val2 > max_val:
                            max_val = val2
                            max_idx = in_y1 * W_in + in_x0

                    # Position (1, 1)
                    if (
                        in_y1 >= 0
                        and in_y1 < H_in
                        and in_x1 >= 0
                        and in_x1 < W_in
                    ):
                        var idx3 = in_nc_base + in_y1 * in_stride_H + in_x1
                        var val3 = input_ptr[unsafe_offset=idx3]
                        if val3 > max_val:
                            max_val = val3
                            max_idx = in_y1 * W_in + in_x1

                    output_ptr[unsafe_offset=out_y_base + out_x] = max_val
                    argmax_ptr[unsafe_offset=argmax_y_base + out_x] = Int64(
                        max_idx
                    )

        parallelize(pool_nc, N * C)

    # OPTIMIZED 3×3 POOLING (fully unrolled)
    @staticmethod
    def _pool_3x3(
        input_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        output_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        argmax_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
        N: Int,
        C: Int,
        H_in: Int,
        W_in: Int,
        H_out: Int,
        W_out: Int,
        stride: Int,
        pad: Int,
        in_stride_N: Int,
        in_stride_C: Int,
        in_stride_H: Int,
        out_stride_N: Int,
        out_stride_C: Int,
        out_stride_H: Int,
    ):
        """Fully unrolled 3×3 max pooling."""

        def pool_nc(idx: Int) {imm}:
            var n = idx // C
            var c = idx % C

            var in_nc_base = n * in_stride_N + c * in_stride_C
            var out_nc_base = n * out_stride_N + c * out_stride_C
            var argmax_nc_base = n * C * H_out * W_out + c * H_out * W_out

            for out_y in range(H_out):
                var in_y_start = out_y * stride - pad
                var out_y_base = out_nc_base + out_y * out_stride_H
                var argmax_y_base = argmax_nc_base + out_y * W_out

                for out_x in range(W_out):
                    var in_x_start = out_x * stride - pad

                    var max_val = neg_inf[Self.dtype]()
                    var max_idx = -1

                    # Unrolled 3×3 window (9 comparisons)
                    comptime for ky in range(3):
                        comptime for kx in range(3):
                            var in_y = in_y_start + ky
                            var in_x = in_x_start + kx

                            if (
                                in_y >= 0
                                and in_y < H_in
                                and in_x >= 0
                                and in_x < W_in
                            ):
                                var idx = in_nc_base + in_y * in_stride_H + in_x
                                var val = input_ptr[unsafe_offset=idx]
                                if val > max_val:
                                    max_val = val
                                    max_idx = in_y * W_in + in_x

                    output_ptr[unsafe_offset=out_y_base + out_x] = max_val
                    argmax_ptr[unsafe_offset=argmax_y_base + out_x] = Int64(
                        max_idx
                    )

        parallelize(pool_nc, N * C)

    # GENERIC POOLING (arbitrary kernel size)
    @staticmethod
    def _pool_generic(
        input_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        output_ptr: Pointer[Scalar[Self.dtype], MutAnyOrigin],
        argmax_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
        N: Int,
        C: Int,
        H_in: Int,
        W_in: Int,
        H_out: Int,
        W_out: Int,
        KH: Int,
        KW: Int,
        stride: Int,
        pad: Int,
        in_stride_N: Int,
        in_stride_C: Int,
        in_stride_H: Int,
        out_stride_N: Int,
        out_stride_C: Int,
        out_stride_H: Int,
    ):
        """Generic pooling for arbitrary kernel sizes."""

        def pool_nc(idx: Int) {imm}:
            var n = idx // C
            var c = idx % C

            var in_nc_base = n * in_stride_N + c * in_stride_C
            var out_nc_base = n * out_stride_N + c * out_stride_C
            var argmax_nc_base = n * C * H_out * W_out + c * H_out * W_out

            for out_y in range(H_out):
                var in_y_start = out_y * stride - pad
                var out_y_base = out_nc_base + out_y * out_stride_H
                var argmax_y_base = argmax_nc_base + out_y * W_out

                for out_x in range(W_out):
                    var in_x_start = out_x * stride - pad

                    var max_val = neg_inf[Self.dtype]()
                    var max_idx = -1

                    for ky in range(KH):
                        for kx in range(KW):
                            var in_y = in_y_start + ky
                            var in_x = in_x_start + kx

                            if (
                                in_y >= 0
                                and in_y < H_in
                                and in_x >= 0
                                and in_x < W_in
                            ):
                                var idx = in_nc_base + in_y * in_stride_H + in_x
                                var val = input_ptr[unsafe_offset=idx]
                                if val > max_val:
                                    max_val = val
                                    max_idx = in_y * W_in + in_x

                    output_ptr[unsafe_offset=out_y_base + out_x] = max_val
                    argmax_ptr[unsafe_offset=argmax_y_base + out_x] = Int64(
                        max_idx
                    )

        parallelize(pool_nc, N * C)

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False
