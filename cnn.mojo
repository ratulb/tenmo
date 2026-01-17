from tenmo import Tensor
from backpropagation import (
    BackwardFn,
    Delegate,
    BACKWARD_FUSED_CONV,
    BACKWARD_MAXPOOL2D,
)
from operators import AddTensor
from common_utils import panic
from forwards import Pad
from shapes import Shape
from gradbox import Gradbox
from forwards import Padding
from algorithm import parallelize
from sys import simd_width_of
from ndbuffer import NDBuffer
from intarray import IntArray


@fieldwise_init
struct Conv2dFused[dtype: DType](ImplicitlyCopyable):
    """
    Batched, multi-channel, multi-filter 2D convolution using fused im2col + matmul + bias.
    Args:
        image: (N, C_in, H_in, W_in).
        kernel: (C_out, C_in, KH, KW).
        bias: Optional (C_out,).
        stride: Stride for spatial dimensions.
        dilation: Dilation factor for atrous convolution.
        padding: 'valid', 'same', int, tuple, or list of tuples.
    Returns:
        output: (N, C_out, H_out, W_out).
    """

    var initialized: Bool
    var pad_spec: List[Tuple[Int, Int]]

    fn __copyinit__(out self, other: Self):
        self.initialized = other.initialized
        self.pad_spec = other.pad_spec.copy()

    fn __init__(out self):
        self.initialized = False
        self.pad_spec = List[Tuple[Int, Int]]()

    fn __call__[
        track_grad: Bool
    ](
        mut self,
        image: Tensor[Self.dtype],
        mut kernel: Tensor[Self.dtype],
        bias: Optional[Tensor[Self.dtype]] = None,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Padding = Padding("valid"),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var out: Tensor[Self.dtype]
        if self.initialized:
            var output = Self.invoke[track_grad=track_grad](
                image=image,
                kernel=kernel,
                bias=bias,
                stride=stride,
                dilation=dilation,
                pad_spec=self.pad_spec.copy(),
                requires_grad=requires_grad,
            )
            out = output^
        else:
            var result = Self.invoke[track_grad=track_grad](
                image=image,
                kernel=kernel,
                bias=bias,
                stride=stride,
                dilation=dilation,
                padding=padding,
                requires_grad=requires_grad,
            )
            self.pad_spec = result[1].copy()
            self.initialized = True
            out = result[0]
        return out^

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        mut kernel: Tensor[Self.dtype],
        bias: Optional[Tensor[Self.dtype]] = None,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Padding = Padding("valid"),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var output, _ = Self.invoke[track_grad=track_grad](
            image=image,
            kernel=kernel,
            bias=bias,
            stride=stride,
            dilation=dilation,
            padding=padding,
            requires_grad=requires_grad,
        )
        return output^

    @staticmethod
    fn invoke[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        mut kernel: Tensor[Self.dtype],
        bias: Optional[Tensor[Self.dtype]] = None,
        stride: Int = 1,
        dilation: Int = 1,
        pad_spec: List[Tuple[Int, Int]] = [],
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var C_out = kernel.shape()[0]
        var padded_image = Pad[Self.dtype].forward[track_grad=track_grad](
            image,
            pad_spec.copy(),
            mode="constant",
            value=0.0,
            requires_grad=requires_grad.or_else(image.requires_grad),
        )

        # Setup bias
        var expected_bias_shape = Shape(C_out)
        var bias_tensor = bias.or_else(
            Tensor[Self.dtype].zeros(expected_bias_shape, requires_grad=False)
        )
        if not bias_tensor.shape() == expected_bias_shape:
            panic(
                "Invalid bias tensor shape: ",
                bias_tensor.shape().__str__(),
                ". Should be (C_out,)",
            )

        # Fused forward

        var output = FusedIm2Col[Self.dtype].forward[track_grad=track_grad](
            padded_image^,
            kernel,
            bias_tensor^,
            stride,
            dilation,
            requires_grad=requires_grad,
        )
        return output^

    @staticmethod
    fn invoke[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        mut kernel: Tensor[Self.dtype],
        bias: Optional[Tensor[Self.dtype]] = None,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Padding = Padding("valid"),
        requires_grad: Optional[Bool] = None,
    ) -> Tuple[Tensor[Self.dtype], List[Tuple[Int, Int]]]:
        ref image_shape = image.shape()
        ref kernel_shape = kernel.shape()

        # Validation
        if image_shape.rank() != 4:
            panic("Image must be 4D: (N, C_in, H_in, W_in)")
        if kernel_shape.rank() != 4:
            panic("Kernel must be 4D: (C_out, C_in, KH, KW)")
        # var N = image_shape[0]
        var C_in = image_shape[1]
        if kernel_shape[1] != C_in:
            panic("Kernel input channels must match input channels")
        var H_in = image_shape[2]
        var W_in = image_shape[3]
        var C_out = kernel_shape[0]
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]
        var dil = dilation
        var dilated_KH = KH + (KH - 1) * (dil - 1)
        var dilated_KW = KW + (KW - 1) * (dil - 1)

        # Parse Padding
        var pad_top: Int = 0
        var pad_bottom: Int = 0
        var pad_left: Int = 0
        var pad_right: Int = 0
        if padding.isa[String]():
            var mode = padding[String]
            if mode == "valid":
                pass
            elif mode == "same":
                var H_out_target = (H_in + stride - 1) // stride
                var W_out_target = (W_in + stride - 1) // stride
                var pad_h_total = (
                    (H_out_target - 1) * stride + dilated_KH - H_in
                )
                var pad_w_total = (
                    (W_out_target - 1) * stride + dilated_KW - W_in
                )
                pad_top = pad_h_total // 2
                pad_bottom = pad_h_total - pad_top
                pad_left = pad_w_total // 2
                pad_right = pad_w_total - pad_left
            else:
                panic("Unsupported padding mode: use 'valid' or 'same'")
        elif padding.isa[Int]():
            var p = padding[Int]
            pad_top = pad_bottom = pad_left = pad_right = p
        elif padding.isa[Tuple[Int, Int]]():
            var t = padding[Tuple[Int, Int]]
            pad_top = pad_bottom = t[0]
            pad_left = pad_right = t[1]
        elif padding.isa[List[Tuple[Int, Int]]]():
            if len(padding[List[Tuple[Int, Int]]]) != 2:
                panic("Padding list must contain exactly 2 tuples")
            var lst = padding[List[Tuple[Int, Int]]].copy()
            pad_top = lst[0][0]
            pad_bottom = lst[0][1]
            pad_left = lst[1][0]
            pad_right = lst[1][1]
        else:
            panic("Invalid padding type")

        # Pad the image
        var pad_spec = List[Tuple[Int, Int]]()
        pad_spec.append((0, 0))  # No padding on batch
        pad_spec.append((0, 0))  # No padding on channels
        pad_spec.append((pad_top, pad_bottom))  # Pad height
        pad_spec.append((pad_left, pad_right))  # Pad width
        var padded_image = Pad[Self.dtype].forward[track_grad=track_grad](
            image,
            pad_spec.copy(),
            mode="constant",
            value=0.0,
            requires_grad=requires_grad.or_else(image.requires_grad),
        )

        # Compute output shape
        var H_out = (H_in + pad_top + pad_bottom - dilated_KH) // stride + 1
        var W_out = (W_in + pad_left + pad_right - dilated_KW) // stride + 1
        if H_out <= 0 or W_out <= 0:
            panic(
                "Invalid convolution parameters lead to non-positive output"
                " size"
            )

        # Setup bias
        var expected_bias_shape = Shape(C_out)
        var bias_tensor = bias.or_else(
            Tensor[Self.dtype].zeros(expected_bias_shape, requires_grad=False)
        )
        if not bias_tensor.shape() == expected_bias_shape:
            panic(
                "Invalid bias tensor shape: ",
                bias_tensor.shape().__str__(),
                ". Should be (C_out,)",
            )

        # Fused forward

        var output = FusedIm2Col[Self.dtype].forward[track_grad=track_grad](
            padded_image,
            kernel,
            bias_tensor,
            stride,
            dilation,
            requires_grad=requires_grad,
        )
        return output^, pad_spec^


@fieldwise_init
@register_passable
struct FusedIm2Col[dtype: DType](ImplicitlyCopyable):
    """
    Fused Im2Col + Conv + Bias operation.

    Combines patch extraction, matrix multiplication, and bias addition
    into a single optimized kernel.

    Key optimizations:
    1. Direct buffer access (no tensor indexing overhead)
    2. SIMD vectorization over output channels
    3. Optimal parallelization strategy (N × spatial positions)
    4. Cache-friendly memory access patterns
    5. Precomputed strides and offsets

    Returns (N, C_out, H_out, W_out) - ready for next layer.
    """

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        padded_image: Tensor[Self.dtype],  # (N, C_in, H_pad, W_pad)
        kernel: Tensor[Self.dtype],  # (C_out, C_in, KH, KW)
        bias: Tensor[Self.dtype],  # (C_out,)
        stride: Int = 1,
        dilation: Int = 1,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """
        Fused convolution forward pass.

        Args:
            padded_image: Input tensor (N, C_in, H_pad, W_pad).
            kernel: Convolution weights (C_out, C_in, KH, KW).
            bias: Bias terms (C_out,).
            stride: Convolution stride.
            dilation: Dilation factor for atrous convolution.
            requires_grad: Override gradient tracking.

        Returns:
            Output tensor (N, C_out, H_out, W_out).
        """
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Validate inputs and extract dimensions
        # ═══════════════════════════════════════════════════════════
        ref padded_shape = padded_image.shape()
        ref kernel_shape = kernel.shape()

        if padded_shape.rank() != 4:
            panic("FusedIm2Col expects 4D input: (N, C_in, H_pad, W_pad)")
        if kernel_shape.rank() != 4:
            panic("Kernel must be 4D: (C_out, C_in, KH, KW)")

        var N = padded_shape[0]
        var C_in = padded_shape[1]
        var H_pad = padded_shape[2]
        var W_pad = padded_shape[3]

        var C_out = kernel_shape[0]
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]

        if kernel_shape[1] != C_in:
            panic("Kernel input channels must match image channels")

        # ═══════════════════════════════════════════════════════════
        # STEP 2: Compute output dimensions
        # ═══════════════════════════════════════════════════════════
        var dilated_KH = KH + (KH - 1) * (dilation - 1)
        var dilated_KW = KW + (KW - 1) * (dilation - 1)

        var H_out = (H_pad - dilated_KH) // stride + 1
        var W_out = (W_pad - dilated_KW) // stride + 1

        if H_out <= 0 or W_out <= 0:
            panic("Invalid output dimensions")

        # ═══════════════════════════════════════════════════════════
        # STEP 3: Allocate output tensor
        # ═══════════════════════════════════════════════════════════
        var output = Tensor[Self.dtype].zeros(N, C_out, H_out, W_out)

        # ═══════════════════════════════════════════════════════════
        # STEP 4: Get raw pointers for direct access
        # ═══════════════════════════════════════════════════════════
        var img_ptr = padded_image.buffer.data_buffer().data
        var kernel_ptr = kernel.buffer.data_buffer().data
        var bias_ptr = bias.buffer.data_buffer().data
        var out_ptr = output.buffer.data_buffer().data

        # ═══════════════════════════════════════════════════════════
        # STEP 5: Precompute strides
        # ═══════════════════════════════════════════════════════════
        var img_stride_N = C_in * H_pad * W_pad
        var img_stride_C = H_pad * W_pad
        var img_stride_H = W_pad

        var kernel_stride_Co = C_in * KH * KW
        var kernel_stride_Ci = KH * KW
        var kernel_stride_H = KW

        var out_stride_N = C_out * H_out * W_out
        var out_stride_Co = H_out * W_out
        var out_stride_H = W_out

        alias simd_w = simd_width_of[Self.dtype]()

        # ═══════════════════════════════════════════════════════════
        # STEP 6: Parallel computation
        # ═══════════════════════════════════════════════════════════
        # Strategy: Parallelize over (N, output spatial positions)
        # Each thread computes all C_out channels for one position
        # This allows vectorization over C_out dimension

        var total_work = N * H_out * W_out

        @parameter
        fn compute_position(work_idx: Int):
            """
            Compute convolution for one output spatial position.
            Processes all output channels with SIMD vectorization.
            """
            # Decode spatial position
            var n = work_idx // (H_out * W_out)
            var spatial = work_idx % (H_out * W_out)
            var out_y = spatial // W_out
            var out_x = spatial % W_out

            # Input window starting position
            var img_y_base = out_y * stride
            var img_x_base = out_x * stride

            # Base offsets
            var img_n_base = n * img_stride_N
            var out_n_base = n * out_stride_N + out_y * out_stride_H + out_x

            # ───────────────────────────────────────────────────────
            # Vectorized loop over output channels
            # ───────────────────────────────────────────────────────
            var co = 0
            var vec_end = (C_out // simd_w) * simd_w

            # Process simd_w output channels at once
            for _ in range(vec_end // simd_w):
                # Initialize accumulator with bias
                var accum = bias_ptr.load[width=simd_w](co)

                # Accumulate over input channels
                for ci in range(C_in):
                    var img_ci_base = img_n_base + ci * img_stride_C

                    # Kernel base for these output channels
                    # We need kernel[co:co+simd_w, ci, ky, kx]
                    var kernel_ci_base = ci * kernel_stride_Ci

                    # Loop over kernel spatial dimensions
                    for ky in range(KH):
                        var img_y = img_y_base + ky * dilation
                        var img_y_base_idx = img_ci_base + img_y * img_stride_H

                        for kx in range(KW):
                            var img_x = img_x_base + kx * dilation

                            # Load input value (same for all output channels)
                            var img_val = img_ptr[img_y_base_idx + img_x]

                            # Load kernel weights for simd_w output channels
                            # kernel[co:co+simd_w, ci, ky, kx]
                            var kernel_base = (
                                kernel_ci_base + ky * kernel_stride_H + kx
                            )
                            var kernel_vec = SIMD[Self.dtype, simd_w](0)

                            @parameter
                            for v in range(simd_w):
                                var k_idx = (
                                    co + v
                                ) * kernel_stride_Co + kernel_base
                                kernel_vec[v] = kernel_ptr[k_idx]

                            # Accumulate: accum += img_val * kernel_vec
                            accum += img_val * kernel_vec

                # Store results for simd_w output channels
                @parameter
                for v in range(simd_w):
                    var out_idx = out_n_base + (co + v) * out_stride_Co
                    out_ptr[out_idx] = accum[v]

                co += simd_w

            # ───────────────────────────────────────────────────────
            # Scalar tail for remaining output channels
            # ───────────────────────────────────────────────────────
            for co_tail in range(vec_end, C_out):
                var accum = bias_ptr[co_tail]

                for ci in range(C_in):
                    var img_ci_base = img_n_base + ci * img_stride_C
                    var kernel_co_ci_base = (
                        co_tail * kernel_stride_Co + ci * kernel_stride_Ci
                    )

                    for ky in range(KH):
                        var img_y = img_y_base + ky * dilation
                        var img_y_base_idx = img_ci_base + img_y * img_stride_H
                        var kernel_ky_base = (
                            kernel_co_ci_base + ky * kernel_stride_H
                        )

                        for kx in range(KW):
                            var img_x = img_x_base + kx * dilation
                            var img_val = img_ptr[img_y_base_idx + img_x]
                            var kernel_val = kernel_ptr[kernel_ky_base + kx]
                            accum += img_val * kernel_val

                var out_idx = out_n_base + co_tail * out_stride_Co
                out_ptr[out_idx] = accum

        parallelize[compute_position](total_work)

        # ═══════════════════════════════════════════════════════════
        # STEP 7: Setup gradient tracking
        # ═══════════════════════════════════════════════════════════
        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(
                padded_image.requires_grad
                or kernel.requires_grad
                or bias.requires_grad
            )

            if grad_required:
                output.requires_grad_(True)

                var backward_fn = FusedCol2ImBackward[Self.dtype](
                    N=N,
                    C_in=C_in,
                    H_pad=H_pad,
                    W_pad=W_pad,
                    C_out=C_out,
                    KH=KH,
                    KW=KW,
                    H_out=H_out,
                    W_out=W_out,
                    stride=stride,
                    dilation=dilation,
                ).into_backward_fn()

                output.backwardFn = Optional(backward_fn^)
                output.add_ancestry(padded_image)
                output.add_ancestry(kernel)
                output.add_ancestry(bias)

        return output^


@fieldwise_init
@register_passable
struct FusedCol2ImBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Convolution backward pass.

    1. Direct buffer access (no repeated indexing)
    2. SIMD vectorization on innermost loops
    3. Separate kernel/input gradients (no race conditions)
    4. Work with 4D directly (no reshape overhead)
    5. Cache-friendly access patterns

    Parallelization strategy:
    - Bias grad: Over C_out (each thread owns one channel)
    - Kernel grad: Over C_out (each thread owns grad_kernel[co, :, :, :])
    - Input grad: Over N (each thread owns grad_padded[n, :, :, :])
    """

    alias TAG = BACKWARD_FUSED_CONV
    var N: Int
    var C_in: Int
    var H_pad: Int
    var W_pad: Int
    var C_out: Int
    var KH: Int
    var KW: Int
    var H_out: Int
    var W_out: Int
    var stride: Int
    var dilation: Int

    fn backward(
        self,
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref grad_output = output.gradients()[]
        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()
        var padded_image = output.ancestry().get(0)
        var kernel = output.ancestry().get(1)
        var bias = output.ancestry().get(2)

        # BIAS GRADIENT
        if bias.requires_grad:
            var grad_bias = Self.compute_bias_gradient(
                grad_output, self.N, self.C_out, self.H_out, self.W_out
            )
            results.append((bias^, grad_bias^, AddTensor))

        # KERNEL GRADIENT
        if kernel.requires_grad:
            var grad_kernel = Self.compute_kernel_gradient(
                grad_output,
                padded_image,
                self.N,
                self.C_in,
                self.C_out,
                self.H_pad,
                self.W_pad,
                self.H_out,
                self.W_out,
                self.KH,
                self.KW,
                self.stride,
                self.dilation,
            )
            results.append((kernel, grad_kernel^, AddTensor))

        # INPUT GRADIENT
        if padded_image.requires_grad:
            var grad_padded = Self.compute_input_gradient(
                grad_output,
                kernel,
                self.N,
                self.C_in,
                self.C_out,
                self.H_pad,
                self.W_pad,
                self.H_out,
                self.W_out,
                self.KH,
                self.KW,
                self.stride,
                self.dilation,
            )
            results.append((padded_image^, grad_padded^, AddTensor))

        return results^

    # BIAS GRADIENT
    @always_inline
    @staticmethod
    fn compute_bias_gradient(
        grad_output: Gradbox[dtype], N: Int, C_out: Int, H_out: Int, W_out: Int
    ) -> Gradbox[dtype]:
        """
        Vectorized bias gradient: sum over (N, H_out, W_out) for each C_out.

        Memory layout: grad_output is contiguous (N, C_out, H_out, W_out).
        For each channel co, data is at indices:
          [n*C_out*H_out*W_out + co*H_out*W_out + spatial_pos]
        """
        var grad_bias = Gradbox[dtype].zeros(Shape(C_out), share=False)
        var grad_ptr = grad_output.buffer.data_buffer().data
        var bias_ptr = grad_bias.buffer.data_buffer().data

        alias simd_w = simd_width_of[dtype]()

        var stride_N = C_out * H_out * W_out
        var stride_C = H_out * W_out
        var num_spatial = H_out * W_out

        @parameter
        fn accumulate_bias(co: Int):
            var accum_vec = SIMD[dtype, simd_w](0)
            var accum_scalar: Scalar[dtype] = 0
            var base_co = co * stride_C

            for n in range(N):
                var base = n * stride_N + base_co

                # Vectorized accumulation over spatial dimensions
                var idx = 0
                var vec_end = (num_spatial // simd_w) * simd_w

                for _ in range(vec_end // simd_w):
                    accum_vec += grad_ptr.load[width=simd_w](base + idx)
                    idx += simd_w

                # Scalar tail
                for i in range(vec_end, num_spatial):
                    accum_scalar += grad_ptr[base + i]

            bias_ptr[co] = accum_vec.reduce_add() + accum_scalar

        parallelize[accumulate_bias](C_out)
        return grad_bias^

    @always_inline
    @staticmethod
    fn compute_kernel_gradient(
        grad_output: Gradbox[dtype],
        padded_image: Tensor[dtype],
        N: Int,
        C_in: Int,
        C_out: Int,
        H_pad: Int,
        W_pad: Int,
        H_out: Int,
        W_out: Int,
        KH: Int,
        KW: Int,
        stride: Int,
        dilation: Int,
    ) -> Gradbox[dtype]:
        """
        We need to transpose (N,C_out,H,W) → (C_out,N,H,W) before matmul.

        Strategy:
        1. Convert Gradbox → Tensor (to use view operations)
        2. Transpose + reshape Tensor (views, no copy!)
        3. Matmul
        4. Convert result back to Gradbox
        """

        # Extract patches
        var num_patches = N * H_out * W_out
        var patch_size = C_in * KH * KW
        var patches = Tensor[dtype].zeros(num_patches, patch_size)

        var input_ptr = padded_image.buffer.data_buffer().data
        var patch_ptr = patches.buffer.data_buffer().data

        var input_stride_N = C_in * H_pad * W_pad
        var input_stride_C = H_pad * W_pad

        var total_work = N * H_out * W_out

        @parameter
        fn extract_patch(work_idx: Int):
            var n = work_idx // (H_out * W_out)
            var spatial = work_idx % (H_out * W_out)
            var oy = spatial // W_out
            var ox = spatial % W_out

            var patch_base = work_idx * patch_size
            var input_n_base = n * input_stride_N

            var elem_idx = 0
            for ci in range(C_in):
                var input_ci_base = input_n_base + ci * input_stride_C

                for ky in range(KH):
                    var img_y = oy * stride + ky * dilation
                    var input_row_base = input_ci_base + img_y * W_pad

                    for kx in range(KW):
                        var img_x = ox * stride + kx * dilation
                        patch_ptr[patch_base + elem_idx] = input_ptr[
                            input_row_base + img_x
                        ]
                        elem_idx += 1

        parallelize[extract_patch](total_work)

        # Convert grad_output to Tensor (consumes Gradbox)
        var grad_tensor = grad_output.as_tensor(requires_grad=False)
        # grad_tensor shape: (N, C_out, H_out, W_out)

        # Transpose (N, C_out, H_out, W_out) → (C_out, N, H_out, W_out)
        # This creates a VIEW with different strides (no data copy!)
        var grad_transposed = grad_tensor.transpose(IntArray(1, 0, 2, 3))
        # Shape: (C_out, N, H_out, W_out)

        # Reshape (C_out, N, H_out, W_out) → (C_out, N*H_out*W_out)
        var grad_reshaped = grad_transposed.reshape(C_out, num_patches)
        # Shape: (C_out, num_patches)

        # Matrix multiplication
        # grad_kernel_flat = grad_reshaped @ patches
        # Shape: (C_out, patch_size) where patch_size = C_in*KH*KW
        var grad_kernel_flat_tensor = grad_reshaped.matmul(patches)

        # Reshape to (C_out, C_in, KH, KW)
        var grad_kernel_tensor = grad_kernel_flat_tensor.reshape(
            C_out, C_in, KH, KW
        )

        # Convert back to Gradbox (consumes Tensor)
        # contiguous=True ensures the result is contiguous (will copy if needed)
        # share=False means the buffer is NOT ref-counted (owned by this Gradbox)
        return grad_kernel_tensor.as_gradbox(share=False, contiguous=True)

    # INPUT GRADIENT
    @always_inline
    @staticmethod
    fn compute_input_gradient(
        grad_output: Gradbox[dtype],
        kernel: Tensor[dtype],
        N: Int,
        C_in: Int,
        C_out: Int,
        H_pad: Int,
        W_pad: Int,
        H_out: Int,
        W_out: Int,
        KH: Int,
        KW: Int,
        stride: Int,
        dilation: Int,
    ) -> Gradbox[dtype]:
        """
        Vectorized input gradient.

        Parallelized over N (no race - each n owns grad_padded[n, :, :, :]).

        Formula:
          grad_input[n, ci, y, x] += sum_{co, ky, kx | y=oy*s+ky*d, x=ox*s+kx*d}
            kernel[co, ci, ky, kx] * grad_output[n, co, oy, ox]

        Note: Hard to vectorize due to scatter pattern. Focus on cache optimization.
        """
        var grad_padded = Gradbox[dtype].zeros(
            Shape(N, C_in, H_pad, W_pad), share=False
        )

        var grad_ptr = grad_output.buffer.data_buffer().data
        var kernel_ptr = kernel.buffer.data_buffer().data
        var grad_in_ptr = grad_padded.buffer.data_buffer().data

        # Strides
        var grad_stride_N = C_out * H_out * W_out
        var grad_stride_C = H_out * W_out

        var kernel_stride_Co = C_in * KH * KW
        var kernel_stride_Ci = KH * KW

        var input_stride_N = C_in * H_pad * W_pad
        var input_stride_C = H_pad * W_pad

        @parameter
        fn compute_input_for_n(n: Int):
            var grad_n_base = n * grad_stride_N
            var input_n_base = n * input_stride_N

            # Iterate over output spatial positions
            for oy in range(H_out):
                var grad_oy_base = grad_n_base + oy * W_out

                for ox in range(W_out):
                    var grad_spatial_idx = grad_oy_base + ox

                    # For each output channel
                    for co in range(C_out):
                        var grad_val = grad_ptr[
                            grad_spatial_idx + co * grad_stride_C
                        ]

                        # Early skip for zeros
                        if grad_val == Scalar[dtype](0):
                            continue

                        var kernel_co_base = co * kernel_stride_Co

                        # Scatter to input
                        for ci in range(C_in):
                            var kernel_ci_base = (
                                kernel_co_base + ci * kernel_stride_Ci
                            )
                            var input_ci_base = (
                                input_n_base + ci * input_stride_C
                            )

                            # Inner loops: can potentially vectorize over KW
                            for ky in range(KH):
                                var img_y = oy * stride + ky * dilation
                                var kernel_ky_base = kernel_ci_base + ky * KW
                                var input_y_base = input_ci_base + img_y * W_pad

                                # Vectorize over KW if it's large enough
                                # @parameter
                                if KW >= 4:  # Worth vectorizing
                                    alias kw_simd_w = min(
                                        4, simd_width_of[dtype]()
                                    )
                                    var kx = 0
                                    var vec_end = (KW // kw_simd_w) * kw_simd_w

                                    for _ in range(vec_end // kw_simd_w):
                                        # Load kernel weights
                                        var kernel_vec = kernel_ptr.load[
                                            width=kw_simd_w
                                        ](kernel_ky_base + kx)

                                        # Compute contributions
                                        var contrib_vec = grad_val * kernel_vec

                                        # Scatter (can't vectorize scatter, but at least vectorize multiply)
                                        @parameter
                                        for v in range(kw_simd_w):
                                            var img_x = (
                                                ox * stride
                                                + (kx + v) * dilation
                                            )
                                            grad_in_ptr[
                                                input_y_base + img_x
                                            ] += contrib_vec[v]

                                        kx += kw_simd_w

                                    # Scalar tail
                                    for kx_tail in range(vec_end, KW):
                                        var img_x = (
                                            ox * stride + kx_tail * dilation
                                        )
                                        var kernel_val = kernel_ptr[
                                            kernel_ky_base + kx_tail
                                        ]
                                        grad_in_ptr[input_y_base + img_x] += (
                                            grad_val * kernel_val
                                        )
                                else:
                                    # Small KW - just use scalar
                                    for kx in range(KW):
                                        var img_x = ox * stride + kx * dilation
                                        var kernel_val = kernel_ptr[
                                            kernel_ky_base + kx
                                        ]
                                        grad_in_ptr[input_y_base + img_x] += (
                                            grad_val * kernel_val
                                        )

        parallelize[compute_input_for_n](N)
        return grad_padded^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)
