from tenmo import Tensor
from backpropagation import (
    BackwardFn,
    Delegate,
    BACKWARD_FUSED_CONV,
    BACKWARD_MAXPOOL2D,
)
from operators import AddTensor
from common_utils import panic, now
from forwards import Pad
from shapes import Shape
from gradbox import Gradbox
from forwards import Padding
from algorithm import parallelize
from sys import simd_width_of
from ndbuffer import NDBuffer
from memory import memset_zero, memcpy
from intarray import IntArray
from os import Atomic
from os.atomic import Consistency


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
        start = now()
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
        print("Conv2dFused forward took: ", (now() - start) * 1000, "ms")
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
    Production-grade fused Im2Col + Conv + Bias operation.

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
        start = now()
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

        print("FusedIm2Col forward took: ", (now() - start) * 1000, "ms")
        return output^


@fieldwise_init
@register_passable
struct FusedCol2ImBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Convolution backward pass.

    Key optimizations:
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
        start = now()
        var padded_image = output.ancestry().get(0)
        var kernel = output.ancestry().get(1)
        var bias = output.ancestry().get(2)

        # ═══════════════════════════════════════════════════════════
        # 1. BIAS GRADIENT - Vectorized
        # ═══════════════════════════════════════════════════════════
        if bias.requires_grad:
            var grad_bias = Self.compute_bias_gradient(
                grad_output, self.N, self.C_out, self.H_out, self.W_out
            )
            results.append((bias^, grad_bias^, AddTensor))

        # ═══════════════════════════════════════════════════════════
        # 2. KERNEL GRADIENT - Vectorized
        # ═══════════════════════════════════════════════════════════
        if kernel.requires_grad:
            var grad_kernel = Self.compute_kernel_gradient_im2col_gemm(
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

        # ═══════════════════════════════════════════════════════════
        # 3. INPUT GRADIENT
        # ═══════════════════════════════════════════════════════════
        if padded_image.requires_grad:
            var grad_padded = Self.compute_input_gradient(  # best so far
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

        print("FusedCol2ImBackward took: ", (now() - start) * 1000, "ms")

        return results^

    # ═══════════════════════════════════════════════════════════════════
    # BIAS GRADIENT
    # ═══════════════════════════════════════════════════════════════════
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

    # ==============================================================================
    # ALTERNATIVE: Im2Col + GEMM approach (What PyTorch uses!)
    # ==============================================================================
    @always_inline
    @staticmethod
    fn compute_kernel_gradient_im2col_gemm(
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
        PyTorch-style kernel gradient using im2col + matmul.

        Key insight: We need to transpose (N,C_out,H,W) → (C_out,N,H,W) before matmul

        Strategy:
        1. Convert Gradbox → Tensor (to use view operations)
        2. Transpose + reshape Tensor (views, no copy!)
        3. Matmul
        4. Convert result back to Gradbox
        """

        # ═══════════════════════════════════════════════════════════════
        # Step 1: Extract patches (im2col) - Same as before
        # ═══════════════════════════════════════════════════════════════
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

        # ═══════════════════════════════════════════════════════════════
        # Step 2: Convert grad_output to Tensor (consumes Gradbox)
        # ═══════════════════════════════════════════════════════════════
        var grad_tensor = grad_output.as_tensor(requires_grad=False)
        # grad_tensor shape: (N, C_out, H_out, W_out)

        # ═══════════════════════════════════════════════════════════════
        # Step 3: Transpose (N, C_out, H_out, W_out) → (C_out, N, H_out, W_out)
        # ═══════════════════════════════════════════════════════════════
        # This creates a VIEW with different strides (no data copy!)
        var grad_transposed = grad_tensor.transpose(IntArray(1, 0, 2, 3))
        # Shape: (C_out, N, H_out, W_out)

        # ═══════════════════════════════════════════════════════════════
        # Step 4: Reshape (C_out, N, H_out, W_out) → (C_out, N*H_out*W_out)
        # ═══════════════════════════════════════════════════════════════
        # This is just a metadata change (VIEW!)
        var grad_reshaped = grad_transposed.reshape(C_out, num_patches)
        # Shape: (C_out, num_patches)

        # ═══════════════════════════════════════════════════════════════
        # Step 5: Matrix multiplication
        # ═══════════════════════════════════════════════════════════════
        # grad_kernel_flat = grad_reshaped @ patches
        # Shape: (C_out, patch_size) where patch_size = C_in*KH*KW
        var grad_kernel_flat_tensor = grad_reshaped.matmul(patches)

        # ═══════════════════════════════════════════════════════════════
        # Step 6: Reshape to (C_out, C_in, KH, KW)
        # ═══════════════════════════════════════════════════════════════
        # This is a VIEW (no copy)
        var grad_kernel_tensor = grad_kernel_flat_tensor.reshape(
            C_out, C_in, KH, KW
        )

        # ═══════════════════════════════════════════════════════════════
        # Step 7: Convert back to Gradbox (consumes Tensor)
        # ═══════════════════════════════════════════════════════════════
        # contiguous=True ensures the result is contiguous (will copy if needed)
        # share=False means the buffer is NOT ref-counted (owned by this Gradbox)
        return grad_kernel_tensor.as_gradbox(share=False, contiguous=True)

    # ═══════════════════════════════════════════════════════════════════
    # INPUT GRADIENT WITH BATCHED MATMUL (OPTIMAL!)
    # ═══════════════════════════════════════════════════════════════════

    @always_inline
    @staticmethod
    fn compute_input_gradient_gemm_batched(
        grad_output: Gradbox[dtype],
        mut kernel: Tensor[dtype],
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
        Compute input gradient using batched matmul.

        This is BETTER than the flattened approach because:
        1. No transpose of grad_output needed
        2. Col2im can be parallelized over batches (no race conditions)
        3. Better cache locality

        Steps:
        1. Reshape kernel: (C_out, C_in, KH, KW) → (C_in*KH*KW, C_out)
        2. Reshape grad_output: (N, C_out, H_out, W_out) → (N, C_out, H_out*W_out)
        3. Batched matmul: (C_in*KH*KW, C_out) @ (N, C_out, H_out*W_out)
           Result: (N, C_in*KH*KW, H_out*W_out)
        4. Col2im: Scatter patches back to (N, C_in, H_pad, W_pad)
        """

        var patch_size = C_in * KH * KW
        var num_spatial = H_out * W_out

        # ═══════════════════════════════════════════════════════════════
        # Step 1: Reshape kernel to (C_out, patch_size) then transpose
        # ═══════════════════════════════════════════════════════════════
        var kernel_reshaped = kernel.reshape(C_out, patch_size)
        # Shape: (C_out, C_in*KH*KW)

        var kernel_transposed = kernel_reshaped.transpose(IntArray(1, 0))
        # Shape: (C_in*KH*KW, C_out)

        # ═══════════════════════════════════════════════════════════════
        # Step 2: Convert grad_output to Tensor and reshape
        # NO TRANSPOSE NEEDED!
        # ═══════════════════════════════════════════════════════════════
        var grad_tensor = grad_output.as_tensor(requires_grad=False)
        # Shape: (N, C_out, H_out, W_out)

        var grad_reshaped = grad_tensor.reshape(N, C_out, num_spatial)
        # Shape: (N, C_out, H_out*W_out)

        # ═══════════════════════════════════════════════════════════════
        # Step 3: Batched matrix multiplication
        # (C_in*KH*KW, C_out) @ (N, C_out, H_out*W_out)
        # ═══════════════════════════════════════════════════════════════
        var patches_grad = kernel_transposed.matmul(grad_reshaped)
        # Shape: (N, C_in*KH*KW, H_out*W_out)

        # ═══════════════════════════════════════════════════════════════
        # Step 4: Col2im - Scatter patches back to image
        # Parallelize over batches (no race conditions!)
        # ═══════════════════════════════════════════════════════════════
        var grad_padded = Gradbox[dtype].zeros(
            Shape(N, C_in, H_pad, W_pad), share=False
        )

        var patches_ptr = patches_grad.buffer.data_buffer().data
        var grad_in_ptr = grad_padded.buffer.data_buffer().data

        # Strides for grad_padded (output)
        var input_stride_N = C_in * H_pad * W_pad
        var input_stride_C = H_pad * W_pad

        # Strides for patches_grad (input)
        # Shape: (N, patch_size, num_spatial)
        var patch_stride_N = patch_size * num_spatial
        var patch_stride_elem = num_spatial

        @parameter
        fn col2im_for_batch(n: Int):
            """Scatter patches for batch n back to grad_input[n]."""
            var patches_n_base = n * patch_stride_N
            var input_n_base = n * input_stride_N

            # Process each output spatial position
            for spatial_idx in range(num_spatial):
                var oy = spatial_idx // W_out
                var ox = spatial_idx % W_out

                var patch_elem = 0

                # Process each element in the patch
                for ci in range(C_in):
                    var input_ci_base = input_n_base + ci * input_stride_C

                    for ky in range(KH):
                        var img_y = oy * stride + ky * dilation
                        var input_row_base = input_ci_base + img_y * W_pad

                        for kx in range(KW):
                            var img_x = ox * stride + kx * dilation

                            # Get gradient value from patch
                            # patches_grad layout: (N, patch_size, num_spatial)
                            # Index: patches_n_base + patch_elem*num_spatial + spatial_idx
                            var patch_idx = (
                                patches_n_base
                                + patch_elem * patch_stride_elem
                                + spatial_idx
                            )
                            var grad_val = patches_ptr[patch_idx]

                            # Accumulate to input gradient
                            # Note: Within a single batch, patches can overlap if stride < kernel_size
                            # But since we're processing one batch at a time, no race between batches!
                            grad_in_ptr[input_row_base + img_x] += grad_val

                            patch_elem += 1

        # Parallelize over batches - completely safe!
        parallelize[col2im_for_batch](N)

        return grad_padded^

    @always_inline
    @staticmethod
    fn compute_kernel_gradient_im2col_gemm_backup(
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
        PyTorch-style kernel gradient using im2col + matmul.

        This is what PyTorch actually does and why it's fast!

        Mathematical insight:
        grad_kernel[co, ci, ky, kx] = sum_{n, oy, ox}
            grad_output[n, co, oy, ox] * input[n, ci, oy*s+ky*d, ox*s+kx*d]

        Rewrite as matrix multiplication:
        grad_kernel_flat = grad_output_reshaped @ input_col

        Where:
        - grad_output_reshaped: (C_out, N*H_out*W_out)
        - input_col: (N*H_out*W_out, C_in*KH*KW)  ← im2col of input
        - grad_kernel_flat: (C_out, C_in*KH*KW) → reshape to (C_out, C_in, KH, KW)
        """

        # Step 1: Extract patches (im2col)
        var num_patches = N * H_out * W_out
        var patch_size = C_in * KH * KW
        var patches = Tensor[dtype].zeros(num_patches, patch_size)

        var input_ptr = padded_image.buffer.data_buffer().data
        var patch_ptr = patches.buffer.data_buffer().data

        var input_stride_N = C_in * H_pad * W_pad
        var input_stride_C = H_pad * W_pad

        # Extract patches in parallel
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

        # Step 2: Reshape grad_output to (C_out, num_patches)
        var grad_reshaped = grad_output.reshape(Shape(C_out, num_patches))

        print("Grad reshaped shape: ", grad_reshaped.shape())
        # var num_patches = N * H_out * W_out
        # var patch_size = C_in * KH * KW
        # var patches = Tensor[dtype].zeros(num_patches, patch_size)

        # Step 3: Matrix multiplication
        # grad_kernel_flat = grad_reshaped @ patches
        # Result shape: (C_out, patch_size)
        var grad_kernel_flat = grad_reshaped.matmul(patches)

        # Step 4: Reshape to (C_out, C_in, KH, KW)
        return grad_kernel_flat.reshape(Shape(C_out, C_in, KH, KW), share=False)

    @always_inline
    @staticmethod
    fn compute_input_gradient_gemm_atomic(
        grad_output: Gradbox[dtype],
        mut kernel: Tensor[dtype],
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
        GEMM-based input gradient with atomic operations for overlapping patches.

        Same as compute_input_gradient_gemm but uses atomic adds to handle
        race conditions when stride < kernel_size.
        """

        var num_patches = N * H_out * W_out
        var patch_size = C_in * KH * KW

        # Steps 1-3: Same as before (reshape + matmul)
        var kernel_reshaped = kernel.reshape(C_out, patch_size)
        var kernel_transposed = kernel_reshaped.transpose(IntArray(1, 0))

        var grad_tensor = grad_output.as_tensor(requires_grad=False)
        var grad_transposed = grad_tensor.transpose(IntArray(1, 0, 2, 3))
        var grad_reshaped = grad_transposed.reshape(C_out, num_patches)

        var patches_grad = kernel_transposed.matmul(grad_reshaped)

        # Step 4: Col2im with atomic operations
        var grad_padded = Gradbox[dtype].zeros(
            Shape(N, C_in, H_pad, W_pad), share=False
        )

        var patches_ptr = patches_grad.buffer.data_buffer().data
        var grad_in_ptr = grad_padded.buffer.data_buffer().data

        var input_stride_N = C_in * H_pad * W_pad
        var input_stride_C = H_pad * W_pad

        var total_work = N * H_out * W_out

        @parameter
        fn scatter_patch_atomic(work_idx: Int):
            var n = work_idx // (H_out * W_out)
            var spatial = work_idx % (H_out * W_out)
            var oy = spatial // W_out
            var ox = spatial % W_out

            var input_n_base = n * input_stride_N
            var patch_elem = 0

            for ci in range(C_in):
                var input_ci_base = input_n_base + ci * input_stride_C

                for ky in range(KH):
                    var img_y = oy * stride + ky * dilation
                    var input_row_base = input_ci_base + img_y * W_pad

                    for kx in range(KW):
                        var img_x = ox * stride + kx * dilation
                        var grad_val = patches_ptr[
                            patch_elem * num_patches + work_idx
                        ]

                        if dtype == DType.float32:
                            # Use atomic add for float32
                            var addr = grad_in_ptr + (input_row_base + img_x)
                            Atomic.store[ordering = Consistency.SEQUENTIAL](
                                addr, grad_val
                            )
                        patch_elem += 1

        # Always parallelize with atomic operations
        parallelize[scatter_patch_atomic](total_work)

        return grad_padded^

    # ═══════════════════════════════════════════════════════════════════
    # INPUT GRADIENT - SIMD Vectorized
    # ═══════════════════════════════════════════════════════════════════
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


fn main() raises:
    alias dtype = DType.float32
    # Batch of 2, 3 input channels, 4x5 image
    # var x = Tensor[dtype].rand(2, 3, 4, 5, requires_grad=True)
    return
    var x = Tensor[dtype].d4(
        [
            [
                [
                    [1.9269, 1.4873, 0.9007, -2.1055, 0.6784],
                    [-1.2345, -0.0431, -1.6047, -0.7521, 1.6487],
                    [-0.3925, -1.4036, -0.7279, -0.5594, -0.7688],
                    [0.7624, 1.6423, -0.1596, -0.4974, 0.4396],
                ],
                [
                    [-0.7581, 1.0783, 0.8008, 1.6806, 1.2791],
                    [1.2964, 0.6105, 1.3347, -0.2316, 0.0418],
                    [-0.2516, 0.8599, -1.3847, -0.8712, -0.2234],
                    [1.7174, 0.3189, -0.4245, 0.3057, -0.7746],
                ],
                [
                    [-1.5576, 0.9956, -0.8798, -0.6011, -1.2742],
                    [2.1228, -1.2347, -0.4879, -0.9138, -0.6581],
                    [0.0780, 0.5258, -0.4880, 1.1914, -0.8140],
                    [-0.7360, -1.4032, 0.0360, -0.0635, 0.6756],
                ],
            ],
            [
                [
                    [-0.0978, 1.8446, -1.1845, 1.3835, 1.4451],
                    [0.8564, 2.2181, 0.5232, 0.3466, -0.1973],
                    [-1.0546, 1.2780, -0.1722, 0.5238, 0.0566],
                    [0.4263, 0.5750, -0.6417, -2.2064, -0.7508],
                ],
                [
                    [0.0109, -0.3387, -1.3407, -0.5854, 0.5362],
                    [0.5246, 1.1412, 0.0516, 0.7440, -0.4816],
                    [-1.0495, 0.6039, -1.7223, -0.8278, 1.3347],
                    [0.4835, -2.5095, 0.4880, 0.7846, 0.0286],
                ],
                [
                    [0.6408, 0.5832, 1.0669, -0.4502, 1.0311],
                    [-0.7048, 1.0131, -0.3308, 0.5177, 0.3878],
                    [-0.5797, -0.1691, -0.5733, 0.5069, -0.4752],
                    [-0.4920, 0.2704, -0.5628, 0.6793, 0.4405],
                ],
            ],
        ],
        requires_grad=True,
    )

    # 4 output filters, 3 input channels, 3x3 kernel
    # var kernel = Tensor[dtype].rand(4, 3, 3, 3, requires_grad=True)
    var kernel = Tensor[dtype].d4(
        [
            [
                [
                    [-0.3609, -0.0606, 0.0733],
                    [0.8187, 1.4805, 0.3449],
                    [-1.4241, -0.1163, 0.2176],
                ],
                [
                    [-0.0467, -1.4335, -0.5665],
                    [-0.4253, 0.2625, -1.4391],
                    [0.5214, 1.0414, -0.3997],
                ],
                [
                    [-2.2933, 0.4976, -0.4257],
                    [-1.3371, -0.1933, 0.6526],
                    [-0.3063, -0.3302, -0.9808],
                ],
            ],
            [
                [
                    [0.1947, -1.6535, 0.6814],
                    [1.4611, -0.3098, 0.9633],
                    [-0.3095, 0.5712, 1.1179],
                ],
                [
                    [-1.2956, 0.0503, -0.5855],
                    [-0.3900, 0.9812, -0.6401],
                    [-0.4908, 0.2080, -1.1586],
                ],
                [
                    [-0.9637, -0.3750, 0.8033],
                    [0.7165, 1.5335, -1.4510],
                    [-0.7861, -0.9563, -1.2476],
                ],
            ],
            [
                [
                    [-0.7499, -0.5922, -1.5326],
                    [-0.7251, 0.4664, 0.6667],
                    [-0.0439, 0.2368, -0.7061],
                ],
                [
                    [-0.7169, -0.1593, -0.4249],
                    [0.9442, -0.1849, 1.0608],
                    [0.2083, -0.5778, 0.3255],
                ],
                [
                    [0.2618, -0.7599, -2.0461],
                    [-1.5295, 0.4049, 0.6319],
                    [0.3125, -0.0335, 1.3032],
                ],
            ],
            [
                [
                    [0.4879, 1.1340, -0.3556],
                    [0.3618, 1.9993, 0.6630],
                    [0.7047, 0.0213, -0.8293],
                ],
                [
                    [-1.0809, -0.7839, -0.8719],
                    [-0.0271, -0.3532, 1.4639],
                    [0.1729, 1.0514, 0.0075],
                ],
                [
                    [-0.0774, 0.5397, 0.5655],
                    [0.5058, 0.2225, -0.9143],
                    [1.4840, -0.9109, -0.5291],
                ],
            ],
        ],
        requires_grad=True,
    )

    # Bias per output channel
    # var bias = Tensor[dtype].rand(4, requires_grad=True)
    var bias = Tensor[dtype].d1(
        [1.2818, -1.5952, -1.0648, 0.1055], requires_grad=True
    )

    # var output = Conv2DBackward[dtype].forward(
    var output = Conv2dFused[dtype].forward(
        image=x,
        kernel=kernel,
        bias=bias,
        stride=1,
        dilation=1,
        padding=Padding("same"),  # or Padding(1), etc.
    )

    output.backward(Tensor[dtype].ones_like(output))

    print("Input grad shape:", x.grad().shape())
    print("Kernel grad shape:", kernel.grad().shape())
    print("Bias grad shape:", bias.grad().shape())

    print("\noutput\n")
    output.print()

    print()
    x.grad().print()
    print()
    kernel.grad().print()
    print()
    bias.grad().print()


from tenmo import Tensor
from backpropagation import BackwardFn, Delegate  # , BACKWARD_CONV2D
from operators import AddTensor
from common_utils import panic, now
from gradbox import Gradbox
from utils import Variant
from shapes import Shape
from algorithm import parallelize, vectorize
from sys import simd_width_of
from forwards import Padding
from os import Atomic


@fieldwise_init
@register_passable
struct Conv2DBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Custom backward for batched, multi-channel, multi-filter Conv2dForward.
    Correctly computes gradients for input, kernel, and bias.
    Optimized with SIMD vectorization where beneficial.
    """

    alias TAG = -9999  # BACKWARD_CONV2D
    var stride: Int
    var dilation: Int
    var pad_top: Int
    var pad_bottom: Int
    var pad_left: Int
    var pad_right: Int

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref grad_output = output.gradients()[]
        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()
        start = now()
        # Ancestry: 0=input, 1=kernel, 2=bias (bias may not exist if None)
        var image = output.ancestry().get(0)
        var kernel = output.ancestry().get(1)
        var has_bias = len(output.ancestry()) > 2

        ref input_shape = image.shape()  # [N, C_in, H_in, W_in]
        ref kernel_shape = kernel.shape()  # [C_out, C_in, KH, KW]
        var bias = (
            output.ancestry()
            .get(2) if has_bias else Tensor[Self.dtype]
            .full(Shape(kernel_shape[0]), 0, requires_grad=False)
        )
        ref output_shape = output.shape()  # [N, C_out, H_out, W_out]

        var N = input_shape[0]
        var C_in = input_shape[1]
        var H_in = input_shape[2]
        var W_in = input_shape[3]

        var C_out = kernel_shape[0]
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]

        var H_out = output_shape[2]
        var W_out = output_shape[3]

        var dil = self.dilation
        var stride_val = self.stride  # ← Hoist for repeated use

        alias simd_w = simd_width_of[Self.dtype]()
        # ═══════════════════════════════════════════════════════════
        # 1. BIAS GRADIENT (Fully Optimized SIMD)
        # ═══════════════════════════════════════════════════════════
        if has_bias and bias.requires_grad:
            var grad_bias = Gradbox[Self.dtype].zeros(bias.shape(), share=False)

            @parameter
            fn compute_bias_channel(o: Int):
                # Contiguous memory layout: can use raw pointer
                var ptr = grad_output.buffer.buffer.data + o * (
                    N * H_out * W_out
                )
                var size = N * H_out * W_out
                var accum = SIMD[Self.dtype, simd_w](0)

                # Vectorized main loop
                var i = 0
                var vec_end = (size // simd_w) * simd_w
                for _ in range(vec_end // simd_w):
                    accum += ptr.load[width=simd_w](i)
                    i += simd_w

                # Horizontal reduction
                var scalar_accum = accum.reduce_add()

                # Scalar tail
                for j in range(i, size):
                    scalar_accum += ptr[j]

                grad_bias[o] = scalar_accum

            parallelize[compute_bias_channel](C_out)
            results.append((bias^, grad_bias^, AddTensor))

        # ═══════════════════════════════════════════════════════════
        # 2. KERNEL GRADIENT (Optimized SIMD)
        # ═══════════════════════════════════════════════════════════
        if kernel.requires_grad:
            var grad_kernel = Gradbox[Self.dtype].zeros(
                kernel_shape, share=False
            )

            @parameter
            fn compute_kernel_channel(o: Int):
                for i in range(C_in):
                    for ky in range(KH):
                        for kx in range(KW):
                            # Separate accumulators for vector and scalar portions
                            var accum_vec = SIMD[Self.dtype, simd_w](0)
                            var accum_scalar: Scalar[Self.dtype] = 0

                            # Hoist loop-invariant calculations
                            var ky_off = ky * dil
                            var kx_off = kx * dil

                            for n in range(N):
                                for y in range(H_out):
                                    var iy = (
                                        y * stride_val + ky_off - self.pad_top
                                    )

                                    # Early exit if row is out of bounds
                                    if iy < 0 or iy >= H_in:
                                        continue

                                    var ix_base = kx_off - self.pad_left
                                    var vec_end = (W_out // simd_w) * simd_w

                                    # Precompute the starting pointer for this (n,o,y) row
                                    var row_offset = (
                                        n * (C_out * H_out * W_out)
                                        + o * (H_out * W_out)
                                        + y * W_out
                                    )
                                    var row_ptr = (
                                        grad_output.buffer.buffer.data
                                        + row_offset
                                    )

                                    # Vectorized loop
                                    for x in range(0, vec_end, simd_w):
                                        var grad_vec = row_ptr.load[
                                            width=simd_w
                                        ](x)

                                        var img_vec = SIMD[Self.dtype, simd_w](
                                            0
                                        )
                                        for v in range(simd_w):
                                            var ix = (
                                                x + v
                                            ) * self.stride + ix_base
                                            if ix >= 0 and ix < W_in:
                                                img_vec[v] = image[n, i, iy, ix]

                                        accum_vec += img_vec * grad_vec

                                    # Vectorized loop - processes simd_w output positions at once
                                    # Scalar tail for remaining elements
                                    for x in range(vec_end, W_out):
                                        var ix = x * stride_val + ix_base
                                        if ix >= 0 and ix < W_in:
                                            accum_scalar += (
                                                image[n, i, iy, ix]
                                                * grad_output[n, o, y, x]
                                            )

                            # Combine vector and scalar results
                            grad_kernel[o, i, ky, kx] = (
                                accum_vec.reduce_add() + accum_scalar
                            )

            parallelize[compute_kernel_channel](C_out)
            results.append((kernel, grad_kernel^, AddTensor))

        # ═══════════════════════════════════════════════════════════
        # 3. INPUT GRADIENT (Scalar - memory pattern too irregular)
        # ═══════════════════════════════════════════════════════════
        if image.requires_grad:
            var grad_input = Gradbox[Self.dtype].zeros(input_shape, share=False)

            @parameter
            fn compute_input_batch(n: Int):
                for o in range(C_out):
                    for y in range(H_out):
                        for x in range(W_out):
                            var g = grad_output[n, o, y, x]
                            for i in range(C_in):
                                for ky in range(KH):
                                    for kx in range(KW):
                                        var iy = (
                                            y * stride_val
                                            + ky * dil
                                            - self.pad_top
                                        )
                                        var ix = (
                                            x * stride_val
                                            + kx * dil
                                            - self.pad_left
                                        )
                                        if (
                                            iy >= 0
                                            and iy < H_in
                                            and ix >= 0
                                            and ix < W_in
                                        ):
                                            grad_input[n, i, iy, ix] += (
                                                g * kernel[o, i, ky, kx]
                                            )

            parallelize[compute_input_batch](N)
            results.append((image^, grad_input^, AddTensor))

        end = now()
        print(
            "Conv2DBackward(convolution) backward took: ",
            end * 1000 - start * 1000,
            "ms",
        )
        return results^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Conv2dForward[dtype: DType](ImplicitlyCopyable):
    """
    Batched, multi-channel, multi-filter 2D convolution with optimized loops.

    Args:
        image:  (N, C_in, H_in, W_in)
        kernel: (C_out, C_in, KH, KW)
        bias:   Optional (C_out,)
        stride: Stride for spatial dimensions
        dilation: Dilation factor for atrous convolution
        padding: 'valid', 'same', int, tuple, or list of tuples

    Returns:
        output: (N, C_out, H_out, W_out)
    """

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        kernel: Tensor[Self.dtype],
        bias: Optional[Tensor[Self.dtype]] = None,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Padding = Padding("valid"),
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        start = now()

        ref input_shape = image.shape()
        ref kernel_shape = kernel.shape()

        # ═══════════════════════════════════════════════════════════
        # Validation
        # ═══════════════════════════════════════════════════════════
        if input_shape.rank() != 4:
            panic("Image must be 4D: (N, C_in, H_in, W_in)")
        if kernel_shape.rank() != 4:
            panic("Kernel must be 4D: (C_out, C_in, KH, KW)")

        var N = input_shape[0]
        var C_in = input_shape[1]
        var H_in = input_shape[2]
        var W_in = input_shape[3]

        var C_out = kernel_shape[0]
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]

        if kernel_shape[1] != C_in:
            panic("Kernel input channels must match input channels")

        var dil = dilation
        var dilated_KH = KH + (KH - 1) * (dil - 1)
        var dilated_KW = KW + (KW - 1) * (dil - 1)

        # ═══════════════════════════════════════════════════════════
        # Parse Padding
        # ═══════════════════════════════════════════════════════════
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
            var lst = padding[List[Tuple[Int, Int]]].copy()
            if len(lst) != 2:
                panic("Padding list must contain exactly 2 tuples")
            pad_top = lst[0][0]
            pad_bottom = lst[0][1]
            pad_left = lst[1][0]
            pad_right = lst[1][1]
        else:
            panic("Invalid padding type")

        # ═══════════════════════════════════════════════════════════
        # Compute Output Shape
        # ═══════════════════════════════════════════════════════════
        var H_out = (H_in + pad_top + pad_bottom - dilated_KH) // stride + 1
        var W_out = (W_in + pad_left + pad_right - dilated_KW) // stride + 1

        if H_out <= 0 or W_out <= 0:
            panic(
                "Invalid convolution parameters lead to non-positive output"
                " size"
            )

        var output = Tensor[Self.dtype].zeros(N, C_out, H_out, W_out)

        # ═══════════════════════════════════════════════════════════
        # Setup Bias
        # ═══════════════════════════════════════════════════════════
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

        # ═══════════════════════════════════════════════════════════
        # Forward Convolution (Scalar - simple and correct)
        # ═══════════════════════════════════════════════════════════
        @parameter
        fn compute_output_batch(n: Int):
            for o in range(C_out):
                var bias_val = bias_tensor[o]

                for y in range(H_out):
                    for x in range(W_out):
                        var accum = bias_val

                        for i in range(C_in):
                            for ky in range(KH):
                                for kx in range(KW):
                                    var iy = y * stride + ky * dil - pad_top
                                    var ix = x * stride + kx * dil - pad_left

                                    if (
                                        iy >= 0
                                        and iy < H_in
                                        and ix >= 0
                                        and ix < W_in
                                    ):
                                        accum += (
                                            image[n, i, iy, ix]
                                            * kernel[o, i, ky, kx]
                                        )

                        output[n, o, y, x] = accum

        parallelize[compute_output_batch](N)

        # ═══════════════════════════════════════════════════════════
        # Gradient Setup
        # ═══════════════════════════════════════════════════════════
        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(
                image.requires_grad
                or kernel.requires_grad
                or bias_tensor.requires_grad
            )
            if grad_required:
                output.requires_grad_(True)
                var backward_fn = Conv2DBackward[Self.dtype](
                    stride=stride,
                    dilation=dilation,
                    pad_top=pad_top,
                    pad_bottom=pad_bottom,
                    pad_left=pad_left,
                    pad_right=pad_right,
                ).into_backward_fn()
                output.backwardFn = Optional(backward_fn^)
                output.add_ancestry(image)
                output.add_ancestry(kernel)
                if bias:
                    output.add_ancestry(bias_tensor)

        end = now()
        print(
            "Conv2dForward(convolution) forward took: ",
            end * 1000 - start * 1000,
            "ms",
        )
        return output^
