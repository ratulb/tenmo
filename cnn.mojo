from tenmo import Tensor
from backpropagation import (
    BackwardFn,
    Delegate,
    BACKWARD_COL2IM,
    BACKWARD_CONV2DMM,
    BACKWARD_FUSED_CONV,
    BACKWARD_MAXPOOL2D,
)
from operators import AddTensor
from common_utils import panic, now
from forwards import Pad
from common_utils import s, i
from shapes import Shape
from gradbox import Gradbox
from forwards import Padding
from intarray import IntArray
from algorithm import parallelize


@fieldwise_init
@register_passable
struct MaxPool2dBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Backward for batched, multi-channel MaxPool2d.
    Uses saved argmax indices to route gradients.
    Parallelized over (N * C) to avoid race conditions.
    """

    alias TAG = BACKWARD_MAXPOOL2D
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var input_shape: Shape  # (N, C, H_in, W_in)

    fn backward(
        self,
        output: Tensor[Self.dtype],  # (N, C, H_out, W_out)
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        start = now()

        ref grad_output = output.gradients()[]
        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        var input_tensor = output.ancestry().get(0)
        var argmax_mask = output.ancestry().get(
            1
        )  # (N, C, H_out, W_out) - stores flattened input indices

        if input_tensor.requires_grad:
            var N = self.input_shape[0]
            var C = self.input_shape[1]
            var H_in = self.input_shape[2]
            var W_in = self.input_shape[3]

            ref output_shape = grad_output.shape()
            var H_out = output_shape[2]
            var W_out = output_shape[3]

            # Initialize gradient tensor
            var grad_input = Gradbox[Self.dtype].zeros(
                self.input_shape, share=False
            )

            # Parallelize over (N * C) to avoid race conditions
            # Each thread handles one (batch, channel) pair exclusively
            @parameter
            fn scatter_gradients_for_batch_channel(idx: Int):
                var n = idx // C
                var c = idx % C

                # Process all output spatial positions for this (n, c)
                for out_y in range(H_out):
                    for out_x in range(W_out):
                        # Get the flattened index of the max element from forward pass
                        var max_idx = Int(argmax_mask[n, c, out_y, out_x])

                        if max_idx >= 0:  # Valid index (not from padding)
                            # Decode flattened index back to (in_y, in_x)
                            var in_y = max_idx // W_in
                            var in_x = max_idx % W_in

                            # Route gradient only to the max position
                            # Safe: no race condition since each (n,c) handled by single thread
                            grad_input[n, c, in_y, in_x] += grad_output[
                                n, c, out_y, out_x
                            ]

            # Parallelize over all (batch, channel) combinations
            parallelize[scatter_gradients_for_batch_channel](N * C)

            results.append((input_tensor^, grad_input^, AddTensor))

        end = now()
        print(
            "MaxPool2dBackward (parallelized over N*C) -> backward took: ",
            (end - start) * 1000,
            " ms",
        )
        return results^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Col2ImBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Backward for batched, multi-channel im2col.
    Parallelized over batch dimension (N).
    """

    alias TAG = BACKWARD_COL2IM
    var kernel_shape: Shape  # (C_out, C_in, KH, KW)
    var stride: Int
    var dilation: Int

    fn backward(
        self,
        output: Tensor[Self.dtype],  # im2_cols: (N, C_in*KH*KW, num_patches)
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        start = now()

        ref grad_im2_cols = output.gradients()[]
        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        var padded_image = output.ancestry().get(0)
        ref padded_shape = padded_image.shape()
        if padded_image.requires_grad:
            var N = padded_shape[0]
            var C_in = padded_shape[1]
            var H_pad = padded_shape[2]
            var W_pad = padded_shape[3]
            var KH = self.kernel_shape[2]
            var KW = self.kernel_shape[3]
            var dil = self.dilation
            var dilated_KH = KH + (KH - 1) * (dil - 1)
            var dilated_KW = KW + (KW - 1) * (dil - 1)
            var H_out = (H_pad - dilated_KH) // self.stride + 1
            var W_out = (W_pad - dilated_KW) // self.stride + 1

            # Initialize full gradient tensor
            var grad_padded_image = Gradbox[Self.dtype].zeros(
                Shape(N, C_in, H_pad, W_pad), share=False
            )

            # Define per-batch work function
            @parameter
            fn scatter_gradients_for_batch(n: Int):
                var col_idx = 0
                for out_y in range(0, H_pad - dilated_KH + 1, self.stride):
                    for out_x in range(0, W_pad - dilated_KW + 1, self.stride):
                        var elem_idx = 0
                        for c in range(C_in):
                            for ky in range(KH):
                                for kx in range(KW):
                                    var img_y = out_y + ky * dil
                                    var img_x = out_x + kx * dil
                                    grad_padded_image[
                                        n, c, img_y, img_x
                                    ] += grad_im2_cols[n, elem_idx, col_idx]
                                    elem_idx += 1
                        col_idx += 1

            # Parallelize over batch
            parallelize[scatter_gradients_for_batch](N)

            results.append((padded_image^, grad_padded_image^, AddTensor))

        end = now()  # Time in seconds
        print(
            "Col2ImBackward (parallelized over batch) -> backward took: ",
            (end - start) * 1000,
            " ms",
        )
        return results^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct FusedCol2ImBackward[dtype: DType](ImplicitlyCopyable & Movable):
    alias TAG = BACKWARD_FUSED_CONV
    var N: Int  # Batch size
    var C_in: Int  # Channel in
    var H_pad: Int
    var W_pad: Int
    var C_out: Int  # Channel out
    var KH: Int
    var KW: Int
    var dilated_KH: Int
    var dilated_KW: Int
    # var H_out: Int
    # var W_out: Int
    var num_patches: Int
    var stride: Int
    var dilation: Int

    fn backward(
        self,
        output: Tensor[Self.dtype],
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        start = now()
        ref grad_output = output.gradients()[]
        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        var padded_image = output.ancestry().get(0)
        ref padded_image_shape = padded_image.shape()
        var kernel = output.ancestry().get(1)
        ref kernel_shape = kernel.shape()  # (C_out, C_in, KH, KW)
        var bias = output.ancestry().get(2)

        _ = """var N = padded_image_shape[0]  # batch
        var C_in = padded_image_shape[1]  # C_in
        var H_pad = padded_image_shape[2]
        var W_pad = padded_image_shape[3]

        var C_out = kernel_shape[0]
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]
        var dil = self.dilation

        var dilated_KH = KH + (KH - 1) * (dil - 1)
        var dilated_KW = KW + (KW - 1) * (dil - 1)
        var H_out = (H_pad - dilated_KH) // self.stride + 1
        var W_out = (W_pad - dilated_KW) // self.stride + 1
        var num_patches = H_out * W_out"""

        var grad_out_flat = grad_output.reshape(
            Shape(self.N, self.C_out, self.num_patches), validated=True
        )

        # 1. Bias gradient — always simple and separate
        if bias.requires_grad:
            var grad_bias = grad_out_flat.sum(axes=IntArray(0, 2))
            results.append((bias^, grad_bias^, AddTensor))

        # Prepare gradients only if needed
        var grad_kernel = (
            Gradbox[Self.dtype]
            .zeros(
                kernel_shape, share=False
            ) if kernel.requires_grad else Gradbox[dtype]
            .zeros(Shape(), share=False)
        )
        var grad_padded = (
            Gradbox[Self.dtype]
            .zeros(
                padded_image_shape, share=False
            ) if padded_image.requires_grad else Gradbox[dtype]
            .zeros(Shape(), share=False)
        )

        # Main fused accumulation loop
        @parameter
        fn accum_gradients(n: Int):
            var patch_idx = 0
            for out_y in range(
                0, self.H_pad - self.dilated_KH + 1, self.stride
            ):
                for out_x in range(
                    0, self.W_pad - self.dilated_KW + 1, self.stride
                ):
                    for co in range(self.C_out):
                        var g = grad_out_flat[n, co, patch_idx]
                        if g == Scalar[Self.dtype](0):
                            continue  # Skip zero gradients early
                        for c in range(self.C_in):
                            for ky in range(self.KH):
                                for kx in range(self.KW):
                                    var img_y = out_y + ky * self.dilation
                                    var img_x = out_x + kx * self.dilation
                                    var input_val = padded_image[
                                        n, c, img_y, img_x
                                    ]
                                    var kernel_val = kernel[co, c, ky, kx]

                                    if kernel.requires_grad:
                                        grad_kernel[co, c, ky, kx] += (
                                            g * input_val
                                        )
                                    if padded_image.requires_grad:
                                        grad_padded[n, c, img_y, img_x] += (
                                            g * kernel_val
                                        )
                    patch_idx += 1

        parallelize[accum_gradients](self.N)

        # Append results
        if kernel.requires_grad:
            results.append((kernel^, grad_kernel^, AddTensor))
        if padded_image.requires_grad:
            results.append((padded_image^, grad_padded^, AddTensor))

        end = now()
        print(
            "FusedCol2ImBackward (fused kernel+input grads) -> backward took: ",
            (end - start) * 1000,
            " ms",
        )
        return results^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct FusedIm2Col[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut padded_image: Tensor[Self.dtype],  # (N, C_in, H_pad, W_pad)
        kernel: Tensor[Self.dtype],  # (C_out, C_in, KH, KW)
        bias: Tensor[Self.dtype],  # (C_out,)
        stride: Int = 1,
        dilation: Int = 1,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """
        Fused multi-channel, batched im2col + matmul + bias.
        Input: (N, C_in, H_pad, W_pad).
        Returns: (N, C_out, num_patches)  # Flattened output features.
        """
        start = now()
        ref padded_shape = padded_image.shape()
        if padded_shape.rank() != 4:
            panic("FusedIm2Col expects 4D input: (N, C_in, H_pad, W_pad)")

        var N = padded_shape[0]
        var C_in = padded_shape[1]
        var H_pad = padded_shape[2]
        var W_pad = padded_shape[3]

        ref kernel_shape = kernel.shape()
        if kernel_shape.rank() != 4:
            panic("Kernel expects 4D: (C_out, C_in, KH, KW)")
        var C_out = kernel_shape[0]
        if kernel_shape[1] != C_in:
            panic("Kernel input channels must match input channels")
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]

        var dil = dilation
        var dilated_KH = KH + (KH - 1) * (dil - 1)
        var dilated_KW = KW + (KW - 1) * (dil - 1)

        var H_out = (H_pad - dilated_KH) // stride + 1
        var W_out = (W_pad - dilated_KW) // stride + 1
        var num_patches = H_out * W_out

        # Output: (N, C_out, num_patches)
        var output_flat = Tensor[Self.dtype].zeros(N, C_out, num_patches)

        # Parallelize over batch N
        @parameter
        fn compute_for_batch(n: Int):
            var patch_idx = 0
            for out_y in range(0, H_pad - dilated_KH + 1, stride):
                for out_x in range(0, W_pad - dilated_KW + 1, stride):
                    # Parallelize over output channels C_out
                    @parameter
                    fn compute_for_channel(co: Int):  # Output channel
                        var accum = bias[co]
                        for c in range(C_in):
                            for ky in range(KH):
                                for kx in range(KW):
                                    var img_y = out_y + ky * dil
                                    var img_x = out_x + kx * dil
                                    accum += (
                                        padded_image[n, c, img_y, img_x]
                                        * kernel[co, c, ky, kx]
                                    )
                        output_flat[n, co, patch_idx] = accum

                    parallelize[compute_for_channel](C_out)
                    patch_idx += 1

        parallelize[compute_for_batch](N)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(
                padded_image.requires_grad
                or kernel.requires_grad
                or bias.requires_grad
            )
            if grad_required:
                output_flat.requires_grad_(True)
                var backward_fn = FusedCol2ImBackward[Self.dtype](
                    N=N,
                    C_in=C_in,
                    H_pad=H_pad,
                    W_pad=W_pad,
                    C_out=C_out,
                    KH=KH,
                    KW=KW,
                    dilated_KH=dilated_KH,
                    dilated_KW=dilated_KW,
                    # H_out = H_out,
                    # W_out = W_out,
                    num_patches=num_patches,
                    stride=stride,
                    dilation=dilation,
                ).into_backward_fn()
                output_flat.backwardFn = Optional(backward_fn^)
                output_flat.add_ancestry(padded_image)
                output_flat.add_ancestry(kernel)
                output_flat.add_ancestry(bias)

        end = now()
        print("FusedIm2Col - forward took: ", (end - start) * 1000, " ms")
        return output_flat^


@fieldwise_init
@register_passable
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
        start = now()
        ref image_shape = image.shape()
        ref kernel_shape = kernel.shape()

        # Validation
        if image_shape.rank() != 4:
            panic("Image must be 4D: (N, C_in, H_in, W_in)")
        if kernel_shape.rank() != 4:
            panic("Kernel must be 4D: (C_out, C_in, KH, KW)")
        var N = image_shape[0]
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
            var lst = padding[List[Tuple[Int, Int]]].copy()
            if len(lst) != 2:
                panic("Padding list must contain exactly 2 tuples")
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
            pad_spec^,
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
        var output_flat = FusedIm2Col[Self.dtype].forward[
            track_grad=track_grad
        ](
            padded_image,
            kernel,
            bias_tensor,
            stride,
            dilation,
            requires_grad=requires_grad,
        )

        # Reshape to spatial
        var output = output_flat.reshape[track_grad=track_grad](
            N, C_out, H_out, W_out
        )

        end = now()
        print("Conv2dFused -> forward took: ", (end - start) * 1000, " ms")
        return output^


@fieldwise_init
@register_passable
struct Im2Col[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut padded_image: Tensor[Self.dtype],  # (N, C_in, H_pad, W_pad)
        kernel_shape: Shape,  # (C_out, C_in, KH, KW) — we only use [2],[3]
        stride: Int = 1,
        dilation: Int = 1,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        start = now()

        ref padded_shape = padded_image.shape()
        if padded_shape.rank() != 4:
            panic("Im2Col expects 4D input: (N, C_in, H_pad, W_pad)")

        var N = padded_shape[0]
        var C_in = padded_shape[1]
        var H_pad = padded_shape[2]
        var W_pad = padded_shape[3]

        var KH = kernel_shape[2]
        var KW = kernel_shape[3]
        var dil = dilation

        var dilated_KH = KH + (KH - 1) * (dil - 1)
        var dilated_KW = KW + (KW - 1) * (dil - 1)

        var H_out = (H_pad - dilated_KH) // stride + 1
        var W_out = (W_pad - dilated_KW) // stride + 1
        var num_patches = H_out * W_out
        var patch_size = C_in * KH * KW

        var im2_cols = Tensor[Self.dtype].zeros(N, patch_size, num_patches)

        # Parallelize over batch dimension
        @parameter
        fn extract_patches_for_batch(n: Int):
            var col_idx = 0
            for out_y in range(0, H_pad - dilated_KH + 1, stride):
                for out_x in range(0, W_pad - dilated_KW + 1, stride):
                    var elem_idx = 0
                    for c in range(C_in):
                        for ky in range(KH):
                            for kx in range(KW):
                                var img_y = out_y + ky * dil
                                var img_x = out_x + kx * dil
                                im2_cols[n, elem_idx, col_idx] = padded_image[
                                    n, c, img_y, img_x
                                ]
                                elem_idx += 1
                    col_idx += 1

        parallelize[extract_patches_for_batch](N)  # Uses all available cores

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(
                padded_image.requires_grad
            )
            if grad_required:
                im2_cols.requires_grad_(True)
                var backward_fn = Col2ImBackward[Self.dtype](
                    kernel_shape=kernel_shape,
                    stride=stride,
                    dilation=dilation,
                ).into_backward_fn()
                im2_cols.backwardFn = Optional(backward_fn^)
                im2_cols.add_ancestry(padded_image)

        end = now()
        print(
            "Im2Col (parallelized over batch) - forward took: ",
            (end - start) * 1000,
            " ms",
        )
        return im2_cols^


@fieldwise_init
@register_passable
struct Conv2dMM[dtype: DType](ImplicitlyCopyable):
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
        mut kernel: Tensor[Self.dtype],
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

        if kernel_shape[1] != C_in:
            panic("Kernel input channels must match input channels")

        var H_in = input_shape[2]
        var W_in = input_shape[3]

        var C_out = kernel_shape[0]
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]

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
        # Pad
        # ═══════════════════════════════════════════════════════════

        var pad_spec = List[Tuple[Int, Int]]()
        pad_spec.append((0, 0))  # No padding on batch
        pad_spec.append((0, 0))  # No padding on channels
        pad_spec.append((pad_top, pad_bottom))  # Pad height
        pad_spec.append((pad_left, pad_right))  # Pad width

        var padded_image = Pad[Self.dtype].forward[track_grad=track_grad](
            image,
            pad_spec^,
            mode="constant",
            value=0.0,
            requires_grad=requires_grad.or_else(image.requires_grad),
        )

        # ═══════════════════════════════════════════════════════════
        # im2col
        # ═══════════════════════════════════════════════════════════

        var im2_cols = Im2Col[Self.dtype].forward[track_grad=track_grad](
            padded_image,
            kernel_shape,
            stride,
            dilation,
            requires_grad=requires_grad,
        )
        # num_patches = H_out * W_out
        # im2_cols shape: (N, C_in * KH * KW, num_patches)
        # kernel shape: (C_out, C_in, KH, KW)
        # output shape: (N, C_out, H_out, W_out)

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

        var patch_size = C_in * KH * KW
        var kernel_2d = kernel.reshape[track_grad=False](C_out, patch_size)
        # Shape: (C_out, patch_size)

        # Matmul with broadcasting!
        var output_flat = kernel_2d.matmul[track_grad=False](im2_cols)
        # (C_out, patch_size) @ (N, patch_size, num_patches)
        # → (N, C_out, num_patches)

        if bias:
            var bias_broadcast = bias_tensor.reshape[track_grad=False](
                1, C_out, 1
            )
            output_flat = output_flat.__add__[track_grad=False](bias_broadcast)

        # ═══════════════════════════════════════════════════════════
        # Reshape to spatial
        # ═══════════════════════════════════════════════════════════
        var output = output_flat.reshape[track_grad=False](
            N, C_out, H_out, W_out
        )

        # ═══════════════════════════════════════════════════════════
        # Gradient Setup
        # ═══════════════════════════════════════════════════════════
        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(
                im2_cols.requires_grad
                or kernel.requires_grad
                or bias_tensor.requires_grad
            )
            if grad_required:
                output.requires_grad_(True)
                var backward_fn = Conv2dMMBackward[
                    Self.dtype
                ]().into_backward_fn()
                output.backwardFn = Optional(backward_fn^)
                output.add_ancestry(im2_cols)
                output.add_ancestry(kernel)
                if bias:
                    output.add_ancestry(bias_tensor)

        end = now()
        print(
            "Conv2dMM -> forward took: ",
            end * 1000 - start * 1000,
            "ms",
            "Output shape: ",
            output.shape(),
        )
        return output^


@fieldwise_init
@register_passable
struct Conv2dMMBackward[dtype: DType](ImplicitlyCopyable & Movable):
    alias TAG = BACKWARD_CONV2DMM

    fn backward(
        self, output: Tensor[Self.dtype]  # (N, C_out, H_out, W_out)
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        start = now()
        ref grad_output = output.gradients()[]
        var results = List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]](
            capacity=3
        )

        var im2_cols = output.ancestry().get(0)
        var kernel_tensor = output.ancestry().get(1)
        ref kernel_shape = kernel_tensor.shape()
        var has_bias = len(output.ancestry()) > 2
        ref grad_shape = grad_output.shape()
        var N = grad_output.shape()[0]
        var C_out = grad_shape[1]
        var H_out = grad_shape[2]
        var W_out = grad_shape[3]
        var num_patches = H_out * W_out

        var C_in = kernel_shape[1]
        var KH = kernel_shape[2]
        var KW = kernel_shape[3]
        var patch_size = C_in * KH * KW

        # ═══════════════════════════════════════════════════════════
        # Gradient 1: BIAS (ONE LINE!)
        # ═══════════════════════════════════════════════════════════

        if has_bias:
            var bias_tensor = output.ancestry().get(2)
            if bias_tensor.requires_grad:
                # Sum across batch (0), height (2), width (3) dimensions
                var grad_bias = grad_output.sum(
                    axes=IntArray(0, 2, 3)
                )  # → (C_out,)
                results.append((bias_tensor^, grad_bias^, AddTensor))

        # ═══════════════════════════════════════════════════════════
        # Gradient 2: KERNEL (using cached im2_cols)
        # ═══════════════════════════════════════════════════════════

        # Flatten grad_output upfront - it is potentially being in 2 places
        var grad_output_flat = grad_output.reshape(
            Shape(N, C_out, num_patches), validated=True
        )
        if kernel_tensor.requires_grad:
            # Flatten grad_output: (N, C_out, H_out, W_out) → (N, C_out, num_patches)

            # Transpose im2_cols for matmul
            var im2_cols_T = im2_cols.transpose[track_grad=False](-1, -2)
            # (N, patch_size, num_patches) → (N, num_patches, patch_size)

            # Matmul: (N, C_out, num_patches) @ (N, num_patches, patch_size)
            var grad_kernel_batched = grad_output_flat.matmul(im2_cols_T)
            # → (N, C_out, patch_size)

            # Sum across batch dimension
            var grad_kernel_flat = grad_kernel_batched.sum(
                axes=IntArray(0)
            )  # → (C_out, patch_size)

            # Reshape to kernel shape
            var grad_kernel = grad_kernel_flat.reshape(
                kernel_shape, validated=True
            )
            results.append((kernel_tensor, grad_kernel^, AddTensor))

        # ═══════════════════════════════════════════════════════════
        # Gradient 3: INPUT (broadcasting matmul)
        # ═══════════════════════════════════════════════════════════

        if im2_cols.requires_grad:
            # Reshape kernel

            # var kernel_2d = kernel_tensor.reshape[track_grad=False](C_out, patch_size)
            # var kernel_T = kernel_2d.transpose[track_grad=False]()  # (patch_size, C_out)

            kernel_T = kernel_tensor.reshape[track_grad=False](
                patch_size, C_out
            )
            # Broadcasting matmul
            var grad_im2_cols = kernel_T.matmul(grad_output_flat)
            # (patch_size, C_out) @ (N, C_out, num_patches) → (N, patch_size, num_patches)

            # This flows to im2_cols which has Col2ImBackward
            # which converts to grad_padded_image
            # which flows through Pad.backward to grad_input
            results.append((im2_cols^, grad_im2_cols^, AddTensor))

        end = now()
        print(
            "Conv2dMMBackward -> backward took: ",
            end * 1000 - start * 1000,
            "ms",
        )

        return results^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


fn main() raises:
    alias dtype = DType.float32

    # Batch of 2, 3 input channels, 4x5 image
    # var x = Tensor[dtype].rand(2, 3, 4, 5, requires_grad=True)
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

    # var output = Conv2dMM[dtype].forward(
    var output = Conv2dMM[dtype].forward(
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
