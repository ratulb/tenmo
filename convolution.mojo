from tenmo import Tensor
from backpropagation import BackwardFn, Delegate, BACKWARD_CONV2D
from operators import AddTensor
from common_utils import panic, now
from gradbox import Gradbox
from utils import Variant
from shapes import Shape
from algorithm import parallelize, vectorize
from sys import simd_width_of
from forwards import Padding


@fieldwise_init
@register_passable
struct Conv2DBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Custom backward for batched, multi-channel, multi-filter Conv2dForward.
    Correctly computes gradients for input, kernel, and bias.
    Optimized with SIMD vectorization where beneficial.
    """

    alias TAG = BACKWARD_CONV2D
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

    var output = Conv2dForward[dtype].forward(
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
