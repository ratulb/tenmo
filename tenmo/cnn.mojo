from .tensor import Tensor
from .shared.panic import panic
from .pad import Padding
from .shared.shapes import Shape
from .conv import Conv


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

    def __init__(out self, *, copy: Self):
        self.initialized = copy.initialized
        self.pad_spec = copy.pad_spec.copy()

    def __init__(out self):
        self.initialized = False
        self.pad_spec = List[Tuple[Int, Int]]()

    def __call__[
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
        sync: Bool = True,
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
                sync=sync,
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
                sync=sync,
            )
            self.pad_spec = result[1].copy()
            self.initialized = True
            out = result[0]
        return out^

    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        kernel: Tensor[Self.dtype],
        bias: Optional[Tensor[Self.dtype]] = None,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Padding = Padding("valid"),
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var output, _ = Self.invoke[track_grad=track_grad](
            image=image,
            kernel=kernel,
            bias=bias,
            stride=stride,
            dilation=dilation,
            padding=padding,
            requires_grad=requires_grad,
            sync=sync,
        )
        return output^

    @staticmethod
    def invoke[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        mut kernel: Tensor[Self.dtype],
        bias: Optional[Tensor[Self.dtype]] = None,
        stride: Int = 1,
        dilation: Int = 1,
        pad_spec: List[Tuple[Int, Int]] = [],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var C_out = kernel.shape()[0]
        var pad_top = pad_spec[2][0]
        var pad_bottom = pad_spec[2][1]
        var pad_left = pad_spec[3][0]
        var pad_right = pad_spec[3][1]

        # Setup bias (lazy: no zeros alloc when bias is provided).
        # The fallback zeros live on the image's device so GPU conv never
        # goes mixed-device.
        var expected_bias_shape = Shape(C_out)
        var bias_tensor: Tensor[Self.dtype]
        if bias:
            bias_tensor = bias.value()
        else:
            bias_tensor = Tensor[Self.dtype].zeros(
                expected_bias_shape,
                requires_grad=False,
                device=image.device(),
            )
        if not bias_tensor.shape() == expected_bias_shape:
            panic(
                "Invalid bias tensor shape: ",
                String(bias_tensor.shape()),
                ". Should be (C_out,)",
            )

        # New core: fused pad-by-index, no Pad node.
        var output = Conv[Self.dtype].forward[track_grad=track_grad](
            image,
            kernel,
            bias_tensor,
            stride,
            dilation,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            requires_grad=requires_grad,
            sync=sync,
        )
        return output^

    @staticmethod
    def invoke[
        track_grad: Bool = True
    ](
        image: Tensor[Self.dtype],
        kernel: Tensor[Self.dtype],
        bias: Optional[Tensor[Self.dtype]] = None,
        stride: Int = 1,
        dilation: Int = 1,
        padding: Padding = Padding("valid"),
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tuple[Tensor[Self.dtype], List[Tuple[Int, Int]]]:
        ref image_shape = image.shape()
        ref kernel_shape = kernel.shape()

        # Validation
        if image_shape.rank() != 4:
            panic("Image must be 4D: (N, C_in, H_in, W_in)")
        if kernel_shape.rank() != 4:
            panic("Kernel must be 4D: (C_out, C_in, KH, KW)")
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

        # Pad the image (fused by index in the new core — no Pad node)
        var pad_spec = List[Tuple[Int, Int]]()
        pad_spec.append((0, 0))  # No padding on batch
        pad_spec.append((0, 0))  # No padding on channels
        pad_spec.append((pad_top, pad_bottom))  # Pad height
        pad_spec.append((pad_left, pad_right))  # Pad width

        # Compute output shape
        var H_out = (H_in + pad_top + pad_bottom - dilated_KH) // stride + 1
        var W_out = (W_in + pad_left + pad_right - dilated_KW) // stride + 1
        if H_out <= 0 or W_out <= 0:
            panic(
                "Invalid convolution parameters lead to non-positive output"
                " size"
            )

        # Setup bias (lazy: no zeros alloc when bias is provided).
        # The fallback zeros live on the image's device so GPU conv never
        # goes mixed-device.
        var expected_bias_shape = Shape(C_out)
        var bias_tensor: Tensor[Self.dtype]
        if bias:
            bias_tensor = bias.value()
        else:
            bias_tensor = Tensor[Self.dtype].zeros(
                expected_bias_shape,
                requires_grad=False,
                device=image.device(),
            )
        if not bias_tensor.shape() == expected_bias_shape:
            panic(
                "Invalid bias tensor shape: ",
                String(bias_tensor.shape()),
                ". Should be (C_out,)",
            )

        # Fused forward (new core)

        var output = Conv[Self.dtype].forward[track_grad=track_grad](
            image,
            kernel,
            bias_tensor,
            stride,
            dilation,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            requires_grad=requires_grad,
            sync=sync,
        )
        return output^, pad_spec^


