from tenmo import Tensor
from backpropagation import BackwardFn, Delegate, BACKWARD_COL2IM
from operators import AddTensor
from common_utils import panic
from forwards import Pad
from common_utils import s, i
from shapes import Shape
from gradbox import Gradbox


@fieldwise_init
@register_passable
struct Col2ImBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """
    Custom backward for im2col → col2im.
    Handles ONLY the input gradient computation.
    Now supports dilation.
    """
    alias TAG = BACKWARD_COL2IM
    var kernel_shape: Shape
    var stride: Int
    var dilation: Int

    fn backward(
        self, output: Tensor[Self.dtype]  # This is im2_cols
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref grad_im2_cols = output.gradients()[]

        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        var padded_image = output.ancestry().get(0)
        ref padded_shape = padded_image.shape()
        if padded_image.requires_grad:
            var H_pad = padded_shape[0]
            var W_pad = padded_shape[1]
            var KH = self.kernel_shape[0]
            var KW = self.kernel_shape[1]
            var dil = self.dilation

            # Effective kernel size after dilation
            var dilated_KH = KH + (KH - 1) * (dil - 1)
            var dilated_KW = KW + (KW - 1) * (dil - 1)

            var grad_padded_image = Gradbox[Self.dtype].zeros(
                Shape(H_pad, W_pad), share=False
            )

            var col_idx = 0
            for out_y in range(0, H_pad - dilated_KH + 1, self.stride):
                for out_x in range(0, W_pad - dilated_KW + 1, self.stride):
                    var row_idx = 0
                    for ky in range(KH):
                        for kx in range(KW):
                            var img_y = out_y + ky * dil
                            var img_x = out_x + kx * dil
                            grad_padded_image[img_y, img_x] += grad_im2_cols[
                                row_idx, col_idx
                            ]
                            row_idx += 1
                    col_idx += 1

            results.append((padded_image^, grad_padded_image^, AddTensor))

        return results^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Im2Col[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        mut padded_image: Tensor[Self.dtype],
        kernel_shape: Shape,
        stride: Int = 1,
        dilation: Int = 1,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        ref padded_shape = padded_image.shape()
        var H_pad, W_pad = padded_shape[0], padded_shape[1]
        var KH, KW = kernel_shape[0], kernel_shape[1]
        var dil = dilation

        # Effective kernel size after dilation
        var dilated_KH = KH + (KH - 1) * (dil - 1)
        var dilated_KW = KW + (KW - 1) * (dil - 1)

        # Output spatial dimensions (with dilation)
        var H_out = (H_pad - dilated_KH) // stride + 1
        var W_out = (W_pad - dilated_KW) // stride + 1

        var num_rows = KH * KW
        var num_cols = H_out * W_out
        var im2_cols = Tensor[Self.dtype].zeros(num_rows, num_cols)

        var col = 0
        for out_y in range(0, H_pad - dilated_KH + 1, stride):
            for out_x in range(0, W_pad - dilated_KW + 1, stride):
                var row_idx = 0
                for ky in range(KH):
                    for kx in range(KW):
                        var img_y = out_y + ky * dil
                        var img_x = out_x + kx * dil
                        im2_cols[row_idx, col] = padded_image[img_y, img_x]
                        row_idx += 1
                col += 1

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(padded_image.requires_grad)
            if grad_required:
                im2_cols.requires_grad_(True)
                backward_fn = Col2ImBackward[Self.dtype](
                    kernel_shape=kernel_shape,
                    stride=stride,
                    dilation=dilation
                ).into_backward_fn()
                im2_cols.backwardFn = Optional(backward_fn^)
                im2_cols.add_ancestry(padded_image)

        return im2_cols


@fieldwise_init
@register_passable
struct Conv2D[dtype: DType](ImplicitlyCopyable):
    alias BIAS = Tensor[Self.dtype].scalar(0, requires_grad=True)

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        input_image: Tensor[Self.dtype],
        mut kernel: Tensor[Self.dtype],
        bias: Tensor[Self.dtype] = Self.BIAS,
        stride: Int = 1,
        dilation: Int = 1,
        pad_top: Int = 0,
        pad_bottom: Int = 0,
        pad_left: Int = 0,
        pad_right: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var pad = List[Tuple[Int, Int]]()
        pad.append((pad_top, pad_bottom))
        pad.append((pad_left, pad_right))

        var padded_image = Pad[Self.dtype].forward[track_grad=track_grad](
            input_image,
            pad^,
            mode="constant",
            value=0.0,
            requires_grad=requires_grad.or_else(input_image.requires_grad),
        )

        var im2_cols = Im2Col[Self.dtype].forward[track_grad=track_grad](
            padded_image,
            kernel.shape(),
            stride,
            dilation,
            requires_grad=requires_grad
        )

        var kernel_flattened = kernel.reshape(1, -1)
        var output_flat = kernel_flattened.matmul(im2_cols)
        var output = output_flat + bias

        # Recompute output shape for reshaping
        ref padded_shape = padded_image.shape()
        ref kernel_shape = kernel.shape()
        var H_pad, W_pad = padded_shape[0], padded_shape[1]
        var KH, KW = kernel_shape[0], kernel_shape[1]
        var dilated_KH = KH + (KH - 1) * (dilation - 1)
        var dilated_KW = KW + (KW - 1) * (dilation - 1)
        var H_out = (H_pad - dilated_KH) // stride + 1
        var W_out = (W_pad - dilated_KW) // stride + 1

        var output_reshaped = output.reshape(H_out, W_out)
        return output_reshaped^

from random import seed
fn main() raises:
    alias dtype = DType.float32
    var x = Tensor[dtype].d2(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True
    )

    var kernel = Tensor[dtype].d2([[1, 1], [1, -1]], requires_grad=True)
    var bias = Tensor[dtype].scalar(10, requires_grad=True)
    seed(42)
    var output = Conv2D[dtype].forward[track_grad=True](
        input_image=x,
        kernel=kernel,
        bias=bias,
        stride=1,
        pad_top=1,
        pad_bottom=1,
        pad_left=1,
        pad_right=1,
        requires_grad=True,
    )

    output.print()

    output.backward(42)
    print("\nx's grad\n")
    x.grad().print()
    print("\nkernel's grad\n")
    kernel.grad().print()
    print("\nbias's grad")
    bias.grad().print()

    _="""x.zero_grad()
    kernel.zero_grad()
    bias.zero_grad()

    print("\nPost zero grading\n")

    print("\nx's grad\n")
    x.grad().print()
    print("\nkernel's grad\n")
    kernel.grad().print()
    print("\nbias's grad")
    bias.grad().print()"""


    var output_dil = Conv2D[dtype].forward[track_grad=True](
        input_image=x,
        kernel=kernel,
        bias=bias,
        stride=1,
        dilation=2,
        pad_top=0, pad_bottom=0, pad_left=0, pad_right=0,  # Adjust padding as needed
        requires_grad=True,
    )

    output_dil.backward(21)

    print("\nx's grad\n")
    x.grad().print()
    print("\nkernel's grad\n")
    kernel.grad().print()
    print("\nbias's grad")
    bias.grad().print()

