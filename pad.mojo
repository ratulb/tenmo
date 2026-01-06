# Generalized Padding Implementation for Mojo Tensor Library
# Supports arbitrary dimensions with asymmetric padding and gradient flow

"""
PADDING SPECIFICATION:

For N-dimensional tensor, padding is specified as a list of tuples:
[(before_0, after_0), (before_1, after_1), ..., (before_N-1, after_N-1)]

Or as a flat list (PyTorch style, applied from last to first dimension):
[before_last, after_last, before_second_last, after_second_last, ...]

Examples:
- 2D tensor (H, W): pad = [(1, 2), (3, 4)] means:
  - Dimension 0 (H): add 1 before, 2 after
  - Dimension 1 (W): add 3 before, 4 after

- 4D tensor (N, C, H, W): pad = [(0, 0), (0, 0), (1, 1), (2, 2)] means:
  - No padding on batch and channel dimensions
  - Pad H with 1 on each side
  - Pad W with 2 on each side
"""

from tenmo import Tensor
from shapes import Shape
from gradbox import Gradbox
from backpropagation import BackwardFn, Delegate, BACKWARD_PAD
from operators import AddTensor
from common_utils import panic
from intarray import IntArray
from utils import Variant

alias Padding = Variant[String, Int, Tuple[Int, Int], List[Tuple[Int, Int]]]


@fieldwise_init
struct PadBackward[dtype: DType](ImplicitlyCopyable & Movable):
    """Backward pass for padding operation - handles all modes."""

    alias TAG = BACKWARD_PAD

    var pad: List[Tuple[Int, Int]]
    var mode: String

    fn __copyinit__(out self, other: Self):
        self.pad = other.pad.copy()
        self.mode = other.mode

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """
        Backward pass: Accumulate gradients based on padding mode.
        """
        ref grad_out = output.gradients()[]
        var parent = output.ancestry().get(0)

        var results = List[
            Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]
        ]()

        if parent.requires_grad:
            ref parent_shape = parent.shape()
            var grad_parent = Gradbox[Self.dtype].zeros(
                parent_shape, share=False
            )

            # Different backward pass based on mode
            if self.mode == "constant":
                Self._extract_constant(
                    grad_out, grad_parent, self.pad, parent_shape
                )
            elif self.mode == "circular":
                Self._extract_circular(
                    grad_out, grad_parent, self.pad, parent_shape
                )
            elif self.mode == "replicate":
                Self._extract_replicate(
                    grad_out, grad_parent, self.pad, parent_shape
                )
            elif self.mode == "reflect":
                Self._extract_reflect(
                    grad_out, grad_parent, self.pad, parent_shape
                )

            results.append((parent^, grad_parent^, AddTensor))

        return results^

    @staticmethod
    fn _extract_constant(
        grad_out: Gradbox[Self.dtype],
        grad_parent: Gradbox[Self.dtype],
        pad: List[Tuple[Int, Int]],
        parent_shape: Shape,
    ):
        """Extract gradients for constant padding - simple extraction from center.
        """
        var ndim = parent_shape.rank()

        # Calculate offset where input data starts in padded output
        var offset_list = List[Int]()
        for i in range(ndim):
            offset_list.append(pad[i][0])

        # Iterate over parent's shape and extract gradients
        for coord in parent_shape:
            var grad_out_coord = coord
            grad_out_coord += offset_list
            grad_parent[coord] += grad_out[grad_out_coord^]

    @staticmethod
    fn _extract_circular(
        grad_out: Gradbox[Self.dtype],
        grad_parent: Gradbox[Self.dtype],
        pad: List[Tuple[Int, Int]],
        parent_shape: Shape,
    ):
        """Extract gradients for circular padding - accumulate from all wrapped positions.
        """
        var ndim = parent_shape.rank()
        var grad_out_shape = grad_out.shape()

        # Iterate over ALL output positions and accumulate gradients
        for out_coord in grad_out_shape:
            # Map output coordinate back to input coordinate (same logic as forward)
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = parent_shape[i]

                # Same wrapping logic as forward pass
                var in_idx = (out_idx - before) % in_size
                if in_idx < 0:
                    in_idx += in_size
                in_coord.append(in_idx)

            # ACCUMULATE gradient (not replace!)
            grad_parent[in_coord] += grad_out[out_coord]

    @staticmethod
    fn _extract_replicate(
        grad_out: Gradbox[Self.dtype],
        grad_parent: Gradbox[Self.dtype],
        pad: List[Tuple[Int, Int]],
        parent_shape: Shape,
    ):
        """Extract gradients for replicate padding - accumulate from all replicated positions.
        """
        var ndim = parent_shape.rank()
        var grad_out_shape = grad_out.shape()

        # Iterate over ALL output positions and accumulate gradients
        for out_coord in grad_out_shape:
            # Map output coordinate back to input coordinate (same logic as forward)
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = parent_shape[i]

                # Same clamping logic as forward pass (replicate edges)
                var in_idx = out_idx - before
                in_idx = max(0, min(in_size - 1, in_idx))
                in_coord.append(in_idx)

            # ACCUMULATE gradient
            grad_parent[in_coord] += grad_out[out_coord]

    @staticmethod
    fn _extract_reflect(
        grad_out: Gradbox[Self.dtype],
        grad_parent: Gradbox[Self.dtype],
        pad: List[Tuple[Int, Int]],
        parent_shape: Shape,
    ):
        """Extract gradients for reflect padding - accumulate from all reflected positions.
        """
        var ndim = parent_shape.rank()
        var grad_out_shape = grad_out.shape()

        # Iterate over ALL output positions and accumulate gradients
        for out_coord in grad_out_shape:
            # Map output coordinate back to input coordinate (same logic as forward)
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = parent_shape[i]

                # Same reflection logic as forward pass
                var in_idx: Int
                if out_idx < before:
                    # Reflect from left border
                    in_idx = before - out_idx
                    in_idx = min(in_idx, in_size - 1)
                elif out_idx >= before + in_size:
                    # Reflect from right border
                    var offset = out_idx - (before + in_size)
                    in_idx = in_size - 2 - offset
                    in_idx = max(0, in_idx)
                else:
                    # Inside original region
                    in_idx = out_idx - before

                # Final clamp
                in_idx = max(0, min(in_size - 1, in_idx))
                in_coord.append(in_idx)

            # ACCUMULATE gradient
            grad_parent[in_coord] += grad_out[out_coord]

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Pad[dtype: DType](ImplicitlyCopyable):
    """
    Generalized padding operation supporting:
    - Arbitrary dimensions.
    - Asymmetric padding (different on each side).
    - Multiple padding modes.
    - Proper gradient flow in backward pass.
    """

    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        x: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        mode: String = "constant",
        value: Scalar[Self.dtype] = 0.0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Pad tensor along specified dimensions."""
        var x_shape = x.shape()
        var ndim = x_shape.rank()

        # Validate padding specification
        if len(pad) != ndim:
            panic("Pad: padding must be specified for all dimensions")

        # Calculate output shape
        var out_shape = List[Int]()
        for i in range(ndim):
            var before = pad[i][0]
            var after = pad[i][1]
            out_shape.append(x_shape[i] + before + after)

        # Create output tensor
        var result = Tensor[Self.dtype].zeros(out_shape)

        # Apply padding based on mode
        if mode == "constant":
            Self._pad_constant(x, result, pad, value)
        elif mode == "reflect":
            Self._pad_reflect(x, result, pad)
        elif mode == "replicate":
            Self._pad_replicate(x, result, pad)
        elif mode == "circular":
            Self._pad_circular(x, result, pad)
        else:
            panic("Pad: unsupported mode")

        # Setup backward
        @parameter
        if track_grad:
            var req_grad = requires_grad.or_else(x.requires_grad)
            if req_grad:
                result.requires_grad_(True)
                # PASS MODE TO BACKWARD!
                var backward_fn = PadBackward[Self.dtype](
                    pad.copy(), mode  # <-- Add mode here
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                result.add_ancestry(x)

        return result^

    @staticmethod
    fn _pad_constant(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
        value: Scalar[Self.dtype],
    ):
        """Apply constant padding (most common for CNNs)."""
        # Fill with pad value
        result.fill(value)

        # Copy input data to center region
        # We need to map input indices to output indices
        Self._copy_to_padded_region(x, result, pad)

    @staticmethod
    fn _copy_to_padded_region(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ):
        """Copy input tensor to the non-padded region of output."""
        var x_shape = x.shape()
        # var result_shape = result.shape()
        var ndim = x_shape.rank()

        # Calculate offset in output for where input data starts
        var offset_list = List[Int]()
        for i in range(ndim):
            offset_list.append(pad[i][0])  # before padding

        # Iterate over all elements of input
        for coord in x_shape:
            var result_indices = coord
            result_indices += offset_list
            result[result_indices] = x[coord]

    @staticmethod
    fn _pad_replicate(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ):
        """Apply replicate padding - repeat edge values using coordinate iteration.
        """
        var x_shape = x.shape()
        var result_shape = result.shape()
        var ndim = x_shape.rank()

        # Iterate over all output coordinates
        for out_coord in result_shape:
            # Map to input with edge replication
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = x_shape[i]

                # Clamp to valid input range (replicate edges)
                var in_idx = out_idx - before
                in_idx = max(0, min(in_size - 1, in_idx))
                in_coord.append(in_idx)

            result[out_coord] = x[in_coord]

    @staticmethod
    fn _pad_reflect(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ):
        """Apply reflect padding - mirror at borders using coordinate iteration.
        """
        var x_shape = x.shape()
        var result_shape = result.shape()
        var ndim = x_shape.rank()

        # Iterate over all output coordinates
        for out_coord in result_shape:
            # Map output coordinates to input coordinates with reflection
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = x_shape[i]

                var in_idx: Int
                if out_idx < before:
                    # Reflect from left border
                    in_idx = before - out_idx
                    # Clamp to avoid out of bounds
                    in_idx = min(in_idx, in_size - 1)
                elif out_idx >= before + in_size:
                    # Reflect from right border
                    var offset = out_idx - (before + in_size)
                    in_idx = in_size - 2 - offset
                    # Clamp to avoid out of bounds
                    in_idx = max(0, in_idx)
                else:
                    # Inside original region
                    in_idx = out_idx - before

                # Final clamp to ensure valid range
                in_idx = max(0, min(in_size - 1, in_idx))
                in_coord.append(in_idx)

            result[out_coord] = x[in_coord]

    @staticmethod
    fn _pad_circular(
        x: Tensor[Self.dtype],
        mut result: Tensor[Self.dtype],
        pad: List[Tuple[Int, Int]],
    ):
        """Apply circular padding - wrap around using coordinate iteration."""
        var x_shape = x.shape()
        var result_shape = result.shape()
        var ndim = x_shape.rank()

        # Iterate over all output coordinates
        for out_coord in result_shape:
            # Map to input with wrapping
            var in_coord = IntArray.with_capacity(ndim)

            for i in range(ndim):
                var before = pad[i][0]
                var out_idx = out_coord[i]
                var in_size = x_shape[i]

                # Wrap around using modulo
                var in_idx = (out_idx - before) % in_size
                if in_idx < 0:
                    in_idx += in_size
                in_coord.append(in_idx)

            result[out_coord] = x[in_coord]


fn main() raises:
    alias dtype = DType.float32

    # var input_image: Tensor[dtype]  # Shape: (2, 3, 28, 28)
    # 2 images, 3 channels, 28×28 pixels

    var input_image = Tensor[dtype].arange(1 * 2 * 3 * 3)
    input_image = input_image.reshape(1, 2, 3, 3)
    print("Is image contiguous: ", input_image.is_contiguous())
    input_image.print()

    var pad_spec = List[Tuple[Int, Int]]()
    pad_spec.append((0, 0))  # Batch: DON'T pad
    pad_spec.append((0, 0))  # Channels: DON'T pad
    pad_spec.append((1, 1))  # Height: pad by 1
    pad_spec.append((1, 1))  # Width: pad by 1

    var padded = Pad.forward(input_image, pad_spec, mode="constant", value=-9)

    print()
    padded.print()
