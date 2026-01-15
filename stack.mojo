from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_STACK,
)
from operators import AddTensor
from common_utils import panic
from gradbox import Gradbox
from intarray import IntArray
from indexhelper import IndexCalculator
from shapes import Shape
from forwards import Concate


@fieldwise_init
@register_passable
struct StackBackward[dtype: DType](ImplicitlyCopyable):
    """Backward pass for stack operation."""

    alias TAG = BACKWARD_STACK

    var axis: Int
    var num_tensors: Int

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        """
        Split gradient and squeeze the stacked dimension.

        Forward:  stack([A, B, C], axis=1) → Result(d0, 3, d1, d2)
        Backward: grad_Result(d0, 3, d1, d2) → [grad_A(d0, d1, d2),
                                                  grad_B(d0, d1, d2),
                                                  grad_C(d0, d1, d2)]
        """

        ref grad_output = output.gradients()[]
        var grad_data = grad_output.buffer.buffer.data
        ref grad_shape = grad_output.shape()
        ref grad_strides = grad_output.strides()

        var count = len(output.ancestry())
        var result = List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]()

        # Size of stacked dimension should equal num_tensors
        var stack_size = grad_shape[self.axis]
        if stack_size != self.num_tensors:
            panic(
                "StackBackward: Expected stack dimension size",
                self.num_tensors.__str__(),
                "but got",
                stack_size.__str__(),
            )

        # ===== SPLIT GRADIENT ALONG STACKED AXIS =====
        for tensor_idx in range(count):
            var tensor = output.ancestry().get(tensor_idx)

            if not tensor.requires_grad:
                continue

            # Build grad_input shape (without the stacked dimension)
            var grad_input_shape_dims = List[Int]()
            for d in range(grad_shape.rank()):
                if d != self.axis:
                    grad_input_shape_dims.append(grad_shape[d])

            var grad_input_shape = Shape(grad_input_shape_dims)
            var grad_input = Gradbox[dtype].zeros(grad_input_shape)
            var grad_input_data = grad_input.buffer.buffer.data

            # ===== EXTRACT SLICE: grad_output[..., tensor_idx, ...] =====
            var elem_idx = 0

            # Iterate over all elements in the OUTPUT gradient shape
            for coord in grad_shape:
                # Only process elements where coord[self.axis] == tensor_idx
                if coord[self.axis] == tensor_idx:
                    # Get flat index in grad_output
                    var src_idx = IndexCalculator.flatten_index(
                        grad_shape, coord, grad_strides, 0
                    )

                    # Copy to grad_input (contiguous)
                    grad_input_data[elem_idx] = grad_data[src_idx]
                    elem_idx += 1

            result.append((tensor^, grad_input^, AddTensor))

        return result^

    fn into_backward_fn(self) -> BackwardFn[dtype]:
        return BackwardFn[dtype](Delegate[dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Stack[dtype: DType](ImplicitlyCopyable):
    @always_inline
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensors: List[Tensor[Self.dtype]],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """
        Stack tensors along a new axis.

        Args:
            tensors: List of tensors to stack (must have identical shapes).
            axis: Position where new dimension is inserted.
            requires_grad: Whether to track gradients.

        Returns:
            Stacked tensor with new dimension at position 'axis'.

        Example:
            A: (2, 3), B: (2, 3)
            stack([A, B], axis=0) → (2, 2, 3)
            stack([A, B], axis=1) → (2, 2, 3)
            stack([A, B], axis=2) → (2, 3, 2).
        """
        if len(tensors) == 0:
            panic("Cannot stack empty list")

        var grad_required = requires_grad.or_else(False)

        # ===== 1. VALIDATE: All tensors must have same shape =====
        var first_shape = tensors[0].shape()
        var ndim = first_shape.rank()

        for i in range(1, len(tensors)):
            var shape = tensors[i].shape()
            if shape.rank() != ndim:
                panic("All tensors must have same rank")
            for d in range(ndim):
                if shape[d] != first_shape[d]:
                    panic("All tensors must have identical shapes")

        # Normalize negative axis: allowed range is [-ndim-1, ndim]
        var stack_axis = axis if axis >= 0 else (ndim + 1 + axis)
        if stack_axis < 0 or stack_axis > ndim:
            panic("Axis out of bounds for stack")

        # ===== 2. UNSQUEEZE: Add dimension at stack_axis to each tensor =====
        var expanded_tensors = List[Tensor[dtype]]()

        for i in range(len(tensors)):
            var to_be_expanded = tensors[i]
            grad_required = grad_required or to_be_expanded.requires_grad
            expanded = to_be_expanded.unsqueeze[track_grad=False]([stack_axis])
            # expanded = to_be_expanded.unsqueeze([stack_axis])
            expanded_tensors.append(expanded^)

        # ===== 3. CONCATENATE: Along the new dimension =====
        var result = Concate[Self.dtype].forward[track_grad=False](
            expanded_tensors^, axis=stack_axis, requires_grad=False
        )

        # ===== 4. SETUP AUTOGRAD =====
        @parameter
        if track_grad:
            if grad_required:
                result.requires_grad_(True)
                var backward_fn = StackBackward[Self.dtype](
                    stack_axis, len(tensors)
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)

                # Add original tensors (not expanded) to ancestry
                for i in range(len(tensors)):
                    result.add_ancestry(tensors[i])

        return result^

    @always_inline
    @staticmethod
    fn vstack[
        track_grad: Bool = True
    ](
        tensors: List[Tensor[Self.dtype]], requires_grad: Optional[Bool] = None
    ) -> Tensor[dtype]:
        """
        Stack tensors vertically (row-wise).

        For 2D+ tensors: equivalent to concat(axis=0).
        For 1D tensors: reshapes to 2D then stacks.

        Args:
            tensors: List of tensors to stack vertically.
            requires_grad: Whether to track gradients.

        Returns:
            Vertically stacked tensor.

        Example:
            A: (3,), B: (3,)
            vstack([A, B]) → (2, 3)

            A: (2, 3), B: (2, 3)
            vstack([A, B]) → (4, 3).
        """

        if len(tensors) == 0:
            panic("Cannot vstack empty list")

        var first_ndim = tensors[0].rank()

        # Special case: 1D tensors
        if first_ndim == 1:
            # Reshape each (N,) → (1, N) then concatenate
            var reshaped = List[Tensor[dtype]]()
            for i in range(len(tensors)):
                var tensor = tensors[i]
                var size = tensor.shape()[0]
                # var reshaped_tensor = tensor.reshape[track_grad=False](1, size)
                var reshaped_tensor = tensor.reshape(1, size)
                reshaped.append(reshaped_tensor^)

            return Concate[Self.dtype].forward[track_grad](
                reshaped, axis=0, requires_grad=requires_grad
            )

        # For 2D+: use CONCATENATION along axis 0, not STACKING
        # All tensors must have same number of columns (all dimensions except axis 0)
        var first_shape = tensors[0].shape()

        for i in range(1, len(tensors)):
            var shape = tensors[i].shape()
            if shape.rank() != first_shape.rank():
                panic("All tensors must have same rank for vstack")
            # Check all dimensions except the first (rows) must match
            for d in range(1, first_shape.rank()):
                if shape[d] != first_shape[d]:
                    panic(
                        "All tensors must have same dimensions except rows for"
                        " vstack"
                    )

        return Concate[Self.dtype].forward[track_grad](
            tensors, axis=0, requires_grad=requires_grad
        )

    @always_inline
    @staticmethod
    fn hstack[
        track_grad: Bool = True
    ](
        tensors: List[Tensor[Self.dtype]], requires_grad: Optional[Bool] = None
    ) -> Tensor[Self.dtype]:
        """
        Stack tensors horizontally (column-wise).

        For 1D tensors: equivalent to concat(axis=0).
        For 2D+ tensors: equivalent to concat(axis=1).

        Args:
            tensors: List of tensors to stack horizontally.
            requires_grad: Whether to track gradients.

        Returns:
            Horizontally stacked tensor.

        Example:
            A: (3,), B: (3,)
            hstack([A, B]) → (6,)

            A: (2, 3), B: (2, 5)
            hstack([A, B]) → (2, 8).
        """

        if len(tensors) == 0:
            panic("Cannot hstack empty list")

        var first_ndim = tensors[0].rank()

        # For 1D: concatenate along axis 0
        if first_ndim == 1:
            return Concate.forward[track_grad](
                tensors, axis=0, requires_grad=requires_grad
            )

        # For 2D+: concatenate along axis 1
        # All tensors must have same number of rows (all dimensions except axis 1 must match)
        var first_shape = tensors[0].shape()

        for i in range(1, len(tensors)):
            var shape = tensors[i].shape()
            if shape.rank() != first_shape.rank():
                panic("All tensors must have same rank for hstack")
            # Check all dimensions except axis 1 (columns) must match
            for d in range(first_shape.rank()):
                if d != 1:  # Skip dimension 1 (columns)
                    if shape[d] != first_shape[d]:
                        panic(
                            "All tensors must have same dimensions except"
                            " columns for hstack"
                        )

        return Concate.forward[track_grad](
            tensors, axis=1, requires_grad=requires_grad
        )
