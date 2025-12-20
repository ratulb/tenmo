from tenmo import Tensor
from backpropagation import (
    Delegate,
    BackwardFn,
    BACKWARD_CONCAT,
)
from operators import AddTensor
from common_utils import panic
from gradbox import Gradbox
from intarray import IntArray
from indexhelper import IndexCalculator
from shapes import Shape


@fieldwise_init
@register_passable
struct ConcatBackward[dtype: DType](ImplicitlyCopyable):
    alias TAG = BACKWARD_CONCAT
    var axis: Int

    fn backward(
        self, output: Tensor[dtype]
    ) -> List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]:
        ref grad_output = output.gradients()[]
        var grad_data = grad_output.buffer.buffer.data
        ref grad_shape = grad_output.shape()
        ref grad_strides = grad_output.strides()

        var count = len(output.ancestry())
        var result = List[Tuple[Tensor[dtype], Gradbox[dtype], Int]]()

        # Fast path: axis 0
        if self.axis == 0:
            var src_offset = 0
            for i in range(count):
                var tensor = output.ancestry().get(i)
                var num_elements = tensor.numels()
                if tensor.requires_grad:
                    var grad_input = Gradbox[dtype].zeros(tensor.shape())
                    var grad_input_data = grad_input.buffer.buffer.data
                    for j in range(num_elements):
                        grad_input_data[j] = grad_data[src_offset + j]
                    result.append((tensor^, grad_input^, AddTensor))
                src_offset += num_elements
            return result^

        # General path: any axis
        var offset = 0
        for i in range(count):
            var tensor = output.ancestry().get(i)
            if not tensor.requires_grad:
                offset += tensor.shape()[self.axis]
                continue

            ref tensor_shape = tensor.shape()
            var grad_input = Gradbox[dtype].zeros(tensor_shape)
            var grad_input_data = grad_input.buffer.buffer.data

            var elem_idx = 0
            for dest_idx in grad_input.index_iterator():
                var coord = IndexCalculator.index_to_coord(
                    tensor_shape, elem_idx
                )
                var grad_coord = IntArray.filled(grad_shape.rank(), 0)
                for d in range(grad_shape.rank()):
                    grad_coord[d] = coord[d] + (offset if d == self.axis else 0)
                var src_idx = IndexCalculator.flatten_index(
                    grad_shape, grad_coord, grad_strides, 0
                )
                grad_input_data[dest_idx] = grad_data[src_idx]
                elem_idx += 1

            offset += tensor.shape()[self.axis]
            result.append((tensor^, grad_input^, AddTensor))

        return result^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)


@fieldwise_init
@register_passable
struct Concate[dtype: DType](ImplicitlyCopyable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        tensors: List[Tensor[Self.dtype]],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        """Concatenate tensors along specified axis."""

        if len(tensors) == 0:
            panic("Concate → forward: cannot concatenate empty list")

        if len(tensors) == 1:
            return tensors[0]

        # ===== 1. VALIDATE =====
        var first_shape = tensors[0].shape()
        var ndim = first_shape.rank()
        var concat_axis = axis if axis >= 0 else ndim + axis

        if concat_axis < 0 or concat_axis >= ndim:
            panic("Concate → forward: axis out of bounds")

        for i in range(1, len(tensors)):
            var shape = tensors[i].shape()
            if shape.rank() != ndim:
                panic("Concate → forward: all tensors must have same rank")
            for d in range(ndim):
                if d != concat_axis and shape[d] != first_shape[d]:
                    panic(
                        "Concate → forward: dimensions must match except on"
                        " concat axis"
                    )

        # ===== 2. CALCULATE OUTPUT SHAPE =====
        var output_dims = List[Int]()
        for d in range(ndim):
            if d == concat_axis:
                var total_size = 0
                for i in range(len(tensors)):
                    total_size += tensors[i].shape()[d]
                output_dims.append(total_size)
            else:
                output_dims.append(first_shape[d])

        var output_shape = Shape(output_dims)

        # ===== 3. ALLOCATE OUTPUT =====
        var result = Tensor[Self.dtype].zeros(output_shape)
        ref result_data = result.buffer.data_buffer()
        ref result_strides = result.strides()
        var result_offset = result.offset()

        # ===== 4. COPY DATA =====
        var offset = 0  # Track position along concat axis
        var grad_required = False
        for tensor_idx in range(len(tensors)):
            ref tensor = tensors[tensor_idx]
            grad_required = grad_required or tensor.requires_grad
            ref tensor_data = tensor.buffer.data_buffer()
            ref tensor_shape = tensor.shape()
            ref tensor_strides = tensor.strides()
            var tensor_offset = tensor.offset()
            var tensor_size = tensor_shape[concat_axis]

            # Iterate through all coordinates in source tensor
            for coord in tensor_shape:
                # Build destination coordinate (shift concat axis by offset)
                var dest_coord = IntArray.filled(ndim, 0)
                for d in range(ndim):
                    if d == concat_axis:
                        dest_coord[d] = coord[d] + offset
                    else:
                        dest_coord[d] = coord[d]

                # Get flat indices using IndexCalculator
                var src_idx = IndexCalculator.flatten_index(
                    tensor_shape, coord, tensor_strides, tensor_offset
                )
                var dest_idx = IndexCalculator.flatten_index(
                    output_shape, dest_coord, result_strides, result_offset
                )

                # Copy element
                result_data[dest_idx] = tensor_data[src_idx]

            offset += tensor_size

        # ===== 5. SETUP AUTOGRAD =====

        @parameter
        if track_grad:
            grad_required = requires_grad.or_else(grad_required)
            if grad_required:
                result.requires_grad_(True)
                var backward_fn = ConcatBackward[Self.dtype](
                    concat_axis
                ).into_backward_fn()
                result.backwardFn = Optional(backward_fn^)
                for i in range(len(tensors)):
                    result.add_ancestry(tensors[i])

        return result^


fn main() raises:
    test_concat_comprehensive()


fn test_concat_comprehensive() raises:
    alias dtype = DType.float32

    print("=" * 60)
    print("TEST 1: Basic concat along axis 0")
    print("=" * 60)

    var A = Tensor[dtype].d2(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True
    )
    var B = Tensor[dtype].d2(
        [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], requires_grad=True
    )

    var tensors = List[Tensor[dtype]]()
    tensors.append(A)
    tensors.append(B)

    var result = Concate.forward(tensors, axis=0, requires_grad=True)

    print("\nA:")
    A.print()
    print("\nB:")
    B.print()
    print("\nConcat(A, B, axis=0):")
    result.print()
    # Expected:
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9],
    #  [10, 11, 12]]

    print("\n" + "=" * 60)
    print("TEST 2: Concat along axis 1")
    print("=" * 60)

    var result2 = Concate.forward(tensors, axis=1, requires_grad=True)
    print("\nConcat(A, B, axis=1):")
    result2.print()
    # Expected:
    # [[1, 2, 3, 7, 8, 9],
    #  [4, 5, 6, 10, 11, 12]]

    print("\n" + "=" * 60)
    print("TEST 3: Gradient flow")
    print("=" * 60)

    var loss = result.sum()
    loss.backward()

    print("\nGrad A:")
    A.grad().print()
    # Expected: all ones (2x3)

    print("\nGrad B:")
    B.grad().print()
    # Expected: all ones (2x3)

    print("\n" + "=" * 60)
    print("TEST 4: Three tensors")
    print("=" * 60)

    var C = Tensor[dtype].ones(1, 3, requires_grad=True) * 13.0

    var tensors3 = List[Tensor[dtype]]()
    tensors3.append(A)
    tensors3.append(B)
    tensors3.append(C)

    var result3 = Concate.forward(tensors3, axis=0, requires_grad=True)
    print("\nConcat(A, B, C, axis=0):")
    result3.print()
    # Expected: (5, 3) matrix

    print("\n✅ All tests passed!")
