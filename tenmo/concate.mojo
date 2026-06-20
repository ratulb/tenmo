from .tensor import Tensor
from .backpropagation import (
    BackwardFnArg,
    Integer,
    BACKWARD_CONCAT,
)
from .mnemonics import AddTensor
from .common_utils import panic
from .gradbox import Gradbox
from .intarray import IntArray
from .indexhelper import IndexCalculator
from .shapes import Shape
from .ancestry import Ancestor
from .ndbuffer import NDBuffer
from std.sys import has_accelerator
from tenmo.kernels.concate_kernel import ConcateGpuKernel


@fieldwise_init
struct ConcatBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        var axis = output.ancestry().backward_fn_arg().get[Integer]().value
        ref grad_output = output.gradients()
        var count = len(output.ancestry())

        # ===== GPU BACKWARD PATH =====
        comptime if has_accelerator():
            if grad_output.is_on_gpu():
                try:
                    var grad_shape = grad_output.shape()
                    var output_axis_size = grad_shape[axis]
                    var stride_axis = 1
                    for d in range(axis + 1, grad_shape.rank()):
                        stride_axis *= grad_shape[d]
                    var gpu_device = grad_output.buffer().device()

                    var offset = 0
                    for i in range(count):
                        ref parent = output.ancestry().get(i)
                        ref parent_shape = parent.shape()
                        if not parent.requires_grad:
                            offset += parent_shape[axis]
                            continue

                        var grad_input = Gradbox[Self.dtype].zeros(
                            parent_shape, device=gpu_device
                        )

                        ConcateGpuKernel[Self.dtype].launch_backward(
                            grad_output.buffer(),
                            grad_input.buffer(),
                            parent_shape[axis],
                            output_axis_size,
                            stride_axis,
                            offset,
                        )

                        offset += parent_shape[axis]
                        parent.update_grad(grad_input^, AddTensor, None)
                        parent_ids.append(parent._id)

                    if not retain_graph:
                        grad_output.zero_grad()
                except e:
                    panic(
                        "ConcatBackward GPU backward failed: " + String(e)
                    )
                return

        # ===== CPU BACKWARD PATH =====
        var grad_data = grad_output.data_ptr()
        var grad_shape = grad_output.shape()
        var grad_strides = grad_output.strides()

        # Fast path: axis 0
        if axis == 0:
            var src_offset = 0
            for i in range(count):
                ref parent = output.ancestry().get(i)
                ref parent_shape = parent.shape()
                var num_elements = parent_shape.numels()
                if parent.requires_grad:
                    var grad_input = Gradbox[Self.dtype].zeros(parent_shape)
                    var grad_input_data = grad_input.data_ptr()
                    for j in range(num_elements):
                        grad_input_data[j] = grad_data[src_offset + j]
                    parent.update_grad(grad_input^, AddTensor, None)
                    parent_ids.append(parent._id)
                src_offset += num_elements
            if not retain_graph:
                grad_output.zero_grad()
            return

        # General path: any axis
        var offset = 0
        for i in range(count):
            ref parent = output.ancestry().get(i)
            ref parent_shape = parent.shape()
            if not parent.requires_grad:
                offset += parent_shape[axis]
                continue

            var grad_input = Gradbox[Self.dtype].zeros(parent_shape)
            var grad_input_data = grad_input.data_ptr()

            var elem_idx = 0
            for dest_idx in grad_input.index_iterator():
                var coord = IndexCalculator.index_to_coord(
                    parent_shape, elem_idx
                )
                var grad_coord = IntArray.filled(grad_shape.rank(), 0)
                for d in range(grad_shape.rank()):
                    grad_coord[d] = coord[d] + (offset if d == axis else 0)
                var src_idx = IndexCalculator.flatten_index(
                    grad_shape, grad_coord, grad_strides, 0
                )
                grad_input_data[dest_idx] = grad_data[src_idx]
                elem_idx += 1

            offset += parent_shape[axis]
            parent.update_grad(grad_input^, AddTensor, None)
            parent_ids.append(parent._id)

        if not retain_graph:
            grad_output.zero_grad()


@fieldwise_init
struct Concate[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        tensors: List[Tensor[Self.dtype]],
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
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

        # ===== GPU FORWARD PATH =====
        comptime if has_accelerator():
            var any_gpu = False
            for i in range(len(tensors)):
                if tensors[i].is_on_gpu():
                    any_gpu = True
                    break

            if any_gpu:
                try:
                    # Allocate output on GPU
                    var device = tensors[0].device()
                    var result = Tensor[Self.dtype].zeros(
                        output_shape, device=device
                    )

                    var grad_required = False
                    var offset = 0
                    for tensor_idx in range(len(tensors)):
                        ref tensor = tensors[tensor_idx]
                        grad_required = (
                            grad_required or tensor.requires_grad
                        )
                        var input_axis_size = tensor.shape()[concat_axis]
                        var stride_axis = result.strides()[concat_axis]

                        ConcateGpuKernel[Self.dtype].launch_forward(
                            tensor.buffer,
                            result.buffer,
                            input_axis_size,
                            output_shape[concat_axis],
                            stride_axis,
                            offset,
                        )

                        offset += input_axis_size

                    # Setup autograd
                    comptime if track_grad:
                        grad_required = requires_grad.or_else(
                            grad_required
                        )
                        if grad_required:
                            result.requires_grad_(True)
                            var backwardFnArg = BackwardFnArg[
                                Self.dtype
                            ].integer_arg(BACKWARD_CONCAT, concat_axis)
                            backwardFnArg.needs_parent_data = True
                            for i in range(len(tensors)):
                                result.add_ancestry(
                                    backwardFnArg, tensors[i]
                                )

                    return result^
                except e:
                    panic(
                        "Concate GPU forward failed: " + String(e)
                    )

        # ===== CPU FORWARD PATH =====
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

        comptime if track_grad:
            grad_required = requires_grad.or_else(grad_required)
            if grad_required:
                result.requires_grad_(True)
                var backwardFnArg = BackwardFnArg[Self.dtype].integer_arg(
                    BACKWARD_CONCAT, concat_axis
                )
                backwardFnArg.needs_parent_data = True
                for i in range(len(tensors)):
                    result.add_ancestry(backwardFnArg, tensors[i])

        return result^
