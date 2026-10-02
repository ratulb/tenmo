from .tensor import Tensor
from .backpropagation import (
    BackwardFnType,
    BackwardFn,
    Integer,
)
from .shared.mnemonics import AddTensor
from .shared.panic import panic
from .gradbox import Gradbox
from .shared.intarray import IntArray
from .shared.indexhelper import IndexCalculator
from .shared.shapes import Shape
from .ancestry import Ancestor
from std.sys import has_accelerator
from std.memory import unsafe_memcpy
from std.sys.info import num_physical_cores
from max.algorithm import parallelize
from .kernels.concate_kernel import ConcatKernel


@fieldwise_init
struct ConcatBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        var axis = output.ancestry().backward_fn().get[Integer]().value
        ref grad_output = output.gradients()
        var count = len(output.ancestry())

        # GPU BACKWARD PATH
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
                        # Grad is built only if needed; the id is ALWAYS
                        # appended (engine fanin-completion contract).
                        if parent.requires_grad:
                            var grad_input = Gradbox[Self.dtype].zeros(
                                parent_shape, device=gpu_device
                            )

                            ConcatKernel[Self.dtype].launch_backward(
                                grad_output.buffer().layout(),
                                grad_output.buffer().device_state.value(),
                                grad_input.buffer().layout(),
                                grad_input.buffer().device_state.value(),
                                parent_shape[axis],
                                output_axis_size,
                                stride_axis,
                                offset,
                            )

                            parent.update_grad(grad_input^, AddTensor, None)
                        parent_ids.append(parent._id)

                        offset += parent_shape[axis]

                    grad_output.zero_grad()
                except e:
                    panic("ConcatBackward GPU backward failed: " + String(e))
                return

        # CPU BACKWARD PATH
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
                    unsafe_memcpy(
                        dest=grad_input_data,
                        src=grad_data.unsafe_offset(src_offset),
                        count=num_elements,
                    )
                    parent.update_grad(grad_input^, AddTensor, None)
                # Always appended (engine fanin-completion contract).
                parent_ids.append(parent._id)
                src_offset += num_elements
            grad_output.zero_grad()
            return

        # General path: any axis
        var offset = 0
        for i in range(count):
            ref parent = output.ancestry().get(i)
            ref parent_shape = parent.shape()
            # Grad is built only if needed; the id is ALWAYS appended
            # (engine fanin-completion contract).
            if parent.requires_grad:
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
                    grad_input_data[unsafe_offset=dest_idx] = grad_data[
                        unsafe_offset=src_idx
                    ]
                    elem_idx += 1

                parent.update_grad(grad_input^, AddTensor, None)
            parent_ids.append(parent._id)

            offset += parent_shape[axis]

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

        # 1. VALIDATE
        # Axis first: even the single-tensor alias path below must reject
        # out-of-bounds axes instead of silently returning.
        var first_shape = tensors[0].shape()
        var ndim = first_shape.rank()
        var concat_axis = axis if axis >= 0 else ndim + axis

        if concat_axis < 0 or concat_axis >= ndim:
            panic("Concate → forward: axis out of bounds")

        # Single input needs no concat, but an explicit requires_grad
        # override disagreeing with the input still needs the normal wiring
        # below (tracked output + backward), not this raw alias. The alias
        # itself is intentional identity (shared storage/_id, zero-copy).
        if len(tensors) == 1:
            var only_requires_grad = tensors[0].requires_grad
            if requires_grad.or_else(only_requires_grad) == only_requires_grad:
                return tensors[0]

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

        # 1b. DEVICE VALIDATION (GPU builds only)
        # The GPU leg below allocates the output on tensors[0].device() and
        # launches one kernel per input buffer: mixed CPU/GPU (or multi-GPU)
        # inputs would read host memory as device memory, or worse. Same
        # rule as Accuracy (and PyTorch torch.cat): a single device.
        # Backward needs no check — its parents ARE these validated inputs,
        # and grad_output lives on the output's (single) device.
        comptime if has_accelerator():
            var first_device = tensors[0].device()
            for i in range(1, len(tensors)):
                if tensors[i].device() != first_device:
                    panic(
                        "Concate → forward: all tensors must be on the same"
                        " device"
                    )

        # 2. CALCULATE OUTPUT SHAPE
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

        # GPU FORWARD PATH
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
                        grad_required = grad_required or tensor.requires_grad
                        var input_axis_size = tensor.shape()[concat_axis]
                        var stride_axis = result.strides()[concat_axis]

                        ConcatKernel[Self.dtype].launch_forward(
                            tensor.buffer.layout(),
                            tensor.buffer.device_state.value(),
                            result.buffer.layout(),
                            result.buffer.device_state.value(),
                            input_axis_size,
                            output_shape[concat_axis],
                            stride_axis,
                            offset,
                        )

                        offset += input_axis_size

                    # Setup autograd
                    comptime if track_grad:
                        grad_required = requires_grad.or_else(grad_required)
                        if grad_required:
                            result.requires_grad_(True)
                            var backwardFn = BackwardFn.integer_arg[Self.dtype](
                                concat_axis,
                                ConcatBackward[Self.dtype](),
                            )
                            backwardFn.needs_parent_data = True
                            for i in range(len(tensors)):
                                result.add_ancestry(backwardFn, tensors[i])

                    return result^
                except e:
                    panic("Concate GPU forward failed: " + String(e))

        # CPU FORWARD PATH
        # 3. ALLOCATE OUTPUT
        var result = Tensor[Self.dtype].zeros(output_shape)
        ref result_data = result.buffer.data_buffer()
        ref result_strides = result.strides()
        var result_offset = result.offset()

        # 4. COPY DATA
        var offset = 0  # Track position along concat axis
        var grad_required = False

        # Fast path: with contiguous sources, each tensor contributes whole
        # row-major blocks (one per outer coordinate) that are contiguous in
        # both source and destination — a straight memcpy per block, no
        # per-element index calculation.
        var all_contiguous = True
        for tensor_idx in range(len(tensors)):
            if not tensors[tensor_idx].buffer.is_contiguous():
                all_contiguous = False
                break

        if all_contiguous:
            var stride_below = 1
            for d in range(concat_axis + 1, ndim):
                stride_below *= output_shape[d]
            var outer_count = 1
            for d in range(concat_axis):
                outer_count *= output_shape[d]
            var dest_block_stride = output_shape[concat_axis] * stride_below
            var n_threads = num_physical_cores()

            for tensor_idx in range(len(tensors)):
                ref tensor = tensors[tensor_idx]
                grad_required = grad_required or tensor.requires_grad
                ref tensor_data = tensor.buffer.data_buffer()
                var tensor_size = tensor.shape()[concat_axis]
                var block = tensor_size * stride_below
                var src_base = tensor_data.data.unsafe_value().unsafe_offset(
                    tensor.offset()
                )
                var dest_base = result_data.data.unsafe_value().unsafe_offset(
                    result_offset + offset * stride_below
                )

                def copy_block(ci: Int) {imm}:
                    unsafe_memcpy(
                        dest=dest_base.unsafe_offset(ci * dest_block_stride),
                        src=src_base.unsafe_offset(ci * block),
                        count=block,
                    )

                # memcpy is cheap per byte; only parallelize when there is
                # enough work to amortize the thread-launch cost.
                if (
                    block > 0
                    and outer_count >= n_threads
                    and outer_count * block >= n_threads * 32768
                ):
                    parallelize(copy_block, outer_count, n_threads)
                else:
                    for ci in range(outer_count):
                        copy_block(ci)

                offset += tensor_size
        else:
            # General path: coordinate-by-coordinate copy (any strides)
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

        # 5. SETUP AUTOGRAD

        comptime if track_grad:
            grad_required = requires_grad.or_else(grad_required)
            if grad_required:
                result.requires_grad_(True)
                var backwardFn = BackwardFn.integer_arg[Self.dtype](
                    concat_axis,
                    ConcatBackward[Self.dtype](),
                )
                backwardFn.needs_parent_data = True
                for i in range(len(tensors)):
                    result.add_ancestry(backwardFn, tensors[i])

        return result^
