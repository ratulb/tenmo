from tenmo import Tensor
from mnemonics import AddTensor
from validators import Validator
from backpropagation import Delegate, BackwardFn, BACKWARD_SHUFFLE
from random import shuffle, seed
from gradbox import Gradbox

from gpu import thread_idx, block_idx, block_dim, grid_dim
from gpu.host import DeviceBuffer
from memory import AddressSpace
from sys import has_accelerator
from device import DeviceState, GPU
from ndbuffer import NDBuffer
from intarray import IntArray
from array import Array
from common_utils import panic

# ── GPU Kernels ──────────────────────────────────────────────────────────────


fn shuffle_gather[
    dtype: DType
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    perm_buffer: UnsafePointer[
        Int64, ImmutAnyOrigin
    ],  # permutation as device ptr
    in_shape: Array,  # shape — max rank 8, fine for dims count
    in_strides: Array,  # strides — same
    axis: Int,
    total_elements: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= total_elements:
        return

    var remaining = tid
    var src_flat = 0
    var rank = len(in_shape)

    for k in reversed(range(rank)):
        var coord = remaining % in_shape[k]
        remaining //= in_shape[k]
        var src_coord = Int(perm_buffer[coord]) if k == axis else coord
        src_flat += src_coord * in_strides[k]

    out_buffer[tid] = in_buffer[src_flat]


fn shuffle_scatter[
    dtype: DType
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    perm_buffer: UnsafePointer[
        Int64, ImmutAnyOrigin
    ],  # permutation as device ptr
    in_shape: Array,
    in_strides: Array,
    axis: Int,
    total_elements: Int,
):
    var tid = Int(block_idx.x * block_dim.x + thread_idx.x)
    if tid >= total_elements:
        return

    var remaining = tid
    var dst_flat = 0
    var rank = len(in_shape)

    for k in reversed(range(rank)):
        var coord = remaining % in_shape[k]
        remaining //= in_shape[k]
        # Same permutation as gather — bijection guarantees no write conflicts
        var dst_coord = Int(perm_buffer[coord]) if k == axis else coord
        dst_flat += dst_coord * in_strides[k]

    out_buffer[dst_flat] = in_buffer[tid]


# ── ShuffleGPU launcher ──────────────────────────────────────────────────────


@fieldwise_init
@register_passable
struct ShuffleGPU[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    fn _upload_permutation(
        permutation: List[Int],
        gpu: GPU,
    ) raises -> DeviceBuffer[DType.int64]:
        """Upload permutation list to a DeviceBuffer of Int64."""
        var device_context = gpu()
        var n = len(permutation)
        var perm_buffer = device_context.enqueue_create_buffer[DType.int64](n)
        with perm_buffer.map_to_host() as host:
            for i in range(n):
                host[i] = Int64(permutation[i])
        return perm_buffer^

    @staticmethod
    fn launch_gather(
        A: NDBuffer[Self.dtype],
        permutation: List[Int],
        axis: Int,
    ) raises -> NDBuffer[Self.dtype]:
        var shape = A.shape
        var total_elements = shape.num_elements()

        ref device_state = A.device_state.value()
        ref gpu = device_state.get_gpu()
        var device_context = gpu()

        var in_shape = shape.array()
        var in_strides = A.strides.array()

        # Upload permutation as DeviceBuffer → pass as UnsafePointer
        var perm_device = Self._upload_permutation(permutation, gpu)

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_elements
        )

        var threads_per_block = 256
        var num_blocks = (
            total_elements + threads_per_block - 1
        ) // threads_per_block

        var compiled = device_context.compile_function[
            shuffle_gather[Self.dtype],
            shuffle_gather[Self.dtype],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            device_state.device_buffer(),
            perm_device,
            in_shape,
            in_strides,
            axis,
            total_elements,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(result_state^, shape)

    @staticmethod
    fn launch_scatter(
        grad: NDBuffer[Self.dtype],
        permutation: List[Int],
        axis: Int,
    ) raises -> NDBuffer[Self.dtype]:
        var shape = grad.shape
        var total_elements = shape.num_elements()

        ref device_state = grad.device_state.value()
        ref gpu = device_state.get_gpu()
        var device_context = gpu()

        var in_shape = shape.array()
        var in_strides = grad.strides.array()

        # Upload permutation as DeviceBuffer → pass as UnsafePointer
        var perm_device = Self._upload_permutation(permutation, gpu)

        # Zero-initialise — scatter writes to perm-mapped positions
        # Permutation is a bijection so every slot is written exactly once,
        # but zero-fill is cheap insurance
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_elements
        )
        result_buffer.enqueue_fill(Scalar[Self.dtype](0))

        var threads_per_block = 256
        var num_blocks = (
            total_elements + threads_per_block - 1
        ) // threads_per_block

        var compiled = device_context.compile_function[
            shuffle_scatter[Self.dtype],
            shuffle_scatter[Self.dtype],
        ]()

        device_context.enqueue_function(
            compiled,
            result_buffer,
            device_state.device_buffer(),
            perm_device,
            in_shape,
            in_strides,
            axis,
            total_elements,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        device_context.synchronize()

        var result_state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(result_state^, shape)


@fieldwise_init
struct ShuffleBackward_old[dtype: DType](ImplicitlyCopyable & Movable):
    comptime TAG = BACKWARD_SHUFFLE
    var axis: Int
    var permutation: List[Int]

    fn __copyinit__(out self, other: Self):
        self.axis = other.axis
        self.permutation = other.permutation.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axis = other.axis
        self.permutation = other.permutation^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)

        var shape = gradbox.shape()

        # Allocate gradient w.r.t. ancestor
        var gradbox_parent = Gradbox[Self.dtype].zeros(
            shape, share=False
        )  # parent.shape == gradients.shape, only difference is coord postions along the permuted axis

        # Scatter gradients back using the original permutation
        # For each position in the output gradient, find where it came from in the input
        for grad_coord in shape:
            parent_coord = grad_coord
            parent_coord[self.axis] = self.permutation[grad_coord[self.axis]]
            gradbox_parent[parent_coord] = gradbox[grad_coord]

        return [(parent^, gradbox_parent^, AddTensor)]


# ── Shuffle forward ──────────────────────────────────────────────────────────


@fieldwise_init
@register_passable
struct Shuffle[dtype: DType](Copyable & Movable):
    @staticmethod
    fn forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        perm: List[Int],  # permutation, length == axis length/span/spread
        axis: Int = 0,
        requires_grad: Optional[Bool] = None,
    ) -> Tensor[Self.dtype]:
        var shape = self.shape()
        var axis_length = shape[axis]
        var permutation: List[Int]

        if len(perm) > 0:
            Validator.check_permutation(perm, axis_length)
            permutation = perm.copy()
        else:
            seed()
            permutation = List[Int](capacity=axis_length)
            for i in range(axis_length):
                permutation.append(i)
            shuffle(permutation)

        var result_ndb: NDBuffer[Self.dtype]

        @parameter
        if has_accelerator():
            if self.is_on_gpu():
                try:
                    result_ndb = ShuffleGPU[Self.dtype].launch_gather(
                        self.buffer, permutation, axis
                    )
                except e:
                    panic("Shuffle → forward GPU failed: " + e.__str__())
                    result_ndb = NDBuffer[Self.dtype].Empty()  # unreachable
            else:
                result_ndb = self.buffer.shuffle(permutation, axis)
        else:
            result_ndb = self.buffer.shuffle(permutation, axis)

        var out = Tensor[Self.dtype](result_ndb^, requires_grad=False)

        @parameter
        if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backward_fn = ShuffleBackward[Self.dtype](
                    axis, permutation^
                ).into_backward_fn()
                out.backwardFn = Optional(backward_fn^)
                out.add_ancestry(self)

        return out^


# ── ShuffleBackward ──────────────────────────────────────────────────────────


@fieldwise_init
struct ShuffleBackward[dtype: DType](ImplicitlyCopyable & Movable):
    comptime TAG = BACKWARD_SHUFFLE
    var axis: Int
    var permutation: List[Int]

    fn __copyinit__(out self, other: Self):
        self.axis = other.axis
        self.permutation = other.permutation.copy()

    fn __moveinit__(out self, deinit other: Self):
        self.axis = other.axis
        self.permutation = other.permutation^

    fn into_backward_fn(self) -> BackwardFn[Self.dtype]:
        return BackwardFn[Self.dtype](Delegate[Self.dtype](self), Self.TAG)

    fn backward(
        self, output: Tensor[Self.dtype]
    ) -> List[Tuple[Tensor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref gradbox = output.gradients()[]
        var parent = output.ancestry().get(0)
        var shape = gradbox.shape()
        var gradbox_parent: Gradbox[Self.dtype]

        @parameter
        if has_accelerator():
            if gradbox.is_on_gpu():
                try:
                    var result_ndb = ShuffleGPU[Self.dtype].launch_scatter(
                        gradbox.buffer, self.permutation, self.axis
                    )
                    gradbox_parent = Gradbox[Self.dtype](
                        result_ndb^, share=False
                    )
                except e:
                    panic("ShuffleBackward GPU scatter failed: " + e.__str__())
                    gradbox_parent = Gradbox[Self.dtype].zeros(
                        shape, share=False
                    )
                return [(parent^, gradbox_parent^, AddTensor)]

        # CPU path
        # parent.shape == gradients.shape, only difference is coord postions
        # along the permuted axis
        # Scatter gradients back using the original permutation
        # For each position in the output gradient, find where it came from in the input

        gradbox_parent = Gradbox[Self.dtype].zeros(shape, share=False)
        for grad_coord in shape:
            var parent_coord = grad_coord
            parent_coord[self.axis] = self.permutation[grad_coord[self.axis]]
            gradbox_parent[parent_coord] = gradbox[grad_coord]

        return [(parent^, gradbox_parent^, AddTensor)]


fn main() raises:
    pass
