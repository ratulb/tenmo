from max.gpu import thread_idx, block_dim, grid_dim, block_idx

from ..gpu.device import DeviceState
from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from .kernel_helpers import elementwise_launch_config


def where_forward_kernel[
    dtype: DType,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    a_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    b_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    cond_ptr: Pointer[Scalar[DType.uint8], ImmutAnyOrigin],
    size_: Int64,
):
    """Minimal fused where: result[i] = a[i] if cond[i] else b[i].
    One element per thread. No SIMD. No stride-based broadcast."""
    var size = Int(size_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    for i in range(gtid, size, stride):
        result[unsafe_offset=i] = a_ptr[unsafe_offset=i] if cond_ptr[unsafe_offset=i] else b_ptr[unsafe_offset=i]


struct WhereKernel[dtype: DType](ImplicitlyCopyable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch_forward(
        A_layout: Layout,
        A_device_state: DeviceState[Self.dtype],
        B_layout: Layout,
        B_device_state: DeviceState[Self.dtype],
        Cond_layout: Layout,
        Cond_device_state: DeviceState[DType.bool],
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var shape = A_layout.shape
        var numels = A_layout.numel()

        var (num_blocks, threads_per_block) = elementwise_launch_config(
            numels, 1
        )

        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var contig_a = materialize_contiguous(
            A_device_state, A_layout
        )
        var contig_b = materialize_contiguous(
            B_device_state, B_layout
        )
        var contig_cond = materialize_contiguous(
            Cond_device_state, Cond_layout
        )

        var result_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )

        var compiled = device_context.compile_function[
            where_forward_kernel[Self.datatype],
        ]()
        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_a.device_buffer(),
            contig_b.device_buffer(),
            contig_cond.device_buffer(),
            Int64(numels),
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        # bool-as-uint8: skip sub-buffer creation (buffer dtype != Self.dtype when bool)
        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, gpu
        )
        return (
            Layout(shape),
            result_state^,
        )
