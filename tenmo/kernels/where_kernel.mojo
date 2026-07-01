from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of

from tenmo.ndbuffer import NDBuffer
from . import elementwise_launch_config
from tenmo.device import DeviceState
from tenmo.shapes import Shape
from tenmo.strides import Strides
from tenmo.common_utils import panic


def where_forward_kernel[
    dtype: DType,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cond_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    size: Int,
):
    """Minimal fused where: result[i] = a[i] if cond[i] else b[i].
    One element per thread. No SIMD. No stride-based broadcast."""
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    for i in range(gtid, size, stride):
        result[i] = a_ptr[i] if cond_ptr[i] else b_ptr[i]


struct WhereGpuKernel[dtype: DType](ImplicitlyCopyable & Movable):
    comptime datatype: DType = DType.uint8 if Self.dtype == DType.bool else Self.dtype

    @staticmethod
    def launch_forward(
        A: NDBuffer[Self.dtype],
        B: NDBuffer[Self.dtype],
        Condition: NDBuffer[DType.bool],
        a_is_scalar: Bool,
        b_is_scalar: Bool,
        a_scalar: Scalar[Self.dtype],
        b_scalar: Scalar[Self.dtype],
        sync: Bool = False,
    ) raises -> NDBuffer[Self.dtype]:
        _ = (a_is_scalar, b_is_scalar, a_scalar, b_scalar)
        var shape = A.shape
        var numels = A.numels()

        var (num_blocks, threads_per_block) = elementwise_launch_config(
            numels, 1
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]

        var contig_a = A.contiguous_device_state()
        var contig_b = B.contiguous_device_state()
        var contig_cond = Condition.contiguous_device_state()

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
            numels,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync:
            device_context.synchronize()

        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(result_state^, shape)
