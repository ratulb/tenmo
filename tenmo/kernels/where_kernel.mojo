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
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    cond_ptr: UnsafePointer[Scalar[DType.uint8], ImmutAnyOrigin],
    size: Int,
    cond_numels: Int,
    a_val: Scalar[dtype],
    b_val: Scalar[dtype],
    a_is_scalar: UInt8,
    b_is_scalar: UInt8,
    cond_stride0: Int,
    cond_stride1: Int,
    out_stride1: Int,
):
    """Fused where: result[i] = a[i] if cond[i] else b[i].
    Condition is uint8 (0/1), broadcast via strides."""
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base_idx = gtid * CHUNK_SIZE

    while base_idx < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base_idx + item * simd_width

            if i + simd_width <= size:
                var vec_a: SIMD[dtype, simd_width]
                var vec_b: SIMD[dtype, simd_width]

                if a_is_scalar:
                    vec_a = SIMD[dtype, simd_width](a_val)
                else:
                    vec_a = a_ptr.load[width=simd_width](i)
                if b_is_scalar:
                    vec_b = SIMD[dtype, simd_width](b_val)
                else:
                    vec_b = b_ptr.load[width=simd_width](i)

                var vec_result = vec_a
                for j in range(simd_width):
                    var idx = i + j
                    var cond_idx = idx
                    if cond_numels > 1:
                        var row = idx // out_stride1
                        var col = idx % out_stride1
                        cond_idx = row * cond_stride0 + col * cond_stride1
                        cond_idx = cond_idx % cond_numels
                    vec_result[j] = vec_a[j] if cond_ptr[cond_idx] else vec_b[j]

                result.store[width=simd_width](i, vec_result)

            elif i < size:
                for j in range(size - i):
                    var idx = i + j
                    var a_val_j = a_val if a_is_scalar else a_ptr[idx]
                    var b_val_j = b_val if b_is_scalar else b_ptr[idx]
                    var cond_idx = idx
                    if cond_numels > 1:
                        var row = idx // out_stride1
                        var col = idx % out_stride1
                        cond_idx = row * cond_stride0 + col * cond_stride1
                        cond_idx = cond_idx % cond_numels
                    result[idx] = a_val_j if cond_ptr[cond_idx] else b_val_j

        base_idx += stride * CHUNK_SIZE


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
        var shape = A.shape
        var numels = A.numels()
        comptime simdwidth = simd_width_of[Self.datatype]()

        var (num_blocks, threads_per_block) = elementwise_launch_config(
            numels, simdwidth
        )

        ref device_state = A.device_state.value()
        var device_context = device_state.gpu[]

        var contig_a = A.contiguous_device_state()
        var contig_b = B.contiguous_device_state()
        var contig_cond = Condition.contiguous_device_state()

        var result_buffer = device_context.enqueue_create_buffer[Self.datatype](
            numels
        )

        var cond_rank = Condition.shape.rank()
        var cond_stride0 = Condition.strides[cond_rank - 2] if cond_rank >= 2 else 0
        var cond_stride1 = Condition.strides[cond_rank - 1] if cond_rank >= 1 else 0
        var cond_numels = Condition.numels()
        var out_stride1 = shape[shape.rank() - 1]

        var compiled = device_context.compile_function[
            where_forward_kernel[Self.datatype, simdwidth, 2 * simdwidth],
            where_forward_kernel[Self.datatype, simdwidth, 2 * simdwidth],
        ]()
        device_context.enqueue_function(
            compiled,
            result_buffer,
            contig_a.device_buffer(),
            contig_b.device_buffer(),
            contig_cond.device_buffer(),
            numels,
            cond_numels,
            a_scalar,
            b_scalar,
            UInt8(1) if a_is_scalar else UInt8(0),
            UInt8(1) if b_is_scalar else UInt8(0),
            cond_stride0,
            cond_stride1,
            out_stride1,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        if sync: device_context.synchronize()

        var result_state = DeviceState[Self.dtype].__init__[True](
            result_buffer^, device_state.gpu
        )
        return NDBuffer[Self.dtype].with_device_state(result_state^, shape)
