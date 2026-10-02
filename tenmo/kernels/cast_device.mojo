"""Layer-0 GPU kernel: element-wise dtype cast.

Kernel body only — takes raw pointers. Host-side launch logic lives
in tenmo/kernels/cast_kernel.mojo (CastKernel.launch).

Handles bool dtype: stored as uint8 on GPU (DeviceBuffer[DType.bool]
unsupported). Uses DType.uint8 for bool pointer types, matching
the pattern in compare_kernel.mojo and unary_ops_kernel.mojo.
"""

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.sys import simd_width_of


def dtype_cast[
    src_dtype: DType,
    dst_dtype: DType,
    src_datatype: DType = DType.uint8 if src_dtype == DType.bool else src_dtype,
    dst_datatype: DType = DType.uint8 if dst_dtype == DType.bool else dst_dtype,
    simd_width: Int = simd_width_of[dst_datatype](),
    simd_vectors_per_thread: Int = 2,
](
    dst: Pointer[Scalar[dst_datatype], MutAnyOrigin],
    src: Pointer[Scalar[src_datatype], ImmutAnyOrigin],
    size_: Int64,
):
    """Element-wise cast from src_dtype to dst_dtype on GPU.

    Uses src_datatype/dst_datatype for pointer types (bool→uint8 mapping).
    The actual cast uses dst_datatype for the output type conversion, which
    is correct because uint8→uint8 is identity and float→float/int→int
    truncation works via SIMD.cast.

    Grid-stride loop processes CHUNK_SIZE = simd_vectors_per_thread * simd_width
    elements per thread per pass, matching elementwise_launch_config.
    """
    var size = Int(size_)
    var tid = thread_idx.x
    var gtid = tid + block_dim.x * block_idx.x
    var stride = block_dim.x * grid_dim.x

    comptime CHUNK = simd_vectors_per_thread * simd_width
    var base = gtid * CHUNK

    while base < size:
        comptime for v in range(simd_vectors_per_thread):
            var i = base + v * simd_width
            if i + simd_width <= size:
                var chunk = src.unsafe_load[width=simd_width](i)
                dst.unsafe_store[width=simd_width](i, chunk.cast[dst_datatype]())
            elif i < size:
                for j in range(size - i):
                    dst[unsafe_offset=i + j] = Scalar[dst_datatype](
                        src[unsafe_offset=i + j]
                    )
        base += stride * CHUNK
