# =============================================================================
# gather_kernel.mojo — GPU gather/embedding-bag kernels
# =============================================================================

from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys import simd_width_of, has_accelerator
from tenmo.ndbuffer import NDBuffer
from .kernel_helpers import elementwise_launch_config
from tenmo.device import DeviceState
from tenmo.shapes import Shape
from tenmo.strides import Strides
from tenmo.device import GPU
from tenmo.buffers import Buffer
from tenmo.array import Array
from tenmo.intarray import IntArray
from tenmo.common_utils import panic
from tenmo.shared import Reduction
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE


def gather_gpu_kernel[
    dtype: DType,
    rank: Int,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: Array,
    in_strides: Array,
    in_offset: Int,
    indices_buffer: UnsafePointer[Scalar[index_dtype], ImmutAnyOrigin],
    indices_len: Int,
    axis: Int,
    out_shape: Array,
    out_strides: Array,
    total_output: Int,
):
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var gstride = Int(block_dim.x * grid_dim.x)
    var out_idx = gtid

    while out_idx < total_output:
        var out_coords = Array()
        out_coords.size = rank
        var rem = out_idx
        comptime for d in range(rank - 1, -1, -1):
            out_coords.storage[d] = rem % out_shape[d]
            rem //= out_shape[d]

        var src_coords = out_coords
        var idx_val = indices_buffer[out_coords[axis]]
        if idx_val < 0:
            idx_val += Scalar[index_dtype](in_shape[axis])
        src_coords.storage[axis] = Int(idx_val)

        var src_flat = in_strides.fma(src_coords, in_offset)
        var dst_flat = out_strides.fma(out_coords, 0)
        out_buffer[dst_flat] = in_buffer[src_flat]

        out_idx += gstride


def gather_rows_2d_kernel[
    dtype: DType,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_rows: Int,
    in_cols: Int,
    in_row_stride: Int,
    indices_buffer: UnsafePointer[Scalar[index_dtype], ImmutAnyOrigin],
    out_rows: Int,
    out_row_stride: Int,
):
    var row = Int(block_idx.x)
    var col = Int(thread_idx.x)

    if row >= out_rows:
        return

    var src_row = indices_buffer[row]
    if src_row < 0:
        src_row += Scalar[index_dtype](in_rows)

    var col_stride = Int(block_dim.x)
    var c = col
    while c < in_cols:
        out_buffer[row * out_row_stride + c] = in_buffer[
            Int(src_row) * in_row_stride + c
        ]
        c += col_stride


def embedding_bag_kernel[
    dtype: DType,
    mean: Bool,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    in_rows: Int,
    in_cols: Int,
    in_row_stride: Int,
    indices_buffer: UnsafePointer[Scalar[index_dtype], ImmutAnyOrigin],
    n_indices: Int,
):
    var col = Int(thread_idx.x)
    var col_stride = Int(block_dim.x)
    var c = col
    var divisor = Scalar[dtype](n_indices)
    while c < in_cols:
        var acc = Scalar[dtype](0)
        for k in range(n_indices):
            var src_row = indices_buffer[k]
            if src_row < 0:
                src_row += Scalar[index_dtype](in_rows)
            acc += in_buffer[Int(src_row) * in_row_stride + c]
        comptime if mean:
            out_buffer[c] = acc / divisor
        else:
            out_buffer[c] = acc
        c += col_stride


def _gather_2d_block_cols(in_cols: Int) -> Int:
    if in_cols <= 32:
        return 32
    if in_cols <= 64:
        return 64
    if in_cols <= 128:
        return 128
    if in_cols <= 256:
        return 256
    return 512


def _launch_gather_generic[
    dtype: DType, rank: Int, index_dtype: DType = DEFAULT_INDEX_DTYPE
](
    ctx: DeviceContext,
    out_dev: DeviceBuffer[dtype],
    in_dev: DeviceBuffer[dtype],
    in_shape: Array,
    in_strides: Array,
    in_offset: Int,
    idx_dev: DeviceBuffer[index_dtype],
    indices_len: Int,
    axis: Int,
    out_shape: Array,
    out_strides: Array,
    total_output: Int,
) raises:
    comptime simdwidth = simd_width_of[dtype]()
    var (blocks, tpb) = elementwise_launch_config(total_output, simdwidth)
    var compiled = ctx.compile_function[
        gather_gpu_kernel[dtype, rank, index_dtype],
        gather_gpu_kernel[dtype, rank, index_dtype],
    ]()
    ctx.enqueue_function(
        compiled,
        out_dev,
        in_dev,
        in_shape,
        in_strides,
        in_offset,
        idx_dev,
        indices_len,
        axis,
        out_shape,
        out_strides,
        total_output,
        grid_dim=blocks,
        block_dim=tpb,
    )


@fieldwise_init
struct GatherGpu[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def gather_gpu(
        tensor: NDBuffer[Self.dtype],
        axis: Int,
        indices: IntArray,
        reduction: Reduction = Reduction(0),
        sync: Bool = False,
    ) raises -> NDBuffer[Self.dtype]:
        comptime datatype = DType.uint8 if Self.dtype == DType.bool else Self.dtype

        var rank = tensor.shape.rank()
        if rank > 8:
            panic("gather_gpu: rank ", String(rank), " > 8 not supported")

        var n_indices = len(indices)

        ref ds = tensor.device_state.value()
        ref gpu = ds.get_gpu()
        var ctx = gpu[]

        var idx_dev = ctx.enqueue_create_buffer[Self.index_dtype](n_indices)
        with idx_dev.map_to_host() as host_idx:
            for k in range(n_indices):
                host_idx[k] = Scalar[Self.index_dtype](indices[k])

        var in_dev = ds.buffer

        if (
            (reduction.is_sum() or reduction.is_mean())
            and rank == 2
            and axis == 0
        ):
            var in_cols = tensor.shape[1]
            var block_cols = _gather_2d_block_cols(in_cols)
            var out_dev = ctx.enqueue_create_buffer[datatype](in_cols)
            if reduction.is_mean():
                var compiled = ctx.compile_function[
                    embedding_bag_kernel[datatype, True, Self.index_dtype],
                    embedding_bag_kernel[datatype, True, Self.index_dtype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    out_dev,
                    in_dev,
                    tensor.shape[0],
                    in_cols,
                    tensor.strides[0],
                    idx_dev,
                    n_indices,
                    grid_dim=1,
                    block_dim=block_cols,
                )
            else:
                var compiled = ctx.compile_function[
                    embedding_bag_kernel[datatype, False, Self.index_dtype],
                    embedding_bag_kernel[datatype, False, Self.index_dtype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    out_dev,
                    in_dev,
                    tensor.shape[0],
                    in_cols,
                    tensor.strides[0],
                    idx_dev,
                    n_indices,
                    grid_dim=1,
                    block_dim=block_cols,
                )
            if sync:
                ctx.synchronize()
            var out_shape = Shape(in_cols)
            var result_state = DeviceState[Self.dtype].__init__[special=True](
                out_dev^, gpu
            )
            return NDBuffer[Self.dtype].with_device_state(
                result_state^, out_shape
            )

        var out_shape_arr = IntArray.with_capacity(rank)
        for d in range(rank):
            out_shape_arr.append(n_indices if d == axis else tensor.shape[d])
        var out_shape = Shape(out_shape_arr)
        var out_strides = Strides.default(out_shape)
        var total_output = out_shape.num_elements()

        var out_dev = ctx.enqueue_create_buffer[datatype](total_output)

        if rank == 2 and axis == 0 and tensor.shape[1] <= 512:
            var in_cols = tensor.shape[1]
            var block_cols = _gather_2d_block_cols(in_cols)
            var compiled = ctx.compile_function[
                gather_rows_2d_kernel[datatype, Self.index_dtype],
                gather_rows_2d_kernel[datatype, Self.index_dtype],
            ]()
            ctx.enqueue_function(
                compiled,
                out_dev,
                in_dev,
                tensor.shape[0],
                in_cols,
                tensor.strides[0],
                idx_dev,
                n_indices,
                out_strides[0],
                grid_dim=n_indices,
                block_dim=block_cols,
            )
        elif rank == 1:
            _launch_gather_generic[datatype, 1, Self.index_dtype](
                ctx,
                out_dev,
                in_dev,
                tensor.shape.array(),
                tensor.strides.array(),
                tensor.offset,
                idx_dev,
                n_indices,
                axis,
                out_shape.array(),
                out_strides.array(),
                total_output,
            )
        elif rank == 2:
            _launch_gather_generic[datatype, 2, Self.index_dtype](
                ctx,
                out_dev,
                in_dev,
                tensor.shape.array(),
                tensor.strides.array(),
                tensor.offset,
                idx_dev,
                n_indices,
                axis,
                out_shape.array(),
                out_strides.array(),
                total_output,
            )
        elif rank == 3:
            _launch_gather_generic[datatype, 3, Self.index_dtype](
                ctx,
                out_dev,
                in_dev,
                tensor.shape.array(),
                tensor.strides.array(),
                tensor.offset,
                idx_dev,
                n_indices,
                axis,
                out_shape.array(),
                out_strides.array(),
                total_output,
            )
        elif rank == 4:
            _launch_gather_generic[datatype, 4, Self.index_dtype](
                ctx,
                out_dev,
                in_dev,
                tensor.shape.array(),
                tensor.strides.array(),
                tensor.offset,
                idx_dev,
                n_indices,
                axis,
                out_shape.array(),
                out_strides.array(),
                total_output,
            )
        elif rank == 5:
            _launch_gather_generic[datatype, 5, Self.index_dtype](
                ctx,
                out_dev,
                in_dev,
                tensor.shape.array(),
                tensor.strides.array(),
                tensor.offset,
                idx_dev,
                n_indices,
                axis,
                out_shape.array(),
                out_strides.array(),
                total_output,
            )
        elif rank == 6:
            _launch_gather_generic[datatype, 6, Self.index_dtype](
                ctx,
                out_dev,
                in_dev,
                tensor.shape.array(),
                tensor.strides.array(),
                tensor.offset,
                idx_dev,
                n_indices,
                axis,
                out_shape.array(),
                out_strides.array(),
                total_output,
            )
        elif rank == 7:
            _launch_gather_generic[datatype, 7, Self.index_dtype](
                ctx,
                out_dev,
                in_dev,
                tensor.shape.array(),
                tensor.strides.array(),
                tensor.offset,
                idx_dev,
                n_indices,
                axis,
                out_shape.array(),
                out_strides.array(),
                total_output,
            )
        else:
            _launch_gather_generic[datatype, 8, Self.index_dtype](
                ctx,
                out_dev,
                in_dev,
                tensor.shape.array(),
                tensor.strides.array(),
                tensor.offset,
                idx_dev,
                n_indices,
                axis,
                out_shape.array(),
                out_strides.array(),
                total_output,
            )

        if sync:
            ctx.synchronize()
        var result_state = DeviceState[Self.dtype].__init__[special=True](
            out_dev^, gpu
        )
        return NDBuffer[Self.dtype].with_device_state(result_state^, out_shape)
