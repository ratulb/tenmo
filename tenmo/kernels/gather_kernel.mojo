# gather_kernel.mojo — GPU gather/embedding-bag kernels

from max.gpu import thread_idx, block_dim, grid_dim, block_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import simd_width_of
from .kernel_helpers import elementwise_launch_config
from ..gpu.device import DeviceState
from ..shared.layout import Layout
from ..shared.shapes import Shape
from ..shared.strides import Strides
from ..shared.array import RankArray
from ..shared.constants import MAX_RANK
from ..shared.intarray import IntArray
from ..shared.panic import panic
from ..shared import Reduction
from ..shared.mnemonics import DEFAULT_INDEX_DTYPE


def gather_gpu_kernel[
    dtype: DType,
    rank: Int,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_shape: RankArray,
    in_strides: RankArray,
    in_offset_: Int64,
    indices_buffer: Pointer[Scalar[index_dtype], ImmutAnyOrigin],
    indices_len_: Int64,
    axis_: Int64,
    out_shape: RankArray,
    out_strides: RankArray,
    total_output_: Int64,
):
    var in_offset = Int(in_offset_)
    var indices_len = Int(indices_len_)
    var axis = Int(axis_)
    var total_output = Int(total_output_)
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var gstride = Int(block_dim.x * grid_dim.x)
    var out_idx = gtid

    while out_idx < total_output:
        var out_coords = RankArray()
        out_coords.size = rank
        var rem = out_idx
        comptime for d in range(rank - 1, -1, -1):
            out_coords.storage[d] = rem % out_shape[d]
            rem //= out_shape[d]

        var src_coords = out_coords
        var idx_val = indices_buffer[unsafe_offset=out_coords[axis]]
        if idx_val < 0:
            idx_val += Scalar[index_dtype](in_shape[axis])
        src_coords.storage[axis] = Int(idx_val)

        var src_flat = in_strides.fma(src_coords, in_offset)
        var dst_flat = out_strides.fma(out_coords, 0)
        out_buffer[unsafe_offset=dst_flat] = in_buffer[unsafe_offset=src_flat]

        out_idx += gstride


def gather_rows_2d_kernel[
    dtype: DType,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_rows_: Int64,
    in_cols_: Int64,
    in_row_stride_: Int64,
    indices_buffer: Pointer[Scalar[index_dtype], ImmutAnyOrigin],
    out_rows_: Int64,
    out_row_stride_: Int64,
):
    var in_rows = Int(in_rows_)
    var in_cols = Int(in_cols_)
    var in_row_stride = Int(in_row_stride_)
    var out_rows = Int(out_rows_)
    var out_row_stride = Int(out_row_stride_)
    var row = Int(block_idx.x)
    var col = Int(thread_idx.x)

    if row >= out_rows:
        return

    var src_row = indices_buffer[unsafe_offset=row]
    if src_row < 0:
        src_row += Scalar[index_dtype](in_rows)

    var col_stride = Int(block_dim.x)
    var c = col
    while c < in_cols:
        out_buffer[unsafe_offset=row * out_row_stride + c] = in_buffer[unsafe_offset=
            Int(src_row) * in_row_stride + c
        ]
        c += col_stride


def embedding_bag_kernel[
    dtype: DType,
    mean: Bool,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    out_buffer: Pointer[Scalar[dtype], MutAnyOrigin],
    in_buffer: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_rows_: Int64,
    in_cols_: Int64,
    in_row_stride_: Int64,
    indices_buffer: Pointer[Scalar[index_dtype], ImmutAnyOrigin],
    n_indices_: Int64,
):
    var in_rows = Int(in_rows_)
    var in_cols = Int(in_cols_)
    var in_row_stride = Int(in_row_stride_)
    var n_indices = Int(n_indices_)
    var col = Int(thread_idx.x)
    var col_stride = Int(block_dim.x)
    var c = col
    var divisor = Scalar[dtype](n_indices)
    while c < in_cols:
        var acc = Scalar[dtype](0)
        for k in range(n_indices):
            var src_row = indices_buffer[unsafe_offset=k]
            if src_row < 0:
                src_row += Scalar[index_dtype](in_rows)
            acc += in_buffer[unsafe_offset=Int(src_row) * in_row_stride + c]
        comptime if mean:
            out_buffer[unsafe_offset=c] = acc / divisor
        else:
            out_buffer[unsafe_offset=c] = acc
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
    in_shape: RankArray,
    in_strides: RankArray,
    in_offset: Int,
    idx_dev: DeviceBuffer[index_dtype],
    indices_len: Int,
    axis: Int,
    out_shape: RankArray,
    out_strides: RankArray,
    total_output: Int,
) raises:
    comptime simdwidth = simd_width_of[dtype]()
    var (blocks, tpb) = elementwise_launch_config(total_output, simdwidth)
    var compiled = ctx.compile_function[
        gather_gpu_kernel[dtype, rank, index_dtype],
    ]()
    ctx.enqueue_function(
        compiled,
        out_dev,
        in_dev,
        in_shape,
        in_strides,
        Int64(in_offset),
        idx_dev,
        Int64(indices_len),
        Int64(axis),
        out_shape,
        out_strides,
        Int64(total_output),
        grid_dim=blocks,
        block_dim=tpb,
    )


@fieldwise_init
struct GatherKernel[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def gather_gpu(
        tensor_layout: Layout,
        tensor_device_state: DeviceState[Self.dtype],
        axis: Int,
        indices: IntArray,
        reduction: Reduction = Reduction(0),
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        comptime datatype = DType.uint8 if Self.dtype == DType.bool else Self.dtype

        var rank = tensor_layout.shape.rank()
        if rank > MAX_RANK:
            panic(
                "gather_gpu: rank ",
                String(rank),
                " > MAX_RANK ",
                String(MAX_RANK),
                " not supported",
            )

        var n_indices = len(indices)

        ref ds = tensor_device_state
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
            var in_cols = tensor_layout.shape[1]
            var block_cols = _gather_2d_block_cols(in_cols)
            var out_dev = ctx.enqueue_create_buffer[datatype](in_cols)
            if reduction.is_mean():
                var compiled = ctx.compile_function[
                    embedding_bag_kernel[datatype, True, Self.index_dtype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    out_dev,
                    in_dev,
                    Int64(tensor_layout.shape[0]),
                    Int64(in_cols),
                    Int64(tensor_layout.strides[0]),
                    idx_dev,
                    Int64(n_indices),
                    grid_dim=1,
                    block_dim=block_cols,
                )
            else:
                var compiled = ctx.compile_function[
                    embedding_bag_kernel[datatype, False, Self.index_dtype],
                ]()
                ctx.enqueue_function(
                    compiled,
                    out_dev,
                    in_dev,
                    Int64(tensor_layout.shape[0]),
                    Int64(in_cols),
                    Int64(tensor_layout.strides[0]),
                    idx_dev,
                    Int64(n_indices),
                    grid_dim=1,
                    block_dim=block_cols,
                )
            if sync:
                ctx.synchronize()
            var out_shape = Shape(in_cols)
            # bool-as-uint8: skip sub-buffer creation (buffer dtype != Self.dtype when bool)
            var result_state = DeviceState[Self.dtype].__init__[special=True](
                out_dev^, gpu
            )
            return (
                Layout(out_shape),
                result_state^,
            )

        var out_shape_arr = IntArray.with_capacity(rank)
        for d in range(rank):
            out_shape_arr.append(n_indices if d == axis else tensor_layout.shape[d])
        var out_shape = Shape(out_shape_arr)
        var out_strides = Strides.default(out_shape)
        var total_output = out_shape.num_elements()

        var out_dev = ctx.enqueue_create_buffer[datatype](total_output)

        if rank == 2 and axis == 0 and tensor_layout.shape[1] <= 512:
            var in_cols = tensor_layout.shape[1]
            var block_cols = _gather_2d_block_cols(in_cols)
            var compiled = ctx.compile_function[
                gather_rows_2d_kernel[datatype, Self.index_dtype],
            ]()
            ctx.enqueue_function(
                compiled,
                out_dev,
                in_dev,
                Int64(tensor_layout.shape[0]),
                Int64(in_cols),
                Int64(tensor_layout.strides[0]),
                idx_dev,
                Int64(n_indices),
                Int64(out_strides[0]),
                grid_dim=n_indices,
                block_dim=block_cols,
            )
        # Rank dispatch generated from MAX_RANK (shared.constants) — one
        # arm per rank, selected at runtime. Lowering MAX_RANK keeps the
        # instantiations (dead arms never fire; over-rank inputs panic
        # above); raising it generates new arms automatically.
        comptime for r in range(1, MAX_RANK + 1):
            if rank == r:
                _launch_gather_generic[datatype, r, Self.index_dtype](
                    ctx,
                    out_dev,
                    in_dev,
                    tensor_layout.shape.array(),
                    tensor_layout.strides.array(),
                    tensor_layout.offset,
                    idx_dev,
                    n_indices,
                    axis,
                    out_shape.array(),
                    out_strides.array(),
                    total_output,
                )

        if sync:
            ctx.synchronize()
        # bool-as-uint8: skip sub-buffer creation (buffer dtype != Self.dtype when bool)
        var result_state = DeviceState[Self.dtype].__init__[special=True](
            out_dev^, gpu
        )
        return (
            Layout(out_shape),
            result_state^,
        )
