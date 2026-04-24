from std.gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from std.gpu.host import Dim
from std.memory import AddressSpace, stack_allocation
from .shapes import Shape
from .device import DeviceState
from .ndbuffer import NDBuffer
from .broadcasthelper import ShapeBroadcaster


# 2D Tiled Core Kernel
#
# Computes C = A @ B for a batch of 2D matrix multiplications.
#
# Launch config (set by host):
#   grid_dim.x  = ceil(n / TILE_SIZE)   — tiles across output columns
#   grid_dim.y  = ceil(m / TILE_SIZE)   — tiles across output rows
#   grid_dim.z  = total_batch           — one z-slice per batch element
#   block_dim.x = TILE_SIZE
#   block_dim.y = TILE_SIZE
#
# A_batch_offsets[b] and B_batch_offsets[b] are flat element offsets into
# A_buffer and B_buffer for batch element b, precomputed on the host.
# Output batch b starts at out_buffer + b * m * n (always contiguous).
#
# Shared memory:
#   smem_A: [TILE_SIZE, TILE_SIZE]  — tile of A rows
#   smem_B: [TILE_SIZE, TILE_SIZE]  — tile of B cols
#
# Each thread (ty, tx) owns output element (row, col) = (block_row + ty, block_col + tx).
# It steps over k in chunks of TILE_SIZE, accumulating the dot product.
# TILE SIZE — 32x32 = 1024 threads per block
# Each thread computes one output element
# smem_A: 32*32*4 = 4096 bytes
# smem_B: 32*32*4 = 4096 bytes
# Total smem: 8192 bytes — well within 48KB limit


fn matmul_2d_tiled[
    dtype: DType,
    TILE_SIZE: Int = 32,
](
    A: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    C: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A_batch_offsets: UnsafePointer[Scalar[DType.int64], ImmutAnyOrigin],
    B_batch_offsets: UnsafePointer[Scalar[DType.int64], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    A_row_stride: Int,
    A_col_stride: Int,
    B_row_stride: Int,
    B_col_stride: Int,
):
    comptime assert (
        TILE_SIZE == 16 or TILE_SIZE == 32
    ), "TILE_SIZE must be 16 or 32"

    # Shared memory tiles
    var smem_A = stack_allocation[
        TILE_SIZE * TILE_SIZE,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()
    var smem_B = stack_allocation[
        TILE_SIZE * TILE_SIZE,
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
    ]()

    var tx = Int(thread_idx.x)  # col within tile
    var ty = Int(thread_idx.y)  # row within tile

    var batch = Int(block_idx.z)
    var block_col = Int(block_idx.x) * TILE_SIZE
    var block_row = Int(block_idx.y) * TILE_SIZE

    var row = block_row + ty  # global output row
    var col = block_col + tx  # global output col

    # A_base/B_base are pointers offset from A/B ────────────────────
    var A_base = A + A_batch_offsets[batch]
    var B_base = B + B_batch_offsets[batch]
    var out_base = batch * m * n

    var acc = Scalar[dtype](0)

    var num_k_tiles = (k + TILE_SIZE - 1) // TILE_SIZE

    for k_tile in range(num_k_tiles):
        var k_offset = k_tile * TILE_SIZE

        # Load tile of A
        var a_col = k_offset + tx
        if row < m and a_col < k:
            smem_A[ty * TILE_SIZE + tx] = A_base[
                row * A_row_stride + a_col * A_col_stride
            ]
        else:
            smem_A[ty * TILE_SIZE + tx] = 0

        # Load tile of B
        var b_row = k_offset + ty
        if b_row < k and col < n:
            smem_B[ty * TILE_SIZE + tx] = B_base[
                b_row * B_row_stride + col * B_col_stride
            ]
        else:
            smem_B[ty * TILE_SIZE + tx] = 0

        barrier()

        for kk in range(TILE_SIZE):
            acc += smem_A[ty * TILE_SIZE + kk] * smem_B[kk * TILE_SIZE + tx]

        barrier()

    # use C not out_buffer
    if row < m and col < n:
        C[out_base + row * n + col] = acc


@fieldwise_init
struct MatmulNdGpu[dtype: DType = DType.float32](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    fn launch[
        tile_size: Int = 32,
    ](
        A: NDBuffer[Self.dtype],
        B: NDBuffer[Self.dtype],
    ) raises -> NDBuffer[
        Self.dtype
    ]:
        """ND batched matrix multiply with broadcasting.

        A: [..., m, k]
        B: [..., k, n]
        out: [..., m, n]
        """
        var A_shape = A.shape
        var B_shape = B.shape

        if A_shape.rank() < 2:
            raise Error("MatmulNdGpu: A must have rank >= 2")
        if B_shape.rank() < 2:
            raise Error("MatmulNdGpu: B must have rank >= 2")

        var m = A_shape[-2]
        var k_A = A_shape[-1]
        var k_B = B_shape[-2]
        var n = B_shape[-1]

        if k_A != k_B:
            raise Error(
                "MatmulNdGpu: inner dims must match, got "
                + String(k_A)
                + " and "
                + String(k_B)
            )

        var k = k_A

        var A_batch_shape = A_shape[:-2]
        var B_batch_shape = B_shape[:-2]

        var batch_shape = ShapeBroadcaster.broadcast_shape(
            A_batch_shape, B_batch_shape
        )

        var total_batch = batch_shape.product()
        if total_batch == 0:
            total_batch = 1

        var out_shape = batch_shape + Shape(m, n)
        var total_output = total_batch * m * n

        var A_batch_strides_obj = A.strides[:-2]
        var B_batch_strides_obj = B.strides[:-2]

        var A_batch_rank = A_batch_shape.rank()
        var B_batch_rank = B_batch_shape.rank()
        var batch_rank = batch_shape.rank()

        var A_offsets = List[Int]()
        var B_offsets = List[Int]()

        for b in range(total_batch):
            var coords = List[Int]()
            for _ in range(batch_rank):
                coords.append(0)

            var remaining = b
            for dim in reversed(range(batch_rank)):
                coords[dim] = remaining % batch_shape[dim]
                remaining //= batch_shape[dim]

            var A_off = 0
            var A_rank_off = batch_rank - A_batch_rank
            for i in range(A_batch_rank):
                var coord = (
                    coords[A_rank_off + i] if A_batch_shape[i] > 1 else 0
                )
                A_off += coord * A_batch_strides_obj[i]

            var B_off = 0
            var B_rank_off = batch_rank - B_batch_rank
            for i in range(B_batch_rank):
                var coord = (
                    coords[B_rank_off + i] if B_batch_shape[i] > 1 else 0
                )
                B_off += coord * B_batch_strides_obj[i]

            A_offsets.append(A_off)
            B_offsets.append(B_off)

        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu[]

        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )

        var A_offsets_buf = device_context.enqueue_create_buffer[DType.int64](
            total_batch
        )
        var B_offsets_buf = device_context.enqueue_create_buffer[DType.int64](
            total_batch
        )

        with A_offsets_buf.map_to_host() as h:
            for b in range(total_batch):
                h[b] = Int64(A_offsets[b])
        with B_offsets_buf.map_to_host() as h:
            for b in range(total_batch):
                h[b] = Int64(B_offsets[b])

        ref A_buf = A_device_state.device_buffer()
        ref B_buf = B.device_state.value().device_buffer()

        var A_rank = A_shape.rank()
        var B_rank = B_shape.rank()
        var A_row_stride = A.strides[A_rank - 2]
        var A_col_stride = A.strides[A_rank - 1]
        var B_row_stride = B.strides[B_rank - 2]
        var B_col_stride = B.strides[B_rank - 1]

        var (grid, block) = Self.launch_config[tile_size](m, n, total_batch)

        # Single template arg for compile_function
        var compiled_func = device_context.compile_function[
            matmul_2d_tiled[Self.dtype, tile_size],
            matmul_2d_tiled[Self.dtype, tile_size],
        ]()

        device_context.enqueue_function(
            compiled_func,
            A_buf,
            B_buf,
            result_buffer,
            A_offsets_buf,
            B_offsets_buf,
            m,
            n,
            k,
            A_row_stride,
            A_col_stride,
            B_row_stride,
            B_col_stride,
            grid_dim=grid,
            block_dim=block,
        )

        device_context.synchronize()

        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var out = NDBuffer[Self.dtype].with_device_state(
            device_state^, out_shape
        )
        return out^

    @staticmethod
    fn launch_config[
        tile_size: Int
    ](m: Int, n: Int, total_batch: Int) -> Tuple[Dim, Dim]:
        var block = Dim(tile_size, tile_size, 1)
        var grid = Dim(
            (n + tile_size - 1) // tile_size,
            (m + tile_size - 1) // tile_size,
            total_batch,
        )
        return grid, block
