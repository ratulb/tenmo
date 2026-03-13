from gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from gpu.host import Dim
from memory import AddressSpace, stack_allocation

from array import Array
from device import DeviceState
from ndbuffer import NDBuffer
from intarray import IntArray
from mnemonics import max_rank
from strides import Strides
from broadcasthelper import ShapeBroadcaster


# ── 2D Tiled Core Kernel ──────────────────────────────────────────────────────
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


fn matmul_2d_tiled[
    dtype: DType,
    TILE_SIZE: Int = 32,
](
    out_buffer: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    A_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    B_buffer: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    A_batch_offsets: UnsafePointer[Scalar[DType.int64], ImmutAnyOrigin],
    B_batch_offsets: UnsafePointer[Scalar[DType.int64], ImmutAnyOrigin],
    m: Int,
    k: Int,
    n: Int,
):
    constrained[
        TILE_SIZE == 16 or TILE_SIZE == 32,
        "TILE_SIZE must be 16 or 32",
    ]()

    # Shared memory tiles — square, symmetric load pattern
    var smem_A = stack_allocation[
        TILE_SIZE * TILE_SIZE,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    var smem_B = stack_allocation[
        TILE_SIZE * TILE_SIZE,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()

    var tx = Int(thread_idx.x)  # col within tile
    var ty = Int(thread_idx.y)  # row within tile

    var batch = Int(block_idx.z)
    var block_col = Int(block_idx.x) * TILE_SIZE
    var block_row = Int(block_idx.y) * TILE_SIZE

    var row = block_row + ty  # global output row
    var col = block_col + tx  # global output col

    # Base pointers for this batch element
    var A_base = A_batch_offsets[batch]
    var B_base = B_batch_offsets[batch]
    var out_base = batch * m * n

    var acc = Scalar[dtype](0)

    # ── Step over k in tiles of TILE_SIZE ─────────────────────────────────────
    var num_k_tiles = (k + TILE_SIZE - 1) // TILE_SIZE

    for k_tile in range(num_k_tiles):
        var k_offset = k_tile * TILE_SIZE

        # ── Load tile of A: smem_A[ty, tx] = A[row, k_offset + tx] ──────────
        # Guard: zero-pad if outside valid A bounds
        if row < m and (k_offset + tx) < k:
            smem_A[ty * TILE_SIZE + tx] = A_buffer[
                A_base + row * k + (k_offset + tx)
            ]
        else:
            smem_A[ty * TILE_SIZE + tx] = Scalar[dtype](0)

        # ── Load tile of B: smem_B[ty, tx] = B[k_offset + ty, col] ──────────
        # Guard: zero-pad if outside valid B bounds
        if (k_offset + ty) < k and col < n:
            smem_B[ty * TILE_SIZE + tx] = B_buffer[
                B_base + (k_offset + ty) * n + col
            ]
        else:
            smem_B[ty * TILE_SIZE + tx] = Scalar[dtype](0)

        barrier()

        # ── Accumulate dot product over this k tile ───────────────────────────
        for kk in range(TILE_SIZE):
            acc += smem_A[ty * TILE_SIZE + kk] * smem_B[kk * TILE_SIZE + tx]

        barrier()

    # ── Write result — guard against out-of-bounds output ────────────────────
    if row < m and col < n:
        out_buffer[out_base + row * n + col] = acc


# ── Host-side launch wrapper ──────────────────────────────────────────────────


@fieldwise_init
@register_passable
struct MatmulNdGpu[dtype: DType = DType.float32](ImplicitlyCopyable & Movable):
    @staticmethod
    fn launch[
        tile_size: Int = 16,
    ](
        A: NDBuffer[Self.dtype],
        B: NDBuffer[Self.dtype],
    ) raises -> NDBuffer[
        Self.dtype
    ]:
        # ── Shape extraction ──────────────────────────────────────────────────
        var A_shape = A.shape
        var B_shape = B.shape

        # ── Validation ────────────────────────────────────────────────────────
        if A_shape.rank() < 2:
            raise Error("MatmulNdGpu: A must have rank >= 2")
        if B_shape.rank() < 2:
            raise Error("MatmulNdGpu: B must have rank >= 2")

        var m = A_shape[-2]
        var k_A = A_shape[-1]
        var k_B = B_shape[-2]
        var n = B_shape[-1]

        if k_A != k_B:
            raise Error("MatmulNdGpu: inner dims must match")

        var k = k_A

        # ── Batch shapes ──────────────────────────────────────────────────────
        var A_batch_shape = A_shape[:-2]
        var B_batch_shape = B_shape[:-2]

        var batch_shape = ShapeBroadcaster.broadcast_shape(
            A_batch_shape, B_batch_shape
        )

        var total_batch = batch_shape.product()
        if total_batch == 0:
            total_batch = 1  # no batch dims — single matrix multiply

        # ── Output shape ──────────────────────────────────────────────────────
        var out_shape = batch_shape + [m, n]
        var total_output = total_batch * m * n

        # ── Batch strides for broadcast clamping ──────────────────────────────
        # Sliced from full tensor strides — correct inner dim accounting
        var A_batch_strides_obj = A.strides[:-2]
        var B_batch_strides_obj = B.strides[:-2]

        # ── Precompute batch offsets on host ──────────────────────────────────
        # For each batch element b:
        #   recover batch_coords from flat b using batch_shape
        #   apply broadcast clamping for A and B
        #   store flat element offset into A and B buffers
        var A_offsets = List[Int]()
        var B_offsets = List[Int]()

        var A_batch_rank = A_batch_shape.rank()
        var B_batch_rank = B_batch_shape.rank()
        var batch_rank = batch_shape.rank()

        for b in range(total_batch):
            # Recover batch coords from flat index b
            var coords = List[Int]()
            for _ in range(batch_rank):
                coords.append(0)

            var remaining = b
            for dim in reversed(range(batch_rank)):
                coords[dim] = remaining % batch_shape[dim]
                remaining //= batch_shape[dim]

            # A offset — right-aligned broadcast clamping
            var A_off = 0
            var A_rank_off = batch_rank - A_batch_rank
            for i in range(A_batch_rank):
                var coord = (
                    coords[A_rank_off + i] if A_batch_shape[i] > 1 else 0
                )
                A_off += coord * A_batch_strides_obj[i]

            # B offset — right-aligned broadcast clamping
            var B_off = 0
            var B_rank_off = batch_rank - B_batch_rank
            for i in range(B_batch_rank):
                var coord = (
                    coords[B_rank_off + i] if B_batch_shape[i] > 1 else 0
                )
                B_off += coord * B_batch_strides_obj[i]

            A_offsets.append(A_off)
            B_offsets.append(B_off)

        # ── Device setup ──────────────────────────────────────────────────────
        ref A_device_state = A.device_state.value()
        ref gpu = A_device_state.get_gpu()
        var device_context = gpu()

        # Allocate output buffer
        var result_buffer = device_context.enqueue_create_buffer[Self.dtype](
            total_output
        )

        # Copy batch offset arrays to device
        var A_offsets_buf = device_context.enqueue_create_buffer[DType.int64](
            total_batch
        )
        var B_offsets_buf = device_context.enqueue_create_buffer[DType.int64](
            total_batch
        )
        with A_offsets_buf.map_to_host() as offset_buf_A, B_offsets_buf.map_to_host() as offset_buf_B:
            for b in range(total_batch):
                offset_buf_A[b] = A_offsets[b]
                offset_buf_B[b] = B_offsets[b]

        ref A_buf = A_device_state.device_buffer()
        ref B_buf = B.device_state.value().device_buffer()

        # ── Launch config ─────────────────────────────────────────────────────
        # grid.x = tiles across n, grid.y = tiles across m, grid.z = batch
        #var grid_x = (n + tile_size - 1) // tile_size
        #var grid_y = (m + tile_size - 1) // tile_size
        #var grid_z = total_batch
        var (grid_dim, block_dim) = Self.launch_config[tile_size](m, n, total_batch)

        # ── Compile and enqueue ───────────────────────────────────────────────
        var compiled_func = device_context.compile_function[
            matmul_2d_tiled[Self.dtype, tile_size],
            matmul_2d_tiled[Self.dtype, tile_size],
        ]()

        device_context.enqueue_function(
            compiled_func,
            result_buffer,
            A_buf,
            B_buf,
            A_offsets_buf,
            B_offsets_buf,
            m,
            k,
            n,
            #grid_dim=(grid_x, grid_y, grid_z),
            grid_dim=grid_dim,
            #block_dim=(tile_size, tile_size, 1),
            block_dim=block_dim,
        )

        device_context.synchronize()

        var device_state = DeviceState[Self.dtype](result_buffer^, gpu)
        var out = NDBuffer[Self.dtype].with_device_state(
            device_state^, out_shape
        )
        return out^

    @staticmethod
    fn launch_config[tile_size: Int](
        m: Int, n: Int, total_batch: Int
    ) -> Tuple[Dim, Dim]:
        """Returns (grid_dim, block_dim) for the tiled matmul kernel."""
        var block = Dim(tile_size, tile_size, 1)
        var grid = Dim(
            (n + tile_size - 1) // tile_size,
            #ceildiv(n, TILE_SIZE),
            #ceildiv(m, TILE_SIZE),
            (m + tile_size - 1) // tile_size,
            total_batch,
        )
        return grid, block

from tenmo import Tensor
from testing import assert_true

fn main() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].rand(3, 4, 5, 9, 8)
    var B = Tensor[dtype].rand(4, 1, 8, 20)

    C = A.matmul(B)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    var C_gpu = A_gpu.matmul(B_gpu)
    assert_true(C.all_close(C_gpu.to_cpu()))
