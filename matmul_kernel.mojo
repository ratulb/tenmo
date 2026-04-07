from std.gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from std.gpu.host import Dim
from std.memory import AddressSpace, stack_allocation
from shapes import Shape
from device import DeviceState
from ndbuffer import NDBuffer
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
    comptime assert TILE_SIZE == 16 or TILE_SIZE == 32, "TILE_SIZE must be 16 or 32"

    # Shared memory tiles
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

    # ── FIX 1: A_base/B_base are pointers offset from A/B ────────────────────
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

    # ── FIX 2: use C not out_buffer ───────────────────────────────────────────
    if row < m and col < n:
        C[out_base + row * n + col] = acc


@fieldwise_init
struct MatmulNdGpu[dtype: DType = DType.float32](RegisterPassable, ImplicitlyCopyable):
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
        var device_context = gpu()

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

        # ── 4: consistent names A_buf/B_buf ───────────────────────────────
        ref A_buf = A_device_state.device_buffer()
        ref B_buf = B.device_state.value().device_buffer()

        var A_rank = A_shape.rank()
        var B_rank = B_shape.rank()
        var A_row_stride = A.strides[A_rank - 2]
        var A_col_stride = A.strides[A_rank - 1]
        var B_row_stride = B.strides[B_rank - 2]
        var B_col_stride = B.strides[B_rank - 1]

        var (grid, block) = Self.launch_config[tile_size](m, n, total_batch)

        # ── FIX 3: single template arg for compile_function ───────────────────
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


from tenmo import Tensor
from std.testing import assert_true


fn main() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var B = Tensor[dtype].rand(80, 20)
    print("B.sum(): in driver program")
    B.sum().print()
    print("B[0,0]:", B[[0,0]])

    C = A.matmul(B)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()
    print("B_gpu.sum() in same driver program:")
    B_gpu.sum().print()
    print("B_gpu[0,0]:", B_gpu[[0,0]])
    var C_gpu = A_gpu.matmul(B_gpu)
    assert_true(C.all_close(C_gpu.to_cpu()))

    C.backward()
    var A_cpu_grad = A.grad().copy()
    A.zero_grad()

    C_gpu.backward()

    assert_true(A.grad().all_close(A_cpu_grad))
    print("Here I am ok")

from intarray import IntArray

fn main_200() raises:
    comptime dtype = DType.float32

    # Exact backward dimensions
    var grad_out = Tensor[dtype].rand(9, 20)   # grad_out
    var B = Tensor[dtype].rand(80, 20)          # B

    # CPU: grad_A = grad_out @ B.T
    var BT_cpu = B.transpose(axes=IntArray(-1, -2))  # (20, 80)
    var grad_A_cpu = grad_out.matmul(BT_cpu)          # (9, 80)
    print("CPU grad_A:")
    grad_A_cpu.print()

    # GPU
    var grad_out_gpu = grad_out.to_gpu()
    var B_gpu = B.to_gpu()
    var BT_gpu = B_gpu.buffer.transpose(axes=IntArray(-1, -2))

    print("BT_gpu shape:", BT_gpu.shape.__str__())
    print("BT_gpu strides:", BT_gpu.strides.__str__())

    var grad_A_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_gpu
    )
    var grad_A_gpu = Tensor[dtype](grad_A_ndb^)
    print("GPU grad_A:")
    grad_A_gpu.to_cpu().print()

    # Compare
    assert_true(grad_A_cpu.all_close(grad_A_gpu.to_cpu()))
    print("PASSED")

fn main_good() raises:
    comptime dtype = DType.float32
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var B = Tensor[dtype].rand(80, 20)
    var A_gpu = A.to_gpu()
    var B_gpu = B.to_gpu()

    # Verify B_gpu matches B exactly
    var B_back = B_gpu.to_cpu()
    assert_true(B.all_close(B_back))
    print("B transfer verified")

    # Now matmul backward manually
    var grad_out = Tensor[dtype].ones(9, 20)
    var grad_out_gpu = grad_out.to_gpu()
    var BT = B.transpose(axes=IntArray(-1, -2))
    var BT_gpu_buf = B_gpu.buffer.transpose(axes=IntArray(-1, -2))

    var grad_A_cpu = grad_out.matmul(BT)
    var grad_A_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_gpu_buf
    )
    var grad_A = Tensor[dtype](grad_A_ndb^)
    var grad_A_gpu = grad_A.to_cpu()
    assert_true(grad_A_cpu.all_close(grad_A_gpu))

    print("grad_out_gpu CPU buffer size:", len(grad_out_gpu.buffer.buffer))
    print("grad_out_gpu GPU buffer size:", len(grad_out_gpu.buffer.device_state.value().buffer))
    print("grad_out_gpu device ptr:", grad_out_gpu.buffer.device_state.value().buffer.unsafe_ptr())

    print("Manual backward verified")

fn main_100() raises:
    comptime dtype = DType.float32

    # ── Test 1: Transfer fidelity ─────────────────────────────────────────────
    print("=== Test 1: GPU transfer fidelity ===")
    var B = Tensor[dtype].rand(80, 20)
    var B_gpu = B.to_gpu()
    var B_back = B_gpu.to_cpu()
    assert_true(B.all_close(B_back))
    print("PASSED: B == B_gpu.to_cpu()")

    # ── Test 2: Ancestry storage and retrieval ────────────────────────────────
    print("=== Test 2: Ancestry storage fidelity ===")
    var A = Tensor[dtype].rand(9, 80, requires_grad=True)
    var A_gpu = A.to_gpu()
    var B2 = Tensor[dtype].rand(80, 20)
    var B2_gpu = B2.to_gpu()
    var C_gpu = A_gpu.matmul(B2_gpu)

    # Retrieve B from ancestry and verify data
    var B_from_ancestry = C_gpu.ancestry().get(1)
    var B_ancestry_back = B_from_ancestry.to_cpu()
    assert_true(B2.all_close(B_ancestry_back))
    print("PASSED: B from ancestry == original B")

    # ── Test 3: Forward matmul fidelity ───────────────────────────────────────
    print("=== Test 3: Forward matmul fidelity ===")
    var C_cpu = A.matmul(B2)
    assert_true(C_cpu.all_close(C_gpu.to_cpu()))
    print("PASSED: CPU matmul == GPU matmul")

    # ── Test 4: Backward grad_A fidelity ─────────────────────────────────────
    print("=== Test 4: Backward grad_A fidelity ===")
    C_cpu.backward()
    var A_cpu_grad = A.grad().copy()
    A.zero_grad()

    # Verify A_gpu gradbox starts at zero
    var A_gpu_grad_before = A_gpu.grad()
    print("A_gpu grad before backward (should be 0):")
    A_gpu_grad_before.print()

    C_gpu.backward()

    # Verify A_gpu grad
    print("A_gpu grad after backward:")
    A_gpu.grad().print()
    print("A CPU grad:")
    A_cpu_grad.print()

    assert_true(A.grad().all_close(A_cpu_grad))
    print("PASSED: GPU backward grad_A == CPU backward grad_A")

    # ── Test 5: Matmul with transposed inputs ─────────────────────────────────
    print("=== Test 5: Transposed matmul fidelity ===")
    var B3 = Tensor[dtype].rand(80, 20)
    var B3_gpu = B3.to_gpu()
    var BT_cpu = B3.transpose(axes=IntArray(-1, -2))   # (20, 80)
    var BT_gpu = B3_gpu.buffer.transpose(axes=IntArray(-1, -2))

    # grad_out ones
    var grad_out = Tensor[dtype].ones(9, 20)
    var grad_out_gpu = grad_out.to_gpu()

    # CPU: grad_out @ B.T
    var grad_A_cpu = grad_out.matmul(BT_cpu)

    # GPU: MatmulNdGpu directly
    var grad_A_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_gpu
    )
    var grad_A_GPU = Tensor[dtype](grad_A_ndb^)
    var grad_A_gpu = grad_A_GPU.to_cpu()

    print("CPU grad_A row 0:")
    for i in range(min(8, grad_A_cpu.shape()[-1])):
        print(grad_A_cpu.buffer[[0, i]], end=" ")
    print()
    print("GPU grad_A row 0:")
    for i in range(min(8, grad_A_gpu.shape()[-1])):
        print(grad_A_gpu.buffer[[0, i]], end=" ")
    print()

    assert_true(grad_A_cpu.all_close(grad_A_gpu))
    print("PASSED: GPU transposed matmul == CPU")

    # ── Test 6: B from ancestry transposed matmul ─────────────────────────────
    print("=== Test 6: B from ancestry transposed matmul ===")
    var B_anc = C_gpu.ancestry().get(1)
    var BT_anc = B_anc.buffer.transpose(axes=IntArray(-1, -2))

    var grad_A_anc_ndb = MatmulNdGpu[dtype].launch[tile_size=32](
        grad_out_gpu.buffer, BT_anc
    )
    var grad_A_ANC = Tensor[dtype](grad_A_anc_ndb^)
    var grad_A_anc = grad_A_ANC.to_cpu()

    print("CPU grad_A row 0:")
    for i in range(min(8, grad_A_cpu.shape()[-1])):
        print(grad_A_cpu.buffer[[0, i]], end=" ")
    print()
    print("GPU grad_A from ancestry row 0:")
    for i in range(min(8, grad_A_anc.shape()[-1])):
        print(grad_A_anc.buffer[[0, i]], end=" ")
    print()

    assert_true(grad_A_cpu.all_close(grad_A_anc))
    print("PASSED: B from ancestry transposed matmul == CPU")

    print("\n=== ALL TESTS PASSED ===")
