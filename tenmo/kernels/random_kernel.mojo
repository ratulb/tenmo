"""GPU random fill kernels — uniform and normal (Box-Muller via Philox).

Uniform kernel fills a GPU buffer with i.i.d. values ~ Uniform[min, max).
Normal kernel fills a GPU buffer with i.i.d. values ~ Normal(mean, std).

Design follows dropout_kernel.mojo exactly:
- Per-thread PhiloxRandom seeded by (rng_seed, global_thread_id, 0)
- step_uniform() produces SIMD[float32, 4]; step_normal_4 produces SIMD[float32, 4]
- Chunk-stride pattern: each thread processes chunk_size elements per stride
- elementwise_launch_config for block sizing
"""

from std.random.philox import Random as PhiloxRandom, NormalRandom as PhiloxNormalRandom
from std.sys import simd_width_of
from std.gpu import thread_idx, block_dim, grid_dim, block_idx
from tenmo.ndbuffer import NDBuffer
from . import elementwise_launch_config
from tenmo.device import GPU, DeviceState
from tenmo.shapes import Shape


def fill_uniform_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    size: Int,
    min_val: Scalar[dtype],
    max_val: Scalar[dtype],
    rng_seed: UInt64,
):
    """Fill `result` with i.i.d. Uniform[min_val, max_val) values.

    Each thread owns an independent Philox stream keyed by global thread id.
    Uniform random float32 values from step_uniform() are scaled to [min, max)
    and cast to `dtype`.
    """
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    var rng = PhiloxRandom(seed=rng_seed, subsequence=UInt64(gtid), offset=0)

    var span = max_val - min_val
    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base = gtid * CHUNK_SIZE

    while base < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base + item * simd_width
            if i + simd_width <= size:
                var rand_f32 = rng.step_uniform()
                var out_vec = SIMD[dtype, simd_width](0)
                comptime for lane in range(simd_width):
                    var r = rand_f32[lane % 4].cast[dtype]()
                    out_vec[lane] = min_val + r * span
                result.store[width=simd_width](i, out_vec)
            elif i < size:
                for j in range(size - i):
                    var r = rng.step_uniform()
                    result[i + j] = min_val + r[0].cast[dtype]() * span
        base += stride * CHUNK_SIZE


def fill_normal_kernel[
    dtype: DType,
    simd_width: Int = simd_width_of[dtype](),
    simd_vectors_per_thread: Int = 2 * simd_width,
](
    result: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    size: Int,
    mean: Float32,
    std: Float32,
    rng_seed: UInt64,
):
    """Fill `result` with i.i.d. Normal(mean, std) values.

    Each thread owns an independent Philox stream. NormalRandom.step_normal_4()
    returns 4 float32 normal deviates per call via Box-Muller.
    """
    var gtid = Int(thread_idx.x + block_dim.x * block_idx.x)
    var stride = Int(block_dim.x * grid_dim.x)

    var rng = PhiloxNormalRandom(seed=rng_seed, subsequence=UInt64(gtid), offset=0)

    comptime CHUNK_SIZE = simd_vectors_per_thread * simd_width
    var base = gtid * CHUNK_SIZE

    while base < size:
        comptime for item in range(simd_vectors_per_thread):
            var i = base + item * simd_width
            if i + simd_width <= size:
                var norm_f32 = rng.step_normal_4(mean, std)
                var out_vec = SIMD[dtype, simd_width](0)
                comptime for lane in range(simd_width):
                    out_vec[lane] = norm_f32[lane % 4].cast[dtype]()
                result.store[width=simd_width](i, out_vec)
            elif i < size:
                for j in range(size - i):
                    var n = rng.step_normal_4(mean, std)
                    result[i + j] = n[0].cast[dtype]()
        base += stride * CHUNK_SIZE


struct RandomKernel[dtype: DType](
    ImplicitlyCopyable & Movable
):
    """Launcher for GPU random fill kernels.

    Usage:
        var gpu = GPU()
        var ndb = RandomKernel[dtype].launch_uniform(
            shape, 0.0, 1.0, seed, gpu
        )
    """

    @staticmethod
    def launch_uniform(
        shape: Shape,
        min_val: Scalar[Self.dtype],
        max_val: Scalar[Self.dtype],
        rng_seed: UInt64,
        gpu: GPU,
    ) raises -> NDBuffer[Self.dtype]:
        """Allocate and fill a GPU buffer with Uniform[min, max)."""
        var ctx = gpu[]
        var numels = shape.num_elements()
        comptime sw = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = _launch_config(numels, sw)

        var result_buffer = ctx.enqueue_create_buffer[Self.dtype](numels)

        var compiled = ctx.compile_function[
            fill_uniform_kernel[
                dtype=Self.dtype,
                simd_width=sw,
                simd_vectors_per_thread=2 * sw,
            ],
        ]()

        ctx.enqueue_function(
            compiled,
            result_buffer,
            numels,
            min_val,
            max_val,
            rng_seed,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        ctx.synchronize()

        var state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(state^, shape)

    @staticmethod
    def launch_normal(
        shape: Shape,
        mean: Float32,
        std: Float32,
        rng_seed: UInt64,
        gpu: GPU,
    ) raises -> NDBuffer[Self.dtype]:
        """Allocate and fill a GPU buffer with Normal(mean, std)."""
        var ctx = gpu[]
        var numels = shape.num_elements()
        comptime sw = simd_width_of[Self.dtype]()

        var (num_blocks, threads_per_block) = _launch_config(numels, sw)

        var result_buffer = ctx.enqueue_create_buffer[Self.dtype](numels)

        var compiled = ctx.compile_function[
            fill_normal_kernel[
                dtype=Self.dtype,
                simd_width=sw,
                simd_vectors_per_thread=2 * sw,
            ],
        ]()

        ctx.enqueue_function(
            compiled,
            result_buffer,
            numels,
            mean,
            std,
            rng_seed,
            grid_dim=num_blocks,
            block_dim=threads_per_block,
        )

        ctx.synchronize()

        var state = DeviceState[Self.dtype](result_buffer^, gpu)
        return NDBuffer[Self.dtype].with_device_state(state^, shape)


def _launch_config(
    numels: Int, simdwidth: Int
) -> Tuple[Int, Int]:
    return elementwise_launch_config(numels, simdwidth)
