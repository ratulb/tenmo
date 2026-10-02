from .tensor import Tensor
from .ndbuffer import NDBuffer
from .shared.buffers import Buffer
from .shared.shapes import Shape
from std.sys import has_accelerator
from std.math import log
from std.random.philox import Random as PhiloxRandom
from std.utils.numerics import neg_inf
from std.sys.info import num_physical_cores
from max.algorithm import parallelize
from .kernels.multinomial_kernel import MultinomialKernel
from .shared.mnemonics import DEFAULT_INDEX_DTYPE


@fieldwise_init
struct Multinomial[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE]:
    @staticmethod
    def sample(
        probs: Tensor[Self.dtype],
        num_samples: Int,
        replacement: Bool = False,
        temperature: Scalar[Self.dtype] = 1.0,
        init_seed: Optional[Int] = None,
    ) raises -> Tensor[Self.index_dtype] where Self.dtype.is_floating_point():
        var rank = probs.rank()
        var N = probs.shape()[-1]

        if rank > 2:
            raise Error("multinomial: only 1D and 2D inputs supported")

        if num_samples <= 0:
            raise Error("multinomial: num_samples must be >= 1")

        if not replacement and num_samples > N:
            raise Error(
                "multinomial: num_samples exceeds vocab size without"
                " replacement"
            )

        var last_axis = List[Int]()
        last_axis.append(rank - 1)

        var p = Tensor[Self.dtype](probs.buffer.copy(), requires_grad=False)

        if temperature != 1.0:
            p = p.log[track_grad=False]() / temperature
            p = p.softmax[track_grad=False](last_axis)
        p = p / p.sum[track_grad=False](last_axis, keepdims=True)

        var B = 1 if rank == 1 else probs.shape()[0]

        # GPU path
        comptime if has_accelerator():
            if p.buffer.is_on_gpu():
                var seed_val: UInt64 = 42
                if init_seed:
                    seed_val = UInt64(init_seed.value())

                # Pre-compute log-probabilities on GPU (single existing kernel
                # launch via NDBuffer unary_ops dispatch).
                var p_log = p.log[track_grad=False]()

                var out_shape = Shape(num_samples) if rank == 1 else Shape(
                    B, num_samples
                )
                var (out_layout, out_storage) = MultinomialKernel[
                    Self.dtype, Self.index_dtype
                ].launch(
                    p_log.buffer.layout(),
                    p_log.buffer.device_state.value(),
                    out_shape,
                    num_samples,
                    seed_val,
                    replacement,
                    sync=True,
                )
                var out_ndb = NDBuffer[
                    Self.index_dtype
                ].with_layout_device_state(out_layout, out_storage)
                return Tensor[Self.index_dtype](out_ndb^, requires_grad=False)

        # CPU path
        # Fused Gumbel-max sampler mirroring MultinomialKernel. One pass over
        # (B, N) per sample with per-class Philox noise; without replacement the
        # selected class's log-prob is zeroed to -inf (no renormalisation needed
        # — the Gumbel-max conditional distribution handles it), and the row
        # loop is parallelized across batch rows.
        var p_log = p.log[track_grad=False]()
        var plog_buf = p_log.buffer.data_buffer()
        var out_buf = Buffer[Self.index_dtype](B * num_samples)
        var seed_val: UInt64 = 42
        if init_seed:
            seed_val = UInt64(init_seed.value())

        var n_threads = num_physical_cores()

        def sample_row(b: Int) {imm}:
            var row_base = b * N
            for s in range(num_samples):
                var best_val = neg_inf[Self.dtype]()
                var best_idx = 0
                var c = 0
                while c < N:
                    # One Philox call feeds 4 consecutive classes (SIMD-wide).
                    var rng = PhiloxRandom(
                        seed=seed_val,
                        subsequence=UInt64(b),
                        offset=UInt64(s * N + c),
                    )
                    var u4 = rng.step_uniform()
                    var remaining = N - c
                    for lane in range(4):
                        if lane >= remaining:
                            break
                        var u = max(u4[lane], 1e-10)
                        var gumbel_f32 = -log(-log(u))
                        var gumbel = Scalar[Self.dtype](gumbel_f32)
                        var idx = c + lane
                        var score = plog_buf[row_base + idx] + gumbel
                        if score > best_val:
                            best_val = score
                            best_idx = idx
                    c += 4
                out_buf[b * num_samples + s] = Scalar[Self.index_dtype](
                    best_idx
                )
                if not replacement and s < num_samples - 1:
                    plog_buf[row_base + best_idx] = neg_inf[Self.dtype]()

        if B >= n_threads and B * N * num_samples >= n_threads * 32768:
            parallelize(sample_row, B, n_threads)
        else:
            for b in range(B):
                sample_row(b)

        var out_shape = Shape(num_samples) if rank == 1 else Shape(
            B, num_samples
        )
        var out_ndb = NDBuffer[Self.index_dtype](out_buf^, out_shape^)
        return Tensor[Self.index_dtype](out_ndb^, requires_grad=False)
