from tenmo.tensor import Tensor
from tenmo.ndbuffer import NDBuffer
from tenmo.buffers import Buffer
from tenmo.shapes import Shape
from std.sys import has_accelerator
from tenmo.kernels.multinomial_kernel import MultinomialGpuKernel

@fieldwise_init
struct Multinomial[dtype: DType]:
    @staticmethod
    def sample(
        probs: Tensor[Self.dtype],
        num_samples: Int,
        replacement: Bool = False,
        temperature: Scalar[Self.dtype] = 1.0,
        init_seed: Optional[Int] = None,
    ) raises -> Tensor[DType.int32] where Self.dtype.is_floating_point():
        var rank = probs.rank()
        var N = probs.shape()[-1]

        if rank > 2:
            raise Error("multinomial: only 1D and 2D inputs supported")

        if num_samples <= 0:
            raise Error("multinomial: num_samples must be >= 1")

        if not replacement and num_samples > N:
            raise Error(
                "multinomial: num_samples exceeds vocab size without replacement"
            )

        var last_axis = List[Int]()
        last_axis.append(rank - 1)

        var p = Tensor[Self.dtype](probs.buffer.copy(), requires_grad=False)

        if temperature != 1.0:
            p = p.log[track_grad=False]() / temperature
            p = p.softmax[track_grad=False](last_axis)
        p = p / p.sum[track_grad=False](last_axis, keepdims=True)

        var B = 1 if rank == 1 else probs.shape()[0]

        # ── GPU path ──────────────────────────────────────────────────────────
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
                var out_ndb = MultinomialGpuKernel[Self.dtype].launch(
                    p_log.buffer,
                    out_shape,
                    num_samples,
                    seed_val,
                    replacement,
                    sync=True,
                )
                return Tensor[DType.int32](out_ndb^, requires_grad=False)

        # ── CPU path ──────────────────────────────────────────────────────────
        var out_buf = Buffer[DType.int32](B * num_samples)

        for s in range(num_samples):
            var seed: Optional[Int] = None
            if init_seed:
                seed = Optional[Int](init_seed.value() + s)

            var u = Tensor[Self.dtype].rand(p.shape(), init_seed=seed)
            var g = -(u + Scalar[Self.dtype](1e-7)).log[track_grad=False]()
            g = -g.log[track_grad=False]()

            var p_log = p.log[track_grad=False]()
            var scores = p_log + g
            var idx = scores.argmax(-1)

            if rank == 1:
                out_buf[s] = idx.item()
            else:
                for b in range(B):
                    out_buf[b * num_samples + s] = idx[b]

            if s < num_samples - 1 and not replacement:
                p = Multinomial[Self.dtype]._zero_out(p, idx, B, N, rank)

        var out_shape = Shape(num_samples) if rank == 1 else Shape(B, num_samples)
        var out_ndb = NDBuffer[DType.int32](out_buf^, out_shape^)
        return Tensor[DType.int32](out_ndb^, requires_grad=False)

    @staticmethod
    def _zero_out(
        p: Tensor[Self.dtype],
        idx: Tensor[DType.int32],
        B: Int,
        N: Int,
        rank: Int,
    ) raises -> Tensor[Self.dtype]:
        var mask_buf = Buffer[DType.bool](B * N)
        if rank == 1:
            var sel = idx.item().__int__()
            for i in range(N):
                mask_buf[i] = (i == sel)
        else:
            for b in range(B):
                var sel = idx[b].__int__()
                for i in range(N):
                    mask_buf[b * N + i] = (i == sel)
        var mask_shape = Shape(N) if rank == 1 else Shape(B, N)
        var mask_ndb = NDBuffer[DType.bool](mask_buf^, mask_shape^)
        var mask = Tensor[DType.bool](mask_ndb^, requires_grad=False)
        var result = p.masked_fill[track_grad=False](mask, Scalar[Self.dtype](0.0))
        var last_axis = List[Int]()
        last_axis.append(rank - 1)
        return result / result.sum[track_grad=False](last_axis, keepdims=True)
