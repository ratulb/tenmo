from std.random.philox import Random as PhiloxRandom
from std.random import random_ui64
from std.sys import simd_width_of, has_accelerator
from std.sys.info import num_physical_cores
from max.algorithm import parallelize

from .tensor import Tensor
from .gpu.device import GPU
from .shared.mnemonics import AddTensor, Multiply
from .backpropagation import BackwardFn, ArgumentType, BackwardFnType

from .gradbox import Gradbox
from .shared.panic import panic
from .ndbuffer import NDBuffer
from .shared.buffers import Buffer
from .layer_trait import LayerTrait
from .kernels.dropout_kernel import DropoutKernel
from .ancestry import Ancestor

@fieldwise_init
struct ArgDropout[dtype: DType](ArgumentType):
    """ArgDropout.
    Packed argument struct for BackwardFn
    Stored in ArgumentType. Carries everything DropoutBackward needs:
      mask_cpu  — CPU Buffer mask (set when tensor is on CPU, else empty)
      mask_gpu  — GPU NDBuffer mask (set when tensor is on GPU, else None)
      on_gpu    — which mask arm to read in backward
    (Scale is baked into the mask at forward time, so it is not stored.)
    """
    var mask_cpu: Buffer[Self.dtype]  # valid on CPU path
    var mask_gpu: Optional[NDBuffer[Self.dtype]]  # valid on GPU path
    var on_gpu: Bool


@fieldwise_init
struct DropoutBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref arg_dropout = (
            output.ancestry().backward_fn().get[ArgDropout[Self.dtype]]()
        )
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        ref shape = ancestor.shape()

        var result_ndb: NDBuffer[Self.dtype]

        if arg_dropout.on_gpu:
            # GPU path — mask stays on device
            # NDBuffer arithmetic_ops dispatches through GPU kernel
            var mask_ndb = arg_dropout.mask_gpu.value()
            result_ndb = gradbox.buffer().arithmetic_ops[Multiply](mask_ndb)
        else:
            # CPU path — wrap Buffer mask in NDBuffer, multiply
            var mask_ndb = NDBuffer[Self.dtype](arg_dropout.mask_cpu, shape)
            result_ndb = gradbox.buffer().arithmetic_ops[Multiply](mask_ndb)

        var gradbox_ancestor = Gradbox[Self.dtype](result_ndb^)
        if ancestor.requires_grad:
            ancestor.update_grad(gradbox_ancestor^, AddTensor, None)
        parent_ids.append(ancestor._id)

        gradbox.zero_grad()


@fieldwise_init
struct Dropout[dtype: DType](LayerTrait & RegisterPassable):
    """Dropout layer — CPU and GPU enabled.

    Forward (training):
      CPU: local Philox RNG (subsequence 0) generates the mask in a SIMD
           loop; output and mask written in one pass.
      GPU: dropout_forward_kernel generates mask on-device via per-thread
           Philox subsequences; output and mask returned as GPU NDBuffers.

    Reproducibility: set_seed(s) fixes the mask stream per device — same
      seed gives the same mask every call on the same device. A fixed
      seed is NOT bit-identical across CPU/GPU (different Philox stream
      decomposition), only statistically identical. Without set_seed both
      paths draw a fresh random seed per call.

    Backward:
      grad_input = grad_output * mask   (scale already baked into mask)
      Both CPU and GPU paths use NDBuffer multiply → device-aware dispatch.

    Eval / p==0:
      Identity — returns input, no mask stored, no grad bookkeeping.
    """

    # Homogeneous layer: consumes and produces Tensor[Self.dtype];
    # OutputDType inherits InputDType (LayerTrait default).
    comptime InputDType = Self.dtype

    var training: Bool
    var p: Scalar[Self.dtype]
    var scale: Scalar[Self.dtype]
    var seed: UInt64  # UInt64 to match PhiloxRandom.seed type
    var fixed_seed: Bool

    def __init__(out self, p: Scalar[Self.dtype] = Scalar[Self.dtype](0.5)):
        if p < 0.0 or p >= 1.0:
            panic("Dropout probability must be in [0, 1)")
        self.training = True
        self.p = p
        self.scale = Scalar[Self.dtype](1.0) / (Scalar[Self.dtype](1.0) - p)
        self.seed = 42
        self.fixed_seed = False

    def __init__(out self, *, copy: Self):
        self.training = copy.training
        self.p = copy.p
        self.scale = copy.scale
        self.seed = copy.seed
        self.fixed_seed = copy.fixed_seed

    def __call__(
        mut self, x: Tensor[Self.dtype], sync: Bool = True
    ) -> Tensor[Self.dtype]:
        # Eval / no-op paths
        # Identity: returning the input alias is intentional (PyTorch
        # parity) — shared _id keeps autograd correct, zero-copy.
        if not self.training or self.p == Scalar[Self.dtype](0.0):
            return x

        # No p==1 branch: the constructor rejects p>=1, and even a
        # directly-mutated p==1 flows correctly below (r > 1 never fires,
        # all-zero mask with proper ancestry instead of a severed graph).

        # GPU path
        comptime if has_accelerator():
            if x.buffer.is_on_gpu():
                try:
                    var seed = self.seed if self.fixed_seed else (
                        random_ui64(0, 1000)
                    )
                    var result = DropoutKernel[Self.dtype].launch(
                        x.buffer.layout(),
                        x.buffer.device_state.value(),
                        self.p,
                        self.scale,
                        seed,
                        sync=sync,
                    )
                    var out_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        result[0][0], result[0][1]
                    )
                    var mask_ndb = NDBuffer[
                        Self.dtype
                    ].with_layout_device_state(result[1][0], result[1][1])

                    var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

                    if x.requires_grad:
                        out.requires_grad_(True)
                        var arg = ArgDropout[Self.dtype](
                            mask_cpu=Buffer[Self.dtype](),  # empty — not used
                            mask_gpu=Optional(mask_ndb^),
                            on_gpu=True,
                        )
                        var backwardFn = BackwardFn(
                            arg^,
                            DropoutBackward[Self.dtype](),
                        )
                        backwardFn.needs_parent_data = True
                        out.add_ancestry(backwardFn^, x)

                    return out^

                except e:
                    panic("Dropout GPU forward failed: ", String(e))
                    # Unreachable
                    return Tensor[Self.dtype].zeros(x.shape())

        # CPU path
        var shape = x.shape()
        var numels = x.numels()
        var out_buf = Buffer[Self.dtype](numels)
        var mask_buf = Buffer[Self.dtype](numels)

        # Mask stream: a local Philox generator (NOT the global RNG), so
        # set_seed is honored on CPU. fixed_seed reuses self.seed every
        # call — same mask each time, mirroring the GPU path; otherwise
        # seed from the global RNG so consecutive calls differ.
        # NOTE: a fixed seed is reproducible per device but NOT
        # bit-identical across CPU/GPU — the GPU kernel splits the key
        # into per-thread subsequences with grid-stride chunking, while
        # the CPU draws per-segment subsequence streams (serial: one
        # subsequence-0 stream, as before). Both paths use
        # the same r > p criterion with scale baked in, so masks are
        # statistically identical, just not bitwise equal.
        var philox_seed = self.seed if self.fixed_seed else random_ui64(0, 1000)

        var x_ptr = x.buffer.data_ptr()
        var out_ptr = out_buf.unsafe_ptr()
        var mask_ptr = mask_buf.unsafe_ptr()

        comptime simd_w = simd_width_of[Self.dtype]()

        var p_s = self.p
        var scale_s = self.scale
        var zero_s = Scalar[Self.dtype](0)

        var threshold_vec = SIMD[Self.dtype, simd_w](p_s)
        var scale_vec = SIMD[Self.dtype, simd_w](scale_s)
        var zero_vec = SIMD[Self.dtype, simd_w](zero_s)

        # Per-segment worker: independent Philox subsequence per segment, so
        # workers never share RNG state and the mask is deterministic for a
        # fixed seed (same mask every call, bit-identical across layers with
        # the same seed). Serial path below threshold keeps the legacy
        # single-stream layout exactly.
        def worker(seg: Int, seg_start: Int, seg_end: Int) {imm}:
            var rng = PhiloxRandom(
                seed=philox_seed, subsequence=UInt64(seg), offset=0
            )
            var i = seg_start
            var seg_vec_end = (
                seg_start + ((seg_end - seg_start) // simd_w) * simd_w
            )
            while i < seg_vec_end:
                var x_vec = x_ptr.unsafe_load[width=simd_w](i)

                # Generate random values — Philox stream, one scalar per lane
                var rand_vec = SIMD[Self.dtype, simd_w](0)
                var lane = 0
                while lane < simd_w:
                    var rand_f32 = rng.step_uniform()
                    for k in range(4):
                        if lane + k < simd_w:
                            rand_vec[lane + k] = rand_f32[k].cast[Self.dtype]()
                    lane += 4

                # Create mask: scale where rand > p, else 0
                var mask_vec = rand_vec.gt(threshold_vec).select(
                    scale_vec, zero_vec
                )

                # Apply mask and store both output and mask
                out_ptr.unsafe_store[width=simd_w](i, x_vec * mask_vec)
                mask_ptr.unsafe_store[width=simd_w](i, mask_vec)

                i += simd_w

            # Scalar tail
            for j in range(seg_vec_end, seg_end):
                var r = rng.step_uniform()[0].cast[Self.dtype]()
                var x_v = x_ptr[unsafe_offset=j]
                var m = scale_s if r > p_s else zero_s
                out_ptr[unsafe_offset=j] = x_v * m
                mask_ptr[unsafe_offset=j] = m

        var n_threads = num_physical_cores()
        if numels >= n_threads * 4096 and n_threads > 1:
            # Parallel path: split the flat index space across cores.
            def dispatch(t: Int) {imm}:
                var seg_start = t * numels // n_threads
                var seg_end = (t + 1) * numels // n_threads
                worker(t, seg_start, seg_end)

            parallelize(dispatch, n_threads, n_threads)
        else:
            # Serial path — single subsequence-0 stream (legacy layout).
            worker(0, 0, numels)

        var out_ndb = NDBuffer[Self.dtype](out_buf^, shape)
        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        if x.requires_grad:
            out.requires_grad_(True)
            var arg = ArgDropout[Self.dtype](
                mask_cpu=mask_buf^,
                mask_gpu=None,
                on_gpu=False,
            )
            var backwardFn = BackwardFn(
                arg^,
                DropoutBackward[Self.dtype](),
            )
            backwardFn.needs_parent_data = True
            out.add_ancestry(backwardFn^, x)

        return out^

    def train(mut self):
        self.training = True

    def eval(mut self):
        self.training = False

    def set_seed(mut self, seed_val: UInt64):
        """Set Philox seed for reproducible dropout masks.

        Honored on BOTH paths: the CPU draws a local Philox stream from
        this seed, the GPU forwards it to the kernel. With a fixed seed
        the same call produces the same mask every time (per device —
        CPU and GPU masks from one seed are statistically identical but
        not bit-identical).
        """
        self.seed = seed_val
        self.fixed_seed = True
