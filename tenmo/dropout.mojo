from std.random.philox import Random as PhiloxRandom
from std.random import seed as set_seed, random_float64
from std.sys import simd_width_of, has_accelerator

from .tensor import Tensor
from .mnemonics import AddTensor, DROPOUT
from .backpropagation import BackwardFnArg, ArgumentType, BACKWARD_DROPOUT
from .gradbox import Gradbox
from .common_utils import panic
from .ndbuffer import NDBuffer
from .buffers import Buffer
from .net import Module, Layer
from .dropout_kernel import DropoutKernel
from .ancestry import Ancestor

# =============================================================================
# ArgDropout: packed argument struct for BackwardFnArg
# =============================================================================
#
# Stored in ArgumentType. Carries everything DropoutBackward needs:
#   mask_cpu  — CPU Buffer mask (set when tensor is on CPU, else empty)
#   mask_gpu  — GPU NDBuffer mask (set when tensor is on GPU, else None)
#   scale     — 1 / (1 - p), used to reconstruct gradient scaling
#   on_gpu    — which mask arm to read in backward
#


@fieldwise_init
struct ArgDropout[dtype: DType](ArgumentType):
    var mask_cpu: Buffer[Self.dtype]  # valid on CPU path
    var mask_gpu: Optional[NDBuffer[Self.dtype]]  # valid on GPU path
    var scale: Scalar[Self.dtype]
    var on_gpu: Bool


@fieldwise_init
struct DropoutBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    fn backward(
        output: Ancestor[Self.dtype],
    ) -> List[Tuple[Ancestor[Self.dtype], Gradbox[Self.dtype], Int]]:
        ref arg_dropout = (
            output.ancestry().backward_fn_arg().get[ArgDropout[Self.dtype]]()
        )
        ref gradbox = output.gradients()[]
        var ancestor = output.ancestry().get(0)
        ref shape = ancestor.shape()

        var result_ndb: NDBuffer[Self.dtype]

        if arg_dropout.on_gpu:
            # GPU path — mask stays on device
            # NDBuffer arithmetic_ops dispatches through GPU kernel
            var mask_ndb = arg_dropout.mask_gpu.value()
            result_ndb = gradbox.buffer * mask_ndb
        else:
            # CPU path — wrap Buffer mask in NDBuffer, multiply
            var mask_ndb = NDBuffer[Self.dtype](arg_dropout.mask_cpu, shape)
            result_ndb = gradbox.buffer * mask_ndb

        var gradbox_ancestor = Gradbox[Self.dtype](result_ndb^, share=False)
        return [(ancestor^, gradbox_ancestor^, AddTensor)]


@fieldwise_init
struct Dropout[dtype: DType](RegisterPassable & ImplicitlyCopyable):
    """Dropout layer — CPU and GPU enabled.

    Forward (training):
      CPU: Philox RNG generates mask in a SIMD loop; output and mask written
           in one pass.
      GPU: dropout_forward_kernel generates mask on-device via per-thread
           Philox subsequences; output and mask returned as GPU NDBuffers.

    Backward:
      grad_input = grad_output * mask   (scale already baked into mask)
      Both CPU and GPU paths use NDBuffer multiply → device-aware dispatch.

    Eval / p==0:
      Identity — returns input, no mask stored, no grad bookkeeping.
    """

    var training: Bool
    var p: Scalar[Self.dtype]
    var scale: Scalar[Self.dtype]
    var seed: UInt64  # UInt64 to match PhiloxRandom.seed type

    fn __init__(out self, p: Scalar[Self.dtype] = Scalar[Self.dtype](0.5)):
        if p < 0.0 or p >= 1.0:
            panic("Dropout probability must be in [0, 1)")
        self.training = True
        self.p = p
        self.scale = Scalar[Self.dtype](1.0) / (Scalar[Self.dtype](1.0) - p)
        self.seed = 42

    fn __copyinit__(out self, copy: Self):
        self.training = copy.training
        self.p = copy.p
        self.scale = copy.scale
        self.seed = copy.seed

    fn __call__(self, x: Tensor[Self.dtype]) -> Tensor[Self.dtype]:
        # ── Eval / no-op paths ────────────────────────────────────────────────
        if not self.training or self.p == Scalar[Self.dtype](0.0):
            return x

        if self.p == Scalar[Self.dtype](1.0):
            return Tensor[Self.dtype].zeros(x.shape())

        # ── GPU path ──────────────────────────────────────────────────────────
        comptime if has_accelerator():
            if x.buffer.is_on_gpu():
                try:
                    var result = DropoutKernel[Self.dtype].launch(
                        x.buffer,
                        self.p,
                        self.scale,
                        self.seed,
                    )
                    var out_ndb = result[0]
                    var mask_ndb = result[1]

                    var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

                    if x.requires_grad:
                        out.requires_grad_(True)
                        var arg = ArgDropout[Self.dtype](
                            mask_cpu=Buffer[Self.dtype](),  # empty — not used
                            mask_gpu=Optional(mask_ndb^),
                            scale=self.scale,
                            on_gpu=True,
                        )
                        var backwardFnArg = BackwardFnArg[Self.dtype](
                            BACKWARD_DROPOUT, arg^
                        )
                        out.add_ancestry(backwardFnArg^, x)

                    return out^

                except e:
                    panic("Dropout GPU forward failed: ", String(e))
                    # Unreachable
                    return Tensor[Self.dtype].zeros(x.shape())

        # ── CPU path ──────────────────────────────────────────────────────────
        var shape = x.shape()
        var numels = x.numels()
        var out_buf = Buffer[Self.dtype](numels)
        var mask_buf = Buffer[Self.dtype](numels)

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

        var vec_end = (numels // simd_w) * simd_w

        # SIMD vectorized path — mask generation is scalar (random_float64
        # returns one value at a time), but the mask application and store
        # use SIMD loads/stores for throughput.
        var i = 0
        while i < vec_end:
            var x_vec = x_ptr.load[width=simd_w](i)

            # Generate random values — one scalar per lane, no SIMD RNG
            var rand_vec = SIMD[Self.dtype, simd_w](0)
            for lane in range(simd_w):
                rand_vec[lane] = random_float64(0.0, 1.0).cast[Self.dtype]()

            # Create mask: scale where rand > p, else 0
            # var mask_vec = (rand_vec > threshold_vec).select(scale_vec, zero_vec)
            var mask_vec = rand_vec.gt(threshold_vec).select(
                scale_vec, zero_vec
            )

            # Apply mask and store both output and mask
            out_ptr.store[width=simd_w](i, x_vec * mask_vec)
            mask_ptr.store[width=simd_w](i, mask_vec)

            i += simd_w

        # Scalar tail
        for j in range(vec_end, numels):
            var r = random_float64(0.0, 1.0).cast[Self.dtype]()
            var x_v = x_ptr[j]
            var m = scale_s if r > p_s else zero_s
            out_ptr[j] = x_v * m
            mask_ptr[j] = m

        var out_ndb = NDBuffer[Self.dtype](out_buf^, shape)
        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        if x.requires_grad:
            out.requires_grad_(True)
            var arg = ArgDropout[Self.dtype](
                mask_cpu=mask_buf^,
                mask_gpu=None,
                scale=self.scale,
                on_gpu=False,
            )
            var backwardFnArg = BackwardFnArg[Self.dtype](
                BACKWARD_DROPOUT, arg^
            )
            out.add_ancestry(backwardFnArg^, x)

        return out^

    fn parameters(
        ref self,
    ) -> List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]:
        return List[UnsafePointer[Tensor[Self.dtype], MutAnyOrigin]]()

    fn num_parameters(self) -> Int:
        return 0

    fn train(mut self):
        self.training = True

    fn eval(mut self):
        self.training = False

    fn set_seed(mut self, seed_val: UInt64):
        """Set Philox seed for reproducible dropout masks."""
        self.seed = seed_val

    fn into(self) -> Module[Self.dtype]:
        return Module[Self.dtype](Layer[Self.dtype](self), DROPOUT)

    fn to_gpu(self, gpu: Optional[GPU] = None) raises -> Self:
        """No-op — activation layers have no parameters to move."""
        return self

    fn to_cpu(self) raises -> Self:
        """No-op — no parameters to move."""
        return self
