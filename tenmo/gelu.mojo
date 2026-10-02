from .tensor import Tensor
from .shared.mnemonics import AddTensor, GELU_FORWARD, Multiply
from .backpropagation import (
    BackwardFnType,
    BackwardFn,
    BufferArg,
    NDBufferArg,
)
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .shared.buffers import Buffer
from .ancestry import Ancestor
from .kernels.unary_ops_kernel import UnaryKernel
from .shared.panic import panic
from std.sys import simd_width_of, has_accelerator
from std.math import tanh

from .shared.constants import _GELU_K0, _GELU_C, _GELU_C3


@fieldwise_init
struct GELUBackward[dtype: DType](BackwardFnType, ImplicitlyCopyable):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        # Identical to ReLUBackward — the cached buffer's meaning
        # (0/1 mask vs. continuous derivative) doesn't matter here,
        # it's just multiplied through.
        ref arg = output.ancestry().backward_fn()
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        ref shape = ancestor.shape()

        var result_ndb: NDBuffer[Self.dtype]

        if gradbox.is_on_gpu():
            var deriv_ndb = arg.get[NDBufferArg[Self.dtype]]().ndb
            result_ndb = gradbox.buffer().arithmetic_ops[Multiply](deriv_ndb)
        else:
            var deriv_buf = arg.get[BufferArg[Self.dtype]]().buffer
            var result_buf = gradbox.buffer().data_buffer() * deriv_buf
            result_ndb = NDBuffer[Self.dtype](result_buf^, shape)

        var ancestor_gbx = Gradbox[Self.dtype](result_ndb^)
        ancestor.update_grad(ancestor_gbx^, AddTensor, None)

        parent_ids.append(ancestor._id)
        gradbox.zero_grad()


@fieldwise_init
struct GeLU[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype] where Self.dtype.is_floating_point():
        """Apply GELU (tanh approximation).

            GELU(x) = 0.5*x*(1 + tanh(k0*(x + c*x^3))),
            k0 = sqrt(2/pi), c = 0.044715.

        Fuses output + derivative computation, same shape as ReLU's
        fused output+mask.
        """

        var result = Self._device_forward(self.buffer)
        var out_ndb = result[0]
        var deriv_ndb = result[1]

        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFn: BackwardFn

                if self.buffer.is_on_gpu():
                    backwardFn = BackwardFn.from_ndbuffer[Self.dtype](
                        deriv_ndb^,
                        GELUBackward[Self.dtype](),
                    )
                else:
                    backwardFn = BackwardFn.from_buffer[Self.dtype](
                        deriv_ndb.data_buffer(),
                        GELUBackward[Self.dtype](),
                    )
                backwardFn.needs_parent_data = True
                out.add_ancestry(backwardFn^, self)

        return out^

    @staticmethod
    @always_inline
    def _buffer_forward(
        buf: Buffer[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Tuple[
        Buffer[Self.dtype], Buffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        var extent = end_index.or_else(buf.size) - start_index
        var out = Buffer[Self.dtype](extent)
        var deriv = Buffer[Self.dtype](extent)

        comptime simd_width = simd_width_of[Self.dtype]()
        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width

        var k0 = SIMD[Self.dtype, simd_width](Scalar[Self.dtype](_GELU_K0))
        var c = SIMD[Self.dtype, simd_width](Scalar[Self.dtype](_GELU_C))
        var c3 = SIMD[Self.dtype, simd_width](Scalar[Self.dtype](_GELU_C3))
        var half = SIMD[Self.dtype, simd_width](0.5)
        var one = SIMD[Self.dtype, simd_width](1)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var x = buf.load[simdwidth=simd_width](start_index + idx)
            var x2 = x * x
            var t = tanh(k0 * (x + c * x * x2))
            var result = half * x * (one + t)
            var deriv_val = half * (one + t) + half * x * (one - t * t) * (
                k0 * (one + c3 * x2)
            )
            out.store[simdwidth=simd_width](idx, result)
            deriv.store[simdwidth=simd_width](idx, deriv_val)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var k0_s = Scalar[Self.dtype](_GELU_K0)
            var c_s = Scalar[Self.dtype](_GELU_C)
            var c3_s = Scalar[Self.dtype](_GELU_C3)
            var half_s = Scalar[Self.dtype](0.5)
            var one_s = Scalar[Self.dtype](1)
            for i in range(remainder):
                var idx = start_idx + i
                var x = buf.load[simdwidth=1](start_index + idx)[0]
                var x2 = x * x
                var t = tanh(k0_s * (x + c_s * x * x2))
                var result = half_s * x * (one_s + t)
                var deriv_val = half_s * (one_s + t) + half_s * x * (
                    one_s - t * t
                ) * (k0_s * (one_s + c3_s * x2))
                out.store[simdwidth=1](idx, SIMD[Self.dtype, 1](result))
                deriv.store[simdwidth=1](idx, SIMD[Self.dtype, 1](deriv_val))

        return (out^, deriv^)

    @staticmethod
    @always_inline
    def _device_forward(
        ndb: NDBuffer[Self.dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        var out: NDBuffer[Self.dtype]
        var deriv: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var result = UnaryKernel[Self.dtype].launch_with_mask[
                        GELU_FORWARD
                    ](ndb.layout(), ndb.device_state.value())
                    out = NDBuffer[Self.dtype].with_layout_device_state(
                        result[0], result[1]
                    )
                    deriv = NDBuffer[Self.dtype].with_layout_device_state(
                        result[2], result[3]
                    )
                except e:
                    panic("GELU forward → GPU launch failed: ", String(e))
                    out = NDBuffer[Self.dtype].Empty()
                    deriv = NDBuffer[Self.dtype].Empty()
            else:
                (out, deriv) = Self._cpu_forward(ndb)
        else:
            (out, deriv) = Self._cpu_forward(ndb)

        return (out^, deriv^)

    @staticmethod
    @always_inline
    def _cpu_forward(
        ndb: NDBuffer[Self.dtype],
    ) -> Tuple[
        NDBuffer[Self.dtype], NDBuffer[Self.dtype]
    ] where Self.dtype.is_floating_point():
        if ndb.is_contiguous():
            var start = ndb.offset
            var end = start + ndb.numels()
            var result = Self._buffer_forward(ndb.buffer, start, end)
            return (
                NDBuffer[Self.dtype](result[0], ndb.shape),
                NDBuffer[Self.dtype](result[1], ndb.shape),
            )
        else:
            var numels = ndb.numels()
            var out_buf = Buffer[Self.dtype](numels)
            var deriv_buf = Buffer[Self.dtype](numels)
            var k0 = Scalar[Self.dtype](_GELU_K0)
            var c = Scalar[Self.dtype](_GELU_C)
            var c3 = Scalar[Self.dtype](_GELU_C3)
            var half = Scalar[Self.dtype](0.5)
            var one = Scalar[Self.dtype](1)
            var index = 0
            for idx in ndb.index_iterator():
                var x = ndb.buffer[idx]
                var x2 = x * x
                var t = tanh(k0 * (x + c * x * x2))
                out_buf[index] = half * x * (one + t)
                deriv_buf[index] = half * (one + t) + half * x * (
                    one - t * t
                ) * (k0 * (one + c3 * x2))
                index += 1
            return (
                NDBuffer[Self.dtype](out_buf^, ndb.shape),
                NDBuffer[Self.dtype](deriv_buf^, ndb.shape),
            )
