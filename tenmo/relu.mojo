from .tensor import Tensor
from .mnemonics import AddTensor, RELU_FORWARD
from .backpropagation import (
    BackwardFnArg,
    BufferArg,
    NDBufferArg,
    BACKWARD_RELU,
)
from .gradbox import Gradbox
from .ndbuffer import NDBuffer
from .buffers import Buffer
from .ancestry import Ancestor
from tenmo.kernels.unary_ops_kernel import UnaryOpsKernel
from .common_utils import panic
from std.sys import simd_width_of, has_accelerator


@fieldwise_init
struct ReLUBackward[dtype: DType](ImplicitlyCopyable & Movable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref arg = output.ancestry().backward_fn_arg()
        ref gradbox = output.gradients()
        var ancestor = output.ancestry().get(0)
        ref shape = ancestor.shape()

        var result_ndb: NDBuffer[Self.dtype]

        if gradbox.is_on_gpu():
            var mask_ndb = arg.get[NDBufferArg[Self.dtype]]().ndb
            result_ndb = gradbox.buffer().arithmetic_ops[Multiply](mask_ndb)
        else:
            var mask_buf = arg.get[BufferArg[Self.dtype]]().buffer
            var result_buf = gradbox.buffer().data_buffer() * mask_buf
            result_ndb = NDBuffer[Self.dtype](result_buf^, shape)

        var ancestor_gbx = Gradbox[Self.dtype](result_ndb^)
        ancestor.update_grad(ancestor_gbx^, AddTensor, None)

        parent_ids.append(ancestor._id)
        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct ReLU[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True
    ](
        self: Tensor[Self.dtype],
        requires_grad: Optional[Bool] = None,
        sync: Bool = False,
    ) -> Tensor[Self.dtype]:
        """Apply ReLU activation: max(0, x).

        Computes output and mask simultaneously for efficiency.
        Routes through NDBuffer.unary_ops_with_mask — GPU and CPU aware.

        Args:
            self: Input tensor.
            requires_grad: Override gradient tracking (default: inherit from input).

        Returns:
            Output tensor with ReLU applied.
        """

        var result = Self._device_forward(self.buffer)
        var out_ndb = result[0]  # NDBuffer — output values
        var mask_ndb = result[1]  # NDBuffer — gradient mask

        var out = Tensor[Self.dtype](out_ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var backwardFnArg: BackwardFnArg[Self.dtype]

                if self.buffer.is_on_gpu():
                    backwardFnArg = BackwardFnArg[Self.dtype].from_ndbuffer(
                        BACKWARD_RELU, mask_ndb^
                    )
                else:
                    backwardFnArg = BackwardFnArg[Self.dtype].from_buffer(
                        BACKWARD_RELU, mask_ndb.data_buffer()
                    )
                backwardFnArg.needs_parent_data = True
                out.add_ancestry(backwardFnArg^, self)

        return out^

    @staticmethod
    @always_inline
    def _buffer_forward(
        buf: Buffer[Self.dtype],
        start_index: Int = 0,
        end_index: Optional[Int] = None,
    ) -> Tuple[Buffer[Self.dtype], Buffer[Self.dtype]]:
        var extent = end_index.or_else(buf.size) - start_index
        var out = Buffer[Self.dtype](extent)
        var mask = Buffer[Self.dtype](extent)

        comptime simd_width = 1 if Self.dtype == DType.bool else simd_width_of[
            Self.dtype
        ]()

        var num_full_chunks = extent // simd_width
        var remainder = extent % simd_width
        var zero = SIMD[Self.dtype, simd_width](0)
        var one = SIMD[Self.dtype, simd_width](1)

        for chunk in range(num_full_chunks):
            var idx = chunk * simd_width
            var block = buf.load[simdwidth=simd_width](start_index + idx)
            var result = max(block, zero)
            var mask_block = block.gt(SIMD[Self.dtype, simd_width](0)).select(
                one, zero
            )
            out.store[simdwidth=simd_width](idx, result)
            mask.store[simdwidth=simd_width](idx, mask_block)

        if remainder > 0:
            var start_idx = num_full_chunks * simd_width
            var zero_scalar = Scalar[Self.dtype](0)
            var one_scalar = Scalar[Self.dtype](1)
            for i in range(remainder):
                var idx = start_idx + i
                var val = buf.load[simdwidth=1](start_index + idx)
                var result = max(val, SIMD[Self.dtype, 1](zero_scalar))
                var mask_val = (
                    one_scalar if val[0] > zero_scalar else zero_scalar
                )
                out.store[simdwidth=1](idx, result)
                mask.store[simdwidth=1](idx, SIMD[Self.dtype, 1](mask_val))

        return (out^, mask^)

    @staticmethod
    @always_inline
    def _device_forward(
        ndb: NDBuffer[Self.dtype],
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        var out: NDBuffer[Self.dtype]
        var mask: NDBuffer[Self.dtype]

        comptime if has_accelerator():
            if ndb.is_on_gpu():
                try:
                    var result = UnaryOpsKernel[Self.dtype].launch_with_mask[
                        RELU_FORWARD
                    ](ndb)
                    out = result[0]
                    mask = result[1]
                except e:
                    panic(
                        "ReluNdBuffer forward → GPU launch failed: ",
                        String(e),
                    )
                    out = NDBuffer[Self.dtype].Empty()
                    mask = NDBuffer[Self.dtype].Empty()
            else:
                (out, mask) = Self._cpu_forward(ndb)
        else:
            (out, mask) = Self._cpu_forward(ndb)

        return (out^, mask^)

    @staticmethod
    @always_inline
    def _cpu_forward(
        ndb: NDBuffer[Self.dtype],
    ) -> Tuple[NDBuffer[Self.dtype], NDBuffer[Self.dtype]]:
        if ndb.is_contiguous():
            var start = ndb.offset
            var end = start + ndb.numels()
            var result = Self._buffer_forward(ndb.buffer, start, end)
            var out_ndb = NDBuffer[Self.dtype](result[0], ndb.shape)
            var mask_ndb = NDBuffer[Self.dtype](result[1], ndb.shape)
            return (out_ndb^, mask_ndb^)
        else:
            var numels = ndb.numels()
            var out_buf = Buffer[Self.dtype](numels)
            var mask_buf = Buffer[Self.dtype](numels)
            var zero = Scalar[Self.dtype](0)
            var one = Scalar[Self.dtype](1)
            var index = 0
            for idx in ndb.index_iterator():
                var val = ndb.buffer[idx]
                if val > zero:
                    out_buf[index] = val
                    mask_buf[index] = one
                else:
                    out_buf[index] = zero
                    mask_buf[index] = zero
                index += 1
            return (
                NDBuffer[Self.dtype](out_buf^, ndb.shape),
                NDBuffer[Self.dtype](mask_buf^, ndb.shape),
            )
