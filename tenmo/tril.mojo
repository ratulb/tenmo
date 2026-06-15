from .tensor import Tensor
from .mnemonics import AddTensor
from .backpropagation import BackwardFnArg, BACKWARD_TRIL, TrilArg
from .gradbox import Gradbox
from .ancestry import Ancestor
from .ndbuffer import NDBuffer
from .common_utils import panic
from std.sys import has_accelerator
from std.sys import simd_width_of
from .kernels.tril_kernel import TrilGpuKernel


@fieldwise_init
struct TrilBackward[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
        retain_graph: Bool = False,
    ):
        ref bwd_arg = output.ancestry().backward_fn_arg().get[TrilArg]()
        var diagonal = bwd_arg.diagonal
        var M = bwd_arg.M
        var N = bwd_arg.N
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)

        var grad_ndb = apply_tril_cpu[Self.dtype](
            gradbox.buffer(), M, N, diagonal
        )
        var gradbox_ancestor = Gradbox[Self.dtype](grad_ndb^)

        if parent.requires_grad:
            parent.update_grad(gradbox_ancestor^, AddTensor, None)
        parent_ids.append(parent._id)

        if not retain_graph:
            gradbox.zero_grad()


@fieldwise_init
struct Tril[dtype: DType](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def forward[
        track_grad: Bool = True,
    ](
        self: Tensor[Self.dtype],
        diagonal: Int = 0,
        requires_grad: Optional[Bool] = None,
        sync: Bool = True,
    ) -> Tensor[Self.dtype]:
        var shape = self.shape()
        var rank = shape.rank()
        if rank < 2:
            panic(
                "tril requires at least 2 dimensions, got rank "
                + String(rank)
            )
        var M = shape[rank - 2]
        var N = shape[rank - 1]

        var ndb: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if self.buffer.is_on_gpu():
                try:
                    ndb = TrilGpuKernel[Self.dtype].launch(
                        self.buffer, diagonal, sync
                    )
                except e:
                    panic(
                        "tril GPU forward failed: " + String(e)
                    )
                    ndb = NDBuffer[Self.dtype].Empty()
            else:
                ndb = apply_tril_cpu[Self.dtype](
                    self.buffer, M, N, diagonal
                )
        else:
            ndb = apply_tril_cpu[Self.dtype](self.buffer, M, N, diagonal)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var arg = TrilArg(diagonal, M, N)
                var backwardFnArg = BackwardFnArg[Self.dtype](
                    BACKWARD_TRIL, arg^
                )
                backwardFnArg.needs_parent_data = False
                out.add_ancestry(backwardFnArg^, self)

        return out^


def apply_tril_cpu[
    dtype: DType,
](
    inp: NDBuffer[dtype],
    M: Int,
    N: Int,
    diagonal: Int,
) -> NDBuffer[dtype]:
    var numels = inp.numels()
    var shape = inp.shape
    var out = NDBuffer[dtype].zeros(shape)
    var batch_stride = M * N

    if inp.is_contiguous():
        var in_ptr = inp.data_ptr().unsafe_mut_cast[True]()
        var in_offset = inp.offset
        var out_ptr = out.data_ptr().unsafe_mut_cast[True]()
        var out_offset = out.offset
        comptime simd_width = simd_width_of[dtype]()

        var i = 0
        while i < numels:
            var chunk_end = min(i + simd_width, numels)
            var simd_width_actual = chunk_end - i
            if simd_width_actual == simd_width:
                var vec = in_ptr.load[width=simd_width](in_offset + i)
                var result = SIMD[dtype, simd_width](0)

                for lane in range(simd_width):
                    var within = (i + lane) % batch_stride
                    var row = within // N
                    var col = within % N
                    if col <= row + diagonal:
                        result[lane] = vec[lane]

                out_ptr.store[width=simd_width](out_offset + i, result)
            else:
                for lane in range(simd_width_actual):
                    var idx = i + lane
                    var within = idx % batch_stride
                    var row = within // N
                    var col = within % N
                    if col <= row + diagonal:
                        out_ptr[out_offset + idx] = in_ptr[in_offset + idx]
            i += simd_width_actual
    else:
        var flat_idx = 0
        for buf_idx in inp.index_iterator():
            var within = flat_idx % batch_stride
            var row = within // N
            var col = within % N
            if col <= row + diagonal:
                out.data_ptr()[out.offset + flat_idx] = inp.buffer[buf_idx]
            flat_idx += 1

    return out^
