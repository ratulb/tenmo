from .tensor import Tensor
from .shared.mnemonics import AddTensor
from .backpropagation import ArgumentType, BackwardFn, BackwardFnType

from .gradbox import Gradbox
from .ancestry import Ancestor
from .ndbuffer import NDBuffer
from .shared.panic import panic
from std.sys import has_accelerator
from std.sys import simd_width_of
from std.sys.info import num_physical_cores
from max.algorithm import parallelize
from .kernels.trilu_kernel import TrilKernel


@fieldwise_init
struct TrilArg(ArgumentType):
    var diagonal: Int
    var M: Int
    var N: Int


@fieldwise_init
struct TrilBackward[dtype: DType](
    BackwardFnType, ImplicitlyCopyable, RegisterPassable
):
    comptime datatype = Self.dtype

    @staticmethod
    def backward(
        var output: Ancestor[Self.dtype],
        mut parent_ids: List[UInt],
    ):
        ref bwd_arg = output.ancestry().backward_fn().get[TrilArg]()
        var diagonal = bwd_arg.diagonal
        var M = bwd_arg.M
        var N = bwd_arg.N
        ref gradbox = output.gradients()
        var parent = output.ancestry().get(0)

        var grad_ndb: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if gradbox.is_on_gpu():
                try:
                    var (grad_layout, grad_storage) = TrilKernel[
                        Self.dtype
                    ].launch_backward(
                        gradbox.buffer().layout(),
                        gradbox.buffer().device_state.value(),
                        diagonal,
                    )
                    grad_ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        grad_layout, grad_storage
                    )
                except e:
                    print(e)
                    panic("TrilBackward → GPU backward launch failed")
                    grad_ndb = NDBuffer[Self.dtype].Empty()
            else:
                grad_ndb = apply_tril_cpu[Self.dtype](
                    gradbox.buffer(), M, N, diagonal
                )
        else:
            grad_ndb = apply_tril_cpu[Self.dtype](
                gradbox.buffer(), M, N, diagonal
            )
        var gradbox_ancestor = Gradbox[Self.dtype](grad_ndb^)

        if parent.requires_grad:
            parent.update_grad(gradbox_ancestor^, AddTensor, None)
        # Unconditional: parent_ids is the engine's fanin-completion
        # signal (appended set must equal ancestry set).
        parent_ids.append(parent._id)

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
                "tril requires at least 2 dimensions, got rank " + String(rank)
            )
        var M = shape[rank - 2]
        var N = shape[rank - 1]

        var ndb: NDBuffer[Self.dtype]
        comptime if has_accelerator():
            if self.buffer.is_on_gpu():
                try:
                    var (result_layout, result_storage) = TrilKernel[
                        Self.dtype
                    ].launch(
                        self.buffer.layout(),
                        self.buffer.device_state.value(),
                        diagonal,
                        sync,
                    )
                    ndb = NDBuffer[Self.dtype].with_layout_device_state(
                        result_layout, result_storage
                    )
                except e:
                    panic("tril GPU forward failed: " + String(e))
                    ndb = NDBuffer[Self.dtype].Empty()
            else:
                ndb = apply_tril_cpu[Self.dtype](self.buffer, M, N, diagonal)
        else:
            ndb = apply_tril_cpu[Self.dtype](self.buffer, M, N, diagonal)
        var out = Tensor[Self.dtype](ndb^, requires_grad=False)

        comptime if track_grad:
            var grad_required = requires_grad.or_else(self.requires_grad)
            if grad_required:
                out.requires_grad_(True)
                var arg = TrilArg(diagonal, M, N)
                var backwardFn = BackwardFn(
                    arg^, TrilBackward[Self.dtype]()
                )
                backwardFn.needs_parent_data = False
                out.add_ancestry(backwardFn^, self)

        return out^


def apply_tril_cpu[
    dtype: DType,
](inp: NDBuffer[dtype], M: Int, N: Int, diagonal: Int,) -> NDBuffer[dtype]:
    var in_storage = inp.buffer
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
        var n_rows = numels // N
        var n_threads = num_physical_cores()

        def worker_tril(R: Int) {imm}:
            # Kept region of row R is a contiguous prefix — copy it once, no
            # per-lane div/mod; the rest of the row is already zero.
            var r = R % M
            var kept_end = min(max(r + diagonal + 1, 0), N)
            var row_base = R * N
            var j = 0
            while j + simd_width <= kept_end:
                var val = in_ptr.unsafe_load[width=simd_width](
                    in_offset + row_base + j
                )
                out_ptr.unsafe_store[width=simd_width](
                    out_offset + row_base + j, val
                )
                j += simd_width
            for k in range(j, kept_end):
                var idx = row_base + k
                out_ptr[unsafe_offset=out_offset + idx] = in_ptr[
                    unsafe_offset=in_offset + idx
                ]

        if n_rows >= n_threads and numels >= n_threads * 32768:
            parallelize(worker_tril, n_rows, n_threads)
        else:
            for R in range(n_rows):
                worker_tril(R)
    else:
        var out_offset = out.offset
        var flat_idx = 0
        for buf_idx in inp.index_iterator():
            var within = flat_idx % batch_stride
            var row = within // N
            var col = within % N
            if col <= row + diagonal:
                out.data_ptr()[
                    unsafe_offset=out_offset + flat_idx
                ] = in_storage[buf_idx]
            flat_idx += 1

    return out^
