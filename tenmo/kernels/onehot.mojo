"""Standalone onehot encoding — GPU kernel only.

CPU path lives in NDBuffer.onehot() — this module is a pure GPU launcher.
"""

from max.gpu import block_idx
from ..shared.layout import Layout
from ..gpu.transfer import materialize_contiguous
from ..gpu.device import DeviceState
from ..shared.mnemonics import DEFAULT_INDEX_DTYPE


def onehot_fill_kernel[
    dtype: DType,
    target_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    result: Pointer[Scalar[dtype], MutAnyOrigin],
    indices: Pointer[Scalar[target_dtype], ImmutAnyOrigin],
    M_: Int64,
    C_: Int64,
    ignore_index_: Int64,
):
    """Fill result[row * C + target[row]] = 1 for each valid row.

    One block per row — only thread 0 does work per block.
    Rows where target[row] == ignore_index are skipped (left as zeros).
    """
    var M = Int(M_)
    var C = Int(C_)
    var ignore_index = Int(ignore_index_)
    var row = block_idx.x
    if row >= M:
        return
    var tgt = indices[unsafe_offset=row]
    if tgt == Scalar[target_dtype](ignore_index):
        return
    var c = tgt.__int__()
    if 0 <= c < C:
        result[unsafe_offset=row * C + c] = Scalar[dtype](1)


struct OnehotKernel[dtype: DType, target_dtype: DType = DEFAULT_INDEX_DTYPE]:
    """OnehotKernel encoding GPU launcher.

    Pure GPU kernel — caller (NDBuffer.onehot) handles CPU fallback.
    """

    @staticmethod
    def launch(
        indices_layout: Layout,
        indices_device_state: DeviceState[Self.target_dtype],
        num_classes: Int,
        ignore_index: Int = -1000000,
        sync: Bool = False,
    ) raises -> Tuple[Layout, DeviceState[Self.dtype]]:
        var gpu = indices_device_state.gpu
        var ctx = gpu[]
        var M = indices_layout.shape.num_elements()
        var shape = indices_layout.shape

        var contig = materialize_contiguous(
            indices_device_state,
            indices_layout,
        )

        var out = ctx.enqueue_create_buffer[Self.dtype](M * num_classes)
        out.enqueue_fill(0)

        var kern = ctx.compile_function[
            onehot_fill_kernel[Self.dtype, Self.target_dtype]
        ]()
        ctx.enqueue_function(
            kern,
            out,
            contig.device_buffer(),
            Int64(M),
            Int64(num_classes),
            Int64(ignore_index),
            grid_dim=M,
            block_dim=1,
        )

        var st = DeviceState[Self.dtype](out^, gpu)
        if sync:
            ctx.synchronize()
        return (Layout(shape + [num_classes]), st^)
