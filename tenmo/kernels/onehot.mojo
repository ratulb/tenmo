"""Standalone onehot encoding — CPU and GPU."""

from std.sys import has_accelerator
from std.gpu import block_idx
from tenmo.ndbuffer import NDBuffer
from tenmo.device import Device, DeviceState, CPU
from tenmo.shapes import Shape
from tenmo.common_utils import DEFAULT_INDEX_DTYPE
from tenmo.common_utils import panic

def onehot_fill_kernel[
    dtype: DType,
    target_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    result: UnsafePointer[Scalar[dtype], MutUnsafeAnyOrigin],
    indices: UnsafePointer[Scalar[target_dtype], ImmutUnsafeAnyOrigin],
    M: Int,
    C: Int,
    ignore_index: Int,
):
    """Fill result[row * C + target[row]] = 1 for each valid row.

    One block per row — only thread 0 does work per block.
    Rows where target[row] == ignore_index are skipped (left as zeros).
    """
    var row = block_idx.x
    if row >= M:
        return
    var tgt = indices[row]
    if tgt == Scalar[target_dtype](ignore_index):
        return
    var c = tgt.__int__()
    if 0 <= c < C:
        result[row * C + c] = Scalar[dtype](1)


struct Onehot[dtype: DType, target_dtype: DType = DEFAULT_INDEX_DTYPE]:
    """Onehot encoding launcher — CPU and GPU."""

    @staticmethod
    def launch(
        indices: NDBuffer[Self.target_dtype],
        num_classes: Int,
        device: Optional[Device] = None,
        ignore_index: Optional[Int] = None,
    ) -> NDBuffer[Self.dtype]:
        ref shape = indices.shape
        ref target_device = device.or_else(indices.device())

        # ── GPU path ──
        comptime if has_accelerator():
            if target_device.is_gpu():
                try:
                    var (_, indices_gpu) = indices.to_device(
                        target_device, sync=True
                    )
                    ref ds = indices_gpu.device_state.value()
                    var ctx = ds.gpu[]
                    var M = indices_gpu.shape.num_elements()
                    var flat = indices_gpu.reshape(Shape(M))
                    var contig = flat.contiguous_device_state()

                    var out = ctx.enqueue_create_buffer[Self.dtype](
                        M * num_classes
                    )
                    out.enqueue_fill(0)

                    var ign = ignore_index.or_else(-1000000)
                    var kern = ctx.compile_function[
                        onehot_fill_kernel[Self.dtype, Self.target_dtype]
                    ]()
                    ctx.enqueue_function(
                        kern,
                        out,
                        contig.device_buffer(),
                        M,
                        num_classes,
                        ign,
                        grid_dim=M,
                        block_dim=1,
                    )

                    var st = DeviceState[Self.dtype](out^, ds.gpu)
                    var ndb = NDBuffer[Self.dtype].with_device_state(
                        st^, Shape(M, num_classes)
                    )
                    return ndb.reshape(shape + [num_classes])
                except e:
                    print(
                        "Onehot GPU launch failed: ",
                        e,
                        " — falling back to CPU",
                    )

        # ── CPU path ──
        var result = NDBuffer[Self.dtype].zeros(
            shape + [num_classes], device=CPU().into()
        )
        var ign = ignore_index.or_else(-1000000)
        for coord in shape:
            var ci = indices[coord].__int__()
            if ignore_index and ci == ign:
                continue
            if ci < 0 or ci >= num_classes:
                panic(
                    "Onehot: invalid class",
                    String(ci),
                    "at coordinate",
                    String(coord),
                )
            result[coord + ci] = Scalar[Self.dtype](1)
        return result^
