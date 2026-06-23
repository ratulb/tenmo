from std.gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from std.atomic import Atomic, Ordering
from std.sys import simd_width_of

from tenmo.device import DeviceState
from tenmo.ndbuffer import NDBuffer
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE


def accuracy_kernel[
    dtype: DType,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    result: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    pred: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    labels: UnsafePointer[Scalar[index_dtype], ImmutAnyOrigin],
    batch_size: Int,
    num_classes: Int,
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid >= batch_size:
        return

    if tid == 0:
        result.store(0)
    barrier()

    var base = tid * num_classes
    var max_val = pred[base]
    var max_idx = 0
    for j in range(1, num_classes):
        var val = pred[base + j]
        if val > max_val:
            max_val = val
            max_idx = j

    if max_idx == Int(labels[tid]):
        _ = Atomic.fetch_add[ordering=Ordering.RELAXED](result, 1)


@fieldwise_init
struct AccuracyGpu[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE](
    ImplicitlyCopyable, RegisterPassable
):
    @staticmethod
    def launch(
        pred: NDBuffer[Self.dtype],
        labels: NDBuffer[Self.index_dtype],
        sync: Bool = False,
    ) raises -> Int:
        var batch_size = pred.shape[0]
        var num_classes = pred.shape[1]

        ref pred_dev = pred.device_state.value()
        ref gpu = pred_dev.get_gpu()
        var ctx = gpu[]

        var result_buffer = ctx.enqueue_create_buffer[DType.int64](1)

        ref pred_buf = pred_dev.device_buffer()
        ref labels_dev = labels.device_state.value()
        ref labels_buf = labels_dev.device_buffer()

        var compiled = ctx.compile_function[
            accuracy_kernel[Self.dtype, Self.index_dtype],
            accuracy_kernel[Self.dtype, Self.index_dtype],
        ]()

        ctx.enqueue_function(
            compiled,
            result_buffer,
            pred_buf,
            labels_buf,
            batch_size,
            num_classes,
            grid_dim=1,
            block_dim=batch_size,
        )

        if sync:
            ctx.synchronize()

        with result_buffer.map_to_host() as host_buf:
            return Int(host_buf[0])
