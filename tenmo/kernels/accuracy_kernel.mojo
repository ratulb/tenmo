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
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = block_dim.x * grid_dim.x

    if tid == 0:
        result.store(0)
    barrier()

    for i in range(tid, batch_size, stride):
        var base = i * num_classes
        var max_val = pred[base]
        var max_idx = 0
        for j in range(1, num_classes):
            var val = pred[base + j]
            if val > max_val:
                max_val = val
                max_idx = j

        if max_idx == Int(labels[i]):
            _ = Atomic.fetch_add[ordering=Ordering.RELAXED](result, 1)


def sequence_accuracy_kernel[
    dtype: DType,
    index_dtype: DType = DEFAULT_INDEX_DTYPE,
](
    result: UnsafePointer[Scalar[DType.int64], MutAnyOrigin],
    pred: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    labels: UnsafePointer[Scalar[index_dtype], ImmutAnyOrigin],
    batch_size: Int,
    seq_len: Int,
    num_classes: Int,
):
    var tid = thread_idx.x + block_idx.x * block_dim.x
    var stride = block_dim.x * grid_dim.x

    if tid == 0:
        result.store(0)
    barrier()

    for i in range(tid, batch_size, stride):
        var seq_base = i * seq_len * num_classes
        var correct = True
        for t in range(seq_len):
            var pos_base = seq_base + t * num_classes
            var max_val = pred[pos_base]
            var max_idx = 0
            for j in range(1, num_classes):
                var val = pred[pos_base + j]
                if val > max_val:
                    max_val = val
                    max_idx = j
            if max_idx != Int(labels[i * seq_len + t]):
                correct = False
                break
        if correct:
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
        ]()

        var THREADS_PER_BLOCK = 256
        var num_blocks = (
            batch_size + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK

        ctx.enqueue_function(
            compiled,
            result_buffer,
            pred_buf,
            labels_buf,
            batch_size,
            num_classes,
            grid_dim=num_blocks,
            block_dim=THREADS_PER_BLOCK,
        )

        if sync:
            ctx.synchronize()

        with result_buffer.map_to_host() as host_buf:
            return Int(host_buf[0])


@fieldwise_init
struct SequenceAccuracyGpu[
    dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE
](ImplicitlyCopyable, RegisterPassable):
    @staticmethod
    def launch(
        pred: NDBuffer[Self.dtype],
        labels: NDBuffer[Self.index_dtype],
        sync: Bool = False,
    ) raises -> Int:
        var batch_size = pred.shape[0]
        var seq_len = pred.shape[1]
        var num_classes = pred.shape[2]

        ref pred_dev = pred.device_state.value()
        ref gpu = pred_dev.get_gpu()
        var ctx = gpu[]

        var result_buffer = ctx.enqueue_create_buffer[DType.int64](1)

        ref pred_buf = pred_dev.device_buffer()
        ref labels_dev = labels.device_state.value()
        ref labels_buf = labels_dev.device_buffer()

        var compiled = ctx.compile_function[
            sequence_accuracy_kernel[Self.dtype, Self.index_dtype],
        ]()

        var THREADS_PER_BLOCK = 256
        var num_blocks = (
            batch_size + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK

        ctx.enqueue_function(
            compiled,
            result_buffer,
            pred_buf,
            labels_buf,
            batch_size,
            seq_len,
            num_classes,
            grid_dim=num_blocks,
            block_dim=THREADS_PER_BLOCK,
        )

        if sync:
            ctx.synchronize()

        with result_buffer.map_to_host() as host_buf:
            return Int(host_buf[0])
