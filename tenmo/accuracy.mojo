from std.sys import simd_width_of, has_accelerator
from tenmo.tensor import Tensor
from tenmo.ndbuffer import NDBuffer
from tenmo.mnemonics import DEFAULT_INDEX_DTYPE


struct Accuracy[dtype: DType, index_dtype: DType = DEFAULT_INDEX_DTYPE]:
    @staticmethod
    def compute(
        pred: Tensor[Self.dtype],
        target: Tensor[Self.index_dtype],
        sync: Bool = True,
    ) raises -> Float64:
        """Fraction of rows where argmax(pred, axis=-1) matches target.

        pred: (N, C) — logits, class dim last.
        target: (N,) — ground-truth class indices.
        """
        comptime if has_accelerator():
            if pred.is_on_gpu() or target.is_on_gpu():
                return Self._accuracy_gpu(pred, target, sync)
        return Self._accuracy_cpu(pred, target)

    @staticmethod
    def token_accuracy(
        pred: Tensor[Self.dtype],
        target: Tensor[Self.index_dtype],
        sync: Bool = True,
    ) raises -> Float64:
        """Fraction of individual positions correctly predicted.

        pred: (B, T, ..., C) — any shape, class dim last.
        target: (B, T, ...) — matching shape without class dim.
        Both must be on the same device (panics if mixed).
        """
        comptime if has_accelerator():
            if (pred.is_on_gpu() and target.is_on_cpu()) or (
                pred.is_on_cpu() and target.is_on_gpu()
            ):
                panic(
                    "Accuracy.token_accuracy - both tensors must be on the same"
                    " device"
                )
        var num_classes = pred.shape()[pred.shape().ndim() - 1]
        var flat_pred = Tensor[Self.dtype](copy=pred)
        flat_pred = flat_pred.reshape(target.numels(), num_classes)
        var flat_target = Tensor[Self.index_dtype](copy=target)
        flat_target = flat_target.reshape(target.numels())
        return Self.compute(flat_pred, flat_target, sync)

    @staticmethod
    def sequence_accuracy(
        pred: Tensor[Self.dtype],
        target: Tensor[Self.index_dtype],
        sync: Bool = True,
    ) raises -> Float64:
        """Fraction of sequences where ALL positions are correctly predicted.

        pred: (B, T, C) — 3D, class dim last.
        target: (B, T) — matching sequence labels.
        Both must be on the same device (panics if mixed).
        """
        comptime if has_accelerator():
            if pred.is_on_gpu() and target.is_on_gpu():
                var correct = Self._sequence_accuracy_gpu(pred, target, sync)
                return Float64(correct) / Float64(pred.shape()[0])
            elif (pred.is_on_gpu() and target.is_on_cpu()) or (
                pred.is_on_cpu() and target.is_on_gpu()
            ):
                panic(
                    "Accuracy.sequence_accuracy - both tensors must be on the"
                    " same device"
                )
        var correct = Self._sequence_accuracy_cpu(pred, target)
        return Float64(correct) / Float64(pred.shape()[0])

    @staticmethod
    def _accuracy_cpu(
        pred: Tensor[Self.dtype],
        target: Tensor[Self.index_dtype],
    ) raises -> Float64:
        var pred_ndb = pred.buffer
        var tgt_ndb = target.buffer
        var batch_size = pred_ndb.shape[0]
        var num_classes = pred_ndb.shape[1]
        var s0 = pred_ndb.strides[0]
        var s1 = pred_ndb.strides[1]
        var off = pred_ndb.offset
        var buf = pred_ndb.buffer
        var ts0 = tgt_ndb.strides[0]
        var toff = tgt_ndb.offset
        var tbuf = tgt_ndb.buffer
        var correct = 0
        if s0 == num_classes and s1 == 1:
            comptime simd_w = simd_width_of[Self.dtype]()
            for row in range(batch_size):
                var base = off + row * num_classes
                var max_val = buf[base]
                var max_idx = 0
                var j = 0
                while j + simd_w <= num_classes:
                    var chunk = buf.load[simdwidth=simd_w](base + j)
                    for k in range(simd_w):
                        if chunk[k] > max_val:
                            (max_val, max_idx) = (chunk[k], j + k)
                    j += simd_w
                for k in range(j, num_classes):
                    var val = buf[base + k]
                    if val > max_val:
                        (max_val, max_idx) = (val, k)
                if max_idx == Int(tbuf[toff + row * ts0]):
                    correct += 1
        else:
            for row in range(batch_size):
                var base = off + row * s0
                var max_val = buf[base]
                var max_idx = 0
                for j in range(1, num_classes):
                    var val = buf[base + j * s1]
                    if val > max_val:
                        (max_val, max_idx) = (val, j)
                if max_idx == Int(tbuf[toff + row * ts0]):
                    correct += 1
        return Float64(correct) / Float64(batch_size)

    @staticmethod
    def _accuracy_gpu(
        pred_in: Tensor[Self.dtype],
        target_in: Tensor[Self.index_dtype],
        sync: Bool,
    ) raises -> Float64:
        from tenmo.kernels import AccuracyGpu

        var pred = pred_in
        var target = target_in
        if not pred.is_on_gpu():
            pred = pred.to_gpu(sync=sync)
        if not target.is_on_gpu():
            target = target.to_gpu(sync=sync)

        var correct = AccuracyGpu[Self.dtype, Self.index_dtype].launch(
            pred.buffer, target.buffer, sync=sync
        )
        return Float64(correct) / Float64(pred.shape()[0])

    @staticmethod
    def _sequence_accuracy_cpu(
        pred: Tensor[Self.dtype],
        target: Tensor[Self.index_dtype],
    ) raises -> Int:
        var preds = pred.argmax[Self.index_dtype](axis=-1)
        var B = preds.shape()[0]
        var T = preds.shape()[1]
        var correct = 0
        for b in range(B):
            var ok = True
            for t in range(T):
                if Int(preds[b, t]) != Int(target[b, t]):
                    ok = False
                    break
            if ok:
                correct += 1
        return correct

    @staticmethod
    def _sequence_accuracy_gpu(
        pred: Tensor[Self.dtype],
        target: Tensor[Self.index_dtype],
        sync: Bool,
    ) raises -> Int:
        from tenmo.kernels import SequenceAccuracyGpu

        return SequenceAccuracyGpu[Self.dtype, Self.index_dtype].launch(
            pred.buffer, target.buffer, sync=sync
        )
