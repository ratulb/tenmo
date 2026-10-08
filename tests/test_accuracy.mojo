from tenmo.tensor import Tensor
from tenmo.accuracy import Accuracy
from tenmo.shared.mnemonics import DEFAULT_INDEX_DTYPE
from tenmo.shared.shapes import Shape
from std.testing import assert_equal, TestSuite
from std.sys import has_accelerator


# ── compute (existing API, now returns Float64) ─────────────────────────


def test_perfect() raises:
    var pred = Tensor[DType.float32].d2([[0.1, 0.9], [0.8, 0.2]])
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 1.0)


def test_all_wrong() raises:
    var pred = Tensor[DType.float32].d2([[0.9, 0.1], [0.2, 0.8]])
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 0.0)


def test_mixed() raises:
    var pred = Tensor[DType.float32].d2([[0.1, 0.9], [0.9, 0.1], [0.3, 0.7]])
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0, 0])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 2.0 / 3.0)


def test_single_sample() raises:
    var pred = Tensor[DType.float32].d2([[0.1, 0.9]])
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 1.0)


def test_3_classes() raises:
    var pred = Tensor[DType.float32].d2([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 1.0)


def test_5_classes() raises:
    var pred = Tensor[DType.float32].d2(
        [
            [0.9, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.9, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9],
        ]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1, 2, 3, 4])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 1.0)


def test_large_batch() raises:
    var batch_size = 1000
    var num_classes = 10
    var pred_data = Tensor[DType.float32].ones(Shape(batch_size, num_classes))
    var target_data = Tensor[DEFAULT_INDEX_DTYPE].zeros(Shape(batch_size))
    for i in range(batch_size):
        var correct_class = i % num_classes
        target_data[i] = Int64(correct_class)
        for j in range(num_classes):
            if j == correct_class:
                pred_data[i, j] = 0.9
            else:
                pred_data[i, j] = 0.1
    var acc = Accuracy[DType.float32].compute(pred_data, target_data)
    assert_equal(acc, 1.0)


def test_float64() raises:
    var pred = Tensor[DType.float64].d2([[0.1, 0.9], [0.8, 0.2]])
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
    var acc = Accuracy[DType.float64].compute(pred, target)
    assert_equal(acc, 1.0)


def test_tie_first_wins() raises:
    var pred = Tensor[DType.float32].d2([[0.5, 0.5]])
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 1.0)


def test_tie_target_second() raises:
    var pred = Tensor[DType.float32].d2([[0.5, 0.5]])
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 0.0)


def test_batch_non_pow2() raises:
    var pred = Tensor[DType.float32].d2(
        [[0.1, 0.9], [0.8, 0.2], [0.9, 0.1], [0.3, 0.7], [0.6, 0.4]]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0, 0, 0, 0])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 0.8)


def test_class_non_pow2() raises:
    var pred = Tensor[DType.float32].d2(
        [
            [0.1, 0.2, 0.6, 0.0, 0.0, 0.0, 0.1],
            [0.7, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.7],
            [0.1, 0.6, 0.1, 0.0, 0.0, 0.1, 0.1],
        ]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d1([2, 0, 6, 1])
    var acc = Accuracy[DType.float32].compute(pred, target)
    assert_equal(acc, 1.0)


# ── token_accuracy ──────────────────────────────────────────────────────


def test_token_accuracy_perfect() raises:
    var pred = Tensor[DType.float32].d3(
        [
            [[0.1, 0.9, 0.0], [0.9, 0.0, 0.1]],
            [[0.8, 0.1, 0.1], [0.0, 0.9, 0.1]],
        ]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d2([[1, 0], [0, 1]])
    var acc = Accuracy[DType.float32].token_accuracy(pred, target)
    assert_equal(acc, 1.0)


def test_token_accuracy_half() raises:
    var pred = Tensor[DType.float32].d3(
        [
            [[0.1, 0.9, 0.0], [0.9, 0.0, 0.1]],
            [[0.8, 0.1, 0.1], [0.0, 0.9, 0.1]],
        ]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d2([[1, 0], [0, 0]])
    var acc = Accuracy[DType.float32].token_accuracy(pred, target)
    assert_equal(acc, 0.75)


def test_token_accuracy_all_wrong() raises:
    var pred = Tensor[DType.float32].d3(
        [
            [[0.1, 0.9, 0.0], [0.9, 0.0, 0.1]],
            [[0.8, 0.1, 0.1], [0.0, 0.9, 0.1]],
        ]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d2([[0, 1], [1, 0]])
    var acc = Accuracy[DType.float32].token_accuracy(pred, target)
    assert_equal(acc, 0.0)


# ── sequence_accuracy ────────────────────────────────────────────────────


def test_seq_accuracy_perfect() raises:
    var pred = Tensor[DType.float32].d3(
        [
            [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]],
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
        ]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d2([[0, 1, 2], [0, 1, 2]])
    var acc = Accuracy[DType.float32].sequence_accuracy(pred, target)
    assert_equal(acc, 1.0)


def test_seq_accuracy_half() raises:
    var pred = Tensor[DType.float32].d3(
        [
            [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]],
            [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]],
        ]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d2([[0, 1, 2], [0, 1, 2]])
    var acc = Accuracy[DType.float32].sequence_accuracy(pred, target)
    assert_equal(acc, 0.5)


def test_seq_accuracy_none() raises:
    var pred = Tensor[DType.float32].d3(
        [
            [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1]],
            [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.1, 0.1, 0.8]],
        ]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d2([[0, 1, 2], [0, 1, 2]])
    var acc = Accuracy[DType.float32].sequence_accuracy(pred, target)
    assert_equal(acc, 0.0)


def test_token_vs_seq_mismatch() raises:
    # 2 sequences of 4 positions, 5 classes.
    # Seq 0: first 3 correct, last wrong → tok=3/4, seq=0
    # Seq 1: all 4 correct → tok=4/4, seq=1
    # Overall: tok=7/8=0.875, seq=1/2=0.5
    var pred = Tensor[DType.float32].d3(
        [
            [
                [0.8, 0.1, 0.0, 0.0, 0.1],
                [0.1, 0.8, 0.0, 0.0, 0.1],
                [0.1, 0.0, 0.8, 0.0, 0.1],
                [0.1, 0.0, 0.0, 0.1, 0.8],
            ],
            [
                [0.9, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.9, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.9, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.9, 0.1],
            ],
        ]
    )
    var target = Tensor[DEFAULT_INDEX_DTYPE].d2(
        [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
    )
    var tok_acc = Accuracy[DType.float32].token_accuracy(pred, target)
    var seq_acc = Accuracy[DType.float32].sequence_accuracy(pred, target)
    assert_equal(tok_acc, 0.875)
    assert_equal(seq_acc, 0.5)


# ── GPU parity helpers and tests ─────────────────────────────────────────


def gpu_parity(
    pred: Tensor[DType.float32],
    target: Tensor[DEFAULT_INDEX_DTYPE],
    expected: Float64,
) raises:
    comptime if has_accelerator():
        var pred_gpu = pred.to_gpu()
        var target_gpu = target.to_gpu()
        var gpu_result = Accuracy[DType.float32].compute(pred_gpu, target_gpu)
        assert_equal(gpu_result, expected)


def test_gpu_perfect() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.1, 0.9], [0.8, 0.2]])
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
        gpu_parity(pred, target, 1.0)


def test_gpu_all_wrong() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.9, 0.1], [0.2, 0.8]])
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
        gpu_parity(pred, target, 0.0)


def test_gpu_mixed() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2(
            [[0.1, 0.9], [0.9, 0.1], [0.3, 0.7]]
        )
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0, 0])
        gpu_parity(pred, target, 2.0 / 3.0)


def test_gpu_single_sample() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.1, 0.9]])
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1])
        gpu_parity(pred, target, 1.0)


def test_gpu_3_classes() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
        gpu_parity(pred, target, 1.0)


def test_gpu_5_classes() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2(
            [
                [0.9, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.9, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.9],
            ]
        )
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([0, 1, 2, 3, 4])
        gpu_parity(pred, target, 1.0)


def test_gpu_large_batch() raises:
    comptime if has_accelerator():
        var batch_size = 1000
        var num_classes = 10
        var pred_data = Tensor[DType.float32].ones(
            Shape(batch_size, num_classes)
        )
        var target_data = Tensor[DEFAULT_INDEX_DTYPE].zeros(Shape(batch_size))
        for i in range(batch_size):
            var correct_class = i % num_classes
            target_data[i] = Int64(correct_class)
            for j in range(num_classes):
                if j == correct_class:
                    pred_data[i, j] = 0.9
                else:
                    pred_data[i, j] = 0.1
        gpu_parity(pred_data, target_data, 1.0)


def test_gpu_float64() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float64].d2([[0.1, 0.9], [0.8, 0.2]])
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
        var cpu_result = Accuracy[DType.float64].compute(pred, target)
        var pred_gpu = pred.to_gpu()
        var target_gpu = target.to_gpu()
        var gpu_result = Accuracy[DType.float64].compute(pred_gpu, target_gpu)
        assert_equal(gpu_result, cpu_result)


def test_pred_gpu_target_cpu() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.1, 0.9], [0.8, 0.2]])
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
        var pred_gpu = pred.to_gpu()
        var result = Accuracy[DType.float32].compute(pred_gpu, target)
        assert_equal(result, 1.0)


def test_pred_cpu_target_gpu() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.1, 0.9], [0.8, 0.2]])
        var target = Tensor[DEFAULT_INDEX_DTYPE].d1([1, 0])
        var target_gpu = target.to_gpu()
        var result = Accuracy[DType.float32].compute(pred, target_gpu)
        assert_equal(result, 1.0)


# ── GPU tests for token_accuracy / sequence_accuracy ────────────────────


def test_gpu_token_accuracy() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d3(
            [
                [[0.1, 0.9, 0.0], [0.9, 0.0, 0.1]],
                [[0.8, 0.1, 0.1], [0.0, 0.9, 0.1]],
            ]
        )
        var target = Tensor[DEFAULT_INDEX_DTYPE].d2([[1, 0], [0, 1]])
        var cpu_acc = Accuracy[DType.float32].token_accuracy(pred, target)
        var pred_gpu = pred.to_gpu()
        var target_gpu = target.to_gpu()
        var gpu_acc = Accuracy[DType.float32].token_accuracy(
            pred_gpu, target_gpu
        )
        assert_equal(gpu_acc, cpu_acc)


def test_gpu_seq_accuracy() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d3(
            [
                [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.1, 0.0, 0.9]],
                [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]],
            ]
        )
        var target = Tensor[DEFAULT_INDEX_DTYPE].d2([[0, 1, 2], [0, 1, 2]])
        var cpu_acc = Accuracy[DType.float32].sequence_accuracy(pred, target)
        var pred_gpu = pred.to_gpu()
        var target_gpu = target.to_gpu()
        var gpu_acc = Accuracy[DType.float32].sequence_accuracy(
            pred_gpu, target_gpu
        )
        assert_equal(gpu_acc, cpu_acc)


# ── GPU offset label views (regression) ─────────────────────────────────
# AccuracyKernel.launch read pred/labels from the buffer base, dropping
# Layout.offset — every offset batch scored the wrong rows. The launcher
# now reads from sub-buffers at the view offsets. Decoy first-half labels
# (all 1s) give 0.4 pre-fix vs 0.7 post-fix, so the test distinguishes.


def test_accuracy_gpu_offset_label_view() raises:
    comptime if not has_accelerator():
        return
    from tenmo.gpu.device import GPU

    var gpu = GPU()
    var full = Tensor[DEFAULT_INDEX_DTYPE].d1(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]
    )
    var full_gpu = full.to_gpu(gpu)
    var labs = full_gpu.slice(start=10, end=20, step=1, axis=0)
    # argmax pattern matches the slice in 7 of 10 rows.
    var pred = Tensor[DType.float32].d2(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.9, 0.1],
            [0.1, 0.9],
        ]
    )
    var acc = Accuracy[DType.float32].compute(pred.to_gpu(gpu), labs)
    assert_equal(acc, 7.0 / 10.0)


# ── main ────────────────────────────────────────────────────────────────


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
