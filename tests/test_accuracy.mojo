from tenmo import Tensor, Accuracy
from tenmo.shapes import Shape
from std.testing import assert_equal, TestSuite
from std.sys import has_accelerator


def test_perfect() raises:
    var pred = Tensor[DType.float32].d2(
        [[0.1, 0.9], [0.8, 0.2]]
    )
    var target = Tensor[DType.int32].d1([1, 0])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 2)


def test_all_wrong() raises:
    var pred = Tensor[DType.float32].d2(
        [[0.9, 0.1], [0.2, 0.8]]
    )
    var target = Tensor[DType.int32].d1([1, 0])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 0)


def test_mixed() raises:
    var pred = Tensor[DType.float32].d2(
        [[0.1, 0.9], [0.9, 0.1], [0.3, 0.7]]
    )
    var target = Tensor[DType.int32].d1([1, 0, 0])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 2)


def test_single_sample() raises:
    var pred = Tensor[DType.float32].d2([[0.1, 0.9]])
    var target = Tensor[DType.int32].d1([1])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 1)


def test_3_classes() raises:
    var pred = Tensor[DType.float32].d2(
        [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]
    )
    var target = Tensor[DType.int32].d1([1, 0])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 2)


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
    var target = Tensor[DType.int32].d1([0, 1, 2, 3, 4])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 5)


def test_large_batch() raises:
    var batch_size = 1000
    var num_classes = 10
    var pred_data = Tensor[DType.float32].ones(Shape(batch_size, num_classes))
    var target_data = Tensor[DType.int32].zeros(Shape(batch_size))
    for i in range(batch_size):
        var correct_class = i % num_classes
        target_data[i] = Int32(correct_class)
        for j in range(num_classes):
            if j == correct_class:
                pred_data[i, j] = 0.9
            else:
                pred_data[i, j] = 0.1
    var correct = Accuracy[DType.float32].compute(pred_data, target_data)
    assert_equal(correct, batch_size)


def test_float64() raises:
    var pred = Tensor[DType.float64].d2(
        [[0.1, 0.9], [0.8, 0.2]]
    )
    var target = Tensor[DType.int32].d1([1, 0])
    var correct = Accuracy[DType.float64].compute(pred, target)
    assert_equal(correct, 2)


def test_tie_first_wins() raises:
    var pred = Tensor[DType.float32].d2([[0.5, 0.5]])
    var target = Tensor[DType.int32].d1([0])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 1)


def test_tie_target_second() raises:
    var pred = Tensor[DType.float32].d2([[0.5, 0.5]])
    var target = Tensor[DType.int32].d1([1])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 0)


def test_batch_non_pow2() raises:
    var pred = Tensor[DType.float32].d2(
        [[0.1, 0.9], [0.8, 0.2], [0.9, 0.1], [0.3, 0.7], [0.6, 0.4]]
    )
    var target = Tensor[DType.int32].d1([1, 0, 0, 0, 0])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 4)


def test_class_non_pow2() raises:
    var pred = Tensor[DType.float32].d2(
        [
            [0.1, 0.2, 0.6, 0.0, 0.0, 0.0, 0.1],
            [0.7, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1],
            [0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.7],
            [0.1, 0.6, 0.1, 0.0, 0.0, 0.1, 0.1],
        ]
    )
    var target = Tensor[DType.int32].d1([2, 0, 6, 1])
    var correct = Accuracy[DType.float32].compute(pred, target)
    assert_equal(correct, 4)


def gpu_parity(pred: Tensor[DType.float32], target: Tensor[DType.int32], expected: Int) raises:
    comptime if has_accelerator():
        var pred_gpu = pred.to_gpu()
        var target_gpu = target.to_gpu()
        var gpu_result = Accuracy[DType.float32].compute(pred_gpu, target_gpu)
        assert_equal(gpu_result, expected)


def test_gpu_perfect() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.1, 0.9], [0.8, 0.2]])
        var target = Tensor[DType.int32].d1([1, 0])
        gpu_parity(pred, target, 2)


def test_gpu_all_wrong() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.9, 0.1], [0.2, 0.8]])
        var target = Tensor[DType.int32].d1([1, 0])
        gpu_parity(pred, target, 0)


def test_gpu_mixed() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2(
            [[0.1, 0.9], [0.9, 0.1], [0.3, 0.7]]
        )
        var target = Tensor[DType.int32].d1([1, 0, 0])
        gpu_parity(pred, target, 2)


def test_gpu_single_sample() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.1, 0.9]])
        var target = Tensor[DType.int32].d1([1])
        gpu_parity(pred, target, 1)


def test_gpu_3_classes() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2(
            [[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]
        )
        var target = Tensor[DType.int32].d1([1, 0])
        gpu_parity(pred, target, 2)


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
        var target = Tensor[DType.int32].d1([0, 1, 2, 3, 4])
        gpu_parity(pred, target, 5)


def test_gpu_large_batch() raises:
    comptime if has_accelerator():
        var batch_size = 1000
        var num_classes = 10
        var pred_data = Tensor[DType.float32].ones(Shape(batch_size, num_classes))
        var target_data = Tensor[DType.int32].zeros(Shape(batch_size))
        for i in range(batch_size):
            var correct_class = i % num_classes
            target_data[i] = Int32(correct_class)
            for j in range(num_classes):
                if j == correct_class:
                    pred_data[i, j] = 0.9
                else:
                    pred_data[i, j] = 0.1
        gpu_parity(pred_data, target_data, batch_size)


def test_gpu_float64() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float64].d2([[0.1, 0.9], [0.8, 0.2]])
        var target = Tensor[DType.int32].d1([1, 0])
        var cpu_result = Accuracy[DType.float64].compute(pred, target)
        var pred_gpu = pred.to_gpu()
        var target_gpu = target.to_gpu()
        var gpu_result = Accuracy[DType.float64].compute(pred_gpu, target_gpu)
        assert_equal(gpu_result, cpu_result)


def test_pred_gpu_target_cpu() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.1, 0.9], [0.8, 0.2]])
        var target = Tensor[DType.int32].d1([1, 0])
        var pred_gpu = pred.to_gpu()
        var result = Accuracy[DType.float32].compute(pred_gpu, target)
        assert_equal(result, 2)


def test_pred_cpu_target_gpu() raises:
    comptime if has_accelerator():
        var pred = Tensor[DType.float32].d2([[0.1, 0.9], [0.8, 0.2]])
        var target = Tensor[DType.int32].d1([1, 0])
        var target_gpu = target.to_gpu()
        var result = Accuracy[DType.float32].compute(pred, target_gpu)
        assert_equal(result, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
