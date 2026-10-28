from tenmo.tensor import Tensor
from tenmo.topk import TopK
from tenmo.shared.shapes import Shape
from tenmo.shared.mnemonics import DEFAULT_INDEX_DTYPE
from std.testing import assert_true, TestSuite


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
    print("\nAll topk tests passed!")


def test_topk_1d_basic() raises:
    print("test_topk_1d_basic")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([0.5, 2.3, 1.1, 2.3])
    var (vals, idxs) = t.topk(2)
    assert_true(vals == Tensor[dtype].d1([2.3, 2.3]))
    assert_true(idxs == Tensor[DEFAULT_INDEX_DTYPE].d1([1, 3]))
    print("Passed")


def test_topk_2d_axis() raises:
    print("test_topk_2d_axis")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 5.0, 3.0], [7.0, 2.0, 8.0]])
    var (vals, idxs) = t.topk(2, axis=1)
    assert_true(vals.shape() == Shape(2, 2))
    assert_true(vals == Tensor[dtype].d2([[5.0, 3.0], [8.0, 7.0]]))
    assert_true(idxs == Tensor[DEFAULT_INDEX_DTYPE].d2([[1, 2], [2, 0]]))
    print("Passed")


def test_topk_largest_false() raises:
    print("test_topk_largest_false")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([5.0, 1.0, 3.0, 2.0])
    var (vals, idxs) = t.topk(2, largest=False)
    assert_true(vals == Tensor[dtype].d1([1.0, 2.0]))
    assert_true(idxs == Tensor[DEFAULT_INDEX_DTYPE].d1([1, 3]))
    print("Passed")


def test_topk_k_equals_dim() raises:
    print("test_topk_k_equals_dim")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d1([3.0, 1.0, 2.0])
    var (vals, idxs) = t.topk(3)
    assert_true(vals == Tensor[dtype].d1([3.0, 2.0, 1.0]))
    assert_true(idxs == Tensor[DEFAULT_INDEX_DTYPE].d1([0, 2, 1]))
    print("Passed")


def test_topk_axis_negative() raises:
    print("test_topk_axis_negative")
    comptime dtype = DType.float32
    var t = Tensor[dtype].d2([[1.0, 5.0, 3.0], [7.0, 2.0, 8.0]])
    var (vals, _) = t.topk(1, axis=-1)
    assert_true(vals.shape() == Shape(2, 1))
    assert_true(vals == Tensor[dtype].d2([[5.0], [8.0]]))
    print("Passed")
