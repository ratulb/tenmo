from tenmo.tensor import Tensor
from tenmo.net import ModuleList, Sequential, Linear, ReLU, Sigmoid
from tenmo.shapes import Shape
from tenmo.mnemonics import LINEAR
from std.sys import has_accelerator
from std.testing import assert_true, assert_equal, TestSuite


def test_empty() raises:
    var ml = ModuleList[DType.float32]()
    assert_equal(len(ml), 0)
    assert_equal(ml.num_parameters(), 0)


def test_append() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](10, 5).into())
    ml.append(ReLU[DType.float32]().into())
    assert_equal(len(ml), 2)
    assert_equal(ml.num_parameters(), 55)  # weight(10*5) + bias(5)


def test_extend() raises:
    var ml = ModuleList[DType.float32]()
    ml.extend(
        Linear[DType.float32](10, 5).into(), ReLU[DType.float32]().into()
    )
    assert_equal(len(ml), 2)
    assert_equal(ml.num_parameters(), 55)


def test_insert() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(ReLU[DType.float32]().into())
    ml.insert(0, Linear[DType.float32](10, 5).into())
    assert_equal(len(ml), 2)
    assert_equal(ml.num_parameters(), 55)


def test_module_access() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](10, 5).into())
    assert_equal(ml.modules[0].tag, LINEAR)


def test_module_mutation() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(ReLU[DType.float32]().into())
    ml.modules[0] = Linear[DType.float32](10, 5).into()
    assert_equal(len(ml), 1)
    assert_equal(ml.num_parameters(), 55)


def test_forward_iteration() raises:
    var ml = ModuleList[DType.float32]()
    ml.extend(
        Linear[DType.float32](10, 5).into(),
        ReLU[DType.float32]().into(),
        Linear[DType.float32](5, 2).into(),
    )
    var count = 0
    var it = ml.__iter__()
    while it.__has_next__():
        _ = it.__next__()
        count += 1
    assert_equal(count, 3)


def test_mutable_iteration() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](10, 5).into())
    var count = 0
    var it = ml.__iter__()
    while it.__has_next__():
        count += 1
        _ = it.__next__()
    assert_equal(count, 1)


def test_parameters() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](4, 3).into())
    ml.append(Linear[DType.float32](3, 2).into())
    var params = ml.parameters()
    assert_equal(len(params), 4)  # 2 weights + 2 biases


def test_named_parameters() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](4, 3).into())
    ml.append(ReLU[DType.float32]().into())
    var params = ml.named_parameters("")
    assert_equal(len(params), 2)
    assert_equal(params[0].name, "0.weight")
    assert_equal(params[1].name, "0.bias")


def test_named_parameters_with_prefix() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](4, 3).into())
    var params = ml.named_parameters("features.")
    assert_equal(len(params), 2)
    assert_equal(params[0].name, "features.0.weight")
    assert_equal(params[1].name, "features.0.bias")


def test_train_eval() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](4, 3).into())
    ml.append(Sigmoid[DType.float32]().into())
    ml.eval()
    ml.train()
    assert_equal(len(ml), 2)


def test_zero_grad() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](4, 3).into())
    ml.zero_grad()
    assert_equal(len(ml), 1)


def test_multiple_append() raises:
    var ml = ModuleList[DType.float32]()
    for i in range(5):
        ml.append(Linear[DType.float32](i + 1, i + 2).into())
    assert_equal(len(ml), 5)


def test_iter_len_has_next() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](4, 3).into())
    ml.append(Linear[DType.float32](3, 2).into())
    var it = ml.__iter__()
    assert_equal(it.__len__(), 2)
    assert_true(it.__has_next__())


def test_iter_remaining() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](4, 3).into())
    ml.append(Linear[DType.float32](3, 2).into())
    var it = ml.__iter__()
    assert_equal(it.__len__(), 2)
    _ = it.__next__()
    assert_equal(it.__len__(), 1)
    _ = it.__next__()
    assert_equal(it.__len__(), 0)
    assert_true(not it.__has_next__())


def test_to_cpu() raises:
    var ml = ModuleList[DType.float32]()
    ml.append(Linear[DType.float32](4, 3).into())
    comptime if has_accelerator():
        var ml_cpu = ml.to_cpu()
        assert_equal(len(ml_cpu), 1)
        assert_equal(ml_cpu.num_parameters(), 15)


def test_for_loop_ml() raises:
    """Verify for-loop over ModuleList (Iterable trait)."""
    var ml = ModuleList[DType.float32]()
    ml.extend(
        Linear[DType.float32](10, 5).into(),
        ReLU[DType.float32]().into(),
        Linear[DType.float32](5, 2).into(),
    )
    var count = 0
    for _ in ml:
        count += 1
    assert_equal(count, 3)


def test_for_loop_iterator() raises:
    """Verify for-loop over ModuleListIterator (Iterator trait)."""
    var ml = ModuleList[DType.float32]()
    ml.extend(
        Linear[DType.float32](10, 5).into(),
        ReLU[DType.float32]().into(),
    )
    var it = ml.__iter__()
    var count = 0
    for m in it:
        count += 1
        assert_true(m.tag >= 0)
    assert_equal(count, 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
