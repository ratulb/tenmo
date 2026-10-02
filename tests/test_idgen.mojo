from tenmo.shared.idgen import IDGen
from tenmo.tensor import Tensor
from std.testing import assert_true, TestSuite


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_idgen_returns_increasing_ids() raises:
    """IDGen.generate_id() must strictly increase on every call."""
    print("test_idgen_returns_increasing_ids")
    var prev = IDGen.generate_id()
    for _ in range(100):
        var next_id = IDGen.generate_id()
        assert_true(Int(prev) < Int(next_id))
        prev = next_id


def test_idgen_unique_across_tensors() raises:
    """Every Tensor must receive a distinct id."""
    print("test_idgen_unique_across_tensors")
    var ids = List[UInt]()
    for _ in range(50):
        var t = Tensor[DType.float32].zeros(2, 3)
        ids.append(t.id())
    for i in range(len(ids)):
        for j in range(len(ids)):
            if i != j:
                assert_true(ids[i] != ids[j])


def test_idgen_shared_with_tensor_construction() raises:
    """IDGen and Tensor construction must draw from the same counter."""
    print("test_idgen_shared_with_tensor_construction")
    var before = IDGen.generate_id()
    var t = Tensor[DType.float32].ones(1, 1)
    var after = IDGen.generate_id()
    assert_true(Int(before) < Int(t.id()))
    assert_true(Int(t.id()) < Int(after))
